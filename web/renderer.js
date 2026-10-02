// webgpu renderer: the vulkan viewer's pipeline minus what webgpu lacks (compute.wgsl). per
// frame, twice for occlusion:
//   instance_cull -> expand -> cluster_cull -> args_draw
//   -> software raster (depth, then ids) and hardware raster -> depth pyramid
// then shadow pages (vsm.wgsl) -> shade -> taa -> blit.
import { frustumPlanes, invert, lookTo, mul, perspectiveReverseZ } from './math.js';
import { Streamer } from './streamer.js';

const MAX_WORK = 1 << 20;
const MAX_VISIBLE = 1 << 20;
const MAX_REQUESTS = 1 << 13;
const FRAME_BYTES = 608;
const PASS_STRIDE = 256;  // dynamic uniform offsets must be multiples of this
const MAX_HZB_LEVELS = 16;
const STORAGE_BUFFERS_NEEDED = 16;
// vsm_common.wgsl's and vsm.wgsl's sizes.
const VSM_SIDE = 32, VSM_PHYS = 49152, VSM_HEADER = 872, VSM_MOVING_LIST = 62312, VSM_SLOTS = 12288;
const VSM_WORK_WORDS = 2359300 + 2 * 1048576;  // VW_VISIBLE + 2 * VSM_MAX_VISIBLE
const FLAG_MOVING = 256;

export const FLAG_CONE = 1, FLAG_FRUSTUM = 2, FLAG_SOFTWARE = 4, FLAG_SHADOWS = 16, FLAG_OCCLUSION = 32, FLAG_AO = 512,
  FLAG_SOFT_SHADOWS = 1024;
export const FLAG_TAA = 128;  // cpu only
const FLAG_PREV_VALID = 64;

async function source(name) {
  const r = await fetch(name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.text();
}

export class Renderer {
  static async create(canvas) {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    // the argument-writing culling passes bind 16 storage buffers in compute; default limit
    // 8, desktop adapters allow 16.
    if (adapter.limits.maxStorageBuffersPerShaderStage < STORAGE_BUFFERS_NEEDED)
      throw new Error(`this GPU allows ${adapter.limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; the demo needs ${STORAGE_BUFFERS_NEEDED}`);
    const timestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: timestamps ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: STORAGE_BUFFERS_NEEDED,
      },
    });
    const r = new Renderer();
    r.device = device;
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));
    device.lost.then((info) => console.error('WebGPU device lost:', info.message));
    r.adapterInfo = adapter.info || {};
    r.timestamps = timestamps;
    r.canvas = canvas;
    r.context = canvas?.getContext('webgpu');
    r.format = navigator.gpu.getPreferredCanvasFormat();
    r.context?.configure({ device, format: r.format, alphaMode: 'opaque' });
    const common = await source('common.wgsl') + await source('vsm_common.wgsl');
    r.computeModule = device.createShaderModule({ code: common + await source('compute.wgsl') });
    r.vsmModule = device.createShaderModule({ code: common + await source('vsm.wgsl') });
    r.aoModule = device.createShaderModule({ code: common + await source('ao.wgsl') });
    r.rasterModule = device.createShaderModule({ code: common + await source('raster.wgsl') });
    for (const m of [r.computeModule, r.rasterModule, r.vsmModule, r.aoModule]) {
      const info = await m.getCompilationInfo();
      const errors = info.messages.filter((x) => x.type === 'error');
      if (errors.length) throw new Error(errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
    }
    r.createLayouts();
    return r;
  }

  createLayouts() {
    const d = this.device;
    const C = GPUShaderStage.COMPUTE, V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
    const ro = (binding, visibility) => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
    const rw = (binding) => ({ binding, visibility: C, buffer: { type: 'storage' } });
    this.sceneLayout = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: C | V | F, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 6, 7].map((b) => ro(b, C | V))],
    });
    const passInfo = (binding, visibility) => ({ binding, visibility, buffer: { type: 'uniform', hasDynamicOffset: true } });
    this.workLayout = d.createBindGroupLayout({
      entries: [...[0, 1, 2, 3, 4, 6, 7, 12].map(rw), passInfo(9, C)],
    });
    this.imageLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: C, texture: { sampleType: 'uint' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
        { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },  // shadow page entries
        { binding: 4, visibility: C, buffer: { type: 'read-only-storage' } },  // and their atlas
        { binding: 5, visibility: C, texture: { sampleType: 'unfilterable-float' } },
        { binding: 13, visibility: C, texture: { sampleType: 'unfilterable-float' } },  // ambient occlusion
      ],
    });
    this.hzbLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 6, visibility: C, texture: { sampleType: 'unfilterable-float' } },
        { binding: 7, visibility: C, storageTexture: { access: 'write-only', format: 'r32float' } },
      ],
    });
    this.argsLayout = d.createBindGroupLayout({ entries: [rw(0)] });
    this.rasterLayout = d.createBindGroupLayout({ entries: [ro(0, V), passInfo(2, V), ro(3, V)] });
    this.blitLayout = d.createBindGroupLayout({
      entries: [{ binding: 1, visibility: F, texture: { sampleType: 'float' } }],
    });

    const layout3 = d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.workLayout, this.imageLayout] });
    const layout4 = d.createPipelineLayout({
      bindGroupLayouts: [this.sceneLayout, this.workLayout, this.imageLayout, this.argsLayout],
    });
    const compute = (entryPoint, layout) => d.createComputePipeline({ layout, compute: { module: this.computeModule, entryPoint } });
    this.instanceCull = compute('instance_cull', layout3);
    this.argsBig = compute('args_big', layout4);
    this.expand = compute('expand', layout3);
    this.argsCull = compute('args_cull', layout4);
    this.clusterCull = compute('cluster_cull', layout3);
    this.argsDraw = compute('args_draw', layout4);
    this.swDepth = compute('sw_depth_pass', layout3);
    this.swId = compute('sw_id_pass', layout3);
    this.shade = compute('shade', layout3);
    const hzbLayout = d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.workLayout, this.hzbLayout] });
    this.taaLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: C, texture: { sampleType: 'uint' } },
        { binding: 8, visibility: C, texture: { sampleType: 'float' } },
        { binding: 9, visibility: C, texture: { sampleType: 'float' } },
        { binding: 10, visibility: C, sampler: { type: 'filtering' } },
        { binding: 11, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 12, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      ],
    });
    this.taa = compute('taa', d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.workLayout, this.taaLayout] }));
    this.historySampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.hzbFirst = compute('hzb_first', hzbLayout);
    this.hzbDown = compute('hzb_down', hzbLayout);

    this.raster = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.rasterLayout] }),
      vertex: { module: this.rasterModule, entryPoint: 'vs' },
      fragment: { module: this.rasterModule, entryPoint: 'fs', targets: [{ format: 'r32uint' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    // virtual shadow map passes bind the scene, the shadow pages and the streaming buffers
    // they ask through; argument writers also bind the arguments.
    const rwStorage = (binding) => ({ binding, visibility: C, buffer: { type: 'storage' } });
    this.vsmLayout = d.createBindGroupLayout({
      entries: [rwStorage(0), rwStorage(1), rwStorage(2), rwStorage(3),
        { binding: 4, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } }, rwStorage(6), rwStorage(7)],
    });
    this.vsmArgsLayout = d.createBindGroupLayout({ entries: [rwStorage(0)] });
    // ambient occlusion (ao.wgsl): the depth chain's first level, its later levels, the pass.
    const unfilterable = (binding) => ({ binding, visibility: C, texture: { sampleType: 'unfilterable-float' } });
    const r32 = (binding) => ({ binding, visibility: C, storageTexture: { access: 'write-only', format: 'r32float' } });
    this.aoFirstLayout = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } }, r32(2)],
    });
    this.aoDownLayout = d.createBindGroupLayout({ entries: [r32(2), unfilterable(3)] });
    this.aoLayout = d.createBindGroupLayout({ entries: [unfilterable(4), r32(5)] });
    const aoPipeline = (entryPoint, layout) => d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, layout] }),
      compute: { module: this.aoModule, entryPoint },
    });
    this.aoDepthFirst = aoPipeline('ao_depth_first', this.aoFirstLayout);
    this.aoDepthDown = aoPipeline('ao_depth_down', this.aoDownLayout);
    this.aoPass = aoPipeline('ao', this.aoLayout);
    const vsmPipeline = (entryPoint, args = false) => d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.vsmLayout, ...(args ? [this.vsmArgsLayout] : [])] }),
      compute: { module: this.vsmModule, entryPoint },
    });
    this.vsm = {};
    for (const e of ['mark', 'invalidate', 'alloc_slots', 'alloc_phys', 'alloc_assign', 'clear', 'instance', 'expand', 'cluster', 'raster'])
      this.vsm[e] = vsmPipeline(`vsm_${e}`);
    for (const e of ['args_alloc', 'args_expand', 'args_cull', 'args_raster']) this.vsm[e] = vsmPipeline(`vsm_${e}`, true);
    this.blit = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [d.createBindGroupLayout({ entries: [] }), this.blitLayout] }),
      vertex: { module: this.rasterModule, entryPoint: 'blit_vs' },
      fragment: { module: this.rasterModule, entryPoint: 'blit_fs', targets: [{ format: this.format }] },
    });

    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    this.frameBuffer = d.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | CD });
    // shadow pages: all slots and physical pages empty (all ones is VSM_NONE), the atlas's
    // two layers, culling lists, indirect arguments.
    this.vsmEntries = d.createBuffer({ size: 4 * (VSM_PHYS + 2 * VSM_SIDE * VSM_SIDE), usage: S | CD });
    d.queue.writeBuffer(this.vsmEntries, 0, new Uint32Array(VSM_PHYS + 2 * VSM_SIDE * VSM_SIDE).fill(0xffffffff));
    this.vsmAtlas = d.createBuffer({ size: 2 * 4 * VSM_SIDE * VSM_SIDE * 128 * 128, usage: S });
    this.vsmWork = d.createBuffer({ size: 4 * VSM_WORK_WORDS, usage: S | CD | CS });
    this.vsmArgs = d.createBuffer({ size: 80, usage: S | GPUBufferUsage.INDIRECT });
    this.vsmArgsGroup = d.createBindGroup({ layout: this.vsmArgsLayout, entries: [{ binding: 0, resource: { buffer: this.vsmArgs } }] });
    this.counters = d.createBuffer({ size: 96, usage: S | CD | CS });
    // an entry per pass, then per pyramid level: (pass, level).
    this.passInfo = d.createBuffer({ size: PASS_STRIDE * (2 + MAX_HZB_LEVELS), usage: GPUBufferUsage.UNIFORM | CD });
    for (let k = 0; k < 2 + MAX_HZB_LEVELS; k++)
      d.queue.writeBuffer(this.passInfo, k * PASS_STRIDE, new Uint32Array([k < 2 ? k : 0, k < 2 ? 0 : k - 2, 0, 0]));
    this.hzbDummy = d.createTexture({ size: [1, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
    this.prevValid = false;
    this.work = d.createBuffer({ size: 2 * MAX_WORK * 8, usage: S });  // per pass
    this.hwVisible = d.createBuffer({ size: MAX_VISIBLE * 8, usage: S });
    this.swVisible = d.createBuffer({ size: MAX_VISIBLE * 8, usage: S });
    // [0, 8) culling per pass, [8, 16) hardware draws, [16, 24) software dispatches, [24,
    // 28) pass 2 instance culling, [28, 36) expand per pass.
    this.args = d.createBuffer({ size: 144, usage: S | GPUBufferUsage.INDIRECT });
    this.readbacks = [0, 1, 2].map(() => ({ buffer: d.createBuffer({ size: 96, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
    this.requests = d.createBuffer({ size: 16 + MAX_REQUESTS * 8, usage: S | CD | CS });
    this.frameIndex = 0;
    this.time = 0;
    this.wanted = [];
    if (this.timestamps) {
      this.querySet = d.createQuerySet({ type: 'timestamp', count: 8 });
      this.queryResolve = d.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | CS });
      this.timeReadbacks = [0, 1, 2].map(() => ({ buffer: d.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
    }
    this.stats = null;
    this.gpuMs = null;
  }

  // models: from fetchModel() (geometry.js); pages stream into a pool of poolBytes as views
  // need them.
  loadModels(models, poolBytes) {
    const d = this.device;
    const total = models.reduce((a, m) => ({ c: a.c + m.clusterCount, p: a.p + m.pages.length }), { c: 0, p: 0 });
    const clusters = new ArrayBuffer(total.c * 112);
    const meshes = new ArrayBuffer(models.length * 64);
    const pages = [];
    let clusterBase = 0;
    models.forEach((m, k) => {
      const pageBase = pages.length;
      const dst = new Uint32Array(clusters, clusterBase * 112, m.clusterCount * 28);
      dst.set(new Uint32Array(m.clusters));
      for (let c = 0; c < m.clusterCount; c++) {
        dst[28 * c + 23] += pageBase;                                       // its page
        if (dst[28 * c + 24] !== 0xffffffff) dst[28 * c + 24] += pageBase;  // the finer clusters' page
      }
      m.pages.forEach((p, i) => pages.push({
        url: m.pagesUrl, offset: p.offset, size: p.size, pinned: i === 0,
        deps: p.deps.map((x) => x + pageBase), children: p.children.map((x) => x + pageBase),
      }));
      const mu = new Uint32Array(meshes, k * 64, 4), mf = new Float32Array(meshes, k * 64 + 16, 12);
      mu[0] = clusterBase;
      mu[1] = m.clusterCount;
      mf.set(m.bounds, 0);
      mf.set(m.lodBounds, 4);
      mf.set(m.grid, 8);
      clusterBase += m.clusterCount;
    });
    this.models = models;
    this.streamer = new Streamer(pages, poolBytes);
    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    const upload = (data) => {
      const b = d.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: S | CD });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.clusterBuffer = upload(clusters);
    this.meshBuffer = upload(meshes);
    this.pageTable = d.createBuffer({ size: pages.length * 4, usage: S | CD });
    this.pool = d.createBuffer({ size: this.streamer.slotCount * this.streamer.slotBytes, usage: S | CD });
    // per page: frame last drawn from, then frame last asked for.
    this.pageStamps = d.createBuffer({ size: pages.length * 8, usage: S | CD | CS });
    this.streamReadbacks = [0, 1, 2].map(() => ({
      buffer: d.createBuffer({ size: 16 + MAX_REQUESTS * 8 + pages.length * 4, usage: GPUBufferUsage.MAP_READ | CD }), busy: false,
    }));
    this.workGroup = null;  // remade with the new buffers on the next resize
    this.width = this.height = 0;
  }

  // placements: {model, matrix (3x4 rows), scale, material, anim}.
  setPlacements(placements) {
    const d = this.device;
    const instances = new ArrayBuffer(Math.max(1, placements.length) * 64);
    this.fullDetail = 0;
    placements.forEach((p, i) => {
      new Float32Array(instances, i * 64, 12).set(p.matrix);
      new Uint32Array(instances, i * 64 + 48, 1)[0] = p.model;
      new Float32Array(instances, i * 64 + 52, 1)[0] = p.scale;
      new Uint32Array(instances, i * 64 + 56, 1)[0] = p.material ?? 0;
      new Uint32Array(instances, i * 64 + 60, 1)[0] = p.anim ?? 0;
      this.fullDetail += this.models[p.model].leafTriangles;
    });
    this.instanceCount = placements.length;
    this.late?.destroy();
    // hidden clusters, hidden instances, then each pass's big instances (two entries each).
    this.late = d.createBuffer({ size: (MAX_VISIBLE + 5 * Math.max(1, placements.length)) * 8, usage: GPUBufferUsage.STORAGE });
    this.workGroup = null;  // remade on the next resize
    this.vsmGroup = null;
    this.width = 0;
    this.instanceBuffer?.destroy();
    this.instanceBuffer = d.createBuffer({ size: instances.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.instanceBuffer, 0, instances);
    const sceneGroup = (frame) => d.createBindGroup({
      layout: this.sceneLayout,
      entries: [{ binding: 0, resource: { buffer: frame } },
        ...[[1, this.clusterBuffer], [2, this.pageTable], [3, this.pool], [6, this.meshBuffer], [7, this.instanceBuffer]]
          .map(([binding, buffer]) => ({ binding, resource: { buffer } }))],
    });
    this.sceneGroup = sceneGroup(this.frameBuffer);
    // the shadow pages' lists end with the moving instances, for invalidation.
    const moving = [];
    placements.forEach((p, i) => { if (p.anim) moving.push(i); });
    this.movingCount = moving.length;
    this.vsmLists?.destroy();
    this.vsmLists = d.createBuffer({ size: 4 * (VSM_MOVING_LIST + Math.max(1, moving.length)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (moving.length) d.queue.writeBuffer(this.vsmLists, 4 * VSM_MOVING_LIST, new Uint32Array(moving));
  }

  resize(width, height) {
    if (width === this.width && height === this.height && this.workGroup) return;
    const d = this.device;
    this.width = width;
    this.height = height;
    for (const x of [this.swBuffer, this.depthTexture, this.idTexture, this.image, this.shaded, ...(this.history ?? []),
      this.aoImage, this.aoDepth]) x?.destroy();
    this.vsmGroup = null;
    // software rasterizer depth, then triangle ids.
    this.swBuffer = d.createBuffer({ size: width * height * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const T = GPUTextureUsage;
    this.depthTexture = d.createTexture({ size: [width, height], format: 'depth32float', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    this.idTexture = d.createTexture({ size: [width, height], format: 'r32uint', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    this.image = d.createTexture({ size: [width, height], format: 'rgba8unorm', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC });
    // shading writes here; taa blends it into the history and writes image, for the canvas.
    this.shaded = d.createTexture({ size: [width, height], format: 'rgba8unorm', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING });
    this.history = [0, 1].map(() => d.createTexture({ size: [width, height], format: 'rgba16float', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING }));
    this.taaGroups = [0, 1].map((k) => d.createBindGroup({
      layout: this.taaLayout,
      entries: [{ binding: 0, resource: this.depthTexture.createView() }, { binding: 1, resource: this.idTexture.createView() },
        { binding: 8, resource: this.shaded.createView() },
        { binding: 9, resource: this.history[1 - k].createView() }, { binding: 10, resource: this.historySampler },
        { binding: 11, resource: this.history[k].createView() }, { binding: 12, resource: this.image.createView() }],
    }));
    this.historyValid = false;
    this.workGroup = d.createBindGroup({
      layout: this.workLayout,
      entries: [
        ...[[0, this.counters], [1, this.work], [2, this.hwVisible], [3, this.swVisible], [4, this.swBuffer],
          [6, this.pageStamps], [7, this.requests], [12, this.late]].map(([binding, buffer]) => ({ binding, resource: { buffer } })),
        { binding: 9, resource: { buffer: this.passInfo, size: 16 } },
      ],
    });
    // depth pyramid: level 0 half the screen (rounded up), halving to 1x1.
    this.hzb?.destroy();
    const hw = Math.ceil(width / 2), hh = Math.ceil(height / 2);
    this.hzbLevels = Math.min(MAX_HZB_LEVELS, Math.floor(Math.log2(Math.max(hw, hh))) + 1);
    this.hzb = d.createTexture({ size: [hw, hh], format: 'r32float', mipLevelCount: this.hzbLevels,
      usage: T.STORAGE_BINDING | T.TEXTURE_BINDING });
    const level = (l) => this.hzb.createView({ baseMipLevel: l, mipLevelCount: 1 });
    // ambient occlusion: its half-resolution output and depth chain (four levels).
    const aw = Math.ceil(width / 2), ah = Math.ceil(height / 2);
    this.aoImage = d.createTexture({ size: [aw, ah], format: 'r32float', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING });
    this.aoDepth = d.createTexture({ size: [aw, ah], format: 'r32float', mipLevelCount: 4, usage: T.STORAGE_BINDING | T.TEXTURE_BINDING });
    const aoLevel = (l) => this.aoDepth.createView({ baseMipLevel: l, mipLevelCount: 1 });
    this.aoFirstGroup = d.createBindGroup({
      layout: this.aoFirstLayout,
      entries: [{ binding: 0, resource: this.depthTexture.createView() }, { binding: 1, resource: { buffer: this.swBuffer } },
        { binding: 2, resource: aoLevel(0) }],
    });
    this.aoDownGroups = [1, 2, 3].map((l) => d.createBindGroup({
      layout: this.aoDownLayout, entries: [{ binding: 2, resource: aoLevel(l) }, { binding: 3, resource: aoLevel(l - 1) }],
    }));
    this.aoGroup = d.createBindGroup({
      layout: this.aoLayout, entries: [{ binding: 4, resource: this.aoDepth.createView() }, { binding: 5, resource: this.aoImage.createView() }],
    });
    this.hzbGroups = [];
    for (let l = 0; l < this.hzbLevels; l++)
      this.hzbGroups.push(d.createBindGroup({
        layout: this.hzbLayout,
        entries: [{ binding: 0, resource: this.depthTexture.createView() },
          { binding: 6, resource: l === 0 ? this.hzbDummy.createView() : level(l - 1) }, { binding: 7, resource: level(l) }],
      }));
    this.prevValid = false;
    this.imageGroup = d.createBindGroup({
      layout: this.imageLayout,
      entries: [{ binding: 0, resource: this.depthTexture.createView() }, { binding: 1, resource: this.idTexture.createView() },
        { binding: 2, resource: this.shaded.createView() }, { binding: 3, resource: { buffer: this.vsmEntries } },
        { binding: 4, resource: { buffer: this.vsmAtlas } }, { binding: 5, resource: this.hzb.createView() },
        { binding: 13, resource: this.aoImage.createView() }],
    });
    this.argsGroup = d.createBindGroup({ layout: this.argsLayout, entries: [{ binding: 0, resource: { buffer: this.args } }] });
    this.rasterGroup = d.createBindGroup({
      layout: this.rasterLayout,
      entries: [{ binding: 0, resource: { buffer: this.hwVisible } }, { binding: 2, resource: { buffer: this.passInfo, size: 16 } },
        { binding: 3, resource: { buffer: this.counters } }],
    });
    this.blitGroup = d.createBindGroup({ layout: this.blitLayout, entries: [{ binding: 1, resource: this.image.createView() }] });
    this.emptyGroup = d.createBindGroup({ layout: this.blit.getBindGroupLayout(0), entries: [] });
  }

  // camera: {eye, forward, fov, near}; cull: the same, or a frozen copy.
  draw(camera, cull, settings) {
    const d = this.device;
    const { width, height } = this;
    const aspect = width / height;
    const proj = perspectiveReverseZ(camera.fov, aspect, camera.near);
    const unjittered = mul(proj, lookTo(camera.eye, camera.forward, [0, 1, 0]));
    // taa: sub-pixel offset, halton (2, 3) over eight frames; clip.xy += offset * w.
    const taa = (settings.flags & FLAG_TAA) !== 0 && settings.mode === 0;
    const viewProj = new Float32Array(unjittered);
    let jitter = [0, 0];
    if (taa) {
      const halton = (i, base) => { let f = 1, r = 0; for (; i > 0; i = Math.floor(i / base)) { f /= base; r += f * (i % base); } return r; };
      const k = (this.frameIndex % 8) + 1;
      jitter = [(halton(k, 2) - 0.5) * 2 / width, (halton(k, 3) - 0.5) * 2 / height];
      for (let c = 0; c < 4; c++) {
        viewProj[c * 4] += jitter[0] * viewProj[c * 4 + 3];
        viewProj[c * 4 + 1] += jitter[1] * viewProj[c * 4 + 3];
      }
    }
    const cullViewProj = mul(perspectiveReverseZ(cull.fov, aspect, cull.near), lookTo(cull.eye, cull.forward, [0, 1, 0]));
    const shadows = (settings.flags & FLAG_SHADOWS) !== 0;
    this.frameIndex++;
    // time, for motion: stands still while motion is off.
    const now = performance.now() / 1000;
    const prevTime = this.time;
    if (this.lastNow !== undefined && settings.motion) this.time += now - this.lastNow;
    this.lastNow = now;
    const time = this.time;
    const f = new ArrayBuffer(FRAME_BYTES);
    const ff = new Float32Array(f), fu = new Uint32Array(f);
    const view = lookTo(camera.eye, camera.forward, [0, 1, 0]);
    ff.set(view, 96);
    ff.set(this.prevValid ? this.prevView : view, 112);
    ff[128] = proj[0];
    ff[129] = proj[5];
    fu[130] = this.hzbLevels;
    this.prevView = view;
    ff.set(this.historyValid && this.prevViewProj ? this.prevViewProj : unjittered, 132);
    ff[148] = jitter[0]; ff[149] = jitter[1];
    fu[150] = taa && this.historyValid ? 1 : 0;
    this.prevViewProj = unjittered;
    this.historyValid = taa;
    ff.set(viewProj, 0);
    ff.set(invert(viewProj), 16);
    frustumPlanes(cullViewProj).forEach((p, k) => ff.set(p, 32 + 4 * k));
    ff.set([...cull.eye, 1], 52);
    ff.set([...camera.eye, 1], 56);
    fu[60] = width; fu[61] = height; fu[62] = this.instanceCount;
    // depth from the drawing camera says nothing about a frozen culling camera.
    let flags = settings.flags;
    if (cull !== camera) flags &= ~FLAG_OCCLUSION;
    if (this.prevValid) flags |= FLAG_PREV_VALID;
    if (this.movingCount) flags |= FLAG_MOVING;
    fu[63] = flags;
    this.prevValid = true;
    ff[64] = height / (2 * Math.tan(camera.fov / 2));
    ff[65] = settings.threshold;
    ff[66] = camera.near;
    fu[67] = settings.mode;
    fu[68] = MAX_WORK; fu[69] = MAX_VISIBLE;
    ff[70] = settings.swPixels;
    ff[71] = time;
    ff[151] = prevTime;
    fu[72] = this.frameIndex;
    fu[73] = MAX_REQUESTS;
    d.queue.writeBuffer(this.frameBuffer, 0, f);

    // streaming: what the gpu asked for and drew from a few frames ago, and pages arrived
    // since.
    const uploads = this.streamer.service(this.frameIndex, this.wanted, settings.threshold);
    this.wanted = [];
    for (const u of uploads) d.queue.writeBuffer(this.pool, u.slot * this.streamer.slotBytes, u.data);
    d.queue.writeBuffer(this.pageTable, 0, this.streamer.table);

    const enc = d.createCommandEncoder();
    const ts = (begin, end) => (this.timestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } } : {});
    // culls a pass: instances, clusters, the compute rasterizer's share; the hardware draw
    // follows in a render pass.
    const cullPass = (scene, p, timestamps, software = true) => {
      const cp = enc.beginComputePass(timestamps);
      cp.setBindGroup(0, scene);
      cp.setBindGroup(1, this.workGroup, [p * PASS_STRIDE]);
      cp.setBindGroup(2, this.imageGroup);
      cp.setBindGroup(3, this.argsGroup);
      cp.setPipeline(this.instanceCull);
      if (p === 0) {
        const groups = Math.ceil(this.instanceCount / 64);
        cp.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      } else {
        cp.dispatchWorkgroupsIndirect(this.args, 96);
      }
      cp.setPipeline(this.argsBig);
      cp.dispatchWorkgroups(1);
      cp.setPipeline(this.expand);
      cp.dispatchWorkgroupsIndirect(this.args, 112 + p * 16);
      cp.setPipeline(this.argsCull);
      cp.dispatchWorkgroups(1);
      cp.setPipeline(this.clusterCull);
      cp.dispatchWorkgroupsIndirect(this.args, p * 16);
      cp.setPipeline(this.argsDraw);
      cp.dispatchWorkgroups(1);
      if (software) {
        cp.setPipeline(this.swDepth);
        cp.dispatchWorkgroupsIndirect(this.args, 64 + p * 16);
        cp.setPipeline(this.swId);
        cp.dispatchWorkgroupsIndirect(this.args, 64 + p * 16);
      }
      cp.end();
    };
    const buildHzb = () => {
      const hp = enc.beginComputePass();
      hp.setBindGroup(0, this.sceneGroup);
      for (let l = 0; l < this.hzbLevels; l++) {
        hp.setBindGroup(1, this.workGroup, [(2 + l) * PASS_STRIDE]);
        hp.setBindGroup(2, this.hzbGroups[l]);
        hp.setPipeline(l === 0 ? this.hzbFirst : this.hzbDown);
        hp.dispatchWorkgroups(Math.ceil(Math.max(1, Math.ceil(this.width / 2) >> l) / 8), Math.ceil(Math.max(1, Math.ceil(this.height / 2) >> l) / 8));
      }
      hp.end();
    };
    enc.clearBuffer(this.counters);
    enc.clearBuffer(this.requests, 0, 16);
    enc.clearBuffer(this.swBuffer, 0, width * height * 4);
    // pass 1: what last frame's depth does not hide. then a pyramid from it, and pass 2:
    // what pass 1 wrongly hid. then the pyramid again, for next frame.
    for (let p = 0; p < 2; p++) {
      cullPass(this.sceneGroup, p, p === 0 ? ts(0, 1) : {});
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: this.idTexture.createView(), loadOp: p === 0 ? 'clear' : 'load', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        depthStencilAttachment: { view: this.depthTexture.createView(), depthLoadOp: p === 0 ? 'clear' : 'load', depthStoreOp: 'store', depthClearValue: 0 },
        ...(p === 0 ? ts(2, 3) : {}),
      });
      rp.setPipeline(this.raster);
      rp.setBindGroup(0, this.sceneGroup);
      rp.setBindGroup(1, this.rasterGroup, [p * PASS_STRIDE]);
      rp.drawIndirect(this.args, 32 + p * 16);
      rp.end();
      buildHzb();
    }

    this.shadowsTimed = shadows;
    if (shadows) {
      // shadow pages: mark, invalidate, assign physical pages, clear, render. webgpu orders
      // dispatches within a pass.
      if (!this.vsmGroup) {
        this.vsmGroup = d.createBindGroup({
          layout: this.vsmLayout,
          entries: [
            ...[[0, this.vsmEntries], [1, this.vsmLists], [2, this.vsmAtlas], [3, this.vsmWork], [5, this.swBuffer], [6, this.pageStamps],
              [7, this.requests]].map(([binding, buffer]) => ({ binding, resource: { buffer } })),
            { binding: 4, resource: this.depthTexture.createView() },
          ],
        });
      }
      enc.clearBuffer(this.vsmLists, 0, 4 * VSM_HEADER);
      enc.clearBuffer(this.vsmWork, 0, 16);
      const vp = enc.beginComputePass(ts(6, 7));
      vp.setBindGroup(0, this.sceneGroup);
      vp.setBindGroup(1, this.vsmGroup);
      vp.setBindGroup(2, this.vsmArgsGroup);
      const run = (name, x, y = 1) => { vp.setPipeline(this.vsm[name]); vp.dispatchWorkgroups(x, y); };
      const indirect = (name, offset) => { vp.setPipeline(this.vsm[name]); vp.dispatchWorkgroupsIndirect(this.vsmArgs, offset); };
      run('mark', Math.ceil(width / 8), Math.ceil(height / 8));
      // (nothing moves while the clock stands still.)
      if (this.movingCount && time !== prevTime) run('invalidate', Math.min(this.movingCount, 65535), Math.ceil(this.movingCount / 65535));
      run('alloc_slots', VSM_SLOTS / 64);
      run('alloc_phys', VSM_SIDE * VSM_SIDE / 64);
      run('alloc_assign', VSM_SLOTS / 64);
      run('args_alloc', 1);
      indirect('clear', 16);
      indirect('instance', 0);
      run('args_expand', 1);
      indirect('expand', 32);
      run('args_cull', 1);
      indirect('cluster', 48);
      run('args_raster', 1);
      indirect('raster', 64);
      vp.end();
    }

    if (settings.flags & FLAG_AO) {
      // ambient occlusion: the depth chain, then the pass.
      const ap = enc.beginComputePass();
      ap.setBindGroup(0, this.sceneGroup);
      const aw = Math.ceil(width / 2), ah = Math.ceil(height / 2);
      ap.setPipeline(this.aoDepthFirst);
      ap.setBindGroup(1, this.aoFirstGroup);
      ap.dispatchWorkgroups(Math.ceil(aw / 8), Math.ceil(ah / 8));
      ap.setPipeline(this.aoDepthDown);
      this.aoDownGroups.forEach((g, k) => {
        ap.setBindGroup(1, g);
        ap.dispatchWorkgroups(Math.ceil(Math.max(1, aw >> (k + 1)) / 8), Math.ceil(Math.max(1, ah >> (k + 1)) / 8));
      });
      ap.setPipeline(this.aoPass);
      ap.setBindGroup(1, this.aoGroup);
      ap.dispatchWorkgroups(Math.ceil(aw / 8), Math.ceil(ah / 8));
      ap.end();
    }

    let pass;
    pass = enc.beginComputePass(ts(4, 5));
    pass.setBindGroup(0, this.sceneGroup);
    pass.setBindGroup(1, this.workGroup, [0]);
    pass.setBindGroup(2, this.imageGroup);
    pass.setPipeline(this.shade);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.setBindGroup(2, this.taaGroups[this.frameIndex & 1]);
    pass.setPipeline(this.taa);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();

    if (this.context) {
      const bp = enc.beginRenderPass({
        colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      bp.setPipeline(this.blit);
      bp.setBindGroup(0, this.emptyGroup);
      bp.setBindGroup(1, this.blitGroup);
      bp.draw(3);
      bp.end();
    }

    const sb = this.streamReadbacks.find((x) => !x.busy);
    if (sb) {
      enc.copyBufferToBuffer(this.requests, 0, sb.buffer, 0, 16 + MAX_REQUESTS * 8);
      enc.copyBufferToBuffer(this.pageStamps, 0, sb.buffer, 16 + MAX_REQUESTS * 8, this.pageStamps.size / 2);
    }
    // stats and timings return a few frames late, into a free readback buffer.
    const rb = this.readbacks.find((x) => !x.busy);
    if (rb) {
      // shadow pages rendered and their clusters, into the counters' spare words.
      if (shadows) {
        enc.copyBufferToBuffer(this.vsmLists, 12, this.counters, 88, 4);
        enc.copyBufferToBuffer(this.vsmWork, 8, this.counters, 92, 4);
      }
      enc.copyBufferToBuffer(this.counters, 0, rb.buffer, 0, 96);
    }
    const tb = this.timestamps && this.timeReadbacks.find((x) => !x.busy);
    if (tb) {
      enc.resolveQuerySet(this.querySet, 0, 8, this.queryResolve, 0);
      enc.copyBufferToBuffer(this.queryResolve, 0, tb.buffer, 0, 64);
    }
    d.queue.submit([enc.finish()]);
    if (sb) {
      sb.busy = true;
      sb.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const u = new Uint32Array(sb.buffer.getMappedRange().slice(0));
        sb.buffer.unmap();
        sb.busy = false;
        this.streamer.noteUsed(u.subarray(4 + MAX_REQUESTS * 2));
        const n = Math.min(u[0], MAX_REQUESTS);
        const f32 = new Float32Array(u.buffer);
        for (let k = 0; k < n; k++) this.wanted.push([u[4 + 2 * k], f32[4 + 2 * k + 1]]);
      });
    }
    if (rb) {
      rb.busy = true;
      rb.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const u = new Uint32Array(rb.buffer.getMappedRange().slice(0));
        rb.buffer.unmap();
        rb.busy = false;
        this.stats = {
          work: u[0] + u[1], hw: u[2], sw: u[3], instances: u[4], tested: u[5], triangles: u[6], overflow: u[7],
          hiddenLastFrame: u[9], instancesOccluded: u[10], late: u[11], shadowPages: u[22], shadowClusters: u[23],
        };
      });
    }
    if (tb) {
      tb.busy = true;
      tb.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(tb.buffer.getMappedRange().slice(0));
        tb.buffer.unmap();
        tb.busy = false;
        const ms = (a, b) => Number(t[b] - t[a]) / 1e6;
        this.gpuMs = { cull: ms(0, 1), raster: ms(2, 3), shade: ms(4, 5), total: ms(0, 5), shadows: this.shadowsTimed ? ms(6, 7) : 0 };
      });
    }
  }

  // shaded image as rgba bytes, for tests.
  async readPixels() {
    const d = this.device;
    const bytesPerRow = Math.ceil(this.width * 4 / 256) * 256;
    const b = d.createBuffer({ size: bytesPerRow * this.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.image }, { buffer: b, bytesPerRow }, [this.width, this.height]);
    d.queue.submit([enc.finish()]);
    await b.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(b.getMappedRange());
    const out = new Uint8Array(this.width * this.height * 4);
    for (let y = 0; y < this.height; y++) out.set(src.subarray(y * bytesPerRow, y * bytesPerRow + this.width * 4), y * this.width * 4);
    b.unmap();
    b.destroy();
    return out;
  }
}
