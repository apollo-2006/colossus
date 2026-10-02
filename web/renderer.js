// The WebGPU renderer: the same pipeline as the Vulkan viewer, minus what
// WebGPU lacks (see compute.wgsl). Per frame:
//   instance_cull -> args_cull -> cluster_cull -> args_draw
//   -> software raster (depth, then ids) and hardware raster
//   -> shade -> blit to the canvas.
import { frustumPlanes, invert, lookTo, mul, orthographic, perspectiveReverseZ } from './math.js';
import { Streamer } from './streamer.js';

const MAX_WORK = 1 << 20;
const MAX_VISIBLE = 1 << 20;
const MAX_REQUESTS = 1 << 13;
const FRAME_BYTES = 608;
const PASS_STRIDE = 256;  // Dynamic uniform offsets must be multiples of this
const MAX_HZB_LEVELS = 16;
const STORAGE_BUFFERS_NEEDED = 14;
const SHADOW_SIZE = 2048;
const SHADOW_HALF = 8;  // The shadow map covers 16 x 16 units in front of the camera
const SUN_DIR = [0.75, 0.5, 0.3].map((x) => x / Math.hypot(0.75, 0.5, 0.3));

export const FLAG_CONE = 1, FLAG_FRUSTUM = 2, FLAG_SOFTWARE = 4, FLAG_SHADOW_PASS = 8, FLAG_SHADOWS = 16, FLAG_OCCLUSION = 32;
export const FLAG_TAA = 128;  // Read on the CPU only
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
    // The culling passes bind 14 storage buffers to the compute stage; the
    // default limit is 8, but desktop adapters allow at least 16.
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
    const common = await source('common.wgsl');
    r.computeModule = device.createShaderModule({ code: common + await source('compute.wgsl') });
    r.rasterModule = device.createShaderModule({ code: common + await source('raster.wgsl') });
    for (const m of [r.computeModule, r.rasterModule]) {
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
        { binding: 3, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 4, visibility: C, sampler: { type: 'comparison' } },
        { binding: 5, visibility: C, texture: { sampleType: 'unfilterable-float' } },
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
    // The shadow map: casters' depth from the sun, drawn from the same
    // visible list, by a second run of the culling with the sun's camera.
    this.shadowRaster = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.rasterLayout] }),
      vertex: { module: this.rasterModule, entryPoint: 'shadow_vs' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.shadowMap = d.createTexture({ size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.shadowSampler = d.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.blit = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [d.createBindGroupLayout({ entries: [] }), this.blitLayout] }),
      vertex: { module: this.rasterModule, entryPoint: 'blit_vs' },
      fragment: { module: this.rasterModule, entryPoint: 'blit_fs', targets: [{ format: this.format }] },
    });

    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    this.frameBuffer = d.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | CD });
    this.shadowFrameBuffer = d.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | CD });
    this.counters = d.createBuffer({ size: 96, usage: S | CD | CS });
    // One entry per pass, then one per pyramid level: (pass, level).
    this.passInfo = d.createBuffer({ size: PASS_STRIDE * (2 + MAX_HZB_LEVELS), usage: GPUBufferUsage.UNIFORM | CD });
    for (let k = 0; k < 2 + MAX_HZB_LEVELS; k++)
      d.queue.writeBuffer(this.passInfo, k * PASS_STRIDE, new Uint32Array([k < 2 ? k : 0, k < 2 ? 0 : k - 2, 0, 0]));
    this.hzbDummy = d.createTexture({ size: [1, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
    this.prevValid = false;
    this.work = d.createBuffer({ size: 2 * MAX_WORK * 8, usage: S });  // Per pass
    this.hwVisible = d.createBuffer({ size: MAX_VISIBLE * 8, usage: S });
    this.swVisible = d.createBuffer({ size: MAX_VISIBLE * 8, usage: S });
    // [0, 8) culling dispatches per pass, [8, 16) hardware draws, [16, 24)
    // software dispatches.
    this.args = d.createBuffer({ size: 96, usage: S | GPUBufferUsage.INDIRECT });
    this.readbacks = [0, 1, 2].map(() => ({ buffer: d.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
    this.requests = d.createBuffer({ size: 16 + MAX_REQUESTS * 8, usage: S | CD | CS });
    this.frameIndex = 0;
    this.wanted = [];
    if (this.timestamps) {
      this.querySet = d.createQuerySet({ type: 'timestamp', count: 8 });
      this.queryResolve = d.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | CS });
      this.timeReadbacks = [0, 1, 2].map(() => ({ buffer: d.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
    }
    this.stats = null;
    this.gpuMs = null;
  }

  // models: parsed .cgeo files. placements: {model, matrix (3x4 rows), scale}.
  // models: from fetchModel() (geometry.js). Their pages stream into a
  // pool of poolBytes as the views need them.
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
        dst[28 * c + 23] += pageBase;                                       // Its page
        if (dst[28 * c + 24] !== 0xffffffff) dst[28 * c + 24] += pageBase;  // The finer clusters' page
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
    // Per page, the frame it was last drawn from, then the frame it was last asked for.
    this.pageStamps = d.createBuffer({ size: pages.length * 8, usage: S | CD | CS });
    this.streamReadbacks = [0, 1, 2].map(() => ({
      buffer: d.createBuffer({ size: 16 + MAX_REQUESTS * 8 + pages.length * 4, usage: GPUBufferUsage.MAP_READ | CD }), busy: false,
    }));
    this.workGroup = null;  // Remade with the new buffers on the next resize
    this.width = this.height = 0;
  }

  // placements: {model, matrix (3x4 rows), scale, material}.
  setPlacements(placements) {
    const d = this.device;
    const instances = new ArrayBuffer(Math.max(1, placements.length) * 64);
    this.fullDetail = 0;
    placements.forEach((p, i) => {
      new Float32Array(instances, i * 64, 12).set(p.matrix);
      new Uint32Array(instances, i * 64 + 48, 1)[0] = p.model;
      new Float32Array(instances, i * 64 + 52, 1)[0] = p.scale;
      new Uint32Array(instances, i * 64 + 56, 1)[0] = p.material ?? 0;
      this.fullDetail += this.models[p.model].leafTriangles;
    });
    this.instanceCount = placements.length;
    this.late?.destroy();
    this.late = d.createBuffer({ size: (MAX_VISIBLE + Math.max(1, placements.length)) * 8, usage: GPUBufferUsage.STORAGE });
    this.workGroup = null;  // Remade with it on the next resize
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
    this.shadowSceneGroup = sceneGroup(this.shadowFrameBuffer);
  }

  resize(width, height) {
    if (width === this.width && height === this.height && this.workGroup) return;
    const d = this.device;
    this.width = width;
    this.height = height;
    for (const x of [this.swBuffer, this.depthTexture, this.idTexture, this.image, this.shaded, ...(this.history ?? [])]) x?.destroy();
    // The software rasterizer's depth, then its triangle ids.
    this.swBuffer = d.createBuffer({ size: width * height * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const T = GPUTextureUsage;
    this.depthTexture = d.createTexture({ size: [width, height], format: 'depth32float', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    this.idTexture = d.createTexture({ size: [width, height], format: 'r32uint', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    this.image = d.createTexture({ size: [width, height], format: 'rgba8unorm', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC });
    // Shading writes here; the antialiasing pass blends it into the history
    // and writes the result to image, for the canvas.
    this.shaded = d.createTexture({ size: [width, height], format: 'rgba8unorm', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING });
    this.history = [0, 1].map(() => d.createTexture({ size: [width, height], format: 'rgba16float', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING }));
    this.taaGroups = [0, 1].map((k) => d.createBindGroup({
      layout: this.taaLayout,
      entries: [{ binding: 0, resource: this.depthTexture.createView() }, { binding: 8, resource: this.shaded.createView() },
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
    // The depth pyramid: level 0 half the screen (rounded up), each level
    // half the last, down to 1 x 1.
    this.hzb?.destroy();
    const hw = Math.ceil(width / 2), hh = Math.ceil(height / 2);
    this.hzbLevels = Math.min(MAX_HZB_LEVELS, Math.floor(Math.log2(Math.max(hw, hh))) + 1);
    this.hzb = d.createTexture({ size: [hw, hh], format: 'r32float', mipLevelCount: this.hzbLevels,
      usage: T.STORAGE_BINDING | T.TEXTURE_BINDING });
    const level = (l) => this.hzb.createView({ baseMipLevel: l, mipLevelCount: 1 });
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
        { binding: 2, resource: this.shaded.createView() }, { binding: 3, resource: this.shadowMap.createView() },
        { binding: 4, resource: this.shadowSampler }, { binding: 5, resource: this.hzb.createView() }],
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
    // Temporal antialiasing: nudge the projection by a sub-pixel offset,
    // the Halton (2, 3) sequence over eight frames; clip.xy += offset * w.
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
    // The sun's camera: orthographic, over a square of the ground in front
    // of the camera, looking down the sun's direction.
    const ahead = [camera.eye[0] + camera.forward[0] * SHADOW_HALF * 0.7, 0, camera.eye[2] + camera.forward[2] * SHADOW_HALF * 0.7];
    const reach = 40;
    const sunEye = ahead.map((x, k) => x + SUN_DIR[k] * reach);
    const sunViewProj = mul(orthographic(SHADOW_HALF, 0, 2 * reach), lookTo(sunEye, SUN_DIR.map((x) => -x), [0, 1, 0]));
    const shadows = (settings.flags & FLAG_SHADOWS) !== 0;
    this.frameIndex++;
    if (shadows) {
      const sf = new ArrayBuffer(FRAME_BYTES);
      const sff = new Float32Array(sf), sfu = new Uint32Array(sf);
      sff.set(sunViewProj, 0);
      frustumPlanes(sunViewProj).forEach((p, k) => sff.set(p, 32 + 4 * k));
      // Far behind the sun's camera, so the cone test sees parallel rays.
      sff.set([...ahead.map((x, k) => x + SUN_DIR[k] * 1e4), 1], 52);
      sff.set([...sunEye, 1], 56);
      sfu[60] = SHADOW_SIZE; sfu[61] = SHADOW_SIZE; sfu[62] = this.instanceCount;
      sfu[63] = FLAG_FRUSTUM | FLAG_CONE | FLAG_SHADOW_PASS;
      // Error allowed, in shadow map texels: the lookup filters over three
      // by three, and eight texels looks the same as one at a seventh of
      // the cost (0.17 ms against 2.4 for 900 instances).
      sff[64] = 1; sff[65] = 8; sff[66] = 1e-3;
      sfu[68] = MAX_WORK; sfu[69] = MAX_VISIBLE;
      sfu[72] = this.frameIndex; sfu[73] = MAX_REQUESTS;
      sff[92] = SHADOW_SIZE / (2 * SHADOW_HALF);  // ortho_scale: texels per unit
      d.queue.writeBuffer(this.shadowFrameBuffer, 0, sf);
    }
    const f = new ArrayBuffer(FRAME_BYTES);
    const ff = new Float32Array(f), fu = new Uint32Array(f);
    ff.set(sunViewProj, 76);
    ff[93] = 2 * SHADOW_HALF / SHADOW_SIZE;  // shadow_texel
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
    // Occlusion from the drawing camera says nothing about what a frozen
    // culling camera would see.
    let flags = settings.flags;
    if (cull !== camera) flags &= ~FLAG_OCCLUSION;
    if (this.prevValid) flags |= FLAG_PREV_VALID;
    fu[63] = flags;
    this.prevValid = true;
    ff[64] = height / (2 * Math.tan(camera.fov / 2));
    ff[65] = settings.threshold;
    ff[66] = camera.near;
    fu[67] = settings.mode;
    fu[68] = MAX_WORK; fu[69] = MAX_VISIBLE;
    ff[70] = settings.swPixels;
    ff[71] = performance.now() / 1000;
    fu[72] = this.frameIndex;
    fu[73] = MAX_REQUESTS;
    d.queue.writeBuffer(this.frameBuffer, 0, f);

    // Streaming: what the GPU asked for and drew from, a few frames ago,
    // and the pages that have arrived since.
    const uploads = this.streamer.service(this.frameIndex, this.wanted, settings.threshold);
    this.wanted = [];
    for (const u of uploads) d.queue.writeBuffer(this.pool, u.slot * this.streamer.slotBytes, u.data);
    d.queue.writeBuffer(this.pageTable, 0, this.streamer.table);

    const enc = d.createCommandEncoder();
    const ts = (begin, end) => (this.timestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } } : {});
    // Culls one pass: instances, then clusters, then the compute
    // rasterizer's share; the hardware draw follows in its own render pass.
    const cullPass = (scene, p, timestamps, software = true) => {
      const cp = enc.beginComputePass(timestamps);
      cp.setBindGroup(0, scene);
      cp.setBindGroup(1, this.workGroup, [p * PASS_STRIDE]);
      cp.setBindGroup(2, this.imageGroup);
      cp.setBindGroup(3, this.argsGroup);
      cp.setPipeline(this.instanceCull);
      cp.dispatchWorkgroups(Math.min(this.instanceCount, 65535), Math.ceil(this.instanceCount / 65535));
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
    if (shadows) {
      // Cull with the sun's camera (one pass, no occlusion) and draw the
      // casters' depth, then the view starts over with the same lists.
      enc.clearBuffer(this.counters);
      cullPass(this.shadowSceneGroup, 0, {}, false);
      const sp = enc.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: { view: this.shadowMap.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      sp.setPipeline(this.shadowRaster);
      sp.setBindGroup(0, this.shadowSceneGroup);
      sp.setBindGroup(1, this.rasterGroup, [0]);
      sp.drawIndirect(this.args, 32);
      sp.end();
    }
    enc.clearBuffer(this.counters);
    enc.clearBuffer(this.requests, 0, 16);
    enc.clearBuffer(this.swBuffer, 0, width * height * 4);
    // Pass 1: what last frame's depth does not hide. Then the pyramid from
    // what it drew, and pass 2: what pass 1 thought hidden but is not.
    // Then the pyramid again, for next frame's pass 1.
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
    // Statistics and timings come back a few frames late, into whichever
    // readback buffer is free.
    const rb = this.readbacks.find((x) => !x.busy);
    if (rb) enc.copyBufferToBuffer(this.counters, 0, rb.buffer, 0, 64);
    const tb = this.timestamps && this.timeReadbacks.find((x) => !x.busy);
    if (tb) {
      enc.resolveQuerySet(this.querySet, 0, 6, this.queryResolve, 0);
      enc.copyBufferToBuffer(this.queryResolve, 0, tb.buffer, 0, 48);
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
          hiddenLastFrame: u[9], instancesOccluded: u[10], late: u[11],
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
        this.gpuMs = { cull: ms(0, 1), raster: ms(2, 3), shade: ms(4, 5), total: ms(0, 5) };
      });
    }
  }

  // The shaded image as RGBA bytes, for tests.
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
