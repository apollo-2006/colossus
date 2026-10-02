// The WebGPU renderer: the same pipeline as the Vulkan viewer, minus what
// WebGPU lacks (see compute.wgsl). Per frame:
//   instance_cull -> args_cull -> cluster_cull -> args_draw
//   -> software raster (depth, then ids) and hardware raster
//   -> shade -> blit to the canvas.
import { frustumPlanes, invert, lookTo, mul, perspectiveReverseZ } from './math.js';
import { Streamer } from './streamer.js';

const MAX_WORK = 1 << 20;
const MAX_VISIBLE = 1 << 20;
const MAX_REQUESTS = 1 << 13;
const FRAME_BYTES = 304;

export const FLAG_CONE = 1, FLAG_FRUSTUM = 2, FLAG_SOFTWARE = 4;

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
    const timestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: timestamps ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16),
      },
    });
    const r = new Renderer();
    r.device = device;
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
    this.workLayout = d.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6, 7, 8].map(rw) });
    this.imageLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: C, texture: { sampleType: 'uint' } },
        { binding: 2, visibility: C, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      ],
    });
    this.argsLayout = d.createBindGroupLayout({ entries: [rw(0)] });
    this.rasterLayout = d.createBindGroupLayout({ entries: [ro(0, V)] });
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

    this.raster = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.sceneLayout, this.rasterLayout] }),
      vertex: { module: this.rasterModule, entryPoint: 'vs' },
      fragment: { module: this.rasterModule, entryPoint: 'fs', targets: [{ format: 'r32uint' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    this.blit = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [d.createBindGroupLayout({ entries: [] }), this.blitLayout] }),
      vertex: { module: this.rasterModule, entryPoint: 'blit_vs' },
      fragment: { module: this.rasterModule, entryPoint: 'blit_fs', targets: [{ format: this.format }] },
    });

    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    this.frameBuffer = d.createBuffer({ size: FRAME_BYTES, usage: GPUBufferUsage.UNIFORM | CD });
    this.counters = d.createBuffer({ size: 32, usage: S | CD | CS });
    this.work = d.createBuffer({ size: MAX_WORK * 8, usage: S });
    this.hwVisible = d.createBuffer({ size: MAX_VISIBLE * 8, usage: S });
    this.swVisible = d.createBuffer({ size: MAX_VISIBLE * 8, usage: S });
    this.args = d.createBuffer({ size: 48, usage: S | GPUBufferUsage.INDIRECT });
    this.readbacks = [0, 1, 2].map(() => ({ buffer: d.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
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
    this.pageUsed = d.createBuffer({ size: pages.length * 4, usage: S | CD | CS });
    this.requestStamp = d.createBuffer({ size: pages.length * 4, usage: S | CD });
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
    this.instanceBuffer?.destroy();
    this.instanceBuffer = d.createBuffer({ size: instances.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.instanceBuffer, 0, instances);
    this.sceneGroup = d.createBindGroup({
      layout: this.sceneLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } },
        ...[[1, this.clusterBuffer], [2, this.pageTable], [3, this.pool], [6, this.meshBuffer], [7, this.instanceBuffer]]
          .map(([binding, buffer]) => ({ binding, resource: { buffer } }))],
    });
  }

  resize(width, height) {
    if (width === this.width && height === this.height && this.workGroup) return;
    const d = this.device;
    this.width = width;
    this.height = height;
    for (const x of [this.swDepthBuffer, this.swIdBuffer, this.depthTexture, this.idTexture, this.image]) x?.destroy();
    this.swDepthBuffer = d.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.swIdBuffer = d.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.STORAGE });
    const T = GPUTextureUsage;
    this.depthTexture = d.createTexture({ size: [width, height], format: 'depth32float', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    this.idTexture = d.createTexture({ size: [width, height], format: 'r32uint', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    this.image = d.createTexture({ size: [width, height], format: 'rgba8unorm', usage: T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC });
    this.workGroup = d.createBindGroup({
      layout: this.workLayout,
      entries: [this.counters, this.work, this.hwVisible, this.swVisible, this.swDepthBuffer, this.swIdBuffer, this.pageUsed,
        this.requests, this.requestStamp].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.imageGroup = d.createBindGroup({
      layout: this.imageLayout,
      entries: [{ binding: 0, resource: this.depthTexture.createView() }, { binding: 1, resource: this.idTexture.createView() },
        { binding: 2, resource: this.image.createView() }],
    });
    this.argsGroup = d.createBindGroup({ layout: this.argsLayout, entries: [{ binding: 0, resource: { buffer: this.args } }] });
    this.rasterGroup = d.createBindGroup({ layout: this.rasterLayout, entries: [{ binding: 0, resource: { buffer: this.hwVisible } }] });
    this.blitGroup = d.createBindGroup({ layout: this.blitLayout, entries: [{ binding: 1, resource: this.image.createView() }] });
    this.emptyGroup = d.createBindGroup({ layout: this.blit.getBindGroupLayout(0), entries: [] });
  }

  // camera: {eye, forward, fov, near}; cull: the same, or a frozen copy.
  draw(camera, cull, settings) {
    const d = this.device;
    const { width, height } = this;
    const aspect = width / height;
    const proj = perspectiveReverseZ(camera.fov, aspect, camera.near);
    const viewProj = mul(proj, lookTo(camera.eye, camera.forward, [0, 1, 0]));
    const cullViewProj = mul(perspectiveReverseZ(cull.fov, aspect, cull.near), lookTo(cull.eye, cull.forward, [0, 1, 0]));
    const f = new ArrayBuffer(FRAME_BYTES);
    const ff = new Float32Array(f), fu = new Uint32Array(f);
    ff.set(viewProj, 0);
    ff.set(invert(viewProj), 16);
    frustumPlanes(cullViewProj).forEach((p, k) => ff.set(p, 32 + 4 * k));
    ff.set([...cull.eye, 1], 52);
    ff.set([...camera.eye, 1], 56);
    fu[60] = width; fu[61] = height; fu[62] = this.instanceCount; fu[63] = settings.flags;
    ff[64] = height / (2 * Math.tan(camera.fov / 2));
    ff[65] = settings.threshold;
    ff[66] = camera.near;
    fu[67] = settings.mode;
    fu[68] = MAX_WORK; fu[69] = MAX_VISIBLE;
    ff[70] = settings.swPixels;
    ff[71] = performance.now() / 1000;
    fu[72] = ++this.frameIndex;
    fu[73] = MAX_REQUESTS;
    d.queue.writeBuffer(this.frameBuffer, 0, f);

    // Streaming: what the GPU asked for and drew from, a few frames ago,
    // and the pages that have arrived since.
    const uploads = this.streamer.service(this.frameIndex, this.wanted, settings.threshold);
    this.wanted = [];
    for (const u of uploads) d.queue.writeBuffer(this.pool, u.slot * this.streamer.slotBytes, u.data);
    d.queue.writeBuffer(this.pageTable, 0, this.streamer.table);

    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.counters);
    enc.clearBuffer(this.requests, 0, 16);
    enc.clearBuffer(this.swDepthBuffer);
    const ts = (begin, end) => (this.timestamps ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end } } : {});

    let pass = enc.beginComputePass(ts(0, 1));
    pass.setBindGroup(0, this.sceneGroup);
    pass.setBindGroup(1, this.workGroup);
    pass.setBindGroup(2, this.imageGroup);
    pass.setBindGroup(3, this.argsGroup);
    pass.setPipeline(this.instanceCull);
    pass.dispatchWorkgroups(Math.min(this.instanceCount, 65535), Math.ceil(this.instanceCount / 65535));
    pass.setPipeline(this.argsCull);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.clusterCull);
    pass.dispatchWorkgroupsIndirect(this.args, 0);
    pass.setPipeline(this.argsDraw);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.swDepth);
    pass.dispatchWorkgroupsIndirect(this.args, 32);
    pass.setPipeline(this.swId);
    pass.dispatchWorkgroupsIndirect(this.args, 32);
    pass.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.idTexture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      depthStencilAttachment: { view: this.depthTexture.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 0 },
      ...ts(2, 3),
    });
    rp.setPipeline(this.raster);
    rp.setBindGroup(0, this.sceneGroup);
    rp.setBindGroup(1, this.rasterGroup);
    rp.drawIndirect(this.args, 16);
    rp.end();

    pass = enc.beginComputePass(ts(4, 5));
    pass.setBindGroup(0, this.sceneGroup);
    pass.setBindGroup(1, this.workGroup);
    pass.setBindGroup(2, this.imageGroup);
    pass.setPipeline(this.shade);
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
      enc.copyBufferToBuffer(this.pageUsed, 0, sb.buffer, 16 + MAX_REQUESTS * 8, this.pageUsed.size);
    }
    // Statistics and timings come back a few frames late, into whichever
    // readback buffer is free.
    const rb = this.readbacks.find((x) => !x.busy);
    if (rb) enc.copyBufferToBuffer(this.counters, 0, rb.buffer, 0, 32);
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
        this.stats = { work: u[0], hw: u[1], sw: u[2], instances: u[3], tested: u[4], triangles: u[5], overflow: u[6] };
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
