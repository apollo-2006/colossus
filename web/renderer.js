// The WebGPU renderer: the same pipeline as the Vulkan viewer, minus what
// WebGPU lacks (see compute.wgsl). Per frame:
//   instance_cull -> args_cull -> cluster_cull -> args_draw
//   -> software raster (depth, then ids) and hardware raster
//   -> shade -> blit to the canvas.
import { frustumPlanes, invert, lookTo, mul, perspectiveReverseZ } from './math.js';

const MAX_WORK = 1 << 20;
const MAX_VISIBLE = 1 << 20;
const FRAME_BYTES = 288;

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
        ...[1, 2, 3, 4, 5, 6, 7].map((b) => ro(b, C | V))],
    });
    this.workLayout = d.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5].map(rw) });
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
    if (this.timestamps) {
      this.querySet = d.createQuerySet({ type: 'timestamp', count: 8 });
      this.queryResolve = d.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | CS });
      this.timeReadbacks = [0, 1, 2].map(() => ({ buffer: d.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
    }
    this.stats = null;
    this.gpuMs = null;
  }

  // models: parsed .cgeo files. placements: {model, matrix (3x4 rows), scale}.
  loadScene(models, placements) {
    const d = this.device;
    let vertexBase = 0, cvBase = 0, ctBase = 0, clusterBase = 0;
    const counts = models.reduce((a, m) => ({
      v: a.v + m.positions.length, cv: a.cv + m.clusterVertices.length,
      ct: a.ct + m.clusterTriangles.length, c: a.c + m.clusterCount,
    }), { v: 0, cv: 0, ct: 0, c: 0 });
    const positions = new Float32Array(counts.v), normals = new Float32Array(counts.v);
    const clusterVertices = new Uint32Array(counts.cv), clusterTriangles = new Uint32Array(counts.ct);
    const clusters = new ArrayBuffer(counts.c * 96);
    const meshes = new ArrayBuffer(models.length * 48);
    this.leafTriangles = [];
    models.forEach((m, k) => {
      positions.set(m.positions, vertexBase * 3);
      normals.set(m.normals, vertexBase * 3);
      for (let i = 0; i < m.clusterVertices.length; i++) clusterVertices[cvBase + i] = m.clusterVertices[i] + vertexBase;
      clusterTriangles.set(m.clusterTriangles, ctBase);
      const src = new Uint32Array(m.clusters), dst = new Uint32Array(clusters, clusterBase * 96, m.clusterCount * 24);
      dst.set(src);
      for (let c = 0; c < m.clusterCount; c++) {
        dst[24 * c + 18] += cvBase;
        dst[24 * c + 19] += ctBase;
      }
      const mu = new Uint32Array(meshes, k * 48, 4), mf = new Float32Array(meshes, k * 48 + 16, 8);
      mu[0] = clusterBase;
      mu[1] = m.clusterCount;
      mf.set(m.bounds, 0);
      mf.set(m.lodBounds, 4);
      this.leafTriangles.push(m.leafTriangles);
      vertexBase += m.positions.length / 3;
      cvBase += m.clusterVertices.length;
      ctBase += m.clusterTriangles.length;
      clusterBase += m.clusterCount;
    });
    const instances = new ArrayBuffer(Math.max(1, placements.length) * 64);
    this.fullDetail = 0;
    placements.forEach((p, i) => {
      new Float32Array(instances, i * 64, 12).set(p.matrix);
      new Uint32Array(instances, i * 64 + 48, 1)[0] = p.model;
      new Float32Array(instances, i * 64 + 52, 1)[0] = p.scale;
      this.fullDetail += models[p.model].leafTriangles;
    });
    this.instanceCount = placements.length;

    for (const b of this.sceneBuffers || []) b.destroy();
    const upload = (data) => {
      const b = d.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.sceneBuffers = [clusters, clusterVertices, clusterTriangles, positions, normals, meshes, instances].map(upload);
    this.sceneGroup = d.createBindGroup({
      layout: this.sceneLayout,
      entries: [{ binding: 0, resource: { buffer: this.frameBuffer } },
        ...this.sceneBuffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } }))],
    });
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return;
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
      entries: [this.counters, this.work, this.hwVisible, this.swVisible, this.swDepthBuffer, this.swIdBuffer]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
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
    d.queue.writeBuffer(this.frameBuffer, 0, f);

    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.counters);
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
