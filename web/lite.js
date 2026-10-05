// the fallback's live view, for browsers without webgpu: one model in webgl2. its pages stream
// through the same streamer (streamer.js), and each frame every cluster takes the same lod
// test as the webgpu demo's (lod_test() in compute.wgsl), here on the cpu: own error on
// screen within the threshold, parent error over it. no compute rasterizer, shadows or crowd.
import { fetchModel } from './geometry.js';
import { Streamer } from './streamer.js';
import { lookTo, normalize } from './math.js';

const NONE = 0xffffffff;
const POOL_BYTES = 32 << 20;  // packed pages resident at once; decoded, about four times that
const FOV = 0.9;

const VERTEX = `#version 300 es
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in uint info;  // cluster | level << 24
layout(location = 3) in vec2 uv;
uniform mat4 view_proj;
out vec3 v_normal;
out vec3 v_position;
out vec2 v_uv;
flat out uint v_info;
void main() {
  v_normal = normal;
  v_position = position;
  v_uv = uv;
  v_info = info;
  gl_Position = view_proj * vec4(position, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
in vec3 v_normal;
in vec3 v_position;
in vec2 v_uv;
flat in uint v_info;
uniform int mode;  // 0 shaded, 1 clusters, 2 lod level
uniform vec3 eye;
uniform bool textured;
uniform sampler2D albedo_map;  // as stored: the light here is tuned in display values, as the grey is
out vec4 color;
uint hash(uint x) {
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return x;
}
vec3 hash_color(uint x) {
  uint h = hash(x);
  return vec3(float(h & 255u), float((h >> 8u) & 255u), float((h >> 16u) & 255u)) / 255.0 * 0.75 + 0.2;
}
// blue for full detail through red to magenta for the coarsest, as the viewer's view 4.
vec3 level_color(uint level) {
  float t = clamp(float(level) / 14.0, 0.0, 1.0);
  return t < 0.5 ? mix(vec3(0.15, 0.45, 1.0), vec3(0.2, 0.9, 0.35), t * 2.0)
       : t < 0.8 ? mix(vec3(0.2, 0.9, 0.35), vec3(1.0, 0.3, 0.15), (t - 0.5) / 0.3)
                 : mix(vec3(1.0, 0.3, 0.15), vec3(0.95, 0.2, 0.9), (t - 0.8) / 0.2);
}
void main() {
  vec3 n = normalize(v_normal);
  vec3 to_eye = normalize(eye - v_position);
  if (!gl_FrontFacing) n = -n;
  vec3 albedo = mode == 1 ? hash_color(v_info & 0xffffffu)
              : mode == 2 ? level_color(v_info >> 24u)
              : textured  ? texture(albedo_map, v_uv).rgb
                          : vec3(0.82, 0.8, 0.76);
  vec3 sun = normalize(vec3(0.45, 0.8, 0.4));
  float diffuse = max(dot(n, sun), 0.0);
  vec3 sky = mix(vec3(0.36, 0.33, 0.3), vec3(0.62, 0.7, 0.82), n.y * 0.5 + 0.5);
  vec3 h = normalize(sun + to_eye);
  float spec = mode == 0 ? pow(max(dot(n, h), 0.0), 48.0) * 0.25 : 0.0;
  vec3 c = albedo * (diffuse * vec3(1.0, 0.96, 0.9) * 0.85 + sky * 0.55) + spec;
  color = vec4(pow(c, vec3(1.0 / 1.1)), 1.0);
}`;

// antialiasing after the 4x msaa resolve: fxaa (lottes 2009, the console variant), for thin
// creases msaa samples cannot hold, which showed as dotted lines along the folds.
const POST_VERTEX = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  uv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const POST_FRAGMENT = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D image;
uniform vec2 texel;
out vec4 color;
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void main() {
  vec3 m = texture(image, uv).rgb;
  float nw = luma(texture(image, uv + vec2(-1.0, -1.0) * texel).rgb), ne = luma(texture(image, uv + vec2(1.0, -1.0) * texel).rgb);
  float sw = luma(texture(image, uv + vec2(-1.0, 1.0) * texel).rgb), se = luma(texture(image, uv + vec2(1.0, 1.0) * texel).rgb);
  float lm = luma(m);
  float lo = min(lm, min(min(nw, ne), min(sw, se))), hi = max(lm, max(max(nw, ne), max(sw, se)));
  if (hi - lo < max(0.0312, hi * 0.125)) { color = vec4(m, 1.0); return; }
  vec2 dir = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
  float reduce = max((nw + ne + sw + se) * 0.03125, 1.0 / 128.0);
  dir = clamp(dir / (min(abs(dir.x), abs(dir.y)) + reduce), -8.0, 8.0) * texel;
  vec3 a = 0.5 * (texture(image, uv - dir / 6.0).rgb + texture(image, uv + dir / 6.0).rgb);
  vec3 b = a * 0.5 + 0.25 * (texture(image, uv - dir * 0.5).rgb + texture(image, uv + dir * 0.5).rgb);
  float lb = luma(b);
  color = vec4(lb < lo || lb > hi ? a : b, 1.0);
}`;

function compile(gl, type, source) {
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function decodeOctahedral(uv, out, at) {
  const ex = (uv & 2047) / 2047 * 2 - 1, ey = ((uv >>> 11) & 2047) / 2047 * 2 - 1;
  let x = ex, y = ey;
  const z = 1 - Math.abs(ex) - Math.abs(ey);
  if (z < 0) {
    x = (1 - Math.abs(ey)) * (ex >= 0 ? 1 : -1);
    y = (1 - Math.abs(ex)) * (ey >= 0 ? 1 : -1);
  }
  const l = Math.hypot(x, y, z) || 1;
  out[at] = Math.round(x / l * 127);
  out[at + 1] = Math.round(y / l * 127);
  out[at + 2] = Math.round(z / l * 127);
}

// a model's clusters unpacked once (load_cluster() in common.wgsl): bounds, the lod and parent
// spheres and errors, and where each one's vertices and triangles sit in its page.
function unpackClusters(model) {
  const n = model.clusterCount, w = new Uint32Array(model.clusters), f = new Float32Array(model.clusters);
  const s = model.shared;
  const c = {
    n, center: new Float32Array(4 * n), lod: new Float32Array(4 * n), parent: new Float32Array(4 * n),
    lodError: new Float32Array(n), parentError: new Float32Array(n), page: new Uint32Array(n), creator: new Uint32Array(n),
    level: new Uint32Array(n), offsets: new Uint32Array(n), origin: new Uint32Array(3 * n), counts: new Uint32Array(n),
  };
  const byPage = model.pages.map(() => []);
  for (let i = 0; i < n; i++) {
    const o = 12 * i;
    c.center.set(f.subarray(o, o + 4), 4 * i);
    c.offsets[i] = w[o + 5];
    c.level[i] = w[o + 6];
    const page = w[o + 7], creator = w[o + 8];
    c.page[i] = page;
    c.creator[i] = creator;
    c.origin[3 * i] = w[o + 9] & 0xffffff;
    c.origin[3 * i + 1] = w[o + 10];
    c.origin[3 * i + 2] = w[o + 11];
    c.counts[i] = (w[o + 6] >>> 24) | ((w[o + 9] >>> 24) << 16);  // vertices | triangles << 16
    c.parent.set(s.subarray(5 * page, 5 * page + 4), 4 * i);
    c.parentError[i] = s[5 * page + 4];
    if (creator === NONE) {
      c.lod.set(f.subarray(o, o + 4), 4 * i);
      c.lodError[i] = 0;
    } else {
      c.lod.set(s.subarray(5 * creator, 5 * creator + 4), 4 * i);
      c.lodError[i] = s[5 * creator + 4];
    }
    byPage[page].push(i);
  }
  c.byPage = byPage;
  return c;
}

// bc1 block (8 bytes at `at`) to 16 rgba texels, as decode_bc1() in src/texture_file.cpp.
function decodeBC1(src, at, out, width, x0, y0, skip, keep) {
  const c0 = src[at] | (src[at + 1] << 8), c1 = src[at + 2] | (src[at + 3] << 8);
  const bits = (src[at + 4] | (src[at + 5] << 8) | (src[at + 6] << 16) | (src[at + 7] << 24)) >>> 0;
  const rgb = (v) => [((v >>> 11) * 255 / 31) | 0, (((v >>> 5) & 63) * 255 / 63) | 0, ((v & 31) * 255 / 31) | 0];
  const p = [rgb(c0), rgb(c1), [0, 0, 0], [0, 0, 0]];
  for (let k = 0; k < 3; k++) {
    p[2][k] = c0 > c1 ? ((2 * p[0][k] + p[1][k]) / 3) | 0 : ((p[0][k] + p[1][k]) / 2) | 0;
    p[3][k] = c0 > c1 ? ((p[0][k] + 2 * p[1][k]) / 3) | 0 : 0;
  }
  for (let i = 0; i < 16; i++) {
    const tx = (i & 3) + skip.x, ty = (i >> 2) + skip.y;  // texel in the tile
    if (tx < keep.x0 || tx >= keep.x1 || ty < keep.y0 || ty >= keep.y1) continue;
    const c = p[(bits >>> (2 * i)) & 3], o = 4 * ((y0 + ty - keep.y0) * width + x0 + tx - keep.x0);
    out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = 255;
  }
}

// a texture for webgl: the first level at most `limit` a side, its tiles fetched in one range
// (a level's tiles are consecutive), decoded on the cpu without their borders, and mipmapped.
// webgl2 can't sample from a page pool, and a phone can't hold every level.
async function loadTexture(gl, texture, limit) {
  const levels = texture.levels;
  let level = 0, w = texture.width, h = texture.height;
  while (level + 1 < levels.length && Math.max(w, h) > limit) { level++; w = (w + 1) >> 1; h = (h + 1) >> 1; }
  const l = levels[level], tb = texture.tileBytes;
  const start = l.first * tb, end = (l.first + l.tilesX * l.tilesY) * tb;
  const r = await fetch(texture.tilesUrl, { headers: { Range: `bytes=${start}-${end - 1}` } });
  let bytes = new Uint8Array(await r.arrayBuffer());
  if (r.status === 200) bytes = bytes.subarray(start, end);  // a server ignoring ranges
  const rgba = new Uint8Array(w * h * 4);
  for (let ty = 0; ty < l.tilesY; ty++)
    for (let tx = 0; tx < l.tilesX; tx++) {
      const tile = (ty * l.tilesX + tx) * tb;
      // the tile's payload: 120 texels from 4 in, clipped to the level.
      const keep = { x0: 4, y0: 4, x1: 4 + Math.min(120, w - 120 * tx), y1: 4 + Math.min(120, h - 120 * ty) };
      for (let by = 0; by < 32; by++)
        for (let bx = 0; bx < 32; bx++) {
          if (4 * bx + 4 <= keep.x0 || 4 * bx >= keep.x1 || 4 * by + 4 <= keep.y0 || 4 * by >= keep.y1) continue;
          decodeBC1(bytes, tile + 8 * (by * 32 + bx), rgba, w, 120 * tx, 120 * ty, { x: 4 * bx, y: 4 * by }, keep);
        }
    }
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, Math.floor(Math.log2(Math.max(w, h))) + 1, gl.RGBA8, w, h);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
  if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
  return { texture: tex, width: w, height: h, bytes: end - start };
}

// a page's clusters as webgl buffers (cluster_position(), cluster_normal(), cluster_uv() and
// cluster_triangle() in common.wgsl): positions, normals, cluster | level and texture
// coordinates per vertex, and each cluster's run of indices.
function decodePage(gl, model, c, page, data) {
  const words = new Uint32Array(data);
  const bits = (at, bit, count) => {
    const wi = at + (bit >>> 5), sh = bit & 31;
    let v = words[wi] >>> sh;
    if (sh + count > 32) v |= words[wi + 1] << (32 - sh);
    return (v & ((count === 32 ? 0 : 1 << count) - 1)) >>> 0;
  };
  const list = c.byPage[page];
  let vertices = 0, indices = 0;
  for (const i of list) { vertices += c.counts[i] & 0xffff; indices += 3 * (c.counts[i] >>> 16); }
  const buffer = new ArrayBuffer(vertices * 24);
  const pos = new Float32Array(buffer), bytes = new Int8Array(buffer), info = new Uint32Array(buffer), uvs = new Uint16Array(buffer);
  const index = vertices > 65535 ? new Uint32Array(indices) : new Uint16Array(indices);
  const runs = new Map();  // cluster -> [first index, count]
  const g = model.grid;
  let v = 0, k = 0;
  for (const i of list) {
    const lw = c.level[i], bx = (lw >>> 8) & 15, by = (lw >>> 12) & 15, bz = (lw >>> 16) & 15, ib = (lw >>> 20) & 7;
    const vc = c.counts[i] & 0xffff, tc = c.counts[i] >>> 16;
    const vat = c.offsets[i] & 0xffff, tat = c.offsets[i] >>> 16, stride = bx + by + bz + 22;
    const ox = c.origin[3 * i], oy = c.origin[3 * i + 1], oz = c.origin[3 * i + 2];
    const first = v, tag = (i & 0xffffff) | ((lw & 255) << 24);
    // textured (level bit 23): corner and widths after the vertex run's following word.
    const uvAt = vat + ((vc * stride + 31) >>> 5) + 1;
    const textured = (lw >>> 23) & 1, corner = textured ? words[uvAt] : 0, widths = textured ? words[uvAt + 1] : 0;
    const bu = Math.min(widths & 31, 16), bv = Math.min((widths >>> 5) & 31, 16);
    for (let q = 0; q < vc; q++, v++) {
      const b = q * stride;
      pos[6 * v] = g[0] + g[3] * (ox + bits(vat, b, bx));
      pos[6 * v + 1] = g[1] + g[3] * (oy + bits(vat, b + bx, by));
      pos[6 * v + 2] = g[2] + g[3] * (oz + bits(vat, b + bx + by, bz));
      decodeOctahedral(bits(vat, b + bx + by + bz, 22), bytes, 24 * v + 12);
      info[6 * v + 4] = tag;
      if (textured) {
        uvs[12 * v + 10] = (corner & 0xffff) + bits(uvAt, 64 + q * (bu + bv), bu);
        uvs[12 * v + 11] = (corner >>> 16) + bits(uvAt, 64 + q * (bu + bv) + bu, bv);
      }
    }
    const mask = (1 << ib) - 1;
    runs.set(i, [k, 3 * tc]);
    for (let t = 0; t < tc; t++) {
      const packed = bits(tat, 3 * ib * t, 3 * ib);
      index[k++] = first + (packed & mask);
      index[k++] = first + ((packed >>> ib) & mask);
      index[k++] = first + ((packed >>> (2 * ib)) & mask);
    }
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, buffer, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.BYTE, true, 24, 12);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_INT, 24, 16);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 2, gl.UNSIGNED_SHORT, true, 24, 20);
  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, index, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, vbo, ibo, runs, type: index instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, size: index.BYTES_PER_ELEMENT };
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0]);
}

function multiply(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + i] * b[j * 4 + k];
      r[j * 4 + i] = s;
    }
  return r;
}

// starts the view in `canvas`, with `ui` its controls and readouts (index.html's #lite). false
// without webgl2.
export async function startLite(canvas, ui) {
  const gl = canvas.getContext('webgl2', { antialias: false });
  if (!gl) return false;
  // the scene into a 4x multisampled target, resolved into a texture that fxaa reads.
  const post = gl.createProgram();
  gl.attachShader(post, compile(gl, gl.VERTEX_SHADER, POST_VERTEX));
  gl.attachShader(post, compile(gl, gl.FRAGMENT_SHADER, POST_FRAGMENT));
  gl.linkProgram(post);
  if (!gl.getProgramParameter(post, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(post));
  const postLoc = { image: gl.getUniformLocation(post, 'image'), texel: gl.getUniformLocation(post, 'texel') };
  const samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES));
  const fbo = { width: 0, height: 0, msaa: gl.createFramebuffer(), color: gl.createRenderbuffer(), depth: gl.createRenderbuffer(),
    resolved: gl.createFramebuffer(), texture: gl.createTexture() };
  const sizeTarget = (width, height) => {
    if (fbo.width === width && fbo.height === height) return;
    fbo.width = width;
    fbo.height = height;
    gl.bindRenderbuffer(gl.RENDERBUFFER, fbo.color);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, width, height);
    gl.bindRenderbuffer(gl.RENDERBUFFER, fbo.depth);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.msaa);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, fbo.color);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, fbo.depth);
    // immutable storage, made again at each size (a resolve blit wants formats matched exactly).
    gl.deleteTexture(fbo.texture);
    fbo.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, fbo.texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.resolved);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fbo.texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const loc = { viewProj: gl.getUniformLocation(program, 'view_proj'), mode: gl.getUniformLocation(program, 'mode'), eye: gl.getUniformLocation(program, 'eye'),
    textured: gl.getUniformLocation(program, 'textured'), albedo: gl.getUniformLocation(program, 'albedo_map') };
  const multiDraw = gl.getExtension('WEBGL_multi_draw');

  const state = { yaw: 0.6, pitch: 0.15, distance: 3, threshold: 1, mode: 0, idle: true, dirty: true };
  let current = null;  // {model, c, streamer, gpu: Map(page -> buffers)}

  async function load(stem) {
    ui.status.textContent = 'loading';
    const model = await fetchModel(stem);
    if (current) for (const b of current.gpu.values()) { gl.deleteVertexArray(b.vao); gl.deleteBuffer(b.vbo); gl.deleteBuffer(b.ibo); }
    if (current && current.texture) gl.deleteTexture(current.texture.texture);
    const c = unpackClusters(model);
    const texture = model.texture ? await loadTexture(gl, model.texture, Math.min(2048, gl.getParameter(gl.MAX_TEXTURE_SIZE))) : null;
    const pages = model.pages.map((p, i) => ({ url: model.pagesUrl, offset: p.offset, size: p.size, deps: p.deps, children: p.children, error: p.error, pinned: i === 0 }));
    current = { model, c, texture, streamer: new Streamer(pages, POOL_BYTES, { maxInFlight: 64 }), gpu: new Map(), stamps: new Uint32Array(pages.length), frame: 0 };
    state.distance = model.bounds[3] * 2.6;
    state.dirty = true;
    ui.full.textContent = model.leafTriangles.toLocaleString('en-US');
    ui.status.textContent = '';
  }

  // the lod test (lod_test() and projected_error() in compute.wgsl and common.wgsl) for every
  // cluster: draws, and pages wanted with their error on screen as priority.
  function select(eye, forward, width, height, planes, near) {
    const { c, streamer } = current, table = streamer.table, threshold = state.threshold;
    const lodScale = height / (2 * Math.tan(FOV / 2));
    const corner = Math.sqrt(1 + (width * width + height * height) / (4 * lodScale * lodScale));
    const projected = (s, at, error) => {
      const vx = s[at] - eye[0], vy = s[at + 1] - eye[1], vz = s[at + 2] - eye[2], r = s[at + 3];
      const dist = Math.hypot(vx, vy, vz), along = vx * forward[0] + vy * forward[1] + vz * forward[2];
      const z = Math.max(along - r, near);
      let sec = corner;
      if (r < dist) {
        const cs = along / dist, sn = Math.sqrt(Math.max(1 - cs * cs, 0)), sa = r / dist;
        const cosFar = cs * Math.sqrt(1 - sa * sa) - sn * sa;
        if (cosFar > 1 / corner) sec = 1 / cosFar;
      }
      return error * lodScale * sec / z;
    };
    const draws = [], wanted = new Map();
    let triangles = 0;
    for (let i = 0; i < c.n; i++) {
      const parentError = c.parentError[i];
      if (Number.isFinite(parentError) && parentError < 3e38 && projected(c.parent, 4 * i, parentError) <= threshold) continue;
      // frustum: left, right, bottom, top.
      const at = 4 * i, r = c.center[at + 3];
      let inside = true;
      for (let p = 0; p < 4 && inside; p++) {
        const pl = planes[p];
        inside = pl[0] * c.center[at] + pl[1] * c.center[at + 1] + pl[2] * c.center[at + 2] + pl[3] >= -r;
      }
      if (!inside) continue;
      const self = c.lodError[i] === 0 ? 0 : projected(c.lod, 4 * i, c.lodError[i]);
      const creator = c.creator[i];
      const finerResident = creator !== NONE && table[creator] !== NONE;
      if (self > threshold && creator !== NONE && !finerResident) wanted.set(creator, Math.max(wanted.get(creator) || 0, self));
      if (table[c.page[i]] === NONE || (self > threshold && finerResident)) continue;
      draws.push(i);
      triangles += c.counts[i] >>> 16;
    }
    return { draws, wanted: [...wanted.entries()], triangles };
  }

  // drag to orbit, wheel or pinch to zoom.
  const pointers = new Map();
  let pinch = 0;
  canvas.addEventListener('pointerdown', (e) => { pointers.set(e.pointerId, [e.clientX, e.clientY]); canvas.setPointerCapture(e.pointerId); state.idle = false; });
  const lift = (e) => { pointers.delete(e.pointerId); pinch = 0; };
  canvas.addEventListener('pointerup', lift);
  canvas.addEventListener('pointercancel', lift);
  canvas.addEventListener('pointermove', (e) => {
    const last = pointers.get(e.pointerId);
    if (!last) return;
    if (pointers.size === 1) {
      state.yaw -= (e.clientX - last[0]) * 0.008;
      state.pitch = Math.max(-1.4, Math.min(1.4, state.pitch + (e.clientY - last[1]) * 0.008));
    }
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) zoom(pinch / d);
      pinch = d;
    }
    state.dirty = true;
  });
  const zoom = (f) => {
    const r = current ? current.model.bounds[3] : 1;
    state.distance = Math.max(r * 0.5, Math.min(r * 40, state.distance * f));
    state.dirty = true;
  };
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); state.idle = false; zoom(e.deltaY > 0 ? 1.12 : 1 / 1.12); }, { passive: false });
  ui.model.onchange = () => load(ui.model.value);
  ui.mode.onchange = () => { state.mode = Number(ui.mode.value); state.dirty = true; };
  ui.threshold.oninput = () => {
    state.threshold = 2 ** Number(ui.threshold.value);
    ui.thresholdLabel.textContent = `${state.threshold < 1 ? state.threshold.toFixed(2) : state.threshold} px`;
    state.dirty = true;
  };

  let shown = 0, cpu = 0, frames = 0, picked = null;
  const frame = (now) => {
    requestAnimationFrame(frame);
    if (!current) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr)), height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; state.dirty = true; }
    const { model, c, streamer, gpu } = current;
    // until someone takes over, turn slowly and drift in and out, so the levels change.
    if (state.idle) {
      state.yaw += 0.0025;
      state.distance = model.bounds[3] * (2.2 + 1.4 * Math.sin(now / 5000));
      state.dirty = true;
    }
    const t0 = performance.now();
    const target = [model.bounds[0], model.bounds[1], model.bounds[2]];
    const eye = [
      target[0] + state.distance * Math.cos(state.pitch) * Math.sin(state.yaw),
      target[1] + state.distance * Math.sin(state.pitch),
      target[2] + state.distance * Math.cos(state.pitch) * Math.cos(state.yaw),
    ];
    const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    const near = Math.max(1e-4, (state.distance - model.lodBounds[3]) * 0.5, state.distance * 0.01);
    const viewProj = multiply(perspective(FOV, width / height, near, state.distance + model.lodBounds[3] * 4), lookTo(eye, forward, [0, 1, 0]));
    const m = viewProj, row = (r) => [m[r], m[4 + r], m[8 + r], m[12 + r]];
    const r0 = row(0), r1 = row(1), r3 = row(3);
    const planes = [r3.map((v, i) => v + r0[i]), r3.map((v, i) => v - r0[i]), r3.map((v, i) => v + r1[i]), r3.map((v, i) => v - r1[i])];

    // pick again when the view or the resident pages change.
    if (state.dirty || !picked || streamer.stats.inFlight) picked = select(eye, forward, width, height, planes, near);
    state.dirty = false;
    current.frame++;
    for (const i of picked.draws) current.stamps[c.page[i]] = current.frame;
    streamer.noteUsed(current.stamps);
    const uploads = streamer.service(current.frame, picked.wanted, state.threshold);
    for (const u of uploads) gpu.set(u.page, decodePage(gl, model, c, u.page, u.data));
    if (uploads.length) state.dirty = true;
    // pages the streamer let go of.
    for (const [p, b] of gpu) if (!streamer.resident(p)) { gl.deleteVertexArray(b.vao); gl.deleteBuffer(b.vbo); gl.deleteBuffer(b.ibo); gpu.delete(p); }

    sizeTarget(width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.msaa);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.62, 0.69, 0.77, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    gl.uniformMatrix4fv(loc.viewProj, false, viewProj);
    gl.uniform1i(loc.mode, state.mode);
    gl.uniform3fv(loc.eye, eye);
    gl.uniform1i(loc.textured, current.texture ? 1 : 0);
    gl.uniform1i(loc.albedo, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, current.texture ? current.texture.texture : null);
    // by page, consecutive runs merged; one multi draw per page where the extension is there.
    const byPage = new Map();
    for (const i of picked.draws) {
      const b = gpu.get(c.page[i]);
      if (!b) continue;
      if (!byPage.has(b)) byPage.set(b, []);
      byPage.get(b).push(b.runs.get(i));
    }
    let calls = 0;
    for (const [b, runs] of byPage) {
      runs.sort((x, y) => x[0] - y[0]);
      const firsts = [], counts = [];
      for (const [first, count] of runs) {
        const n = counts.length;
        if (n && firsts[n - 1] + counts[n - 1] === first) counts[n - 1] += count;
        else { firsts.push(first); counts.push(count); }
      }
      gl.bindVertexArray(b.vao);
      if (multiDraw) {
        multiDraw.multiDrawElementsWEBGL(gl.TRIANGLES, counts, 0, b.type, firsts.map((f) => f * b.size), 0, counts.length);
        calls++;
      } else {
        for (let k = 0; k < counts.length; k++, calls++) gl.drawElements(gl.TRIANGLES, counts[k], b.type, firsts[k] * b.size);
      }
    }
    gl.bindVertexArray(null);
    // resolve the samples, then fxaa onto the canvas.
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbo.msaa);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, fbo.resolved);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(post);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fbo.texture);
    gl.uniform1i(postLoc.image, 0);
    gl.uniform2f(postLoc.texel, 1 / width, 1 / height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    cpu += performance.now() - t0;
    frames++;
    if (now - shown > 250) {
      const st = streamer.stats;
      ui.triangles.textContent = picked.triangles.toLocaleString('en-US');
      ui.clusters.textContent = `${picked.draws.length.toLocaleString('en-US')} of ${c.n.toLocaleString('en-US')}`;
      ui.pages.textContent = `${st.resident} pages, ${(st.bytes / 1e6).toFixed(1)} MB${st.inFlight ? ' (streaming)' : ''}`;
      ui.cpu.textContent = `${(cpu / frames).toFixed(1)} ms, ${calls} draw${calls === 1 ? '' : 's'}`;
      shown = now;
      cpu = 0;
      frames = 0;
    }
  };
  await load(ui.model.value);
  requestAnimationFrame(frame);
  return true;
}
