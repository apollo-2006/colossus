// a skinned model's skeleton and animations (name.cskn, include/skeleton.hpp), and posing:
// src/skeleton.cpp ported.

export async function fetchSkeleton(url) {
  const r = await fetch(url).catch(() => null);
  if (!r || !r.ok) return null;
  return parseSkeleton(await r.arrayBuffer());
}

export function parseSkeleton(buffer) {
  const view = new DataView(buffer), bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(0, 8)) !== 'CSKNv001') throw new Error('not a skeleton file (or an old one)');
  let at = 8;
  const u32 = () => { const v = view.getUint32(at, true); at += 4; return v; };
  const f32 = () => { const v = view.getFloat32(at, true); at += 4; return v; };
  const count = () => { const v = Number(view.getBigUint64(at, true)); at += 8; return v; };
  const s = { place: { upZ: u32(), base: [f32(), f32(), f32()], scale: f32() } };
  s.anchor = u32();
  s.nodes = [];
  for (let i = 0, n = count(); i < n; i++) {
    const parent = view.getInt32(at, true);
    at += 4;
    s.nodes.push({ parent, t: [f32(), f32(), f32()], r: [f32(), f32(), f32(), f32()], s: [f32(), f32(), f32()] });
  }
  s.joints = [];
  for (let i = 0, n = count(); i < n; i++) s.joints.push(u32());
  const ibm = count();
  s.inverseBind = new Float32Array(buffer.slice(at, at + 4 * ibm));
  at += 4 * ibm;
  const pages = count();
  s.pageJoints = new Uint32Array(buffer.slice(at, at + 8 * pages));  // lo, hi per page
  at += 8 * pages;
  s.animations = [];
  for (let a = 0, n = count(); a < n; a++) {
    const length = count();
    const name = new TextDecoder().decode(bytes.subarray(at, at + length));
    at += length;
    const anim = { name, duration: f32(), channels: [] };
    for (let c = 0, m = count(); c < m; c++) {
      const ch = { node: u32(), path: u32(), step: u32() };
      const nt = count();
      ch.times = new Float32Array(buffer.slice(at, at + 4 * nt));
      at += 4 * nt;
      const nv = count();
      ch.values = new Float32Array(buffer.slice(at, at + 4 * nv));
      at += 4 * nv;
      anim.channels.push(ch);
    }
    s.animations.push(anim);
  }
  return s;
}

// column major 4x4.
function mul(a, b) {
  const r = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let row = 0; row < 4; row++) {
      let v = 0;
      for (let k = 0; k < 4; k++) v += a[k * 4 + row] * b[c * 4 + k];
      r[c * 4 + row] = v;
    }
  return r;
}

function trs(t, q, s) {
  const [x, y, z, w] = q;
  const rot = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)];
  const r = new Float32Array(16);
  for (let c = 0; c < 3; c++) for (let row = 0; row < 3; row++) r[c * 4 + row] = rot[c * 3 + row] * s[c];
  r[12] = t[0]; r[13] = t[1]; r[14] = t[2]; r[15] = 1;
  return r;
}

// a channel's value at time t: linear between keys (rotations the shorter way, normalized).
function sample(c, t, out) {
  const w = c.path === 1 ? 4 : 3, ts = c.times;
  let k = 0;
  while (k < ts.length && ts[k] <= t) k++;
  if (k === 0 || ts.length === 1) { for (let i = 0; i < w; i++) out[i] = c.values[i]; return; }
  if (k >= ts.length) { for (let i = 0; i < w; i++) out[i] = c.values[(ts.length - 1) * w + i]; return; }
  const span = ts[k] - ts[k - 1], f = c.step || span <= 0 ? 0 : (t - ts[k - 1]) / span;
  let sign = 1;
  if (w === 4) {
    let d = 0;
    for (let i = 0; i < 4; i++) d += c.values[(k - 1) * 4 + i] * c.values[k * 4 + i];
    if (d < 0) sign = -1;
  }
  for (let i = 0; i < w; i++) out[i] = c.values[(k - 1) * w + i] + (sign * c.values[k * w + i] - c.values[(k - 1) * w + i]) * f;
  if (w === 4) {
    const l = Math.hypot(out[0], out[1], out[2], out[3]) || 1;
    for (let i = 0; i < 4; i++) out[i] /= l;
  }
}

// joint matrices (3x4, rows, 12 floats each) of animation `index` at time t (looped), in the
// placed model's space; -1: the rest pose.
export function poseJoints(s, index, t, rows) {
  const n = s.nodes.length;
  const local = s.nodes.map((nd) => ({ t: nd.t.slice(), r: nd.r.slice(), s: nd.s.slice() }));
  if (index >= 0 && index < s.animations.length) {
    const a = s.animations[index];
    const at = a.duration > 0 ? ((t % a.duration) + a.duration) % a.duration : 0;
    for (const c of a.channels) sample(c, at, c.path === 0 ? local[c.node].t : c.path === 1 ? local[c.node].r : local[c.node].s);
  }
  const world = new Array(n);
  const worldOf = (i) => {
    if (!world[i]) {
      const l = trs(local[i].t, local[i].r, local[i].s);
      world[i] = s.nodes[i].parent < 0 ? l : mul(worldOf(s.nodes[i].parent), l);
    }
    return world[i];
  };
  const p = s.place;
  const place = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  let unplace = new Float32Array(place);
  if (p.upZ) {  // (x, y, z) -> (x, z, -y)
    place[5] = 0; place[6] = -1; place[9] = 1; place[10] = 0;
    unplace[5] = 0; unplace[6] = 1; unplace[9] = -1; unplace[10] = 0;
  }
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) place[c * 4 + r] *= p.scale;
  place[12] = -p.base[0] * p.scale; place[13] = -p.base[1] * p.scale; place[14] = -p.base[2] * p.scale;
  const unscale = new Float32Array([1 / p.scale, 0, 0, 0, 0, 1 / p.scale, 0, 0, 0, 0, 1 / p.scale, 0, p.base[0], p.base[1], p.base[2], 1]);
  unplace = mul(unplace, unscale);
  for (let j = 0; j < s.joints.length; j++) {
    const m = mul(place, mul(mul(worldOf(s.joints[j]), s.inverseBind.subarray(16 * j, 16 * j + 16)), unplace));
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) rows[12 * j + 4 * r + c] = m[c * 4 + r];
  }
  return rows;
}

// the largest stretch of a 3x3 matrix (rows of a 3x4 at `at`), less the identity if asked, or
// of the difference of two: power iteration on m'm, a little over.
export function spectralNorm(a, at, lessIdentity, b = null, bt = 0) {
  const m = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m.push(a[at + 4 * r + c] - (b ? b[bt + 4 * r + c] : lessIdentity && r === c ? 1 : 0));
  const ata = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) ata[3 * i + j] += m[3 * k + i] * m[3 * k + j];
  let v = [0.577, 0.577, 0.577], lambda = 0;
  for (let it = 0; it < 32; it++) {
    const w = [0, 1, 2].map((i) => ata[3 * i] * v[0] + ata[3 * i + 1] * v[1] + ata[3 * i + 2] * v[2]);
    const l = Math.hypot(w[0], w[1], w[2]);
    if (l < 1e-20) return 0;
    lambda = l;
    v = w.map((x) => x / l);
  }
  return Math.sqrt(lambda) * 1.01 + 1e-6;
}
