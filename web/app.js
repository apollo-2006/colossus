// The demo page: loads the models, places a crowd of them, and flies a
// camera over it with the WebGPU renderer.
import { fetchModel } from './geometry.js';
import { FLAG_CONE, FLAG_FRUSTUM, FLAG_OCCLUSION, FLAG_SHADOWS, FLAG_SOFTWARE, FLAG_TAA, Renderer } from './renderer.js';

const $ = (id) => document.getElementById(id);
const MODELS = ['models/lucy', 'models/dragon'];
const POOL_BYTES = 192 << 20;  // Pages streamed in; the rest stay on the server

const human = (v) => (v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : `${v}`);

// A small seeded generator, so the crowd is the same on every visit.
function random(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// An n by n grid, models alternating, each turned and sized at random.
export function crowd(n, models, spacing = 1.25) {
  const rand = random(7), materialRand = random(11);
  const out = [];
  for (let z = 0; z < n; z++)
    for (let x = 0; x < n; x++) {
      const angle = n === 1 ? 0 : rand() * Math.PI * 2;
      const scale = n === 1 ? 1 : 0.85 + 0.3 * rand();
      const c = Math.cos(angle) * scale, s = Math.sin(angle) * scale;
      const tx = (x - (n - 1) / 2) * spacing, tz = (z - (n - 1) / 2) * -spacing;
      // Mostly marble, some sandstone, bronze, gold and granite, as the
      // native viewer mixes them (shade materials in compute.wgsl).
      const material = n === 1 ? 1 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5][Math.floor(materialRand() * 20)];
      out.push({ model: (z * n + x) % models, scale, material, matrix: [c, 0, s, tx, 0, scale, 0, 0, -s, 0, c, tz] });
    }
  return out;
}

export const camera = {
  eye: [-1.2, 0.55, 3.2],
  yaw: 0.45,
  pitch: -0.12,
  fov: 1.0,
  near: 0.01,
  get forward() {
    return [Math.cos(this.pitch) * Math.sin(this.yaw), Math.sin(this.pitch), -Math.cos(this.pitch) * Math.cos(this.yaw)];
  },
};

async function main() {
  const canvas = $('view');
  const renderer = await Renderer.create(canvas);
  if (!renderer) {
    $('loading').textContent = 'This page needs WebGPU (Chrome or Edge 113+, or Safari 26). Your browser does not offer it.';
    return;
  }
  const progress = MODELS.map(() => [0, 0]);
  const models = await Promise.all(MODELS.map((url, i) => fetchModel(url, (got, total) => {
    progress[i] = [got, total];
    const g = progress.reduce((a, p) => a + p[0], 0), t = progress.reduce((a, p) => a + p[1], 0);
    $('loading').textContent = `loading models: ${(g / 1e6).toFixed(1)}${t ? ` of ${(t / 1e6).toFixed(1)}` : ''} MB`;
  })));
  $('loading').remove();
  renderer.loadModels(models, Math.min(POOL_BYTES, renderer.device.limits.maxStorageBufferBindingSize));

  const settings = { flags: FLAG_CONE | FLAG_FRUSTUM | FLAG_SOFTWARE | FLAG_SHADOWS | FLAG_OCCLUSION | FLAG_TAA, threshold: 1, mode: 0, swPixels: 32 };
  let frozen = null;
  const build = () => {
    const n = Number($('grid').value);
    renderer.setPlacements(crowd(n, models.length));
    $('grid-label').textContent = `${n * n} instances`;
    $('full').textContent = human(renderer.fullDetail);
  };
  $('grid').oninput = build;
  build();
  $('threshold').oninput = () => {
    settings.threshold = 2 ** Number($('threshold').value);
    $('threshold-label').textContent = `${settings.threshold < 1 ? settings.threshold.toFixed(2) : settings.threshold} px`;
  };
  $('mode').onchange = () => (settings.mode = Number($('mode').value));
  const toggle = (id, flag) => ($(id).onchange = () => (settings.flags ^= flag));
  toggle('software', FLAG_SOFTWARE);
  toggle('cone', FLAG_CONE);
  toggle('shadows', FLAG_SHADOWS);
  toggle('occlusion', FLAG_OCCLUSION);
  toggle('taa', FLAG_TAA);
  $('freeze').onchange = () => {
    frozen = $('freeze').checked ? { eye: [...camera.eye], forward: camera.forward, fov: camera.fov, near: camera.near } : null;
  };

  // Drag to look, WASD (and Q, E) to fly, wheel for speed.
  const keys = new Set();
  let speed = 1.5, dragging = false, last = [0, 0], idle = true;
  addEventListener('keydown', (e) => { if (!e.target.closest('input, select')) { keys.add(e.code); idle = false; } });
  addEventListener('keyup', (e) => keys.delete(e.code));
  canvas.addEventListener('pointerdown', (e) => { dragging = true; idle = false; last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => (dragging = false));
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    camera.yaw += (e.clientX - last[0]) * 0.004;
    camera.pitch = Math.max(-1.55, Math.min(1.55, camera.pitch - (e.clientY - last[1]) * 0.004));
    last = [e.clientX, e.clientY];
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    speed = Math.max(0.05, Math.min(200, speed * (e.deltaY < 0 ? 1.25 : 0.8)));
  }, { passive: false });

  let then = performance.now(), shown = 0, frames = 0, cpuMs = 0;
  const frame = (now) => {
    const dt = Math.min(0.1, (now - then) / 1000);
    then = now;
    const f = camera.forward, r = [-f[2], 0, f[0]];
    const len = Math.hypot(r[0], r[2]) || 1;
    const move = [0, 0, 0];
    const add = (v, s) => { for (let k = 0; k < 3; k++) move[k] += v[k] * s; };
    if (keys.has('KeyW')) add(f, 1);
    if (keys.has('KeyS')) add(f, -1);
    if (keys.has('KeyD')) add(r, 1 / len);
    if (keys.has('KeyA')) add(r, -1 / len);
    if (keys.has('KeyE')) move[1] += 1;
    if (keys.has('KeyQ')) move[1] -= 1;
    const boost = keys.has('ShiftLeft') ? 4 : 1;
    for (let k = 0; k < 3; k++) camera.eye[k] += move[k] * speed * boost * dt;
    // Until someone takes over, drift slowly into the crowd and back.
    if (idle) {
      camera.yaw += 0.04 * dt;
      camera.eye[2] -= Math.sin(now / 4000) * 0.15 * dt;
    }

    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    renderer.resize(w, h);
    const view = { eye: camera.eye, forward: camera.forward, fov: camera.fov, near: camera.near };
    renderer.draw(view, frozen || view, settings);

    frames++;
    cpuMs += dt * 1000;
    if (now - shown > 250 && renderer.stats) {
      const s = renderer.stats, g = renderer.gpuMs;
      $('frame').textContent = g ? `${g.total.toFixed(2)} ms GPU` : `${(cpuMs / frames).toFixed(1)} ms a frame`;
      $('breakdown').textContent = g ? `cull and compute raster ${g.cull.toFixed(2)}, hardware raster ${g.raster.toFixed(2)}, shading ${g.shade.toFixed(2)}` : '';
      $('triangles').textContent = human(s.triangles);
      $('clusters').textContent = `${human(s.hw + s.sw)} (${human(s.sw)} in compute)`;
      $('instances').textContent = `${s.instances} of ${renderer.instanceCount}`;
      $('hidden').textContent = `${human(s.hiddenLastFrame)} clusters, ${s.instancesOccluded} instances`;
      $('size').textContent = `${w} x ${h}`;
      const st = renderer.streamer.stats;
      $('pages').textContent = `${st.resident} pages, ${(st.bytes / 1e6).toFixed(0)} MB fetched${st.inFlight ? ' (streaming)' : ''}`;
      shown = now;
      frames = 0;
      cpuMs = 0;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

if (!window.COLOSSUS_TEST) main().catch((e) => {
  console.error(e);
  const l = $('loading');
  if (l) l.textContent = `error: ${e.message}`;
});
