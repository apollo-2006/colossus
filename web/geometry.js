// reads a model's metadata (name.meta.gz from web/split.py): clusters, pages
// and dependencies, all of a .cgeo but page data, which streams from name.pages
// (web/streamer.js). layout: include/paged_file.hpp.

const MAGIC = 'CGEOv008';

// streams a url, reporting (bytes so far, total or 0), gunzipping .gz names.
export async function fetchBytes(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  let received = 0;
  const counted = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      onProgress?.(received, total);
      controller.enqueue(chunk);
    },
  }));
  const stream = url.endsWith('.gz') ? counted.pipeThrough(new DecompressionStream('gzip')) : counted;
  return new Response(stream).arrayBuffer();
}

export async function fetchModel(stem, onProgress) {
  const model = parseMeta(await fetchBytes(`${stem}.meta.gz`, onProgress));
  model.pagesUrl = `${stem}.pages`;
  return model;
}

export function parseMeta(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== MAGIC) throw new Error(`not a geometry file (or an old one): ${magic}`);
  const view = new DataView(buffer);
  let at = 8;
  const header = new Float32Array(buffer.slice(at, at + 48));
  at += 48;
  const array = (elementBytes) => {
    const n = Number(view.getBigUint64(at, true));
    at += 8;
    const slice = buffer.slice(at, at + n * elementBytes);
    at += n * elementBytes;
    return slice;
  };
  const clusters = array(48);  // packed_cluster
  const shared = new Float32Array(array(20));  // page_bounds: centre, radius, error
  const pageBytes = array(24);
  const deps = new Uint32Array(array(4));
  array(40);  // level stats
  const pageView = new DataView(pageBytes);
  const clusterCount = clusters.byteLength / 48;
  const words = new Uint32Array(clusters);
  const pages = [];
  for (let p = 0; p < pageBytes.byteLength / 24; p++) {
    const o = p * 24;
    const first = pageView.getUint32(o + 12, true), count = pageView.getUint32(o + 16, true);
    pages.push({
      offset: Number(pageView.getBigUint64(o, true)), size: pageView.getUint32(o + 8, true),
      deps: Array.from(deps.subarray(first, first + count)), children: [], error: shared[5 * p + 4],
    });
  }
  // each page's children: the finer pages its clusters stand for (cluster word
  // 8 is the creator page, word 7 its own).
  for (let c = 0; c < clusterCount; c++) {
    const creator = words[12 * c + 8];
    if (creator !== 0xffffffff) pages[words[12 * c + 7]].children.push(creator);
  }
  for (const p of pages) p.children = [...new Set(p.children)];
  let leafTriangles = 0;  // the triangle count rides in word 9's top byte
  for (let c = 0; c < clusterCount; c++) if (words[12 * c + 8] === 0xffffffff) leafTriangles += words[12 * c + 9] >>> 24;
  return {
    bounds: header.slice(0, 4), lodBounds: header.slice(4, 8), grid: header.slice(8, 12),
    clusters, shared, clusterCount, pages, leafTriangles,
  };
}
