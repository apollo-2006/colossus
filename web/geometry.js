// Reads a model's metadata (name.meta.gz, from web/split.py): its clusters,
// pages and their dependencies, all of a .cgeo file but the page data,
// which streams in from name.pages a page at a time (web/streamer.js).
// See include/paged_file.hpp for the layout.

const MAGIC = 'CGEOv006';

// Streams a URL, reporting progress as (bytes so far, total or 0), and
// gunzips it if its name ends in .gz.
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
  const clusters = array(112);
  const pageBytes = array(24);
  const deps = new Uint32Array(array(4));
  array(40);  // Level statistics
  const pageView = new DataView(pageBytes);
  const clusterCount = clusters.byteLength / 112;
  const words = new Uint32Array(clusters);
  const pages = [];
  for (let p = 0; p < pageBytes.byteLength / 24; p++) {
    const o = p * 24;
    const first = pageView.getUint32(o + 12, true), count = pageView.getUint32(o + 16, true);
    pages.push({
      offset: Number(pageView.getBigUint64(o, true)), size: pageView.getUint32(o + 8, true),
      deps: Array.from(deps.subarray(first, first + count)), children: [],
    });
  }
  // Each page's children: the finer pages its clusters stand for (word 24
  // of a cluster is the page of the clusters it stands for, word 23 its own).
  for (let c = 0; c < clusterCount; c++) {
    const creator = words[28 * c + 24];
    if (creator !== 0xffffffff) pages[words[28 * c + 23]].children.push(creator);
  }
  for (const p of pages) p.children = [...new Set(p.children)];
  let leafTriangles = 0;
  for (let c = 0; c < clusterCount; c++) if (words[28 * c + 22] === 0) leafTriangles += words[28 * c + 21];
  return {
    bounds: header.slice(0, 4), lodBounds: header.slice(4, 8), grid: header.slice(8, 12),
    clusters, clusterCount, pages, leafTriangles,
  };
}
