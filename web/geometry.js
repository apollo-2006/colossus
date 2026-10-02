// Reads a .cgeo file (include/paged_file.hpp), gzipped or not: the
// clusters and all their pages, which go to buffers as they are. The page
// does not stream: every page is loaded, and the page table is where each
// one starts in the data.

const MAGIC = 'CGEOv006';

// Streams a URL, reporting progress as (bytes so far, total or 0), and
// gunzips it if its name ends in .gz.
export async function fetchModel(url, onProgress) {
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
  return parseModel(await new Response(stream).arrayBuffer());
}

export function parseModel(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== MAGIC) throw new Error(`not a geometry file (or an old one): ${magic}`);
  const view = new DataView(buffer);
  let at = 8;
  const header = new Float32Array(buffer.slice(at, at + 48));
  at += 48;
  // Each array: a 64-bit element count, then the elements.
  const array = (elementBytes) => {
    const n = Number(view.getBigUint64(at, true));
    at += 8;
    const slice = buffer.slice(at, at + n * elementBytes);
    at += n * elementBytes;
    return slice;
  };
  const clusters = array(112);
  const pages = array(24);
  array(4);   // Page dependencies: everything is loaded here, so unused
  array(40);  // Level statistics
  const dataBytes = Number(view.getBigUint64(at, true));
  at = Math.ceil((at + 8) / 16) * 16;
  const data = new Uint32Array(buffer.slice(at, at + dataBytes));
  // Each page's first word in data: the page table, with nothing to stream.
  const pageView = new DataView(pages);
  const pageCount = pages.byteLength / 24;
  const pageTable = new Uint32Array(pageCount);
  for (let p = 0; p < pageCount; p++) pageTable[p] = Number(pageView.getBigUint64(p * 24, true)) / 4;
  const clusterCount = clusters.byteLength / 112;
  // Leaf triangles: the triangle counts (word 21 of each cluster's 28) of
  // the clusters at level 0 (word 22).
  const words = new Uint32Array(clusters);
  let leafTriangles = 0;
  for (let c = 0; c < clusterCount; c++) if (words[28 * c + 22] === 0) leafTriangles += words[28 * c + 21];
  return {
    bounds: header.slice(0, 4), lodBounds: header.slice(4, 8), grid: header.slice(8, 12),
    clusters, clusterCount, pageTable, data, leafTriangles,
  };
}
