// Reads a .ngeo file (include/geometry_file.hpp), gzipped or not: the
// hierarchy as the GPU reads it, so the arrays go to buffers as they are.

const MAGIC = 'NGEOv002';

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
  const header = new Float32Array(buffer.slice(at, at + 32));
  at += 32;
  // Each array: a 64-bit element count, then the elements.
  const array = (elementBytes) => {
    const n = Number(view.getBigUint64(at, true));
    at += 8;
    const slice = buffer.slice(at, at + n * elementBytes);
    at += n * elementBytes;
    return slice;
  };
  const positions = new Float32Array(array(4));
  const normals = new Float32Array(array(4));
  const clusters = array(96);
  const clusterVertices = new Uint32Array(array(4));
  const clusterTriangles = new Uint32Array(array(4));
  const clusterCount = clusters.byteLength / 96;
  // Leaf triangles: the triangle counts (word 21 of each cluster's 24) of
  // the clusters at level 0 (word 22).
  const words = new Uint32Array(clusters);
  let leafTriangles = 0;
  for (let c = 0; c < clusterCount; c++) if (words[24 * c + 22] === 0) leafTriangles += words[24 * c + 21];
  return {
    bounds: header.slice(0, 4),
    lodBounds: header.slice(4, 8),
    positions, normals, clusters, clusterVertices, clusterTriangles, clusterCount, leafTriangles,
  };
}
