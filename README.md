# colossus

[![ci](https://github.com/apollo-2006/colossus/actions/workflows/ci.yml/badge.svg)](https://github.com/apollo-2006/colossus/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Virtualized geometry from scratch, in C++20 and Vulkan: a builder that turns a scanned
model into a crack-free hierarchy of clusters, and a renderer that draws thousands of
copies of it by picking, cluster by cluster, the coarsest detail that stays within a
pixel of the original. It is the technique behind Unreal Engine 5's Nanite, written
with no libraries beyond Vulkan and GLFW: the clustering, the simplifier, the graph
partitioning, the culling, both rasterizers and the shading are all in this repository.

**[Fly through it in your browser →](https://apollo-2006.github.io/colossus/)** A
WebGPU port of the renderer (see [In the browser](#in-the-browser)), with the debug views,
the error threshold and the crowd size to play with.

![900 instances of Lucy and the XYZ RGB dragon, lit by the sun with ray traced shadows](docs/crowd.png)

900 instances of Lucy (28M triangles) and the XYZ RGB dragon (7.2M), 15.9 billion
triangles at full detail, drawn at 1920x1080 in 0.72 ms on an RX 9070 XT. About 0.86M
triangles reach the screen.

## How it works

### Building the hierarchy (`colossus_build`)

* **Clusters.** The mesh is split into clusters of at most 128 triangles and 128
  vertices (`src/cluster.cpp`), the unit every later step works in. A cluster grows
  greedily: the next triangle is the one touching it that adds the fewest new
  vertices, stays nearest its center, and has the fewest unused neighbours, so notches
  that would become islands are taken while the cluster still reaches them. Each new
  cluster starts on the last one's frontier. On the dragon this fills clusters to 108
  triangles on average; seeding in Morton order alone gave 90.
* **Groups, locked, then simplified.** Each level's clusters are partitioned into
  groups of about eight that share the most edges (`src/dag.cpp`). Every vertex a
  group shares with another group is locked, the group's triangles are merged and
  simplified to half, and the result is split into new clusters: the next level.
  Because group outlines never move, a simplified group meets an unsimplified
  neighbour edge for edge. The next level's groups fall differently, so a border
  locked at one level is free at the next, and no seam lasts more than a level.
* **Quadric simplification.** `src/simplify.cpp` collapses edges cheapest first by
  quadric error (Garland and Heckbert): each vertex carries the summed squared
  distances to the planes around it. Collapses keep one endpoint, so every level
  indexes the original vertices and the whole hierarchy shares one vertex buffer.
  Collapses that would flip a triangle or pinch the surface (the link condition) are
  refused, and a vertex on a hole in the scan may only slide along the hole's edge.
* **Errors that only grow.** A cluster stores its own error and its parent group's,
  each with a bounding sphere, and both only grow toward the root. So the test
  `own error on screen <= 1 pixel < parent error on screen` picks exactly one level
  along every path from a leaf to a root, and every cluster of a group agrees. Each
  cluster is tested on its own, by its own GPU thread, with no tree to walk.
* **Checked for cracks.** Since every level uses the original vertices, a crack is
  exact to detect: an edge used by one triangle of a cut whose ends are not both on a
  hole in the scan. `colossus_build --check` selects 25 cuts across the error range and
  counts them. Lucy builds 18 levels in 24 s and the dragon 20 levels in 5.4 s, every
  cut with 0 cracked edges.

### Drawing it (`colossus`)

1. **Instance culling** (`instance_cull.comp`) drops instances outside the view and,
   for the rest, finds which clusters could be drawn at all. Clusters are stored in
   order of parent error, and one can only be drawn while its parent looks too coarse,
   so a binary search skips every cluster fine enough anywhere on the instance. A far
   instance tests a short tail of its coarsest clusters, not all 573k.
2. **Cluster culling** (`cluster_cull.comp`) tests the rest: the LOD cut, the
   frustum, the normal cone, and occlusion. Survivors are appended to a list with one
   atomic per subgroup. This began as a task shader; on RADV each task workgroup has a
   fixed cost, and testing 630k clusters took 1.13 ms that way and 0.39 ms as compute.
3. **Two rasterizers.** Clusters over 32 pixels on screen go to mesh shaders. Smaller
   ones go to `sw_raster.comp`, a compute rasterizer with an invocation per triangle,
   since their triangles cover a pixel or two and the hardware handles those poorly.
   Both write `depth << 32 | (cluster, triangle)` into one 64-bit visibility buffer
   with an atomic max, so they mix freely. The compute rasterizer snaps to 1/256 pixel
   like the hardware and follows a top-left fill rule, so where the two meet there are
   no gaps. Its edge functions are 32-bit, measured from each triangle's corner: RDNA
   has no 64-bit integer multiply, and the first, 64-bit version lost to the hardware
   at every size.
4. **Occlusion in two passes.** Pass 1 tests against last frame's depth pyramid, from
   last frame's camera, and draws what it cannot prove hidden. A pyramid is built from
   what it drew, and pass 2 tests what pass 1 rejected again, drawing anything this
   frame reveals. The pyramid is built from the visibility buffer, so both
   rasterizers count.
5. **Shading** (`shade.comp`) finds each pixel's triangle again, intersects the
   pixel's ray with it for exact barycentrics, and interpolates normals. Where the GPU
   has ray queries, one shadow ray per pixel is traced toward the sun against a coarser
   copy of each model: the finest cut of its own hierarchy within 256k triangles. The
   hierarchy also says how far that copy can stray from what is drawn, and shadow rays
   start that far above the surface.

6. **Streaming.** Only the clusters' bounds and errors stay resident. Their geometry is
   packed into pages, one per group, and paged from disk into a fixed pool
   (`--pool-mb`, 1 GB by default) as the GPU asks for it. A cluster that is drawn but
   would rather be its finer clusters, whose page is missing, requests that page,
   with its error on screen as the priority. Two frames later `viewer/streamer.hpp`
   reads the requests and issues loads for the most wanted pages, evicting the least
   recently used ones when the pool is full. Loader threads read the pages; the render
   thread only issues loads and publishes the ones that have finished.

   What keeps a streamed cut whole: a cluster is drawn either when it is the right
   level, or when the finer clusters it stands for are not resident. That only works
   if those stand-ins are always there, so a page may only be resident while the
   pages of the coarser clusters its group was simplified into are. Loads are issued
   coarsest first and published in the order they were issued, however the reads
   finish; a page counts as a dependent from the moment its load is issued, and is
   evicted only when nothing resident or loading depends on it. Then along every path
   from leaf to root, exactly one cluster is drawn, whatever has been loaded.

   Pages are compact: a vertex is two words, its position snapped to a grid over the
   model and stored as 14-bit offsets from its cluster's corner, its normal
   octahedral at 11 + 11 bits. A vertex shared by clusters snaps to the same grid
   point in each, so quantizing opens no cracks; Lucy's grid step is 3.3e-5 of her
   height. Her pages are 509 MB.

   The 900 instance view at the top settles in 21 frames with 321 pages, 2 MB, of
   the 640 MB on disk resident, since every instance draws from the same pages; after
   that it draws the same 856k triangles as with everything resident. A 300 frame
   flight through the crowd with a 16 MB pool turns over thousands of pages and shows
   no holes at any frame checked. From a cold file cache, the render thread's worst
   frame in the streamer is 0.25 ms with loader threads, against 11.5 ms reading pages
   itself (`--sync-loads`).

![Lucy, each cluster in its own color](docs/clusters.png)

Each cluster of Lucy in its own color (debug view 2).

![The crowd colored by LOD level, blue for full detail through red to magenta for the coarsest](docs/lod_levels.png)

LOD levels (view 4): the nearest dragons draw mid levels, distant statues the coarsest,
and single instances mix levels where they recede.

![The crowd colored by rasterizer: orange for compute, blue for hardware](docs/rasterizers.png)

Which rasterizer drew each pixel (view 8): orange for the compute rasterizer, blue for
mesh shaders. Only the nearest surfaces are worth sending to the hardware.

## In the browser

`web/` is the renderer ported to WebGPU, which has no mesh shaders, no 64-bit atomics and
no ray queries:

* Large clusters go through an ordinary render pipeline: one instance per visible
  cluster, 384 vertices each, every vertex pulling its cluster, triangle and corner
  from storage buffers.
* The compute rasterizer cannot write depth and triangle in one atomic, so it runs
  twice: a 32-bit atomic max keeps the nearest depth, then a second pass writes the
  triangle wherever its depth won. Shading takes the nearer of the two rasterizers'
  results at each pixel.
* No shadows, no occlusion culling and no streaming yet.

The models are trimmed to 400k triangles at their finest (`colossus_build --max-triangles`,
which keeps the hierarchy above that cut intact), 7.5 MB each gzipped. Every page is
loaded up front, so the page table is just each page's place in the file. 900 instances
take 0.47 ms of GPU time at 1600x813 in Chrome on the RX 9070 XT. `web/build.sh` builds the
models, and `node tests/web_screenshot.mjs` renders the page in a headless Chrome.

## Build & run

Needs a GPU with Vulkan 1.3 and `VK_EXT_mesh_shader`, the Vulkan headers and loader,
GLFW, and `glslc`. Ray queries are optional: without them the viewer draws no shadows.

```bash
git clone https://github.com/apollo-2006/colossus.git
cd colossus
make
models/fetch.sh            # downloads Lucy and the dragon (380 MB) and builds both

./colossus --model models/lucy.cgeo --model models/xyzrgb_dragon.cgeo --grid 30
./colossus --model models/lucy.cgeo                       # one Lucy
./colossus_build any.ply out.cgeo --check                       # your own model, checked for cracks
```

| keys | |
|---|---|
| WASD, Q E | move, down and up; drag to look; scroll for speed, shift to hurry |
| 1 to 8 | shaded, clusters, triangles, LOD level, groups, instances, holes, rasterizer |
| [ and ] | halve or double the error threshold (1 pixel) |
| F | freeze culling, to fly out and watch it from outside |
| C V O R H | cone culling, frustum culling, occlusion, compute rasterizer, shadows |
| T, P | wireframe; print the camera as a `--camera` argument |

`--headless --frames N --screenshot out.png` renders without a window and prints the
median frame's timings. `docs/shots.sh` renders this page's images.

## Tests

```bash
make test
```

`tests/builder_test.cpp` runs on procedural meshes, so it needs no GPU and no
downloads: PLY (ASCII and both byte orders) and OBJ reading, welding, cluster limits
and coverage, the simplifier's locked vertices, flips and area on a flat grid, and a
full hierarchy for a closed sphere, a sphere with holes and a flat grid, each checked
for cracks at 31 cuts, paged, written to a file and read back exactly. With the group
locks removed from the builder, the crack check reports 39,095 cracked edges on the
closed sphere alone.

`tests/streamer_test.cpp` runs the streamer for 3,000 frames of random requests into a
pool a tenth the size of a model, synchronously and with loader threads, checking
after each frame that every resident page's dependencies are resident. It caught one
way to break that (making room for a page could evict a page it was about to depend
on), fixed before the streamer was committed. The builder's tests also check that
every vertex shared by clusters decodes bit for bit the same from each. CI runs both
and builds the viewer on every push.

## Performance

RX 9070 XT (RADV), 1920x1080, the 900 instance scene above, median of 200 frames:

| camera | frame | culling | raster | pass 2 | shading | triangles | clusters |
|---|---|---|---|---|---|---|---|
| beside Lucy (top image) | 0.76 ms | 0.15 | 0.13 | 0.09 | 0.40 | 0.86M | 9.8k |
| raised (LOD image) | 0.82 ms | 0.06 | 0.15 | 0.08 | 0.53 | 1.15M | 16.6k |
| ground level | 0.85 ms | 0.17 | 0.11 | 0.09 | 0.47 | 0.82M | 9.0k |

Measured after 100 frames of streaming. Fetching geometry through the page table costs
2% to 6% against keeping everything resident, which drew the same triangles in 0.72,
0.78 and 0.83 ms. Shadow rays are most of the frame: without them the three cameras
take 0.38, 0.36 and 0.42 ms. Turning the compute rasterizer off costs 0.10 to 0.14 ms (raster time
roughly doubles). Occlusion culling removes 39% of the triangles at ground level
(1.34M to 0.82M) but saves little time here, where instances stand apart. With instances packed tighter
(spacing 0.9, from inside the crowd), the culling and rasterizer changes together took
a frame from 1.37 ms to 0.47 ms; the commit messages have each step's numbers.

## Limitations

* About 8% of leaf clusters are still small pockets walled in by full ones. Swapping
  triangles between neighbours, or a real graph partitioner, would fill them.
* The simplifier's error is a mean over the quadric's planes, not a maximum, so it can
  understate the worst spot, and the 1 pixel bound is not strict.
* Shared vertices are stored once per cluster, and triangles take a word each where
  a byte stream would do; pages could be about half their size again.
* A refinement takes a round trip of a few frames per level, so flying into a statue
  from far away sharpens over twenty frames or so.
* Normals are interpolated per vertex, with no materials or textures.

## Models

Lucy and the XYZ RGB Asian Dragon are from the
[Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/),
courtesy of the Stanford Computer Graphics Laboratory (the dragon with XYZ RGB Inc.),
and are not redistributed here; `models/fetch.sh` downloads them.
