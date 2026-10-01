# nexus_geometry

[![ci](https://github.com/apollo-2006/nexus_geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/apollo-2006/nexus_geometry/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Virtualized geometry from scratch, in C++20 and Vulkan: a builder that turns a scanned
model into a crack-free hierarchy of clusters, and a renderer that draws thousands of
copies of it by picking, cluster by cluster, the coarsest detail that stays within a
pixel of the original. It is the technique behind Unreal Engine 5's Nanite, written
with no libraries beyond Vulkan and GLFW: the clustering, the simplifier, the graph
partitioning, the culling, both rasterizers and the shading are all in this repository.

![900 instances of Lucy and the XYZ RGB dragon, lit by the sun with ray traced shadows](docs/crowd.png)

900 instances of Lucy (28M triangles) and the XYZ RGB dragon (7.2M), 15.9 billion
triangles at full detail, drawn at 1920x1080 in 0.72 ms on an RX 9070 XT. About 0.86M
triangles reach the screen.

## How it works

### Building the hierarchy (`ngeo_build`)

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
  hole in the scan. `ngeo_build --check` selects 25 cuts across the error range and
  counts them. Lucy builds 18 levels in 24 s and the dragon 20 levels in 5.4 s, every
  cut with 0 cracked edges.

### Drawing it (`nexus_view`)

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

![Lucy, each cluster in its own color](docs/clusters.png)

Each cluster of Lucy in its own color (debug view 2).

![The crowd colored by LOD level, blue for full detail through red to magenta for the coarsest](docs/lod_levels.png)

LOD levels (view 4): the nearest dragons draw mid levels, distant statues the coarsest,
and single instances mix levels where they recede.

![The crowd colored by rasterizer: orange for compute, blue for hardware](docs/rasterizers.png)

Which rasterizer drew each pixel (view 8): orange for the compute rasterizer, blue for
mesh shaders. Only the nearest surfaces are worth sending to the hardware.

## Build & run

Needs a GPU with Vulkan 1.3 and `VK_EXT_mesh_shader`, the Vulkan headers and loader,
GLFW, and `glslc`. Ray queries are optional: without them the viewer draws no shadows.

```bash
git clone https://github.com/apollo-2006/nexus_geometry.git
cd nexus_geometry
make
models/fetch.sh            # downloads Lucy and the dragon (380 MB) and builds both

./nexus_view --model models/lucy.ngeo --model models/xyzrgb_dragon.ngeo --grid 30
./nexus_view --model models/lucy.ngeo                       # one Lucy
./ngeo_build any.ply out.ngeo --check                       # your own model, checked for cracks
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
for cracks at 31 cuts and written to a file and read back. CI runs them and builds the
viewer on every push. With the group locks removed from the builder, the crack check
reports 39,095 cracked edges on the closed sphere alone.

## Performance

RX 9070 XT (RADV), 1920x1080, the 900 instance scene above, median of 200 frames:

| camera | frame | culling | raster | pass 2 | shading | triangles | clusters |
|---|---|---|---|---|---|---|---|
| beside Lucy (top image) | 0.72 ms | 0.13 | 0.12 | 0.08 | 0.39 | 0.86M | 9.8k |
| raised (LOD image) | 0.78 ms | 0.04 | 0.15 | 0.08 | 0.51 | 1.15M | 16.6k |
| ground level | 0.83 ms | 0.15 | 0.12 | 0.08 | 0.47 | 0.82M | 9.0k |

Shadow rays are most of the frame: without them the three cameras take 0.35, 0.33
and 0.39 ms. Turning the compute rasterizer off costs 0.10 to 0.14 ms (raster time
roughly doubles). Occlusion culling removes 39% of the triangles at ground level
(1.34M to 0.82M) but saves little time here, where instances stand apart. With instances packed tighter
(spacing 0.9, from inside the crowd), the culling and rasterizer changes together took
a frame from 1.37 ms to 0.47 ms; the commit messages have each step's numbers.

## Limitations

* About 8% of leaf clusters are still small pockets walled in by full ones. Swapping
  triangles between neighbours, or a real graph partitioner, would fill them.
* The simplifier's error is a mean over the quadric's planes, not a maximum, so it can
  understate the worst spot, and the 1 pixel bound is not strict.
* Everything is resident in GPU memory: Lucy's hierarchy is 780 MB. Nanite streams
  clusters from disk on demand; this does not yet.
* Normals are interpolated per vertex, with no materials or textures.

## Models

Lucy and the XYZ RGB Asian Dragon are from the
[Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/),
courtesy of the Stanford Computer Graphics Laboratory (the dragon with XYZ RGB Inc.),
and are not redistributed here; `models/fetch.sh` downloads them.
