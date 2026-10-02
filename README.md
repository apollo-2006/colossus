# colossus

[![ci](https://github.com/apollo-2006/colossus/actions/workflows/ci.yml/badge.svg)](https://github.com/apollo-2006/colossus/actions/workflows/ci.yml)
[![license: mit](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

virtualized geometry from scratch, in c++20 and vulkan. a builder turns a scanned model
into a crack-free hierarchy of clusters; a renderer streams it from disk and draws
thousands of copies, picking per cluster the coarsest detail within a pixel of the
original. it's the idea behind unreal engine 5's nanite, built here with nothing but
vulkan and glfw: clustering, simplification, partitioning, streaming, culling, both
rasterizers, shadows and shading all live in this repository.

**[fly through it in your browser →](https://apollo-2006.github.io/colossus/)** the
webgpu port ([in the browser](#in-the-browser)), with the debug views, the error
threshold and the crowd size there to play with.

![900 instances of lucy and the xyz rgb dragon in sunlight, shadowed by virtual shadow maps](docs/crowd.png)

900 instances of lucy (28 million triangles) and the xyz rgb dragon (7.2 million): 15.9
billion triangles at full detail, drawn at 1920x1080 in 0.75 ms on an rx 9070 xt,
shadows and antialiasing included. about 3 million triangles reach the screen, from 23
mb of the 579 mb on disk. a million instances (17.6 trillion triangles) take 1.06 ms,
and instances can move.

## how it works

### building the hierarchy (`colossus_build`)

* **clusters.** at most 128 triangles and 128 vertices each (`src/cluster.cpp`), by
  recursive bisection of the triangle graph: two far seeds, breadth-first growth, a cut
  at a whole number of clusters, then swaps to shorten it. every cluster comes out full:
  the dragon makes 56,404 leaves at 128.0 triangles on average. the greedy grower before
  it left 8% as small walled-in pockets.
* **groups, locked, simplified.** each level's clusters form groups of about eight by
  shared edges (`src/dag.cpp`). vertices shared between groups are locked, each group is
  merged and simplified to half, and the result becomes the next level's clusters.
  locked outlines meet their neighbours edge for edge; groups fall differently each
  level, so no seam outlives a level.
* **quadric simplification.** `src/simplify.cpp` collapses edges cheapest first by
  quadric error (garland and heckbert). collapses keep an endpoint, so every level
  indexes the original vertices and the hierarchy shares one vertex buffer. no flips,
  no pinches (the link condition), no lost islands; hole rims only slide along
  themselves.
* **measured errors that only grow.** a quadric is a mean and can understate the worst
  spot, so each group is measured against what it replaced: a two-sided distance
  sampled at vertices, edge midpoints and triangle centres. its error is its children's
  plus the larger of that and the estimate; groups measuring over four times their
  estimate are retried more gently. errors only grow toward the root, so
  `own error on screen <= 1 pixel < parent error on screen` picks exactly one level on
  every path, one gpu thread per cluster, no tree to walk.
* **checked for cracks.** every level uses the original vertices, so a crack is exact:
  an edge used by one cut triangle whose ends aren't both on a hole. `--check` tests 25
  cuts. lucy builds 23 levels in 84 s, the dragon 21 in 21 s, every cut with 0 cracked
  edges.
* **one small root.** near the top, thin parts and hole rims stall edge collapses (lucy
  used to stop at three roots of 313 triangles). nothing borders the last group, so
  vertex clustering takes over there, keeping duplicate triangles and hole vertices so
  the crack check still holds. lucy now ends in one root of 20 triangles, the dragon in
  one of 32.

### drawing it (`colossus`)

1. **instance culling.** instances sit in cells of 8x8; `cell_cull.comp` tests each
   cell's sphere against the frustum and depth pyramid before any instance is read.
   `instance_cull.comp` then culls the visible cells' instances, one per invocation, and
   binary searches each one's clusters (stored by parent error) for the ones that could
   draw at its distance: a far instance tests a short tail, not all 445k. near instances,
   thousands of work items each, get a workgroup of their own (`expand.comp`).
2. **cluster culling** (`cluster_cull.comp`): lod cut, frustum, normal cone, occlusion.
   it began as a task shader; on radv that took 1.13 ms for 630k clusters, as compute
   0.39 ms.
3. **two rasterizers.** clusters over 32 pixels go to mesh shaders; smaller ones to
   `sw_raster.comp`, a compute rasterizer, since pixel-sized triangles are what the
   hardware handles worst. both write `depth << 32 | (cluster, triangle)` into one
   64-bit visibility buffer by atomic max. the compute path snaps to 1/256 pixel with a
   top-left rule, so the two meet without gaps, and keeps its edge functions in 32 bits
   (rdna has no 64-bit multiply; the 64-bit version lost at every size).
4. **occlusion in two passes.** pass 1 tests cells, instances and clusters against last
   frame's depth pyramid and draws what it can't prove hidden; a pyramid built from that
   lets pass 2 recover anything newly visible. moving instances are tested where they
   were last frame.
5. **shading** (`shade.comp`) refetches each pixel's triangle and hits it with the
   pixel's ray for exact barycentrics. five materials (marble, sandstone, granite, bronze,
   gold), lambert plus a ggx lobe.
6. **shadows: virtual shadow maps** (`vsm.glsl`, `vsm_*.comp`). the sun's depth lives in
   a clipmap around the camera: 12 levels of 32x32 pages of 128x128 texels, each level's
   texels twice the last's, backed by a pool of physical pages. each frame every pixel
   marks the page it will read, missing pages get physical ones, and only new or invalid
   pages are drawn, from the same hierarchy seen from the sun with errors in that level's
   texels. pages keep two layers: still instances, drawn once (or until their geometry
   has loaded), and moving ones, redrawn where something moved. with the camera still
   they cost 0.06 ms; flying through the crowd draws about 3 pages a frame.

   ray traced shadows are still there (`--shadows rt`): coarser copies of each model,
   half resolution with full resolution at edges, moving instances in a refitted second
   structure. they shade these views in 0.38 to 0.58 ms against 0.23 to 0.29, grow with
   the instance count and need ray queries; the shadow maps draw from the full hierarchy,
   so fine folds shadow themselves in more detail.

7. **antialiasing** (`taa.comp`). eight sub-pixel offsets in turn, each pixel blended 10%
   into a history reprojected through last frame's camera (and a moving instance's last
   transform), catmull-rom sampled and clamped to the 3x3 neighbourhood. 0.05 to 0.07 ms.
8. **streaming.** only bounds and errors stay resident; geometry lives in pages, one per
   group, loaded into a fixed pool (`--pool-mb`, 1 gb by default) as the gpu asks. a
   cluster that would rather be its finer clusters requests their page, with its error
   on screen as priority; `viewer/streamer.hpp` loads the most wanted, evicting least
   recently used, on loader threads.

   cuts stay whole because a page is resident only while the pages of its coarser
   stand-ins are: loads go coarsest first and publish in issue order, and a page is
   evicted only when nothing resident or loading depends on it. whatever has loaded,
   exactly one cluster draws on every path.

   each level roughly halves the error, so the streamer prefetches the pages below a
   request while they'd still be too coarse. from a cold start a lucy close-up settles in
   17 frames instead of 45, the crowd above in 32 instead of 45.

   a vertex is two words: 14-bit offsets on a grid over the model plus an 11 + 11 bit
   octahedral normal; a triangle is three bytes. shared vertices snap to the same grid
   point in each cluster, so quantizing opens no cracks. lucy's pages are 460 mb.

   the crowd above reads 23 mb and holds at most 2,577 pages. a 300 frame flight with a
   16 mb pool evicts 24,085 pages and shows exactly the same empty pixels as a 1 gb pool.
   from a cold file cache the render thread's worst frame in the streamer is 2.0 ms with
   loader threads, 44 ms without (`--sync-loads`).

### motion and scale

`--moving f` sets a share of instances moving: each turns on the spot and drifts round a
small circle, computed in the shaders from the time (`animate()` in `common.glsl`), so a
million moving instances need no uploads. occlusion and antialiasing use each instance's
last transform, and the shadow maps redraw the pages it crossed.

| instances | moving | shadow maps | ray traced |
|---|---|---|---|
| 900 | none | 0.74 ms | 0.85 ms |
| 900 | 1% | 0.93 ms | 1.03 ms |
| 900 | half | 1.32 ms | 1.12 ms |
| 90,000 | 1% | 1.00 ms | 1.33 ms |
| 1,000,000 | none | 1.06 ms | 1.34 ms |
| 1,000,000 | 1% | 1.40 ms | 1.77 ms |

the crowd view at 1920x1080. half the crowd moving is the one case the shadow maps lose:
450 statues redrawn into 156 pages a frame cost 0.62 ms. a ray traced refit on radv costs
by the size of the whole structure (2.7k entries 0.25 ms, 270k 0.94 ms), so still
instances get a structure of their own. at a million, occlusion culling is what makes it
work: without it the view draws 72 million triangles in 3.84 ms.

![lucy, each cluster in its own colour](docs/clusters.png)

each cluster of lucy in its own colour (view 2).

![the crowd coloured by lod level, blue for full detail through red to magenta for the coarsest](docs/lod_levels.png)

lod levels (view 4): near dragons draw mid levels, distant statues the coarsest, and
single instances mix levels as they recede.

![the crowd coloured by rasterizer: orange for compute, blue for hardware](docs/rasterizers.png)

which rasterizer drew each pixel (view 8): orange compute, blue mesh shaders. only the
nearest surfaces are worth the hardware.

## in the browser

`web/` is the renderer in webgpu, which has no mesh shaders, 64-bit atomics, ray queries
or push constants:

* big clusters go through a plain render pipeline, 384 vertices per cluster, each vertex
  pulling its triangle from storage.
* the compute rasterizer runs twice: nearest depth by 32-bit atomic max, then the
  triangle wherever its depth won.
* pages stream over http range requests through a javascript port of the streamer
  (`web/streamer.js`), same rules.
* occlusion runs in the same two passes, the pass number from a uniform bound at an
  offset per dispatch.
* shadows are the viewer's virtual shadow maps, in a module of their own
  (`web/vsm.wgsl`) to keep the bind group small, their indirect arguments in their own
  buffer (a dispatch can't write the buffer it reads arguments from). materials and
  antialiasing match the viewer.
* a quarter of the middle 30x30 instances move, and the crowd slider goes to a million.

the models are trimmed to 4 million triangles at their finest (`--max-triangles`): about
4 mb of gzipped metadata each, up front, and 71 mb of pages, streamed. in chrome on the
rx 9070 xt at 1600x813, 900 instances take 0.79 ms of gpu time once streaming settles,
1.23 ms with the middle moving, and a million 1.85 ms. it needs 16 storage buffers per
shader stage, which desktop gpus allow. `web/build.sh` builds the models;
`node tests/web_screenshot.mjs` renders the page headless.

## build & run

you'll need a gpu with vulkan 1.3 and `VK_EXT_mesh_shader`, the vulkan headers and
loader, glfw and `glslc`. ray queries are optional, only `--shadows rt` uses them.

```bash
git clone https://github.com/apollo-2006/colossus.git
cd colossus
make
models/fetch.sh            # downloads lucy and the dragon (380 mb) and builds both

./colossus --model models/lucy.cgeo --model models/xyzrgb_dragon.cgeo --grid 30
./colossus --model models/lucy.cgeo                       # one lucy
./colossus --model models/lucy.cgeo --model models/xyzrgb_dragon.cgeo --grid 1000 --moving 0.01
./colossus_build any.ply out.cgeo --check                 # your own model, checked for cracks
```

| keys | |
|---|---|
| w a s d, q e | move, down and up; drag to look; scroll for speed, shift to hurry |
| 1 to 9 | shaded, clusters, triangles, lod level, groups, instances, holes, rasterizer, shadow levels |
| [ and ] | halve or double the error threshold (1 pixel) |
| f | freeze culling, then fly out and watch it |
| c v o r h x | cone culling, frustum culling, occlusion, compute rasterizer, shadows, antialiasing |
| t, p | wireframe; print the camera as a `--camera` argument |

`--headless --frames n --screenshot out.png` renders without a window and prints the
median frame's timings. `docs/shots.sh` renders the images on this page.

## tests

```bash
make test
```

`tests/builder_test.cpp` needs no gpu or downloads: ply and obj reading, welding,
cluster limits, the simplifier's locks and flips, and full hierarchies for a closed
sphere, a holed sphere and a flat grid, crack-checked at 31 cuts, paged, written and read
back exactly, shared vertices identical from every cluster. without the group locks, the
crack check finds 39,095 cracked edges on the closed sphere alone.

`tests/streamer_test.cpp` runs 3,000 frames of random requests into a pool a tenth of a
model (synchronous, loader threads, loader threads with prefetch), checking after every
frame that resident pages' dependencies are resident. it caught one real bug before the
streamer landed. ci runs both and builds the viewer on every push.

## performance

rx 9070 xt (radv), 1920x1080, the 900 instance scene, median of 200 frames after 100 of
streaming. shading includes shadows and antialiasing.

| camera | frame | culling | raster | pass 2 | shadow pages | shading | triangles |
|---|---|---|---|---|---|---|---|
| beside lucy (top image) | 0.75 ms | 0.17 | 0.18 | 0.10 | 0.06 | 0.24 | 3.00m |
| raised (lod image) | 0.79 ms | 0.09 | 0.25 | 0.09 | 0.07 | 0.29 | 4.33m |
| ground level | 0.69 ms | 0.12 | 0.18 | 0.10 | 0.06 | 0.23 | 2.59m |

without shadows: 0.61, 0.61 and 0.56 ms; ray traced: 0.85, 1.02 and 0.78. without
antialiasing: 0.67, 0.72 and 0.62. the compute rasterizer saves 0.30 to 0.44 ms. occlusion
culling removes 46% of the triangles at ground level (0.76 to 0.69 ms) and next to
nothing from the raised camera, which sees over the crowd. the culling built for a
million instances costs these 900 about 0.03 ms.

early versions drew fewer triangles (0.86 million beside lucy); measured errors now
refuse levels that strayed several pixels. the commit messages have every step's numbers.

## limitations

* the error is sampled at vertices, edge midpoints and triangle centres, so a narrow
  spike between samples can slip through: a close bound, not a proof.
* refinement takes 17 to 45 frames from a cold start; prefetch only overlaps the levels.
* motion is rigid and built in: nothing deforms. many moving instances are costly in the
  shadow maps (half the crowd: 0.62 ms).
* shadow pages draw from streamed geometry too, so shadows sharpen with everything else,
  and a page that finds the pool full falls back a level.
* materials are a few parameters per instance, with no textures.
* the browser has no instance cells, its shadow pages wait on http streaming (about 40
  seconds headless for a still view to settle), and its pages are only compressed as
  far as the server does.

## models

lucy and the xyz rgb asian dragon come from the
[stanford 3d scanning repository](http://graphics.stanford.edu/data/3Dscanrep/), with
thanks to the stanford computer graphics laboratory (and xyz rgb inc. for the dragon).
they aren't redistributed here; `models/fetch.sh` downloads them.
