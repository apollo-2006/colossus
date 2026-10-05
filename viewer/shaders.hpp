#pragma once
// the shaders, compiled to spir-v by the makefile (obj/shaders/*.inc).
#include <cstdint>

namespace viewer {

const uint32_t vsm_mark_spv[] = {
#include "vsm_mark.comp.inc"
};
const uint32_t vsm_alloc_spv[] = {
#include "vsm_alloc.comp.inc"
};
const uint32_t vsm_clear_spv[] = {
#include "vsm_clear.comp.inc"
};
const uint32_t vsm_args_spv[] = {
#include "vsm_args.comp.inc"
};
const uint32_t vsm_instance_spv[] = {
#include "vsm_instance.comp.inc"
};
const uint32_t vsm_expand_spv[] = {
#include "vsm_expand.comp.inc"
};
const uint32_t vsm_cluster_spv[] = {
#include "vsm_cluster.comp.inc"
};
const uint32_t vsm_raster_spv[] = {
#include "vsm_raster.comp.inc"
};
const uint32_t ao_depth_spv[] = {
#include "ao_depth.comp.inc"
};
const uint32_t ao_spv[] = {
#include "ao.comp.inc"
};
const uint32_t cell_cull_spv[] = {
#include "cell_cull.comp.inc"
};
const uint32_t tlas_spv[] = {
#include "tlas.comp.inc"
};
const uint32_t expand_spv[] = {
#include "expand.comp.inc"
};
const uint32_t instance_cull_spv[] = {
#include "instance_cull.comp.inc"
};
const uint32_t args_spv[] = {
#include "args.comp.inc"
};
const uint32_t cluster_cull_spv[] = {
#include "cluster_cull.comp.inc"
};
const uint32_t draw_mesh_spv[] = {
#include "draw.mesh.inc"
};
const uint32_t vis_frag_spv[] = {
#include "vis.frag.inc"
};
const uint32_t shade_spv[] = {
#include "shade.comp.inc"
};
const uint32_t ao_rt_spv[] = {
#include "ao_rt.comp.inc"
};
const uint32_t shade_rt_spv[] = {
#include "shade_rt.comp.inc"
};
const uint32_t sw_raster_spv[] = {
#include "sw_raster.comp.inc"
};
const uint32_t shadow_spv[] = {
#include "shadow.comp.inc"
};
const uint32_t taa_spv[] = {
#include "taa.comp.inc"
};
const uint32_t hzb_spv[] = {
#include "hzb.comp.inc"
};

}  // namespace viewer
