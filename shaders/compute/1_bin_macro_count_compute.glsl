#version 450

#extension GL_KHR_shader_subgroup_basic  : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 256) in;

#define WG_SIZE   256
#define SG_SIZE   32
#define N_WARPS   (WG_SIZE / SG_SIZE)
#define MAX_MACRO 256

struct Segment {
  vec2 p0;
  vec2 p1;
};

layout(set=0,binding=0,std430) readonly buffer Segments {
  Segment segments[];
};

layout(set = 0, binding = 1, std430) writeonly buffer macrotileNSegments {
  uint macrotile_count[];
};

layout(push_constant) uniform push_constant {
  uint screen_w, screen_h;
  uint n_tiles_x, n_tiles_y;
  uint n_macrotiles_x, n_macrotiles_y;
  uint tile_size, macrotile_size;

  uint n_segments;
  uint n_paths;
  uint fill_rule;
} pc;

shared uint warp_mask[N_WARPS][MAX_MACRO];


void main()
{
  // ==============================================
  // Dispatch: X: DIV_UP(N_segments / WG_SIZE)
  // ==============================================

  uint tid   = gl_LocalInvocationID.x;
  uint lane  = gl_SubgroupInvocationID;
  uint warp  = tid / SG_SIZE;

  // clear the segment "count" mask for each macrotile
  uint n_macro = pc.n_macrotiles_x * pc.n_macrotiles_y;
  for(uint i = lane; i < n_macro; i += SG_SIZE)
    warp_mask[warp][i] = 0;

  barrier();

  uint seg_id = gl_WorkGroupID.x * WG_SIZE + tid;

  if(seg_id < pc.n_segments) { 

  Segment seg = segments[seg_id];

  vec2 seg_min = min(seg.p0, seg.p1);
  vec2 seg_max = max(seg.p0, seg.p1);

  float ms = float(pc.macrotile_size);

  // Eval AABB box of segment
  ivec2 start = ivec2(floor(seg_min / ms));
  ivec2 end = ivec2(floor(seg_max / ms));

  // clamp to avoid UB writing
  start = clamp(start, ivec2(0), ivec2(int(pc.n_macrotiles_x - 1), int(pc.n_macrotiles_y - 1)));
  end = clamp(end, ivec2(0), ivec2(int(pc.n_macrotiles_x - 1), int(pc.n_macrotiles_y - 1)));


  for(int ty = start.y; ty <= end.y; ty++) {
    for(int tx = start.x; tx <= end.x; tx++) {

      // Each lane handles exactly one segment, so we set the bit
      // representing the segment in the mask of this bin.
      // --
      // Because we are subgroup aggregating, each warp has its own mask,
      // so we set the bit inside the current warp.
      uint bin = uint(ty) * pc.n_macrotiles_x + uint(tx);
      atomicOr(warp_mask[warp][bin], 1u << lane);
    }
  }

  }
  barrier();

  for(uint bin = tid; bin < n_macro; bin += WG_SIZE) {
    // Sum up the bit count of all warps in the workgroup for this bin.

    // The bit count per mask is exactly how many segments intersected the 
    // bin's AABB within the respective warp. Adding all warp counts together 
    // results in the total number of segments inside this bin.
    uint count = 0;
    for(uint w = 0; w < N_WARPS; w++)
      count += bitCount(warp_mask[w][bin]);

    // global atomic scales deterministically 
    if(count != 0) {
      atomicAdd(macrotile_count[bin], count);
    }
  }

}

