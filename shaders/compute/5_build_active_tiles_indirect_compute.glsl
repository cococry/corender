#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout(local_size_x = 1) in;

layout(set = 0, binding = INDIRECT_BINDING, std430) buffer Indirect {
  uint sparse_x;
  uint sparse_y;
  uint sparse_z;

  uint dense_x;
  uint dense_y;
  uint dense_z;

  uint analytic_x;
  uint analytic_y;
  uint analytic_z;
};

#if CR_ENABLE_GPU_STATS
layout(set = 0, binding = STATS_BINDING, std430) buffer StatsBuffer {
  GpuStats stats;
};
#endif

void main() {
  if (bump.failed != 0u) {
#if CR_ENABLE_GPU_STATS
    atomicOr(stats.failed, bump.failed);
#endif

    sparse_x = 0u;
    sparse_y = 1u;
    sparse_z = 1u;

    dense_x = 0u;
    dense_y = 1u;
    dense_z = 1u;

    analytic_x = 0u;
    analytic_y = 1u;
    analytic_z = 1u;

    return;
  }

  uint n_sparse = bump.n_active_tiles_sparse;
  uint n_dense = bump.n_active_tiles_dense;

  sparse_x = n_sparse;
  sparse_y = 1u;
  sparse_z = 1u;

  dense_x = n_dense;
  dense_y = 1u;
  dense_z = 1u;

  analytic_x = n_sparse + n_dense;
  analytic_y = 1u;
  analytic_z = 1u;
}
