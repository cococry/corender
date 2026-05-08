#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout(local_size_x = 1) in;


layout(set = 0, binding = INDIRECT_BINDING, std430) buffer ActiveTilesIndirect {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
} active_tiles_indirect;

void main() {
    if (bump.failed != 0u) {
        active_tiles_indirect.dispatch_x = 0u;
        active_tiles_indirect.dispatch_y = 1u;
        active_tiles_indirect.dispatch_z = 1u;
        return;
    }

    // we will dispatch one workgroup per active tile
    // in the next step
    active_tiles_indirect.dispatch_x = bump.n_active_tiles;
    active_tiles_indirect.dispatch_y = 1u;
    active_tiles_indirect.dispatch_z = 1u;
}
