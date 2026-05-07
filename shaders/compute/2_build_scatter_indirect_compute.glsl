#version 450
#extension GL_GOOGLE_include_directive : require
#include "../../shared/pc.glsl"

layout(local_size_x = 1) in;

layout(set = 0, binding = INDIRECT_BINDING, std430) buffer ScatterIndirect {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
} scatter_indirect;

void main() {
    if (bump.failed != 0u) {
        scatter_indirect.dispatch_x = 0u;
        scatter_indirect.dispatch_y = 1u;
        scatter_indirect.dispatch_z = 1u;
        return;
    }

    // this is used in the next pass, scatter_touches, which 
    // runs per touch record.
    scatter_indirect.dispatch_x = (bump.n_touches + 255u) >> 8; 
    scatter_indirect.dispatch_y = 1u;
    scatter_indirect.dispatch_z = 1u;
}
