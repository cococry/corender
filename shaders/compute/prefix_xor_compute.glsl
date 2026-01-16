#version 450
layout(local_size_x = 128, local_size_y = 1) in;

layout(set = 0, binding = 4, std430) readonly buffer TileParity {
    uint tile_parity[];
};

layout(set = 0, binding = 5, std430) buffer ParityIn {
    uint parity_in[];
};

layout(push_constant) uniform PC {
    uint n_tiles_x, n_tiles_y;
    uint tile_size;
} pc;

shared uint s[128];

void main() {
    uint tile_x = gl_LocalInvocationID.x;
    if (tile_x >= pc.n_tiles_x) return;

    uint group = gl_WorkGroupID.y;
    uint tile_y = group / pc.tile_size;
    uint scan   = group % pc.tile_size;

    uint base = (tile_y * pc.n_tiles_x + tile_x) * pc.tile_size + scan;

    s[tile_x] = tile_parity[base];
    barrier();

    for (uint o = 1; o < pc.n_tiles_x; o <<= 1) {
        uint v = (tile_x >= o) ? s[tile_x - o] : 0;
        barrier();
        s[tile_x] ^= v;
        barrier();
    }

    uint out_parity = (tile_x == 0) ? 0 : s[tile_x - 1];
    parity_in[base] = out_parity;
}

