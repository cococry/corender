
#define SW_ONE_MINUS_ULP 0.99999994
#define SW_EPS 2e-7

struct LineWalkParams {
    vec2 s0;
    vec2 s1;

    uint count;
    uint count_x;

    float x_step_rate;
    float x_step_start_offset;
    float x_sign;
    float x0;
    float y0;

    int delta;
};


uint span_cells(float a, float b) {
    return uint(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

bool make_line_walk_params(vec2 p0, vec2 p1, out LineWalkParams lp) {
    float ts = float(pc.tile_size);
    float inv_ts = 1.0 / ts;

    /*
        delta convention:
            original downward edge (1 > 0) -> -1
            original upward edge   (0 > 1) -> +1
    */
    bool is_down = p1.y >= p0.y;

    // normalized so xy0.y <= xy1.y
    vec2 xy0 = is_down ? p0 : p1;
    vec2 xy1 = is_down ? p1 : p0;
    lp.delta = is_down ? -1 : 1;

    // the segment's start and end in tile-space 
    lp.s0 = xy0 * inv_ts;
    lp.s1 = xy1 * inv_ts;

    float dx = abs(lp.s1.x - lp.s0.x);
    float dy = lp.s1.y - lp.s0.y;

    // segment has no length => invalid
    if (dx + dy == 0.0) {
        return false;
    }

    /*
        horizontal segments exactly on a tile boundary are dropped.
        if floor(lp.s0.y) == lp.s0.y, the tile-local y-coord is 
        an integer, meaning it's exactly on a tile border.
    */
    if (dy == 0.0 && floor(lp.s0.y) == lp.s0.y) {
        return false;
    }

    // how many vertical tile boundaries does this segment cross 
    lp.count_x = span_cells(lp.s0.x, lp.s1.x) - 1u;
    // total traversed tile boundaries => total number of scan 
    // steps later
    lp.count = lp.count_x + span_cells(lp.s0.y, lp.s1.y);

    float inverse_dx_plus_dy = 1.0 / (dx + dy);
    // what fraction of the walk steps are x-steps 
    lp.x_step_rate = dx * inverse_dx_plus_dy; // effictively dx / (dx + dy)

    bool x_moves_right = lp.s1.x >= lp.s0.x;
    lp.x_sign = x_moves_right ? 1.0 : -1.0;

    // the tile-local x position, measured in the walk direction
    float tile_local_x = fract(lp.s0.x * lp.x_sign);

    lp.y0 = floor(lp.s0.y);

    // +1 as at this point, the segment has 
    // already been normalized so it moves downward 
    // in tile-space y
    float ytop = lp.y0 + 1.0;

    // how much fractional y in tile space the 
    // segment moves until it crosses the next 
    // horizontal tile boundary.
    float y_to_next_boundary = ytop - lp.s0.y;

    // the segments's start offset for the stepper 
    lp.x_step_start_offset = min(
        (dy * tile_local_x + dx * y_to_next_boundary) * inverse_dx_plus_dy,
        SW_ONE_MINUS_ULP
    );

    float num_x_steps = floor(lp.x_step_rate * (float(lp.count) - 1.0) + lp.x_step_start_offset);
    float x_step_err = num_x_steps - float(lp.count_x);

    if (x_step_err != 0.0) {
        lp.x_step_rate -= SW_EPS * sign(x_step_err);
    }

    // the starting tile x-column for the stepper, 
    // accounted for x-walk-direction.
    lp.x0 = floor(lp.s0.x * lp.x_sign) * lp.x_sign + (x_moves_right ? 0.0 : -1.0);

    return true;
}

ivec2 tile_for_local_tile(LineWalkParams lp, uint local_tile, out float n_x_steps) {
    n_x_steps = floor(lp.x_step_rate * float(local_tile) + lp.x_step_start_offset);

    int x = int(lp.x0 + lp.x_sign * n_x_steps);
    // the remaining steps of float(local_tile) - n_x_steps are 
    // y steps
    int y = int(lp.y0 + float(local_tile) - n_x_steps);

    return ivec2(x, y);
}
