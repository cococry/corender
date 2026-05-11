#include "msaa_lut.h"

#include <cglm/types.h>
#include <cglm/cglm.h>
#include <cglm/struct.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const vec2s FINE_MSAA8_SAMPLE_POS[8] = {
  {0.5625f, 0.3125f},
  {0.4375f, 0.6875f},
  {0.8125f, 0.5625f},
  {0.3125f, 0.1875f},
  {0.1875f, 0.8125f},
  {0.0625f, 0.4375f},
  {0.6875f, 0.9375f},
  {0.9375f, 0.0625f},
};

static float fine_lut_bucket_center(uint32_t ix) {
  const float range =
    FINE_MSAA8_LUT_X_MAX - FINE_MSAA8_LUT_X_MIN;

  return FINE_MSAA8_LUT_X_MIN +
    ((float)ix + 0.5f) *
    (range / (float)FINE_MSAA8_LUT_DIM);
}

static uint32_t fine_msaa8_make_mask_for_line(
    float x_top,
    float x_bottom
    ) {
  uint32_t mask = 0u;

  const float dx_dy = x_bottom - x_top;

  for (uint32_t s = 0u; s < FINE_MSAA8_SAMPLE_COUNT; s++) {
    const float sx = FINE_MSAA8_SAMPLE_POS[s].x;
    const float sy = FINE_MSAA8_SAMPLE_POS[s].y;

    int parity = 0;

    if (fabsf(dx_dy) > 1.0e-20f) {
      const float y_cross = -x_top / dx_dy;

      if (y_cross > 0.0f && y_cross <= sy) {
        parity ^= 1;
      }
    }
    {
      const float x_cross = x_top + dx_dy * sy;

      if (x_cross > 0.0f && x_cross <= sx) {
        parity ^= 1;
      }
    }

    if (parity) {
      mask |= 1u << s;
    }
  }

  return mask & FINE_MSAA8_FULL_MASK;
}

void fine_msaa8_build_cell_mask_lut_packed(uint32_t *dst_words) {
  memset(
      dst_words,
      0,
      sizeof(uint32_t) * FINE_MSAA8_LUT_PACKED_WORD_COUNT
      );

  for (uint32_t top_ix = 0u; top_ix < FINE_MSAA8_LUT_DIM; top_ix++) {
    const float x_top = fine_lut_bucket_center(top_ix);

    for (uint32_t bottom_ix = 0u; bottom_ix < FINE_MSAA8_LUT_DIM; bottom_ix++) {
      const float x_bottom = fine_lut_bucket_center(bottom_ix);

      const uint32_t mask =
        fine_msaa8_make_mask_for_line(x_top, x_bottom);

      const uint32_t entry_ix =
        top_ix * FINE_MSAA8_LUT_DIM + bottom_ix;

      const uint32_t word_ix = entry_ix >> 2u;
      const uint32_t byte_ix = entry_ix & 3u;

      dst_words[word_ix] |= mask << (byte_ix * 8u);
    }
  }
}


static const vec2s FINE_MSAA16_SAMPLE_POS[16] = {
    {{0.1250f, 0.3750f}},
    {{0.3750f, 0.8750f}},
    {{0.6250f, 0.1250f}},
    {{0.8750f, 0.6250f}},

    {{0.1250f, 0.8750f}},
    {{0.3750f, 0.1250f}},
    {{0.6250f, 0.6250f}},
    {{0.8750f, 0.3750f}},

    {{0.1250f, 0.6250f}},
    {{0.3750f, 0.3750f}},
    {{0.6250f, 0.8750f}},
    {{0.8750f, 0.1250f}},

    {{0.1250f, 0.1250f}},
    {{0.3750f, 0.6250f}},
    {{0.6250f, 0.3750f}},
    {{0.8750f, 0.8750f}},
};

static float fine_msaa16_lut_bucket_center(uint32_t ix) {
    const float range =
        FINE_MSAA16_LUT_X_MAX - FINE_MSAA16_LUT_X_MIN;

    return FINE_MSAA16_LUT_X_MIN +
        ((float)ix + 0.5f) *
        (range / (float)FINE_MSAA16_LUT_DIM);
}

static uint32_t fine_msaa16_make_mask_for_line(
    float x_top,
    float x_bottom
) {
    uint32_t mask = 0u;

    const float dx_dy = x_bottom - x_top;

    for (uint32_t s = 0u; s < FINE_MSAA16_SAMPLE_COUNT; s++) {
        const float sx = FINE_MSAA16_SAMPLE_POS[s].x;
        const float sy = FINE_MSAA16_SAMPLE_POS[s].y;

        int parity = 0;

        /*
         * Leg 1:
         *
         * Pixel top-left -> down along x = 0 to sample_y.
         *
         * The line x(y) crosses x = 0 at:
         *
         *     y_cross = -x_top / (x_bottom - x_top)
         */
        if (fabsf(dx_dy) > 1.0e-20f) {
            const float y_cross = -x_top / dx_dy;

            if (y_cross > 0.0f && y_cross <= sy) {
                parity ^= 1;
            }
        }

        /*
         * Leg 2:
         *
         * Pixel-left at sample_y -> sample point.
         *
         * The line's x at sample_y:
         *
         *     x_cross = x_top + dx_dy * sample_y
         */
        {
            const float x_cross = x_top + dx_dy * sy;

            if (x_cross > 0.0f && x_cross <= sx) {
                parity ^= 1;
            }
        }

        if (parity) {
            mask |= 1u << s;
        }
    }

    return mask & FINE_MSAA16_FULL_MASK;
}

void fine_msaa16_build_cell_mask_lut_packed(uint32_t *dst_words) {
    memset(
        dst_words,
        0,
        sizeof(uint32_t) * FINE_MSAA16_LUT_PACKED_WORD_COUNT
    );

    for (uint32_t top_ix = 0u; top_ix < FINE_MSAA16_LUT_DIM; top_ix++) {
        const float x_top = fine_msaa16_lut_bucket_center(top_ix);

        for (uint32_t bottom_ix = 0u; bottom_ix < FINE_MSAA16_LUT_DIM; bottom_ix++) {
            const float x_bottom = fine_msaa16_lut_bucket_center(bottom_ix);

            const uint32_t mask =
                fine_msaa16_make_mask_for_line(x_top, x_bottom);

            const uint32_t entry_ix =
                top_ix * FINE_MSAA16_LUT_DIM + bottom_ix;

            const uint32_t word_ix = entry_ix >> 1u;
            const uint32_t half_ix = entry_ix & 1u;

            dst_words[word_ix] |= mask << (half_ix * 16u);
        }
    }
}
