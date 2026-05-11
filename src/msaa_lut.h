#pragma once
#include <stdint.h>

#define FINE_MSAA8_LUT_DIM 64u
#define FINE_MSAA8_LUT_ENTRY_COUNT \
  (FINE_MSAA8_LUT_DIM * FINE_MSAA8_LUT_DIM)

#define FINE_MSAA8_LUT_PACKED_WORD_COUNT \
  (FINE_MSAA8_LUT_ENTRY_COUNT / 4u)

#define FINE_MSAA8_SAMPLE_COUNT 8u
#define FINE_MSAA8_FULL_MASK 0xffu

#define FINE_MSAA8_LUT_X_MIN (-1.0f)
#define FINE_MSAA8_LUT_X_MAX ( 2.0f)

void fine_msaa8_build_cell_mask_lut_packed(uint32_t *dst_words);


#define FINE_MSAA16_LUT_DIM 256u

#define FINE_MSAA16_LUT_ENTRY_COUNT \
  (FINE_MSAA16_LUT_DIM * FINE_MSAA16_LUT_DIM)

/*
 * 16-bit masks.
 * Two entries packed per uint32_t.
 */
#define FINE_MSAA16_LUT_PACKED_WORD_COUNT \
  (FINE_MSAA16_LUT_ENTRY_COUNT / 2u)

#define FINE_MSAA16_SAMPLE_COUNT 16u
#define FINE_MSAA16_FULL_MASK 0xffffu

#define FINE_MSAA16_LUT_X_MIN (-1.0f)
#define FINE_MSAA16_LUT_X_MAX ( 2.0f)

void fine_msaa16_build_cell_mask_lut_packed(uint32_t *dst_words);
