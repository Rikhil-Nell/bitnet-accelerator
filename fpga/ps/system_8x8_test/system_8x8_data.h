/* Golden vector from tools/golden/vectors/system_8x8.json (capstone_rtl encoding). */
#ifndef SYSTEM_8X8_DATA_H
#define SYSTEM_8X8_DATA_H

#include <stdint.h>

static const int8_t g_activations[8] = {5, -3, 7, 2, 4, -2, 6, 1};

static const uint16_t g_weight_words[8] = {
    0x2461, 0x9214, 0x4185, 0x1852, 0x4918, 0x1249, 0x6126, 0x8584};

static const int32_t g_expected_outputs[8] = {-8, 5, 5, 12, 17, 12, -16, -4};

#endif
