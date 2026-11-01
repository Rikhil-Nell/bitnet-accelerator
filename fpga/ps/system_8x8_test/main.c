/*
 * Bare-metal PS test: drive bitnet_ip_top @ 0x43C00000, print results on UART.
 * Build via Vitis platform from build/vivado/bitnet_zybo.xsa (see fpga/README.md).
 */
#include "system_8x8_data.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "sleep.h"

#define BITNET_BASE 0x43C00000U

#define REG_CTRL   (BITNET_BASE + 0x00U)
#define REG_STATUS (BITNET_BASE + 0x04U)
#define REG_ACT    (BITNET_BASE + 0x10U)
#define REG_WGT    (BITNET_BASE + 0x14U)
#define REG_OUTIDX (BITNET_BASE + 0x18U)
#define REG_OUTDAT (BITNET_BASE + 0x1CU)

static void mmio_write(uint32_t off, uint32_t val) { Xil_Out32(BITNET_BASE + off, val); }

static uint32_t mmio_read(uint32_t off) { return Xil_In32(BITNET_BASE + off); }

static int32_t read_output(int idx) {
    mmio_write(0x18U, (uint32_t)idx);
    return (int32_t)mmio_read(0x1CU);
}

/* STATUS @ 0x04: bit0 done, bit1 busy (see docs/ITERATION.md). */
static int wait_accelerator_done(void) {
    for (int i = 0; i < 5000000; i++) {
        uint32_t st = mmio_read(0x04U);
        if (st & 0x1U) {
            return 0;
        }
    }
    return -1;
}

int main(void) {
    xil_printf("\r\nBitNet system_8x8 PS test\r\n");

    for (int i = 0; i < 8; i++) {
        uint32_t w = ((uint32_t)(uint8_t)g_activations[i]) | ((uint32_t)i << 8);
        mmio_write(0x10U, w);
    }
    for (int i = 0; i < 8; i++) {
        uint32_t w = ((uint32_t)g_weight_words[i]) | ((uint32_t)i << 16);
        mmio_write(0x14U, w);
    }

    mmio_write(0x00U, 1U);

    if (wait_accelerator_done() != 0) {
        xil_printf("FAIL: accelerator timeout (STATUS)\r\n");
        return 1;
    }

    int pass = 1;
    for (int i = 0; i < 8; i++) {
        int32_t got = read_output(i);
        if (got != g_expected_outputs[i]) {
            xil_printf("OUT[%d] got %ld expected %ld\r\n", i, (long)got,
                       (long)g_expected_outputs[i]);
            pass = 0;
        }
    }

    if (pass) {
        xil_printf("PASS: system_8x8 on PL accelerator\r\n");
    } else {
        xil_printf("FAIL: output mismatch\r\n");
    }

    return pass ? 0 : 1;
}
