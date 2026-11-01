/*
 * Second Cortex-A9 core (CPU1) entry. xsdb sets CPU1's PC to cpu1_entry after CPU0's
 * firmware is running (fpga/scripts/run_ps_zybo.tcl; the address comes from the ELF
 * symbol table, tools/setup/run_ps_xsdb.ps1).
 *
 * CPU1 joins CPU0's coherent world: its own L1 / TLB are invalidated, SMP coherency is
 * enabled before the caches, and it uses CPU0's MMU table (MMUTable from the BSP). It
 * never touches the L2 controller, the SCU or the vector table; it runs with interrupts
 * masked and only executes cpu1_main() (main.c).
 *
 * (A naked C function rather than a .S file: the Vitis app build only collects C.)
 */
#include <stdint.h>

void cpu1_main(void);
uint64_t cpu1_stack[2048] __attribute__((aligned(16)));   /* 16 KB */

__attribute__((naked, used)) void cpu1_entry(void) {
    __asm__ volatile(
        "cpsid   if\n"
        "mrs     r0, cpsr\n"                    /* SYS mode */
        "bic     r0, r0, #0x1f\n"
        "orr     r0, r0, #0x1f\n"
        "msr     cpsr_c, r0\n"

        "mrc     p15, 0, r0, c1, c0, 0\n"       /* MMU, D-cache, I-cache off */
        "bic     r0, r0, #0x1\n"
        "bic     r0, r0, #0x4\n"
        "bic     r0, r0, #0x1000\n"
        "mcr     p15, 0, r0, c1, c0, 0\n"
        "isb\n"

        "mov     r0, #0\n"
        "mcr     p15, 0, r0, c8, c7, 0\n"       /* invalidate TLB */
        "mcr     p15, 0, r0, c7, c5, 0\n"       /* invalidate I-cache */
        "mcr     p15, 0, r0, c7, c5, 6\n"       /* invalidate branch predictor */

        /* invalidate this core's L1 D-cache by set/way: 32 KB, 4 ways, 256 sets of 32 B */
        "mov     r2, #0\n"
        "1:\n"
        "mov     r1, #0\n"
        "2:\n"
        "orr     r0, r1, r2\n"
        "mcr     p15, 0, r0, c7, c6, 2\n"
        "add     r1, r1, #32\n"
        "cmp     r1, #8192\n"
        "blt     2b\n"
        "add     r2, r2, #0x40000000\n"
        "cmp     r2, #0\n"
        "bne     1b\n"
        "dsb\n"

        "ldr     sp, =cpu1_stack + 16384\n"

        "mrc     p15, 0, r0, c1, c0, 2\n"       /* CPACR: cp10/cp11 full access (VFP/NEON) */
        "orr     r0, r0, #0xf00000\n"
        "mcr     p15, 0, r0, c1, c0, 2\n"
        "isb\n"
        "mov     r0, #0x40000000\n"             /* FPEXC.EN */
        "vmsr    fpexc, r0\n"

        "mrc     p15, 0, r0, c1, c0, 1\n"       /* ACTLR: SMP, maintenance broadcast, prefetch */
        "orr     r0, r0, #0x47\n"
        "mcr     p15, 0, r0, c1, c0, 1\n"

        "ldr     r0, =MMUTable\n"               /* same translation table as CPU0 */
        "orr     r0, r0, #0x5B\n"
        "mcr     p15, 0, r0, c2, c0, 0\n"
        "mvn     r0, #0\n"
        "mcr     p15, 0, r0, c3, c0, 0\n"       /* all domains: manager */
        "mrc     p15, 0, r0, c1, c0, 0\n"
        "ldr     r1, =0x1005\n"                 /* I, C, M */
        "orr     r0, r0, r1\n"
        "mcr     p15, 0, r0, c1, c0, 0\n"
        "dsb\n"
        "isb\n"

        "bl      cpu1_main\n"
        "3:\n"
        "wfe\n"
        "b       3b\n"
        ".ltorg\n");
}
