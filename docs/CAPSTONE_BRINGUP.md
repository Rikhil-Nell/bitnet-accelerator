# Capstone bring-up checklist (Zybo Z7-10)

Use after host + FPGA artifacts exist (`make capstone-check` — only `program` should be BLOCKED).

Stage bitstream + `ps7_init.tcl` to paths without spaces (optional if already done):

```powershell
make stage-board
```

After reboot (one shot):

```powershell
. tools/setup/post_reboot_verify.ps1 -Com COMx
```

Log: `build/capstone/post_reboot.log`

## 1. USB / JTAG

```powershell
. tools/setup/diagnose_zybo_usb.ps1
```

- **Converter A** = UART (COM port)
- **Converter B** = JTAG (must be **OK**, not Error)
- If Problem is **`CM_PROB_NEED_RESTART`**: **reboot Windows**, replug PROG micro-USB, rerun diagnose.

```powershell
make list-hw
```

Expect a **non-empty** line: `targets on localhost:3121: ...`

After any change to `fpga/ps/system_8x8_test/`, rebuild: `make ps-app`. If Vitis says workspace **already in use**, close the IDE and run `. tools/setup/unlock_vitis_ws.ps1`.

## 2. Program + run + UART

One shot (auto COM when possible):

```powershell
. tools/setup/board_bringup.ps1 -Listen
```

Or stepwise:

| Step | Command |
|------|---------|
| PL bitstream | `make program` (programs PL; PS7 clocks still need init) |
| PS ELF | `make run-ps` (XSDB runs `ps7_init.tcl` from BD export, then `dow`) |
| UART | `make uart-listen COM=COMx` |

**Pass criterion:** UART prints `PASS: system_8x8 on PL accelerator`.

## 3. Full software matrix (no board)

```powershell
make eval
make sim-all
make vivado-synth    # or rely on existing build/vivado/*.rpt
make bitstream     # if RTL/BD changed
make ps-app
```

Optional Microsoft BitNet PC path: `docs/BITNET_PC_INFERENCE.md`, `make test-bitnet-smoke`.
