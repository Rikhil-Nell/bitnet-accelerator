# Zybo Z7 JTAG troubleshooting

## Symptom

`make program` fails with **No JTAG target on localhost**. Vivado sees `hw_server` but **zero `hw_targets`**.

## Check on this machine (2026-10-31)

Device Manager / PnP showed **FTDI dual UART/JTAG** (`USB\VID_0403&PID_6010`):

| Interface | Status |
|-----------|--------|
| USB Serial Converter **A** | OK |
| USB Serial Converter **B** | **Error** (often `CM_PROB_NEED_RESTART`) |

JTAG often uses the second FTDI channel. Fix **Converter B** before expecting Vivado targets.

If Problem is **`CM_PROB_NEED_RESTART`**: **reboot Windows** (required if `pnputil /restart-device` says *pending system reboot*). After reboot, replug PROG USB, run `. tools/setup/diagnose_zybo_usb.ps1`, then `make list-hw`.

## Safe bring-up (especially after a BSOD / unstable USB)

Run **one step at a time** in an **Admin-free** shell; close Vivado IDE first.

1. `. tools/setup/diagnose_zybo_usb.ps1` (USB only, no Vivado)
2. `. tools/setup/stop_hw_server.ps1` (clear stale servers)
3. `make list-hw` — wait for exit; confirm a **non-empty** target line
4. Only if stable: `make program`, then `make run-ps`, then `make uart-listen COM=COMx`

Do **not** chain `post_reboot_verify.ps1` until `make list-hw` alone is stable.

**Event Viewer:** DCOM **10016** (`Windows.SecurityCenter.WscDataProtection`) is a common warning and is **not** Vivado/JTAG. For a real BSOD, look for **BugCheck 1001** or **Kernel-Power 41**; optional: `. tools/setup/find_recent_bugcheck.ps1`

On this machine (2026-11-01): two crashes, **`0x139`** at 00:35 and **`0x50`** at 00:56, each triggered by `make list-hw`. WinDbg `!analyze -v` on both minidumps gives the same result: `PROCESS_NAME: hw_server.exe` and `SYMBOL_NAME: ftdibus+f4d4` (`ftdibus.sys`).

**Root cause:** the FTDI driver versions don't match. `ftdibus.sys` is 2.12.36.20 (2024), `ftser2k.sys` (the COM port driver) is 2.12.28 (2017), and a stale 2017 `ftdibus.inf` package is still in the driver store.

**Fix:** with the board unplugged, install the FTDI CDM "setup executable" from <https://ftdichip.com/drivers/vcp-drivers/> and reboot. Then confirm both drivers report the same version before running `make list-hw` again:

```powershell
Get-Item C:\Windows\System32\drivers\ftdibus.sys, C:\Windows\System32\drivers\ftser2k.sys | % { $_.Name + ' ' + $_.VersionInfo.FileVersion }
```

If the versions still differ, remove the stale 2017 packages from an admin shell (`pnputil /enum-drivers`, then `pnputil /delete-driver oemNN.inf`) and reinstall the CDM package.

To analyse a crash dump yourself, install WinDbg (`winget install Microsoft.WinDbg`), copy the dump out of `C:\Windows\Minidump` (this needs admin), and run `cdb -z <dmp> -c "!analyze -v; q"`.

Host-only audit without launching Vivado:

```powershell
$env:CAPSTONE_SKIP_JTAG_PROBE = "1"
make capstone-check
```

## Steps

1. **Cable:** Zybo **PROG/UART** micro-USB (not a UART-only cable on a wrong port).
2. **Power:** SW7 on, board powered (LDs lit).
3. **Drivers:** Install [Digilent Adept](https://digilent.com/reference/software/adept/start) (includes **Digilent USB Device** for JTAG). If **Converter A/B show OK** but Vivado still lists **zero targets**, Adept/JTAG drivers are the usual fix (generic FTDI UART is not enough).
4. **Device Manager:** Fix **USB Serial Converter B** (Error, often **Problem 14 / CM_PROB_NEED_RESTART** on `oem141.inf` FTDI stack):
   - Uninstall **Converter B** + check **Delete the driver software for this device**
   - Install [Digilent Adept](https://digilent.com/reference/software/adept/start) (Digilent USB Device driver)
   - **Full reboot** (verify `LastBootUpTime` changed), replug PROG USB
   - Try another USB port/cable; disable USB selective suspend
5. **UART vs JTAG:** Converter **A** (OK) is often **COMx** for serial (`make uart-listen COM=COMx`); **B** must be healthy for Vivado JTAG.
6. **Single session:** Close other Vivado / hw_server instances.
7. **Probe:** `make list-hw` — should print a non-empty target list.
8. **Program PL:** `make program` → `build/vivado/bitnet_zybo.bit`
9. **Run PS app:** Vitis workspace `D:\xilinx-tools\bitnet_vitis_ws_clean`, run `system_8x8_test.elf`, then `make uart-listen COM=COMx`.

**Automated PL program (after step 7 works):**

```powershell
. tools/setup/board_bringup.ps1 -Com COMx
```

(`djtgcfg` from Digilent Adept is optional; Vivado `make program` + `make run-ps` are the supported paths.)

**Note (Windows):** Full `settings64.bat` can exceed PATH length; `make run-ps` uses `tools/setup/run_ps_xsdb.ps1` (minimal PATH + `TCL_LIBRARY`).

## Verify

```powershell
. tools/setup/env.ps1
make list-hw
make program
```
