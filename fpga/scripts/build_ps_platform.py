"""Vitis 2026 embedded Python: (re)build the standalone platform from the current XSA
and build the app in fpga/ps/$BITNET_PS_APP.

The platform is rebuilt whenever build/vivado/bitnet_zybo.xsa is newer than it, so a
new bitstream can never be paired with stale xparameters.h / ps7_init.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Required for Zynq standalone flows (xsct is disabled in 2026.1).
os.environ["VITIS_EMBEDDED_INSTALL"] = "true"

REPO = Path(__file__).resolve().parents[2]
XSA = REPO / "build/vivado/bitnet_zybo.xsa"
WORK = Path(os.environ.get("BITNET_WORK", "D:/xilinx-tools"))
XSA_STAGE = Path(os.environ.get("BITNET_XSA", str(WORK / "stage" / "bitnet_zybo.xsa")))
# Avoid spaces in user profile paths — Vitis platform gen can reject them.
WS = Path(os.environ.get("BITNET_VITIS_WS", str(WORK / "vitis_ws")))
APP = os.environ.get("BITNET_PS_APP", "gemv_bench")
PS_SRC = REPO / "fpga/ps" / APP


def main() -> int:
    if not XSA.is_file():
        print(f"Missing {XSA}", file=sys.stderr)
        return 1
    XSA_STAGE.parent.mkdir(parents=True, exist_ok=True)
    if not XSA_STAGE.exists() or XSA_STAGE.stat().st_mtime < XSA.stat().st_mtime:
        XSA_STAGE.write_bytes(XSA.read_bytes())
    hw = str(XSA_STAGE)

    import shutil

    import vitis

    stamp = WS.parent / f"{WS.name}.xsa_stamp"   # outside: Vitis rejects unknown files in a new workspace
    if WS.exists() and (not stamp.is_file() or stamp.read_text().strip() != str(XSA.stat().st_mtime_ns)
                        or not (WS / "_ide").exists()):
        print(f"XSA changed: recreating workspace {WS}")
        def _force(func, path, _exc):   # Vitis leaves read-only files behind
            os.chmod(path, 0o666)
            func(path)

        shutil.rmtree(WS, onexc=_force)
    WS.mkdir(parents=True, exist_ok=True)
    stamp.write_text(str(XSA.stat().st_mtime_ns))
    ide = WS / "_ide"
    for stale in (ide / ".peers.ini", ide / "workspace_journal.py"):
        if stale.is_file():
            stale.unlink(missing_ok=True)

    try:
        client = vitis.create_client(workspace=str(WS))
    except Exception as exc:
        if "already in use" not in str(exc):
            raise
        print(
            "Vitis workspace locked — close Vitis IDE on this workspace, then retry make ps-app.",
            file=sys.stderr,
        )
        raise

    emb_sw = Path(os.environ["XILINX_VITIS"]) / "data" / "embeddedsw"
    if emb_sw.is_dir() and not client.get_embedded_sw_repo():
        client.add_embedded_sw_repo(level="LOCAL", path=str(emb_sw))

    cpu = "ps7_cortexa9_0"
    os_name = "standalone"
    proc_os = client.get_processor_os_list(hw)
    print(f"processor/os hint: {proc_os!r}")

    xpfm = WS / "bitnet_platform/export/bitnet_platform/bitnet_platform.xpfm"
    if not xpfm.is_file():
        platform = client.create_platform_component(
            name="bitnet_platform",
            hw_design=hw,
            os=os_name,
            cpu=cpu,
            domain_name="standalone_domain",
        )
        # SD card weight loading: FatFs with exFAT (big cards ship exFAT-formatted)
        dom = platform.get_domain("standalone_domain")
        dom.set_lib("xilffs")
        for key, val in (("XILFFS_enable_exfat", "true"), ("XILFFS_read_only", "true"),
                         ("XILFFS_use_mkfs", "false"),
                         ("XILFFS_enable_multi_partition", "true")):
            dom.set_config("lib", key, val, "xilffs")
        status = platform.build()
        if status:
            print(f"Platform build status: {status}")
    if not xpfm.is_file():
        print(f"Platform XPFM missing: {xpfm}", file=sys.stderr)
        return 1
    print(f"Using platform: {xpfm}")

    comp = None
    for entry in client.list_components():
        if entry.get("name") == APP:
            comp = client.get_component(name=APP)
            break
    if comp is None:
        comp = client.create_app_component(
            name=APP,
            platform=str(xpfm),
            domain="standalone_domain",
            template="empty_application",
        )

    # Release build: the forward pass runs on the A9, so optimisation matters.
    comp.set_app_config(key="USER_COMPILE_OPTIMIZATION_LEVEL", values="-O3")
    comp.set_app_config(key="USER_COMPILE_OPTIMIZATION_OTHER_FLAGS",
                        values="-mfpu=neon-fp16 -mfp16-format=ieee -ffast-math -funroll-loops")
    comp.set_app_config(key="USER_COMPILE_DEBUG_LEVEL", values="-g1")
    comp.set_app_config(key="USER_LINK_LIBRARIES", values="m")

    app_src = WS / APP / "src"
    app_src.mkdir(parents=True, exist_ok=True)
    for src in PS_SRC.iterdir():
        if src.is_file():
            (app_src / src.name).write_bytes(src.read_bytes())

    comp.build()
    elf = WS / APP / "build" / f"{APP}.elf"
    if elf.is_file():
        print(f"Built PS app: {elf}")
        return 0
    for candidate in WS.rglob(f"{APP}.elf"):
        print(f"Built PS app: {candidate}")
        return 0
    print(f"Build finished but {APP}.elf not found under {WS}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    err_log = REPO / "build/vitis/ps-app-python.err"
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        err_log.parent.mkdir(parents=True, exist_ok=True)
        err_log.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        try:
            import vitis

            vitis.dispose()
        except Exception:
            pass
