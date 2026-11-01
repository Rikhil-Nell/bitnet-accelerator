"""Vitis 2026 embedded Python: platform + standalone app for system_8x8_test."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Required for Zynq standalone flows (xsct is disabled in 2026.1).
os.environ["VITIS_EMBEDDED_INSTALL"] = "true"

REPO = Path(__file__).resolve().parents[2]
XSA = REPO / "build/vivado/bitnet_zybo.xsa"
XSA_STAGE = Path(os.environ.get("BITNET_XSA", "D:/xilinx-tools/bitnet_zybo.xsa"))
# Avoid spaces in user profile paths — Vitis platform gen can reject them.
WS = Path(os.environ.get("BITNET_VITIS_WS", "D:/xilinx-tools/bitnet_vitis_ws_clean"))
APP = os.environ.get("BITNET_PS_APP", "system_8x8_test")
PS_SRC = REPO / "fpga/ps" / APP


def main() -> int:
    if not XSA.is_file():
        print(f"Missing {XSA}", file=sys.stderr)
        return 1
    XSA_STAGE.parent.mkdir(parents=True, exist_ok=True)
    if not XSA_STAGE.exists() or XSA_STAGE.stat().st_mtime < XSA.stat().st_mtime:
        XSA_STAGE.write_bytes(XSA.read_bytes())
    hw = str(XSA_STAGE)

    import vitis

    WS.mkdir(parents=True, exist_ok=True)
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
    print("Build finished but ELF not found under build/vitis", file=sys.stderr)
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
