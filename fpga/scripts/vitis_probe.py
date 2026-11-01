import os
from pathlib import Path

os.environ["VITIS_EMBEDDED_INSTALL"] = "true"
import vitis

REPO = Path(__file__).resolve().parents[2]
WS = REPO / "build/vitis_probe_ws3"
XSA = REPO / "build/vivado/bitnet_zybo.xsa"
emb = Path(os.environ["XILINX_VITIS"]) / "data" / "embeddedsw"

client = vitis.create_client(workspace=str(WS))
client.add_embedded_sw_repo(level="LOCAL", path=str(emb))
print("repos:", client.get_embedded_sw_repo())
print("proc_os:", client.get_processor_os_list(str(XSA)))
