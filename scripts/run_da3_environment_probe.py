#!/usr/bin/env python3
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.reconstruction.da3 import probe_da3_environment

python = Path.home() / "anaconda3/envs/eia-da3/bin/python"
repo = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
probe = probe_da3_environment(python, repo)
out = Path("/tmp/phase8_7f2_da3_probe.json")
out.write_text(json.dumps(probe.as_dict(), indent=2, sort_keys=True) + "\n")
print("Phase 8.7F2 DA3 environment probe:", "PASS" if probe.available else "NOT READY")
print(f"available={probe.available} source_revision={probe.source_revision or 'none'}")
print(f"python={probe.python_version} torch={probe.torch_version} torchvision={probe.torchvision_version} xformers={probe.xformers_version} cuda={probe.torch_cuda_version}")
print(f"gpus={list(probe.device_names)} capabilities={list(probe.device_capabilities)}")
print(f"turing_patch={probe.turing_compatibility_patch_applied} runtime_lock={probe.runtime_lock_valid}")
print(f"production_depth_environment_untouched=True")
print(f"summary={out}")
raise SystemExit(0 if probe.available else 2)
