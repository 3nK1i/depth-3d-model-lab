#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.perception.availability import check_perception_availability
from core.perception.registry import PERCEPTION_MODEL_REGISTRY

def main():
    for d in PERCEPTION_MODEL_REGISTRY.descriptors():
        a = check_perception_availability(d)
        print(f'{d.model_id}: {a.status.value}' + (f' ({a.message})' if a.message else ''))
if __name__ == '__main__':
    main()
