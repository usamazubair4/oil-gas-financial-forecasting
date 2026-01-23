import pickle
from pathlib import Path

MODEL_DIR = Path("trained_models")

for cfg_file in MODEL_DIR.glob("*_config.pkl"):
    with open(cfg_file, "rb") as f:
        cfg = pickle.load(f)

    print(f"KPI: {cfg_file.stem.replace('_config','')}")
    print(f"  Order: {cfg['order']}")
    print(f"  Seasonal: {cfg['seasonal_order']}")
    print(f"  MAPE: {cfg['mape']}\n")
