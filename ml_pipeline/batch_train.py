import os
import yaml
import subprocess

config_dir = "."
configs = [f for f in os.listdir(config_dir) if f.startswith("config_") and f.endswith(".yaml")]

for cfg_file in configs:
    full_path = os.path.join(config_dir, cfg_file)
    print(f"\n🚀 Training from {cfg_file}...")
    result = subprocess.run(["python", "train_and_register.py", full_path])
    if result.returncode != 0:
        print(f"❌ Failed to train with {cfg_file}")
    else:
        print(f"✅ Done with {cfg_file}")
