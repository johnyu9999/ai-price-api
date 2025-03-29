import json
import matplotlib.pyplot as plt

metadata_file = "model_store/metadata.jsonl"
versions, mses, weights_list = [], [], []

with open(metadata_file, "r") as f:
    for line in f:
        entry = json.loads(line)
        versions.append(entry["version"])
        mses.append(entry["mse"])
        weights_list.append(entry["weights"])

# 🎯 MSE 趋势图
plt.figure()
plt.plot(versions, mses, marker="o")
plt.xlabel("Model Version")
plt.ylabel("Mean Squared Error")
plt.title("📉 MSE across model versions")
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_trend.png")
print("✅ Saved mse_trend.png")

if len(weights_list[0]) == 3:
    for i in range(3):
        plt.figure()
        ith_weights = [w[i] for w in weights_list]
        plt.plot(versions, ith_weights, marker="o")
        plt.xlabel("Model Version")
        plt.ylabel(f"Weight[{i}]")
        plt.title(f"⚙️ Trend of Weight[{i}]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"weight_{i}_trend.png")
        print(f"✅ Saved weight_{i}_trend.png")
