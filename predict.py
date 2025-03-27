# predict.py - 使用已训练好的模型进行预测

import numpy as np
import joblib

# 加载模型
model = joblib.load("model.pkl")

# 模拟一个新样本（3 个特征）
X_new = np.array([[0.3, 0.5, 0.8]])

# 预测
y_pred = model.predict(X_new)

# 输出结果
print("✅ 输入特征:", X_new)
print("🎯 预测结果:", y_pred)
