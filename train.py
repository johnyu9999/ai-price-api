# train.py - 训练线性回归模型并保存为 model.pkl

import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 设置随机种子确保结果一致
np.random.seed(42)

# 构造训练数据（200 个样本，3 个特征）
n_samples = 200
n_features = 3
X = np.random.rand(n_samples, n_features)

# 定义真实的权重和偏置
true_w = np.array([1.5, -2.0, 3.0])
true_b = 4.2

# 添加噪声
noise = np.random.randn(n_samples) * 0.1
y = X @ true_w + true_b + noise

# 初始化并训练模型
model = LinearRegression()
model.fit(X, y)

# 打印训练结果
print("Trained weights:", model.coef_)
print("Trained bias:", model.intercept_)

# 保存模型为 model.pkl
joblib.dump(model, "model.pkl")
print("✅ Model saved to model.pkl")
