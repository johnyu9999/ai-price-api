
# 使用官方 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 拷贝本地代码到容器中
COPY . .

# 安装依赖
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn numpy joblib

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
