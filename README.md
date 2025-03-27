
# 🧠 Linear Regression Prediction API

A minimal end-to-end AI service that includes:

- Synthetic data generation
- Model training using `sklearn`
- Model serialization with `joblib`
- Prediction via CLI and FastAPI
- Ready for production-grade extension (Docker, deployment, logging)

---

## 📁 Project Structure

```
linear-regression-service/
├── train.py         # Train the model and save to model.pkl
├── predict.py       # Load model and run a sample prediction
├── serve.py         # FastAPI service exposing /predict endpoint
├── model.pkl        # Trained sklearn LinearRegression model
└── README.md        # This documentation
```

---

## 🚀 How to Use

### 🔧 1. Train the Model

```bash
python train.py
```

This will generate `model.pkl`.

### 🧪 2. Run a Local Prediction

```bash
python predict.py
```

Output:
```
✅ 输入特征: [[0.3 0.5 0.8]]
🎯 预测结果: [6.04]
```

### 🌐 3. Start the FastAPI Server

```bash
uvicorn serve:app --reload
```

Open browser at [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

### 📬 4. Send Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.3, 0.5, 0.8]}'
```

Response:
```json
{"predicted_price": 6.04}
```

---

## 🧠 Interview Talking Points

- "I implemented a complete AI service from scratch using only `numpy`, `sklearn`, and `FastAPI`."
- "The model is trained on synthetic data with noise, then serialized using `joblib`."
- "I designed a modular structure with train/predict/serve separation, ready for production deployment."
- "The API accepts feature vectors and returns real-time predictions. I used `pydantic` for validation."
- "This project demonstrates my understanding of the AI model lifecycle: training, saving, serving, and inference."

---

## ✅ Next Steps (Optional Enhancements)

- Add input validation and error handling
- Include request logging and timing metrics
- Package with Docker for portable deployment
- Deploy to cloud services (e.g., Render, GCP, Hugging Face Spaces)
- Add authentication and rate limiting

---

**Built by Me + ChatGPT 🤖🔥**
