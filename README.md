# 📰 Fake News Detection

A **Machine Learning-powered** web application that detects **fake news** using Flask 

## 🚀 Features

✅ **Real-time Fake News Detection** – Enter a news article, and the AI will classify it as **FAKE** or **REAL**.

✅ **Confidence Score** – Shows the probability of correctness.

✅ **Modern UI** – A sleek interface with animations and a dark mode.

✅ **Preloader Animation** – Enhances user experience with a stylish loading effect.

---

## 🛠️ Tech Stack

### **Frontend ( HTML/CSS)**
-Javascript
- 🌑 **Dark Mode Toggle**

### **Backend (Flask & AI Model)**
- 🐍 **Flask** – Python backend
- 🤖 **Scikit-Learn** – Trained ML model
- 📦 **Pickle** – Model serialization (`trained_model.pkl`)

---

## 🎯 How It Works

1️⃣ **User Inputs News** – Enter a news article in the text box.

2️⃣ **Backend Processing** – Flask processes the input and passes it to the ML model.

3️⃣ **Prediction & Confidence** – The model predicts whether the news is **FAKE** or **REAL** and returns a confidence score.

4️⃣ **UI Animation** – The result is displayed dynamically with **smooth animations**.
 
## 📌 Setup & Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### **2️⃣ Backend Setup (Flask & ML Model)**
```bash
cd backend
pip install -r requirements.txt
python app.py
```
➡ The Flask server will start at `http://127.0.0.1:5000/`

### **3️⃣ Frontend Setup (React)**
```bash
cd frontend
npm install
npm start
```
➡ The React frontend will run at `http://localhost:3000/`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|------------|
| `POST` | `/predict` | Takes news text and returns prediction & confidence score. |

Example Request:
```json
{
  "text": "Breaking news: Scientists discover AI that writes perfect READMEs!"
}
```

Example Response:
```json
{
  "prediction": "FAKE",
  "confidence": 92.5
}
```

---

## 🎓 Model Training
The machine learning model is trained using **TF-IDF Vectorization** and **Logistic Regression** with a dataset of real and fake news articles.

Training Code (inside `train_model.py`):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

## 🚀 Future Enhancements

🔹 **Improve Model Accuracy** – Enhance dataset quality and explore **Deep Learning** models.

🔹 **User Feedback System** – Allow users to report incorrect classifications.

🔹 **Cloud Deployment** – Host the project on **AWS/GCP** for public access.
