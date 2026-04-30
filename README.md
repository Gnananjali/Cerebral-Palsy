# 🧠 CP-Detect AI — Cerebral Palsy Risk Detection System

An AI-powered web application that analyzes infant movement videos to predict the risk of Cerebral Palsy using deep learning models (CNN + LSTM/GRU/BiLSTM).

---

## 🚀 Features

* 🔐 User Authentication (Login/Register)
* 📊 Interactive Dashboard with analytics
* 🎥 Video Upload & Processing
* 🤖 AI-based Risk Prediction
* 📈 Performance Metrics (Accuracy, Recall, F1-score, etc.)
* 🧾 History & Report Tracking
* 🛠 Admin Panel for monitoring system activity

---

## 🧠 AI Model

* CNN for spatial feature extraction
* LSTM / GRU / BiLSTM for temporal analysis
* Trained on RGB + Depth datasets
* Outputs:

  * Risk Score
  * Prediction (Normal / Moderate / High Risk)
  * Confidence & performance metrics

---

## 🛠 Tech Stack

**Frontend**

* HTML, CSS, JavaScript

**Backend**

* Flask (Python)

**AI / ML**

* TensorFlow / Keras
* OpenCV
* NumPy

**Database**

* SQLite

**Deployment**

* Gunicorn (Production server)

---

## 📁 Project Structure

```
Cerebral-palsy/
│
├── app.py / run.py        # Main Flask app
├── datatrain.py          # Model training script
├── requirements.txt      # Dependencies
├── Procfile              # Deployment config
├── static/               # CSS, JS, assets
├── templates/            # HTML templates
├── uploads/              # Uploaded videos
├── models/               # Trained models (.h5)
└── database.db           # SQLite DB
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/Cerebral-Palsy.git
cd Cerebral-Palsy
```

### 2. Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```
python run.py
```

OR (production mode)

```
gunicorn run:app --bind 0.0.0.0:5000
```

Open:

```
http://localhost:5000
```

---

## ⚠️ Note

* If model file (`.h5`) is not present, the system runs in **simulation mode**
* Upload folder uses temporary storage in deployment environments

---

## 🌐 Deployment

Recommended platform:

* Render / Railway / AWS

Start command:

```
gunicorn run:app --workers 1 --timeout 120
```

---

## 📊 Future Improvements

* Real-time video streaming analysis
* Cloud storage integration
* Improved model accuracy with larger datasets
* Mobile app version


