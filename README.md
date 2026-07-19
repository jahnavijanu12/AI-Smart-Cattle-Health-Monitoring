# 🐄 AI Smart Cattle Health Monitoring System Using Machine Learning

An intelligent Machine Learning-based web application that predicts the health status of cattle using physiological parameters. The system assists farmers and veterinarians by providing early disease prediction and classifying cattle into **Healthy**, **At Risk**, or **Sick** categories.


## 📌 Overview

Traditional cattle health monitoring relies on manual observation, which can delay disease detection and increase treatment costs. This project uses a **Random Forest Classifier** to analyze physiological data and provide real-time health predictions through an interactive **Streamlit** interface.


## 🎯 Objectives

- Predict cattle health using Machine Learning.
- Enable early disease detection.
- Support smart livestock management.
- Provide a simple and interactive user interface.
- Assist farmers in making timely healthcare decisions.


## ✨ Features

- Multi-Class Classification (Healthy, At Risk, Sick)
- Interactive Streamlit Web Interface
- Manual User Input
- Real-Time Health Prediction
- Random Forest Classifier
- Feature Importance Visualization
- Lightweight and Easy to Use


## 📊 Input Parameters

The prediction model uses the following physiological parameters:

| Parameter | Unit |
|-----------|------|
| Body Temperature | °C |
| Heart Rate | bpm |
| Respiration Rate | breaths/min |
| Rumination Time | minutes/day |
| Activity Level | Score |
| Weight | kg |
| Age | Years |


## 🎯 Output

The system predicts one of the following health conditions:

- ✅ Healthy
- ⚠️ At Risk
- ❌ Sick


## 🧠 Machine Learning Model

**Algorithm:** Random Forest Classifier

### Why Random Forest?

- High classification accuracy
- Handles multiple input features efficiently
- Reduces overfitting using ensemble learning
- Robust to noisy data
- Provides feature importance analysis


## 🔄 Project Workflow

```
User Input
      │
      ▼
Physiological Parameters
      │
      ▼
Data Preprocessing
      │
      ▼
Random Forest Classifier
      │
      ▼
Health Prediction
      │
      ▼
Healthy / At Risk / Sick
```


## 📐 Mathematical Background

Feature Vector:

```
X = [Temperature, Heart Rate, Respiration Rate,
     Rumination Time, Activity Level,
     Weight, Age]
```

Prediction Function:

```
Y = f(X)
```

Where:

- X → Input physiological parameters
- Y → Predicted health class

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score


## 💡 Project Innovation

- AI-powered early disease prediction
- Multi-parameter physiological analysis
- Multi-class classification approach
- User-friendly Streamlit interface
- Easily extendable for IoT-based smart farming


## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy


## 📁 Project Structure

```
AI-Smart-Cattle-Health-Monitoring/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
│   ├── poster.png
│   └── logo.png
└── venv/ (ignored)
```


## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Smart-Cattle-Health-Monitoring.git
```

Navigate to the project folder:

```bash
cd AI-Smart-Cattle-Health-Monitoring
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```


## 📸 Application

The application allows users to:

- Enter cattle physiological parameters
- Predict cattle health status
- View real-time prediction results
- Support decision-making for livestock management


## 🎯 Applications

- Dairy Farms
- Veterinary Clinics
- Livestock Monitoring
- Smart Farming
- Agricultural Research


## 🔮 Future Enhancements

- IoT Sensor Integration
- Wearable Health Monitoring Devices
- Cloud Deployment
- Mobile Application
- SMS/Email Alert System
- Deep Learning-Based Disease Prediction
- Real-Time Health Monitoring Dashboard


G. Pullaiah College of Engineering and Technology

---

## 📜 License

This project is developed for academic and educational purposes. It may be modified and extended for learning or research with proper attribution.
