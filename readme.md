# 🧠 AI-Based Antibiotic Resistance Prediction System

An AI-powered decision support system that predicts antibiotic resistance and provides explainable insights for better healthcare decisions.

---

## 🚀 Overview

Antibiotic resistance is a critical global health issue. Incorrect antibiotic selection can lead to treatment failure.

This project uses **Machine Learning** to:

* Predict antibiotic resistance
* Provide confidence scores
* Assist in clinical decision-making

---

## 🌐 Live Demo

👉 https://antibiotic-resistance-predictor-f8tp.onrender.com

---

## ✨ Key Features

* Predicts resistance for **Ciprofloxacin**
* Uses **Primary + Secondary datasets**
* Handles real-world noisy data
* Provides **confidence score + recommendation**
* Interactive UI built with **Gradio**
* Auto-fill based on location
* Includes **feature importance graph**
* Includes **resistance network visualization**

---

## 🧪 Input & Output

### 🔹 Inputs

* Location
* IMIPENEM
* CEFTAZIDIME
* GENTAMICIN
* AUGMENTIN

### 🔹 Output

* Prediction for **CIPROFLOXACIN**
* Confidence score
* Recommendation:

  * 🟢 Use
  * 🟡 Use with caution
  * 🔴 Avoid

---

## 🧠 Model

* Algorithm: **Random Forest Classifier**
* Type: Supervised Learning
* Task: Classification

### 📊 Data Handling

* Combined multiple datasets
* Cleaned missing & inconsistent values
* Performed feature engineering

---

## 💻 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Gradio
* Matplotlib
* NetworkX

---

## 📁 Project Structure

```id="n3yoj6"
├── app.py
├── train.py
├── model.pkl
├── Dataset.xlsx
├── Bacteria_dataset_Multiresictance.csv
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Team

* **Tanuj Kumar Singh** (Team Leader)
* Bhavna Agrawal
* Divya Goyal

---

## 🏆 Highlights

* End-to-end ML pipeline
* Real-world dataset integration
* Interactive AI system
* Explainable predictions

---

## 🔮 Future Scope

* Multi-output prediction
* Real-time healthcare integration
* Advanced AI models

---

## 📌 Conclusion

This system demonstrates how AI can help tackle antibiotic resistance by providing **accurate predictions and meaningful insights** for better treatment decisions.

---

⭐ Star this repo if you found it useful!
