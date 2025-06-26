# 🔍 Hate Speech Detection and Classification (Multitask Learning)

This project focuses on identifying and categorizing hate speech from user-generated text using a **multi-task deep learning model**. It not only detects the presence of hate but also classifies its **target** (individual/group/other) and **severity** (low/medium/high).

---

## 🚀 Project Overview

Given the rising volume of online content, it is crucial to detect and analyze hate speech accurately. This project leverages a **multi-output neural network** using the **Universal Sentence Encoder (USE)** as an embedding layer, built with TensorFlow and trained on a labeled dataset.

---

## 🎯 Core Objectives

- ✅ Detect whether a given text contains hate speech.
- ✅ Identify the **target** of the hate (Individual / Group / Other).
- ✅ Assess the **severity** level (Low / Medium / High).
- ✅ Build an end-to-end prediction system using TensorFlow and NLP preprocessing.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow 2.12**
- **Keras Functional API**
- **Google Universal Sentence Encoder**
- **NLTK for Text Preprocessing**
- **Scikit-learn & Pandas**
- **Joblib (for model export)**
- **Google Colab for Training**

---

## 🧠 Model Architecture

- **Input:** 512-d Universal Sentence Embeddings
- **Shared Layers:** Dense → Dropout
- **Outputs:**
  - `hate_output`: Binary classification (Hate / Not Hate)
  - `target_output`: 3-class softmax (Individual / Group / Other)
  - `severity_output`: 3-class softmax (Low / Medium / High)

---

## 📊 Dataset & Preprocessing

- Dataset: CSV file with labeled `Tweet`, `Hate`, `Target`, and `Severity` fields
- Preprocessing:
  - Lowercasing
  - Removing URLs, mentions, and punctuation
  - Tokenization & Stopword removal using NLTK
- Labels encoded and converted to categorical format

---

## 🧪 Training Summary

- Train-test split: 80/20
- Embeddings: USE v4 via TensorFlow Hub
- Loss Functions:
  - Binary Cross-Entropy for hate detection
  - Categorical Cross-Entropy for target & severity
- Metrics: Accuracy for all outputs
- Epochs: 30  
- Batch size: 32

---

## 🧩 How to Use

1. Clean the text input using the `clean_text()` function
2. Generate embeddings using USE
3. Predict using `predict_hate_speech()` function
4. Returns:  
   - Hate/Not Hate  
   - Target (Individual / Group / Other)  
   - Severity (Low / Medium / High)

---

## 🖥️ Sample Usage

```python
hate, target, severity = predict_hate_speech("Example input sentence here.")
print(f"Hate: {hate}, Target: {target}, Severity: {severity}")
