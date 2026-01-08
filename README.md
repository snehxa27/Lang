
# 🌍 Hybrid Language Detection System (Machine Learning)

This project is a **Hybrid Language Detection System** that predicts the language of a given sentence using a combination of **rule-based script detection** and **machine learning** techniques.

The model supports **multiple international and Indian languages** and works interactively from the terminal.

---

## 🧠 Key Idea

Instead of relying only on machine learning, this project uses a **hybrid approach**:

1. **Script Detection (Rule-based)**  
   Detects the writing system of the input text using Unicode ranges  
   (Latin, Devanagari, Tamil, Telugu, Urdu).

2. **Machine Learning Prediction**  
   Uses **TF-IDF character n-grams** with a **Naive Bayes classifier** to predict the exact language.

This approach improves accuracy and handles multilingual inputs more effectively.

---

## ✨ Features

- Hybrid approach: **Rule-based + ML**
- Supports **10 languages**:
  - English
  - Spanish
  - French
  - German
  - Italian
  - Hindi
  - Marathi
  - Tamil
  - Telugu
  - Urdu
- Character-level **TF-IDF vectorization**
- Naive Bayes classification
- Interactive sentence prediction
- Clean and beginner-friendly code

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Regular Expressions (Regex)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/snehxa27/Lang.git
cd Lang

python -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn
python lang.py
🧪 Example Output
📝 Enter sentence: नमस्कार तुम्ही कसे आहात
🧾 Script detected: Devanagari
✅ Predicted Language: Marathi

📝 Enter sentence: hola como estas
🧾 Script detected: Latin
✅ Predicted Language: Spanish
