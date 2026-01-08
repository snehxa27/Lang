import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# 1. CREATE DATASET (10 LANGUAGES)
# -----------------------------
data = {
    "text": [
        # English
        "hello how are you",
        "i love machine learning",
        "python is very powerful",
        "this project predicts language",
        "data science is interesting",

        # Spanish
        "hola como estas",
        "me gusta aprender programacion",
        "el aprendizaje automatico es interesante",
        "este proyecto detecta el idioma",
        "la ciencia de datos es fascinante",

        # French
        "bonjour comment ca va",
        "j aime apprendre la programmation",
        "l apprentissage automatique est interessant",
        "ce projet detecte la langue",
        "la science des donnees est fascinante",

        # German
        "hallo wie geht es dir",
        "ich liebe maschinelles lernen",
        "programmieren mit python macht spass",
        "dieses projekt erkennt die sprache",
        "datenwissenschaft ist sehr interessant",

        # Italian
        "ciao come stai",
        "mi piace programmare in python",
        "l apprendimento automatico e interessante",
        "questo progetto rileva la lingua",
        "la scienza dei dati e affascinante",

        # Hindi
        "नमस्ते आप कैसे हैं",
        "मुझे मशीन लर्निंग पसंद है",
        "पायथन एक शक्तिशाली भाषा है",
        "यह परियोजना भाषा पहचानती है",
        "डेटा विज्ञान रोचक है",

        # Marathi
        "नमस्कार तुम्ही कसे आहात",
        "मला मशीन लर्निंग आवडते",
        "पायथन ही शक्तिशाली भाषा आहे",
        "हा प्रकल्प भाषा ओळखतो",
        "डेटा सायन्स खूप रोचक आहे",

        # Tamil
        "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்",
        "எனக்கு மெஷின் லெர்னிங் பிடிக்கும்",
        "பைத்தான் ஒரு சக்திவாய்ந்த மொழி",
        "இந்த திட்டம் மொழியை கண்டறிகிறது",
        "டேட்டா சயின்ஸ் மிகவும் சுவாரசியமானது",

        # Telugu
        "నమస్తే మీరు ఎలా ఉన్నారు",
        "నాకు మెషిన్ లెర్నింగ్ ఇష్టం",
        "పైథాన్ ఒక శక్తివంతమైన భాష",
        "ఈ ప్రాజెక్ట్ భాషను గుర్తిస్తుంది",
        "డేటా సైన్స్ ఆసక్తికరంగా ఉంటుంది",

        # Urdu
        "ہیلو آپ کیسے ہیں",
        "مجھے مشین لرننگ پسند ہے",
        "پائتھن ایک طاقتور زبان ہے",
        "یہ منصوبہ زبان کی شناخت کرتا ہے",
        "ڈیٹا سائنس دلچسپ ہے"
    ],

    "language": (
        ["English"] * 5 +
        ["Spanish"] * 5 +
        ["French"] * 5 +
        ["German"] * 5 +
        ["Italian"] * 5 +
        ["Hindi"] * 5 +
        ["Marathi"] * 5 +
        ["Tamil"] * 5 +
        ["Telugu"] * 5 +
        ["Urdu"] * 5
    )
}

df = pd.DataFrame(data)

# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+|http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# 3. SCRIPT DETECTION (RULE-BASED)
# -----------------------------
def detect_script(text):
    if re.search(r'[अ-ह]', text):
        return "Devanagari"
    elif re.search(r'[அ-ஹ]', text):
        return "Tamil"
    elif re.search(r'[అ-హ]', text):
        return "Telugu"
    elif re.search(r'[؀-ۿ]', text):
        return "Urdu"
    else:
        return "Latin"

# -----------------------------
# 4. VECTORIZE & TRAIN ML MODEL
# -----------------------------
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5)
)

X = vectorizer.fit_transform(df["text"])
model = MultinomialNB()
model.fit(X, df["language"])

# -----------------------------
# 5. INTERACTIVE PREDICTION
# -----------------------------
print("\n🌍 HYBRID LANGUAGE DETECTION SYSTEM")
print("Rule-based Script Detection + ML Prediction")
print("Type a sentence and press Enter")
print("Type 'exit' to quit\n")

while True:
    sentence = input("📝 Enter sentence: ")

    if sentence.lower() == "exit":
        print("👋 Exiting...")
        break

    script = detect_script(sentence)
    print(f"🧾 Script detected: {script}")

    cleaned = clean_text(sentence)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    print(f"✅ Predicted Language: {prediction}\n")
