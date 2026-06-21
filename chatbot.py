import json
import pickle
import random
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CONFIDENCE_THRESHOLD = 0.018


# ---------------- AUTO-RETRAIN IF STALE ----------------
def train_model():
    """Train and save a fresh model using the current sklearn version."""
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            labels.append(intent["tag"])

    for p in ["what do you mean", "i don't get it", "huh", "random stuff",
              "tell me something", "blah blah", "xyzzy", "asdfgh",
              "not sure what to ask", "something else", "other"]:
        texts.append(p)
        labels.append("fallback")

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(texts)

    mdl = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    mdl.fit(X, labels)

    joblib.dump(mdl, "model.pkl")
    joblib.dump(vec, "vectorizer.pkl")
    print("✅ Model retrained successfully.")
    return mdl, vec


def load_model():
    """Load model, retraining automatically if the saved file is incompatible."""
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        print("⚙️  No model found. Training now...")
        return train_model()
    try:
        mdl = joblib.load("model.pkl")
        vec = joblib.load("vectorizer.pkl")
        # Probe to catch version mismatch before it causes a crash
        vec.transform(["test probe"])
        mdl.predict_proba(vec.transform(["test probe"]))
        return mdl, vec
    except Exception:
        print("⚙️  Saved model is incompatible with your sklearn version. Retraining...")
        return train_model()


# ---------------- LOAD ----------------
model, vectorizer = load_model()

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)


# ---------------- FUNCTIONS ----------------
def predict_intent(text):
    try:
        X = vectorizer.transform([text.lower().strip()])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            confidence = float(max(probs))
            tag = str(model.classes_[probs.argmax()])
            if confidence < CONFIDENCE_THRESHOLD:
                return "fallback", confidence
            return tag, confidence

        return str(model.predict(X)[0]), 1.0

    except Exception:
        return "fallback", 0.0


def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    for intent in data["intents"]:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])

    return "I don't understand that. Please rephrase your question."


# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    print("🤖 Nigerian SME Business Chatbot Ready!")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("Bot: Goodbye 👋")
            break

        tag, confidence = predict_intent(user_input)
        response = get_response(tag)

        print(f"Bot: {response} (confidence: {confidence:.3f})")
