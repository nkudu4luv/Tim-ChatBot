import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Add fallback training examples so the model knows the class exists
fallback_patterns = [
    "what do you mean", "i don't get it", "huh", "random stuff",
    "tell me something", "blah blah", "xyzzy", "asdfgh",
    "not sure what to ask", "something else", "other"
]
for p in fallback_patterns:
    texts.append(p.lower())
    labels.append("fallback")

# VECTORIZER
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

# MODEL
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0
)

model.fit(X, labels)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Training completed")
print(f"   Classes trained: {len(model.classes_)}")
