# app.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ===== LOAD DATA =====
# Read True.csv (first 4 columns only)
true_df = pd.read_csv("True.csv", encoding="latin1", usecols=[0, 1, 2, 3])
# Read Fake.csv (first 4 columns only)
fake_df = pd.read_csv("Fake.csv", encoding="latin1", usecols=[0, 1, 2, 3])

# Add label: 1 for True news, 0 for Fake news
true_df["label"] = 1
fake_df["label"] = 0

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# ===== SPLIT DATA =====
X = df["text"]  # news content
y = df["label"]  # labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== VECTORIZATION =====
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# ===== TRAIN MODEL =====
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# ===== PREDICT & EVALUATE =====
y_pred = model.predict(tfidf_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"Model Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)

# ===== TEST WITH CUSTOM INPUT =====
print("\n=== Test Your Own News ===")
user_news = input("Enter a news article: ")

# Convert to TF-IDF
user_tfidf = vectorizer.transform([user_news])

# Predict
prediction = model.predict(user_tfidf)[0]

if prediction == 1:
    print("✅ This news is REAL")
else:
    print("❌ This news is FAKE")
