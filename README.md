# 📰 Fake News Detection using PassiveAggressiveClassifier

A machine learning project that classifies news articles as real or fake based on text content. Built with TF-IDF vectorization and PassiveAggressiveClassifier, this system allows real-time user input for headline validation.

## 🔍 Overview
This project merges labeled real and fake news datasets, trains a text classifier, and enables live testing via command-line input. It showcases an end-to-end ML pipeline—from data ingestion to evaluation and user prediction.

## 💡 Key Features
- Dataset merge and label assignment
- TF-IDF vectorization with English stop-word removal
- Classification using PassiveAggressiveClassifier
- Performance evaluation with accuracy and confusion matrix
- Real-time testing of user-submitted news content

## 📁 File Summary
- `app.py` – Main script with full pipeline and user input prediction
- `True.csv`, `Fake.csv` – News datasets from Kaggle (first 4 columns used)
- `requirements.txt` – Python dependencies (e.g., pandas, scikit-learn)

## 🧠 Model Highlights
- **Vectorizer**: `TfidfVectorizer(stop_words="english", max_df=0.7)`
- **Classifier**: `PassiveAggressiveClassifier(max_iter=50)`
- **Metrics**: `accuracy_score`, `confusion_matrix`

## 📊 How It Works
1. Loads and concatenates True and Fake datasets
2. Shuffles and splits into training/testing sets
3. Transforms text using TF-IDF
4. Trains PassiveAggressiveClassifier
5. Prints accuracy and confusion matrix
6. Accepts real-time news input and outputs prediction

