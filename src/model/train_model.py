import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import os
from src.components.preprocessing import clean_text

# Load the alternated dataset
df = pd.read_csv('data/processed/alternated_combined_data.csv')

# Automatically combine text fields
df['text'] = df[['title', 'description']].fillna('').agg(' '.join, axis=1)

# Clean text
df['text'] = df['text'].apply(clean_text)

# Labels and features
X_text = df['text']
y = df['fraudulent']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=8000)
X = tfidf.fit_transform(X_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Save model and TF-IDF
joblib.dump(model, 'model/fake_job_model.pkl')
joblib.dump(tfidf, 'model/tfidf.pkl')

print("✅ Model and TF-IDF saved in 'model/' using alternated dataset.")
