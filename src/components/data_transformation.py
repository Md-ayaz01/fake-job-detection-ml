import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.components.preprocessing import clean_text

class DataTransformation:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000)

    def transform_data(self, df, tfidf_save_path):
        df['text'] = df['text'].apply(clean_text)

        X = self.tfidf.fit_transform(df['text'])
        y = pd.to_numeric(df['fraudulent'], errors='coerce').fillna(0).astype(int)

        joblib.dump(self.tfidf, tfidf_save_path)
        print("✅ TF-IDF vectorizer saved at:", tfidf_save_path)

        return X, y
