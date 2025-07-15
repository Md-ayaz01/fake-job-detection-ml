from flask import Flask, render_template, request
import pandas as pd
import joblib
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Load alternated dataset for dropdowns
df = pd.read_csv('data/processed/alternated_combined_data.csv')
titles = df['title'].dropna().unique().tolist()
descriptions = df['description'].dropna().unique().tolist()

# Load prediction pipeline
pipeline = PredictPipeline('model/fake_job_model.pkl', 'model/tfidf.pkl')

@app.route('/')
def index():
    return render_template('index.html', titles=titles, descriptions=descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    description = request.form['description']
    text_input = title + " " + description
    result = pipeline.predict(text_input)
    output = "🚨 Fake Job Detected!" if result[0] == 1 else "✅ Legitimate Job Posting"
    return render_template('result.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
