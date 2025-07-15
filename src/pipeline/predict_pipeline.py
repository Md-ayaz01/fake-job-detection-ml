import joblib

class PredictPipeline:
    def __init__(self, model_path, tfidf_path):
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)

    def predict(self, text):
        X = self.tfidf.transform([text])
        return self.model.predict(X)
