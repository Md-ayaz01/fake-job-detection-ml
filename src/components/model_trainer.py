import xgboost as xgb
import joblib
from sklearn.metrics import classification_report

class ModelTrainer:
    def __init__(self, model_save_path, df_combined):
        fraud = df_combined['fraudulent'].sum()
        legit = len(df_combined) - fraud

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            scale_pos_weight=legit / fraud,   # ✅ Handle imbalance
            eval_metric='logloss',
            random_state=42
        )
        self.model_save_path = model_save_path

    def train(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print("✅ Model Training Completed.")
print(classification_report(y_test, y_pred))

joblib.dump(self.model, self.model_save_path)
print("✅ Model saved at:", self.model_save_path)