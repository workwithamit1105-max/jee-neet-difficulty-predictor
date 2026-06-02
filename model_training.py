import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.utils import logger, MODELS_DIR
from src.data_preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer

class ModelTrainer:
    """Class to manage training, comparison, and evaluation of ML models."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        self.difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
        self.inv_difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
        
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric='mlogloss'
            )
        }
        self.best_model_name = None
        self.metrics_results = {}
        
    def prepare_data(self, df: pd.DataFrame):
        """Preprocesses text and extracts features from the dataset."""
        logger.info("Starting data preparation...")
        
        # 1. Map target difficulty
        df['label'] = df['Difficulty'].map(self.difficulty_map)
        
        # 2. Preprocess text
        logger.info("Preprocessing text...")
        df['Clean_Question_Text'] = df['Question_Text'].apply(self.preprocessor.tokenize_and_lemmatize)
        
        # 3. Split data before fitting feature engineering to avoid leakage
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # 4. Fit feature engineering on train and transform both
        self.feature_engineer.fit(train_df, text_col='Clean_Question_Text')
        
        X_train = self.feature_engineer.transform(train_df, text_col='Clean_Question_Text')
        X_test = self.feature_engineer.transform(test_df, text_col='Clean_Question_Text')
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        return X_train, X_test, y_train, y_test, train_df, test_df

    def train_and_compare(self, X_train, X_test, y_train, y_test):
        """Trains all models, evaluates them, and determines the best one."""
        logger.info("Training and comparing models...")
        
        best_score = -1
        best_model_obj = None
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            self.metrics_results[name] = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm
            }
            
            logger.info(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            # Check if this is the best model (using F1 score)
            if f1 > best_score:
                best_score = f1
                self.best_model_name = name
                best_model_obj = model
                
        logger.info(f"Best model selected: {self.best_model_name} with F1-Score: {best_score:.4f}")
        
        # Save best model details
        self.metrics_results["best_model"] = self.best_model_name
        self.metrics_results["dataset_metrics"] = {
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "num_features": int(X_train.shape[1])
        }
        
        # Save preprocessors and the best model
        self.save_artifacts(best_model_obj)
        
        return self.metrics_results

    def save_artifacts(self, best_model_obj):
        """Saves preprocessor, feature engineer, best model, and training metadata to disk."""
        logger.info("Saving training artifacts...")
        
        joblib.dump(self.preprocessor, MODELS_DIR / "preprocessor.joblib")
        joblib.dump(self.feature_engineer, MODELS_DIR / "feature_engineer.joblib")
        joblib.dump(best_model_obj, MODELS_DIR / f"best_model_{self.best_model_name.lower().replace(' ', '_')}.joblib")
        # Also save as a generic best_model.joblib for easy loading
        joblib.dump(best_model_obj, MODELS_DIR / "best_model.joblib")
        
        # Save metrics as json
        with open(MODELS_DIR / "training_metrics.json", "w") as f:
            json.dump(self.metrics_results, f, indent=4)
            
        logger.info("Artifacts saved successfully!")

    def predict_single(self, question: str, subject: str, topic: str):
        """Predicts the difficulty of a single question using the saved best model."""
        # Check if artifacts exist
        if not (MODELS_DIR / "best_model.joblib").exists():
            raise FileNotFoundError("Model files not found. Please train models first.")
            
        # Load artifacts
        prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
        fe = joblib.load(MODELS_DIR / "feature_engineer.joblib")
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        
        # Preprocess
        clean_q = prep.tokenize_and_lemmatize(question)
        
        # Create single row dataframe
        df_single = pd.DataFrame([{
            'Question_Text': question,
            'Clean_Question_Text': clean_q,
            'Subject': subject,
            'Topic': topic
        }])
        
        # Transform features
        X_single = fe.transform(df_single, text_col='Clean_Question_Text')
        
        # Predict class and probability
        pred_class_idx = model.predict(X_single)[0]
        pred_class = self.inv_difficulty_map[pred_class_idx]
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_single)[0]
            confidence = float(probs[pred_class_idx])
        else:
            confidence = 1.0  # fallback if model doesn't support probability
            
        return {
            "difficulty": pred_class,
            "confidence": confidence,
            "probabilities": {self.inv_difficulty_map[i]: float(p) for i, p in enumerate(probs)} if hasattr(model, "predict_proba") else {}
        }
