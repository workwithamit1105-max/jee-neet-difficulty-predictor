import argparse
import sys
import pandas as pd
from pathlib import Path

from src.utils import logger, MODELS_DIR
from src.database import init_db, get_combined_dataset
from src.model_training import ModelTrainer

def run_training():
    """Combines dataset, trains models, evaluates performance, and saves the best model."""
    logger.info("Starting training pipeline...")
    
    # Get combined dataset
    df = get_combined_dataset()
    if df.empty:
        logger.error("No questions found in dataset! Please generate questions or check data/questions.csv.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df)} questions for training.")
    
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test, train_df, test_df = trainer.prepare_data(df)
    
    # Train and compare
    metrics = trainer.train_and_compare(X_train, X_test, y_train, y_test)
    
    # Print resume-ready metrics to stdout
    print("\n" + "="*50)
    print("           RESUME-READY PERFORMANCE METRICS")
    print("="*50)
    print(f"Dataset Size:        {len(df)} questions")
    print(f"Training Set Size:   {metrics['dataset_metrics']['train_size']} questions")
    print(f"Test Set Size:       {metrics['dataset_metrics']['test_size']} questions")
    print(f"Number of Features:  {metrics['dataset_metrics']['num_features']}")
    print(f"Best Model Selected: {metrics['best_model']}")
    print("\nModel Performance Comparison:")
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-"*50)
    for model_name, model_metrics in metrics.items():
        if model_name not in ["best_model", "dataset_metrics"]:
            print(f"{model_name:<25} | {model_metrics['accuracy']:.4f}     | {model_metrics['f1_score']:.4f}")
    print("="*50 + "\n")
    
    logger.info("Training pipeline completed successfully.")

def run_prediction(question: str, subject: str, topic: str):
    """Predicts difficulty of a single question and prints result."""
    trainer = ModelTrainer()
    try:
        result = trainer.predict_single(question, subject, topic)
        print(f"\nQuestion: {question}")
        print(f"Subject: {subject} | Topic: {topic}")
        print(f"Predicted Difficulty: {result['difficulty']}")
        print(f"Confidence Score: {result['confidence'] * 100:.2f}%\n")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run training first using: python main.py --train")

def main():
    parser = argparse.ArgumentParser(description="JEE/NEET Question Difficulty Predictor CLI")
    parser.add_argument("--init-db", action="store_true", help="Initialize SQLite Database")
    parser.add_argument("--train", action="store_true", help="Train and compare models")
    parser.add_argument("--predict", type=str, help="Predict difficulty of a question text")
    parser.add_argument("--subject", type=str, default="Physics", help="Subject for prediction")
    parser.add_argument("--topic", type=str, default="Mechanics", help="Topic for prediction")
    
    args = parser.parse_args()
    
    # Default behavior: if no arguments are provided, initialize DB and train models
    if len(sys.argv) == 1:
        logger.info("No arguments provided. Executing default bootstrap sequence (Init DB + Train Models)...")
        init_db()
        run_training()
        sys.exit(0)
        
    if args.init_db:
        init_db()
        
    if args.train:
        run_training()
        
    if args.predict:
        run_prediction(args.predict, args.subject, args.topic)

if __name__ == "__main__":
    main()
