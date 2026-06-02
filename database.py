import sqlite3
import pandas as pd
from datetime import datetime
from src.utils import logger, DATA_DIR

DB_PATH = DATA_DIR / "project.db"

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema."""
    logger.info("Initializing SQLite Database...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Prediction History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            predicted_difficulty TEXT NOT NULL,
            confidence REAL NOT NULL,
            actual_difficulty TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Custom Labeled Questions Table (for growing training set)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS custom_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_text TEXT NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def log_prediction(question_text: str, subject: str, topic: str, predicted_diff: str, confidence: float) -> int:
    """Logs a single prediction request into the history database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_history (question_text, subject, topic, predicted_difficulty, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (question_text, subject, topic, predicted_diff, confidence))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def submit_feedback(prediction_id: int, actual_difficulty: str):
    """Updates the actual difficulty of a logged prediction (user feedback)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE prediction_history
        SET actual_difficulty = ?
        WHERE id = ?
    """, (actual_difficulty, prediction_id))
    conn.commit()
    conn.close()
    logger.info(f"Feedback submitted for prediction ID: {prediction_id} -> Actual: {actual_difficulty}")

def add_custom_question(question_text: str, subject: str, topic: str, difficulty: str):
    """Adds a new training question to the custom database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO custom_questions (question_text, subject, topic, difficulty)
        VALUES (?, ?, ?, ?)
    """, (question_text, subject, topic, difficulty))
    conn.commit()
    conn.close()
    logger.info(f"Custom question added to DB: {subject} - {topic} - {difficulty}")

def get_prediction_history(limit: int = 100) -> pd.DataFrame:
    """Fetches prediction history as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(f"""
        SELECT id, question_text, subject, topic, predicted_difficulty, confidence, actual_difficulty, timestamp 
        FROM prediction_history 
        ORDER BY timestamp DESC 
        LIMIT {limit}
    """, conn)
    conn.close()
    return df

def get_custom_questions() -> pd.DataFrame:
    """Fetches custom questions as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT question_text, subject, topic, difficulty 
        FROM custom_questions
    """, conn)
    conn.close()
    return df

def get_combined_dataset() -> pd.DataFrame:
    """Combines baseline CSV questions and custom SQLite questions for retraining."""
    csv_path = DATA_DIR / "questions.csv"
    
    # Load baseline
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path)
    else:
        df_csv = pd.DataFrame(columns=["Question_Text", "Subject", "Topic", "Difficulty"])
        
    # Load custom
    df_db = get_custom_questions()
    if not df_db.empty:
        df_db.columns = ["Question_Text", "Subject", "Topic", "Difficulty"]
        df_combined = pd.concat([df_csv, df_db], ignore_index=True)
    else:
        df_combined = df_csv
        
    return df_combined
