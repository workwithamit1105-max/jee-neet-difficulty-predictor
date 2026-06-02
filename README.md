# JEE/NEET Question Difficulty Predictor

An AI-powered machine learning web application that classifies JEE (engineering) and NEET (medical) exam questions into **Easy**, **Medium**, or **Hard** categories based on textual content, mathematical complexity, and subject-specific vocabulary.

The project implements a complete, modular, and production-ready machine learning pipeline featuring data preprocessing, statistical feature engineering, model training/comparison across multiple algorithms, and a sleek, interactive Streamlit frontend backed by an SQLite database.

---

## 🚀 Key Features

1. **Robust Feature Engineering**: Computes character/word counts, extracts mathematical symbol frequencies, counts subject-specific keywords (Physics, Chemistry, Mathematics, Biology), and computes TF-IDF representations.
2. **Model Comparison Suite**: Trains and compares **Logistic Regression**, **Random Forest**, **Support Vector Machines (SVM)**, and **XGBoost**, automatically selecting the best-performing model.
3. **Interactive Streamlit Dashboard**:
   - **Project Overview**: Live system metrics, architecture flowchart, and project roadmap.
   - **Dataset Statistics**: Exploratory data analysis charts (distribution of subjects, topics, question lengths, and a word-frequency bar chart).
   - **Model Training**: Evaluation metrics table, Confusion Matrix heatmap, and Feature Importance bar chart. Includes a button to trigger model retraining on demand.
   - **Difficulty Prediction**: Interactive text area for difficulty predictions with real-time class probability charts.
4. **Active Feedback Loop (SQLite)**: Logs all predicted questions, confidence scores, and timestamps in an SQLite database. Allows users to submit feedback (actual difficulty) and add new labeled questions to expand the training dataset.

---

## 📁 Project Structure

```text
project/
│
├── data/                    # SQLite databases and baseline CSV files
│   ├── questions.csv        # Baseline generated question dataset
│   └── project.db           # SQLite database for history logging & feedback
│
├── notebooks/               # Research and exploratory notebooks
│
├── models/                  # Serialized classifiers, preprocessors, and metrics
│   ├── best_model.joblib    # Deployed best model (XGBoost)
│   ├── preprocessor.joblib  # Trained NLTK text preprocessor
│   ├── feature_engineer.joblib # Fitted feature engineer & encoders
│   └── training_metrics.json # Saved model performance scores
│
├── app/                     # Streamlit frontend source code
│   └── app.py               # 4-page Streamlit dashboard
│
├── src/                     # Core python package modules
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic dataset generator
│   ├── data_preprocessing.py# Clean text, tokenization, lemmatization
│   ├── feature_engineering.py# Math symbols, keyword extractor, and TF-IDF
│   ├── model_training.py    # Train loop, metrics exporter, predictor
│   ├── database.py          # SQLite connection and query operations
│   └── utils.py             # Global logging and directory creators
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── main.py                  # Main bootstrap and CLI script
```

---

## 📈 Model Performance Summary

| Model Name | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **91.18%** | **91.31%** | **91.18%** | **91.21%** | 🏆 **Selected** |
| **Random Forest** | 88.24% | 88.35% | 88.24% | 87.93% | Available |
| **Logistic Regression** | 86.27% | 86.81% | 86.27% | 86.30% | Available |
| **SVM** | 84.31% | 84.58% | 84.31% | 84.14% | Available |

---

## 🔧 Installation and Running Guide

### Prerequisites
- Python 3.8+ (Tested on Python 3.10 and 3.14)
- Git (optional)

### Setup Instructions

1. **Clone or copy the project files** to your local directory.
2. **Navigate to the project root directory**:
   ```bash
   cd project
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Bootstrap & Train Models** (Initializes the SQLite tables, generates the dataset, runs the feature engineering pipeline, trains models, and saves the best model):
   ```bash
   python main.py
   ```
5. **Start the Streamlit Application**:
   ```bash
   streamlit run app/app.py
   ```
6. **Open the browser** at the default Streamlit URL: `http://localhost:8501`.

---

## 💼 Resume-Ready Bullet Points

For machine learning or full-stack AI roles, you can describe your experience building this project with these impactful bullet points:

* **Engineered a production-ready Question Difficulty Predictor** using Python, Scikit-learn, XGBoost, and Streamlit, achieving a **91.2% F1-score** across 3 class levels (Easy, Medium, Hard).
* **Designed a custom natural language pipeline** using NLTK to perform lowercasing, tokenization, stopword removal, and word lemmatization, combined with statistical features counting mathematical symbol densities and subject-specific vocabulary.
* **Implemented an active learning database framework** using SQLite to log live prediction queries, capture student/educator feedback on prediction correctness, and dynamically expand the training set for on-demand retraining.
* **Developed a modular, OOP-compliant software architecture** with a separation of concerns across text preprocessing, feature engineering, model training, and web serving, complete with robust logging and validation.
* **Built an interactive analytics dashboard** in Streamlit displaying class balance, word frequency metrics, model confusion matrices, and dynamic feature importances using Seaborn.

---

## 🔮 Future AI Roadmap

- **LLM Explanation Generation**: Add a feature leveraging Gemini 1.5 Flash to automatically generate step-by-step solutions and theoretical explanations for any question predicted as "Hard".
- **NCERT RAG System**: Integrate ChromaDB or Pinecone vector stores containing the NCERT textbook curriculum to retrieve exact chapter and page numbers related to the input question.
- **Personalized Recommendations**: Provide curated study schedules, relevant practice sheets, and video lecture recommendations based on a user's logged question history.
