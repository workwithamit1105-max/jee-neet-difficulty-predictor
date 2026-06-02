import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from src.utils import logger

class FeatureEngineer:
    """Class to extract features from text and metadata, and prepare feature matrices."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        
        # Mathematical symbols to count
        self.math_patterns = [
            r'\+', r'-', r'\*', r'/', r'\^', r'_', r'=', r'<', r'>', r'\\', 
            r'sqrt', r'int', r'd/d', r'lim', r'theta', r'pi', r'sin', r'cos', 
            r'tan', r'log', r'\[', r'\]', r'\{', r'\}', r'\d+'
        ]
        
        # Subject-specific keywords
        self.keywords = {
            'physics': ['force', 'velocity', 'acceleration', 'mass', 'friction', 'spring', 'energy', 
                        'gravity', 'torque', 'inertia', 'pendulum', 'electric', 'charge', 'potential', 
                        'capacitor', 'magnetic', 'current', 'inductance', 'thermodynamics', 'entropy', 'gas'],
            'chemistry': ['organic', 'isomerism', 'carbon', 'hybridization', 'reagent', 'alkene', 'reaction', 
                          'product', 'benzene', 'aniline', 'stereochemical', 'synthesis', 'complex', 'diborane', 
                          'molarity', 'ph', 'equilibrium', 'kinetics', 'crystal', 'splitting'],
            'math': ['derivative', 'integral', 'limit', 'differential', 'equation', 'bounded', 'quadratic', 
                     'progression', 'binomial', 'matrix', 'determinant', 'hyperbola', 'parabola', 'ellipse', 
                     'tangent', 'locus'],
            'biology': ['genetics', 'genotype', 'phenotype', 'dna', 'transcription', 'translation', 'operon', 
                        'replication', 'hybridization', 'ecosystem', 'biodiversity', 'succession', 'biomagnification', 
                        'physiology', 'kidney', 'hormone', 'nephron', 'heart', 'cardiac']
        }

    def count_math_symbols(self, text: str) -> int:
        """Counts mathematical symbols and numbers in text."""
        if not isinstance(text, str):
            return 0
        count = 0
        for pattern in self.math_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def count_keywords(self, text: str, category: str) -> int:
        """Counts occurrences of specific keywords in text."""
        if not isinstance(text, str):
            return 0
        count = 0
        words = self.keywords.get(category, [])
        for word in words:
            # Using word boundaries to match whole words
            count += len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))
        return count

    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts dense numerical features (length, symbol counts, keyword counts)."""
        features_list = []
        
        for idx, row in df.iterrows():
            text = row['Question_Text']
            
            char_len = len(str(text))
            word_len = len(str(text).split())
            math_syms = self.count_math_symbols(text)
            
            phy_kw = self.count_keywords(text, 'physics')
            chem_kw = self.count_keywords(text, 'chemistry')
            math_kw = self.count_keywords(text, 'math')
            bio_kw = self.count_keywords(text, 'biology')
            
            # Combine all numerical features
            row_features = [
                char_len,
                word_len,
                math_syms,
                phy_kw,
                chem_kw,
                math_kw,
                bio_kw
            ]
            features_list.append(row_features)
            
        return np.array(features_list)

    def fit(self, df: pd.DataFrame, text_col: str = 'Clean_Question_Text'):
        """Fits the TF-IDF vectorizer and One-Hot encoder on the training data."""
        logger.info("Fitting TF-IDF and One-Hot Encoders...")
        
        # Fit TF-IDF on preprocessed text
        self.tfidf_vectorizer.fit(df[text_col].fillna(''))
        
        # Fit One-Hot on Subject and Topic
        self.one_hot_encoder.fit(df[['Subject', 'Topic']])
        
        return self

    def transform(self, df: pd.DataFrame, text_col: str = 'Clean_Question_Text') -> csr_matrix:
        """Transforms data into a single sparse matrix containing all features."""
        # 1. TF-IDF features
        tfidf_feats = self.tfidf_vectorizer.transform(df[text_col].fillna(''))
        
        # 2. Categorical features (One-Hot)
        cat_feats = self.one_hot_encoder.transform(df[['Subject', 'Topic']])
        
        # 3. Dense numerical features
        num_feats = self.extract_numerical_features(df)
        # Normalize numerical features using standard scaling or pass as-is (we convert to sparse CSR)
        # For text models, passing them as sparse is fine.
        num_feats_sparse = csr_matrix(num_feats)
        
        # Combine all features horizontally
        combined_features = hstack([tfidf_feats, cat_feats, num_feats_sparse]).tocsr()
        
        return combined_features

    def get_feature_names(self) -> list:
        """Returns names of all features in the final matrix."""
        tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        cat_names = [f"cat_{name}" for name in self.one_hot_encoder.get_feature_names_out()]
        num_names = [
            "char_count", "word_count", "math_symbol_count",
            "physics_keyword_count", "chemistry_keyword_count",
            "math_keyword_count", "biology_keyword_count"
        ]
        return tfidf_names + cat_names + num_names
