import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.utils import logger

class TextPreprocessor:
    """Class to preprocess raw question text for NLP models."""
    
    def __init__(self):
        self._download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        # Custom mathematical characters to keep if needed, but remove normal punctuation
        self.stop_words = set(stopwords.words('english'))
        
        # Remove important physics/chemistry/math stopwords that might be in standard stopwords list
        # but are actually meaningful keywords for us
        meaningful_words = {'not', 'no', 'above', 'below', 'between', 'against'}
        self.stop_words = self.stop_words - meaningful_words

    def _download_nltk_resources(self):
        """Downloads required NLTK resources safely."""
        resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                if resource == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                elif resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                else:
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

    def clean_text(self, text: str) -> str:
        """Removes basic noise, website URLs, and normalizes spacing."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenizes, filters punctuation and stop words, lemmatizes, and rejoins."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        
        # Tokenize
        tokens = word_tokenize(cleaned)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Strip punctuation from within token, keep alphanumeric
            clean_token = token.translate(str.maketrans('', '', string.punctuation))
            if clean_token and clean_token not in self.stop_words:
                # Lemmatize
                lemma = self.lemmatizer.lemmatize(clean_token)
                processed_tokens.append(lemma)
                
        return " ".join(processed_tokens)
