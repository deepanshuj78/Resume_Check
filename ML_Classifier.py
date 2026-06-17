from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from Config import DOMAIN_TRAINING_DATA, DOMAIN_LABELS
class DomainClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = LogisticRegression()
        self._bootstrap_classifier()
    def _bootstrap_classifier(self):
        """Pre-computes and saves model weights on system initialization."""
        X_train = self.vectorizer.fit_transform(DOMAIN_TRAINING_DATA)
        self.model.fit(X_train, DOMAIN_LABELS)
    def predict_profile_tag(self, clean_text):
        """Instantly maps candidate text fields directly to categorical strings."""
        if not clean_text.strip():
            return "General Profile"
        X_target = self.vectorizer.transform([clean_text])
        return str(self.model.predict(X_target)[0])
classifier_pipeline = DomainClassifier()