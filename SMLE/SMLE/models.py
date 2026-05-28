import json
import numpy as np
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    reports = db.relationship('DetectionReport', backref='author', lazy=True)

class DetectionReport(db.Model):
    __tablename__ = 'detection_report'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    input_text = db.Column(db.Text, nullable=False)
    label_results = db.Column(db.Text, nullable=False) # JSON dictionary
    shap_scores = db.Column(db.Text, nullable=True)    # JSON list
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_processed = db.Column(db.Boolean, default=False) # Flag for Algorithm 2

class SMLEModel:
    def __init__(self):
        self.labels = ['AI_Generated', 'from_GPT4', 'Plagiarized']

    def predict(self, text):
        """Simulates Multi-Label Classification probabilities """
        probs = np.random.uniform(0.1, 0.98, size=3)
        return {self.labels[i]: round(float(probs[i]), 4) for i in range(len(self.labels))}

    def get_shap_values(self, text):
        """Simulates SHAP Explainability scores for text hotspots """
        words = text.split()
        return [{"word": w, "score": round(float(np.random.uniform(-1, 1)), 4)} for w in words]