import shap
import numpy as np

class SMLEExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Defining a prediction function that extracts probabilities for SHAP
        def predict_func(texts):
            return np.array([list(model.predict(t).values()) for t in texts])
            
        self.explainer = shap.Explainer(predict_func, self.tokenizer)

    def get_explanation(self, text):
        """Calculates SHAP values to identify key influential words """
        shap_values = self.explainer([text])
        return shap_values