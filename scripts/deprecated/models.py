import numpy as np
import xgboost as xgb

def logit(p): 
    return np.log(p / (1 - p + 1e-12) + 1e-12)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class WeightedLogisticModels:
    def __init__(self, models, model_weights):
        if len(models) != len(model_weights):
            raise ValueError(f"There should be the same number of models as model weights. Found {len(models)} models but {len(model_weights)} model weights.")
        self.models = models
        self.model_weights = model_weights
    
    def predict(self, df):
        weighted_logits = 0.0
        for i, m in enumerate(self.models):
            sub = df[m.feature_names_in_]
            logit_i = logit(self.models[i].predict_proba(sub)[:,1])
            weighted_logits += self.model_weights[i] * logit_i
        return sigmoid(weighted_logits)
