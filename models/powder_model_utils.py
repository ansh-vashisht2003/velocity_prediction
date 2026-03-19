import joblib
import numpy as np
import pandas as pd


# LOAD TRAINED MODELS
models = joblib.load("models/powder_trained_models.pkl")

# LOAD MODEL RESULTS
results = pd.read_csv("models/powder_model_results.csv")


# SELECT BEST MODEL
best_model_name = results.sort_values("R2",ascending=False).iloc[0]["Model"]

print("Best powder model:",best_model_name)

best_model = models[best_model_name]


# -------------------------
# PREDICT POWDER MASS
# -------------------------

def predict_powder(inputs):

    inputs = np.array(inputs).reshape(1,-1)

    powder = best_model.predict(inputs)

    return powder[0]