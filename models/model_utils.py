import joblib
import numpy as np
import pandas as pd

# load trained models
models = joblib.load("models/trained_models.pkl")

# load model accuracy results
results = pd.read_csv("models/model_results.csv")

# choose best model automatically (highest R2)
best_model_name = results.sort_values("R2", ascending=False).iloc[0]["Model"]

print("Best model selected:", best_model_name)

best_model = models[best_model_name]


def predict_velocity(inputs):

    inputs = np.array(inputs).reshape(1, -1)

    velocity = best_model.predict(inputs)

    return velocity[0]