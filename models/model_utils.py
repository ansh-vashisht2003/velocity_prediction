import joblib
import numpy as np

models = joblib.load("models/trained_models.pkl")

best_model = models["Random Forest"]


def predict_velocity(inputs):

    inputs = np.array(inputs).reshape(1,-1)

    velocity = best_model.predict(inputs)

    return velocity[0]