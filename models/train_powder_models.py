import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb


# -----------------------------
# LOAD AND CLEAN DATA
# -----------------------------
def load_data():

    df = pd.read_csv("data/fake_gas_gun_dataset.csv")

    # remove spaces from column names
    df.columns = df.columns.str.strip()

    # convert calibre "40 mm" → 40
    df["Calibre"] = df["Calibre"].str.replace(" mm", "").astype(float)

    # encode categorical columns
    df["Projectile Type"] = df["Projectile Type"].astype("category").cat.codes
    df["Shape"] = df["Shape"].astype("category").cat.codes
    df["Material"] = df["Material"].astype("category").cat.codes
    df["s_type"] = df["s_type"].astype("category").cat.codes

    # -----------------------------
    # HANDLE MISSING VALUES (Standard ML Practice)
    # -----------------------------
    df = df.fillna(df.mean(numeric_only=True))

    # features used for training
    X = df[
        [
            "Calibre",
            "Projectile Type",
            "Projectile Dimension",
            "dai",
            "projectile mass",
            "total mass with sabbot",
            "Petal burst pressure",
            "Actual Velocity",
            "Shape",
            "s_type",
            "Breadth",
            "Height",
            "Material",
            "c_drag",
            "Surface Area",
            "Volume",
            "SA/vol",
            "Density",
            "Moment of inerta",
            "cd",
            "sabo length"
        ]
    ]

    # target variable
    y = df["powder mass"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# MACHINE LEARNING MODELS
# -----------------------------
def get_models():

    models = {

        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Huber": HuberRegressor(),
        "Bayesian Ridge": BayesianRidge(),

        "Polynomial Regression":
            Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("linear", LinearRegression())
            ]),

        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=200),
        "Extra Trees": ExtraTreesRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "MLP Regressor": MLPRegressor(max_iter=2000),

        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror"),
        "LightGBM": lgb.LGBMRegressor()
    }

    return models


# -----------------------------
# TRAIN ALL MODELS
# -----------------------------
def train():

    X_train, X_test, y_train, y_test = load_data()

    models = get_models()

    trained_models = {}
    results = []

    print("\nTraining Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)

        trained_models[name] = model

        results.append({
            "Model": name,
            "R2": r2,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae
        })

        print(f"{name} | R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    # save trained models
    joblib.dump(trained_models, "models/powder_trained_models.pkl")

    # save accuracy report
    report = pd.DataFrame(results)
    report.to_csv("models/powder_model_results.csv", index=False)

    print("\nTraining Complete")
    print("Models saved to models/powder_trained_models.pkl")
    print("Accuracy report saved to models/powder_model_results.csv")


# -----------------------------
# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train()