import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

    # features used for training
    X = df[
        [
            "Calibre",
            "Projectile Dimension",
            "projectile mass",
            "powder mass",
            "Density",
            "Surface Area",
            "Volume"
        ]
    ]

    # target variable
    y = df["Actual Velocity"]

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
    results = {}

    print("\nTraining Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        score = r2_score(y_test, preds)

        trained_models[name] = model
        results[name] = score

        print(f"{name} Accuracy: {score:.4f}")

    # save models
    joblib.dump(trained_models, "models/trained_models.pkl")

    # save accuracy report
    report = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
    report.to_csv("models/model_results.csv", index=False)

    print("\nTraining Complete")
    print("Models saved to models/trained_models.pkl")
    print("Accuracy report saved to models/model_results.csv")


# -----------------------------
# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train()