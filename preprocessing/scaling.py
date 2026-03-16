from sklearn.preprocessing import StandardScaler

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler