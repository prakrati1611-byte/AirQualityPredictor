import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("AirQualityUCI.csv", sep=";")

# Drop empty columns
df = df.dropna(axis=1, how="all")

# Replace -200 with NaN and drop missing rows
df = df.replace(-200, pd.NA)
df = df.dropna()

# Convert numeric columns with commas to floats
for col in ["CO(GT)", "C6H6(GT)", "T", "RH", "AH"]:
    df[col] = df[col].str.replace(",", ".").astype(float)

# Features (pollutants + weather)
X = df[["CO(GT)", "PT08.S5(O3)", "T", "RH", "AH"]]

# Target pollutant (NO2 concentration)
y = df["NO2(GT)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "aqi_model.pkl")
print("Model saved as aqi_model.pkl")
