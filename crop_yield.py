import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("D:/Pranshu Docs/SSH/crop_yield.csv")

# Features and target
X = df.drop("Yield", axis=1)
df["Yield"] = df["Yield"].clip(upper=20)
y = df["Yield"]



# Categorical & numerical columns
cat_cols = ["Crop", "Season", "State"]
num_cols = ["Crop_Year", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]

# Preprocessing: OneHotEncode categoricals, pass through numericals
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# Define model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

import joblib

# Save the trained pipeline
joblib.dump(pipeline, "crop_yield_model.joblib")

# Later, to load it
# loaded_model = joblib.load("crop_yield_model.joblib")

# Make predictions on new data
# predictions = loaded_model.predict(new_data)
