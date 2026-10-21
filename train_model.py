import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# Read the dataset
df = pd.read_csv("Updated Quality of Life Data.csv")

os.makedirs("models", exist_ok=True)

# ---------------- REGRESSION MODEL ----------------
# Predict age_at_death (a number)
regression_features = [
    "gender", "occupation_type", "avg_work_hours_per_day",
    "avg_rest_hours_per_day", "avg_sleep_hours_per_day",
    "avg_exercise_hours_per_day"
]
X_reg = df[regression_features]
y_reg = df["age_at_death"]

reg_cat = ["gender", "occupation_type"]
reg_num = [c for c in regression_features if c not in reg_cat]

reg_preprocess = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), reg_cat),
    ("numbers", SimpleImputer(strategy="mean"), reg_num)
])

reg_model = Pipeline([
    ("preprocess", reg_preprocess),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)
print("Regression R²:", round(r2_score(y_test, reg_predictions), 3))
joblib.dump(reg_model, "models/regression_model.joblib")

# ---------------- CLASSIFICATION MODEL ----------------
# Predict occupation_type (a category)
class_features = [
    "gender", "avg_work_hours_per_day", "avg_rest_hours_per_day",
    "avg_sleep_hours_per_day", "avg_exercise_hours_per_day"
]
X_class = df[class_features]
y_class = df["occupation_type"]

class_preprocess = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
    ("numbers", SimpleImputer(strategy="mean"),
     [c for c in class_features if c != "gender"])
])

class_model = Pipeline([
    ("preprocess", class_preprocess),
    ("model", DecisionTreeClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)
class_model.fit(X_train, y_train)
class_predictions = class_model.predict(X_test)
print("Classification accuracy:", round(accuracy_score(y_test, class_predictions), 3))
joblib.dump(class_model, "models/classification_model.joblib")

print("Both models saved in the models folder.")
