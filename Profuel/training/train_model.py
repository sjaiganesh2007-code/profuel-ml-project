"""
Reproducible Training Pipeline for ProFuel ML Recommendation Engine.
Trains a Scikit-Learn suitability regression model on user-food interaction features,
evaluates metrics honestly on an 80/20 test split, and exports trained artifacts.
"""

import os
import sys
import json
import random
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import local src modules
from src.data_loader import load_food_data
from src.preprocessing import clean_food_dataset
from src.nutrition import calculate_nutrition_targets
from src.features import (
    build_user_food_features,
    compute_derived_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS
)

def generate_synthetic_user_profiles(n_samples=50):
    """Generates diverse user profiles covering various demographic and fitness goals."""
    random.seed(42)
    np.random.seed(42)

    genders = ["Male", "Female"]
    goals = ["Weight Loss", "Maintenance", "Muscle Gain"]
    diet_prefs = ["Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian"]
    activity_levels = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]

    profiles = []
    for _ in range(n_samples):
        age = int(random.randint(18, 65))
        gender = random.choice(genders)
        height = float(random.randint(150, 195))
        weight = float(random.randint(45, 105))
        goal = random.choice(goals)
        diet = random.choice(diet_prefs)
        activity = random.choice(activity_levels)
        budget = float(random.choice([100, 150, 200, 250, 300, 400, 500]))

        targets = calculate_nutrition_targets(weight, height, age, gender, activity, goal)
        targets.update({
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "fitness_goal": goal,
            "dietary_preference": diet,
            "activity_level": activity,
            "budget": budget
        })
        profiles.append(targets)
    return profiles

def calculate_derived_suitability_target(user_profile, food_df):
    """
    Computes a transparent, domain-grounded suitability index score (0 to 100)
    for model supervision based on measurable nutritional and budget alignment.
    """
    df = compute_derived_features(user_profile, food_df)

    goal_comp = df["goal_compatibility_score"].values
    macro_align = df["macro_alignment_score"].values * 100.0

    # Protein efficiency norm (0-100)
    p_per_cal = df["protein_per_calorie"].values
    p_norm = np.clip(p_per_cal / 0.15 * 100.0, 0.0, 100.0)

    # Budget suitability (penalize if cost > meal budget target)
    user_budget = float(user_profile.get("budget", 200))
    meal_budget = max(1.0, user_budget / 4.0)
    cost_ratio = df["cost_inr"].values / meal_budget
    budget_score = np.where(cost_ratio <= 1.0, 100.0, np.maximum(0.0, 100.0 - (cost_ratio - 1.0) * 80.0))

    # Diet preference match penalty (0 if mismatch, 1 if match)
    pref = str(user_profile.get("dietary_preference", "Vegetarian")).strip().lower()
    food_diets = df["diet_type"].str.lower().values

    diet_match = np.ones(len(df))
    for i, fd in enumerate(food_diets):
        if pref == "vegan" and fd != "vegan":
            diet_match[i] = 0.0
        elif pref in ["vegetarian", "veg"] and fd not in ["vegetarian", "vegan"]:
            diet_match[i] = 0.0
        elif pref in ["eggetarian", "egg"] and fd not in ["vegetarian", "vegan", "eggetarian"]:
            diet_match[i] = 0.0

    # Composite Suitability Score
    raw_suitability = (0.35 * goal_comp) + (0.30 * macro_align) + (0.20 * p_norm) + (0.15 * budget_score)
    final_target = np.clip(raw_suitability * diet_match, 0.0, 100.0)

    return final_target

def train_model():
    print("==================================================")
    print("      ProFuel Model Training Pipeline           ")
    print("==================================================")

    data_path = Path("data/indian_foods.csv")
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing at {data_path}. Run dataset generator first.")

    # 1. Load & Clean Data
    print(f"Loading food dataset from {data_path}...")
    raw_df = load_food_data(data_path)
    food_df = clean_food_dataset(raw_df)
    print(f"Loaded {len(food_df)} food items after cleaning.")

    # 2. Build User-Food Pair Dataset
    print("Generating synthetic user profiles and user-food interaction feature vectors...")
    user_profiles = generate_synthetic_user_profiles(n_samples=60)

    feature_dfs = []
    targets = []

    for user_prof in user_profiles:
        f_matrix = build_user_food_features(user_prof, food_df)
        t_values = calculate_derived_suitability_target(user_prof, food_df)
        feature_dfs.append(f_matrix)
        targets.extend(t_values)

    X = pd.concat(feature_dfs, ignore_index=True)
    y = np.array(targets, dtype=float)

    print(f"Total training dataset size: {len(X)} user-food pair records.")

    # 3. Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Train split size: {len(X_train)} | Test split size: {len(X_test)}")

    # 4. Preprocessing Pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    # 5. Fit Preprocessor
    print("Fitting preprocessor pipeline...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 6. Train Regressor Model
    print("Training HistGradientBoostingRegressor model...")
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_depth=7,
        min_samples_leaf=15,
        random_state=42
    )
    model.fit(X_train_proc, y_train)

    # 7. Evaluate Model
    y_pred_train = model.predict(X_train_proc)
    y_pred_test = model.predict(X_test_proc)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("\n--------------------------------------------------")
    print("           MODEL EVALUATION METRICS              ")
    print("--------------------------------------------------")
    print(f"Train MAE  : {train_mae:.4f} | Test MAE  : {test_mae:.4f}")
    print(f"Train RMSE : {train_rmse:.4f} | Test RMSE : {test_rmse:.4f}")
    print(f"Train R²   : {train_r2:.4f} | Test R²   : {test_r2:.4f}")
    print("--------------------------------------------------\n")

    # 8. Save Artifacts
    model_path = model_dir / "model.joblib"
    preprocessor_path = model_dir / "preprocessor.joblib"
    metadata_path = model_dir / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "dataset_name": "ProFuel Indian Food Catalog",
        "num_food_items": len(food_df),
        "total_pairs_trained": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(ALL_FEATURE_COLUMNS),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": {
            "train_mae": round(train_mae, 4),
            "test_mae": round(test_mae, 4),
            "train_rmse": round(train_rmse, 4),
            "test_rmse": round(test_rmse, 4),
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4)
        },
        "target_description": "Derived transparent food suitability index S ∈ [0, 100] based on macro alignment, goal compatibility, protein efficiency, and budget tolerance."
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")
    print(f"Metadata saved to {metadata_path}")
    print("Training complete successfully!")

if __name__ == "__main__":
    train_model()
