"""
ML Recommender Module for ProFuel.
Loads trained ML models, predicts suitability scores for candidate food items,
and ranks food candidates based on user-food interaction features.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.features import build_user_food_features
from src.data_loader import filter_by_diet, filter_by_meal_category

def load_trained_artifacts(model_dir="model"):
    """
    Loads trained scikit-learn model, preprocessor pipeline, and metadata JSON.
    """
    dir_path = Path(model_dir)
    model_path = dir_path / "model.joblib"
    prep_path = dir_path / "preprocessor.joblib"
    meta_path = dir_path / "metadata.json"

    if not (model_path.exists() and prep_path.exists()):
        raise FileNotFoundError(
            f"Trained model artifacts not found in '{model_dir}'. Run training/train_model.py first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, preprocessor, metadata

def predict_food_scores(user_profile, food_df, model, preprocessor):
    """
    Generates feature matrix, runs model inference, and attaches predicted suitability score (0-100).
    """
    if food_df.empty:
        return food_df.copy()

    df = food_df.copy().reset_index(drop=True)
    X_feat = build_user_food_features(user_profile, df)

    X_proc = preprocessor.transform(X_feat)
    raw_scores = model.predict(X_proc)

    # Scale / clip predicted suitability scores strictly between 0 and 100
    df["ml_suitability_score"] = np.clip(raw_scores, 0.0, 100.0)
    df["ml_suitability_score"] = df["ml_suitability_score"].round(1)

    # Sort descending by ML suitability score
    ranked_df = df.sort_values(by="ml_suitability_score", ascending=False).reset_index(drop=True)
    return ranked_df

def rank_candidates(user_profile, food_df, model, preprocessor, meal_category=None):
    """
    Filters candidates by dietary preference and optional meal category, then ranks using ML model.
    """
    diet_pref = user_profile.get("dietary_preference", "Vegetarian")
    filtered = filter_by_diet(food_df, diet_pref)

    if meal_category:
        filtered = filter_by_meal_category(filtered, meal_category)

    if filtered.empty:
        return filtered

    return predict_food_scores(user_profile, filtered, model, preprocessor)
