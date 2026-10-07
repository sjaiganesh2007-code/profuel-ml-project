"""
Preprocessing and Data Validation Module for ProFuel.
Contains cleaning utilities, feature scalers, and outlier handling.
"""

import pandas as pd
import numpy as np

def clean_food_dataset(df):
    """
    Performs data cleaning, removing extreme outliers or invalid macro combinations.
    """
    cleaned = df.copy()
    
    # Protein + Carbs + Fat cannot exceed total serving weight (with small margin)
    macro_sum = cleaned["protein"] + cleaned["carbohydrates"] + cleaned["fat"]
    mask = macro_sum <= (cleaned["serving_weight_g"] * 1.05 + 5.0)
    cleaned = cleaned[mask].copy()

    # Calories must be physically reasonable relative to macros
    # Calorie estimate = 4*P + 4*C + 9*F
    estimated_cals = (4.0 * cleaned["protein"]) + (4.0 * cleaned["carbohydrates"]) + (9.0 * cleaned["fat"])
    # Allow reasonable variation due to fiber and rounding
    cal_diff_ratio = np.abs(cleaned["calories"] - estimated_cals) / (cleaned["calories"] + 1.0)
    cleaned = cleaned[cal_diff_ratio <= 0.6].copy()

    return cleaned.reset_index(drop=True)

def normalize_text_features(df):
    """
    Standardizes categorical string features.
    """
    processed = df.copy()
    for col in ["meal_category", "diet_type", "food_category", "region"]:
        if col in processed.columns:
            processed[col] = processed[col].astype(str).str.strip().str.title()
    return processed
