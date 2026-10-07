"""
Data Loading Layer for ProFuel application.
Provides efficient loading, column mapping, normalization, and validation of Indian Food Datasets.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "food_id", "food_name", "meal_category", "diet_type",
    "calories", "protein", "carbohydrates", "fat", "cost_inr"
]

DEFAULT_COLUMN_MAPPING = {
    "name": "food_name",
    "item_name": "food_name",
    "meal": "meal_category",
    "category": "meal_category",
    "diet": "diet_type",
    "type": "diet_type",
    "cal": "calories",
    "energy": "calories",
    "prot": "protein",
    "carbs": "carbohydrates",
    "carb": "carbohydrates",
    "fats": "fat",
    "price": "cost_inr",
    "cost": "cost_inr",
    "price_rs": "cost_inr"
}

def load_food_data(file_path="data/indian_foods.csv", column_mapping=None):
    """
    Loads food dataset from Parquet or CSV file with validation and normalization.
    """
    path = Path(file_path)
    
    # Check Parquet counterpart first for high efficiency
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif path.exists():
        df = pd.read_csv(path)
    else:
        raise FileNotFoundError(f"Food dataset file not found at: {path}")

    # Column mapping for custom schemas
    mapping = DEFAULT_COLUMN_MAPPING.copy()
    if column_mapping:
        mapping.update(column_mapping)
        
    df = df.rename(columns=mapping)

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Fill optional missing metadata if absent
    if "food_category" not in df.columns:
        df["food_category"] = "General"
    if "region" not in df.columns:
        df["region"] = "Pan-Indian"
    if "serving_size" not in df.columns:
        df["serving_size"] = "1 serving"
    if "serving_weight_g" not in df.columns:
        df["serving_weight_g"] = 200.0
    if "fiber" not in df.columns:
        df["fiber"] = 3.0
    if "sugar" not in df.columns:
        df["sugar"] = 2.0
    if "glycemic_index" not in df.columns:
        df["glycemic_index"] = "Medium"

    # Clean & Convert data types
    numeric_cols = ["calories", "protein", "carbohydrates", "fat", "fiber", "sugar", "cost_inr", "serving_weight_g"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Sanity filtering for invalid/impossible nutritional records
    df = df[(df["calories"] > 0) & (df["cost_inr"] > 0)].copy()

    # Clean text values
    df["meal_category"] = df["meal_category"].astype(str).str.strip().str.title()
    df["diet_type"] = df["diet_type"].astype(str).str.strip().str.title()
    df["food_name"] = df["food_name"].astype(str).str.strip()

    # Drop exact duplicate items
    df = df.drop_duplicates(subset=["food_name", "meal_category"]).reset_index(drop=True)

    return df

def filter_by_diet(df, diet_preference):
    """
    Strictly filters food items by dietary preference.
    Preferences:
    - Vegetarian: Veg or Vegan
    - Vegan: Vegan only
    - Eggetarian: Veg, Vegan, or Eggetarian
    - Non-Vegetarian: All items allowed
    """
    pref = str(diet_preference).strip().lower()
    
    if pref == "vegan":
        return df[df["diet_type"].str.lower() == "vegan"].copy()
    elif pref in ["vegetarian", "veg"]:
        return df[df["diet_type"].str.lower().isin(["vegetarian", "vegan"])].copy()
    elif pref in ["eggetarian", "egg"]:
        return df[df["diet_type"].str.lower().isin(["vegetarian", "vegan", "eggetarian"])].copy()
    else:  # Non-Vegetarian / All
        return df.copy()

def filter_by_meal_category(df, meal_category):
    """
    Filters food dataset by meal category: Breakfast, Lunch, Snack, Dinner.
    """
    cat = str(meal_category).strip().title()
    return df[df["meal_category"] == cat].copy()
