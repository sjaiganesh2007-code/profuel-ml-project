"""
Feature Engineering Engine for ProFuel.
Constructs tabular user-food feature vectors combining user metadata, food nutritional attributes,
and derived domain features for Machine Learning model prediction.
"""

import pandas as pd
import numpy as np

# Feature Column Names
NUMERIC_FEATURES = [
    "user_age", "user_height", "user_weight", "user_bmi",
    "target_calories", "target_protein", "target_carbs", "target_fat", "user_budget",
    "food_calories", "food_protein", "food_carbs", "food_fat", "food_fiber", "food_sugar", "food_cost",
    "calorie_ratio", "protein_per_calorie", "protein_per_rupee", "calorie_density",
    "budget_compatibility", "goal_compatibility_score", "macro_alignment_score"
]

CATEGORICAL_FEATURES = [
    "user_gender", "user_goal", "user_diet_preference", "user_activity",
    "meal_category", "food_category", "diet_type"
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def compute_derived_features(user_profile, food_df):
    """
    Computes derived interaction features between user profile and candidate food items.
    """
    df = food_df.copy()
    
    # User Profile Extraction
    target_cal = float(user_profile.get("target_calories", 2000))
    target_prot = float(user_profile.get("target_protein", 100))
    target_carbs = float(user_profile.get("target_carbs", 250))
    target_fat = float(user_profile.get("target_fat", 60))
    user_budget = float(user_profile.get("budget", 200))
    goal = str(user_profile.get("fitness_goal", "Maintenance")).strip().title()

    # Meal target share (approx 1/4th of daily target per meal slot)
    meal_cal_target = max(1.0, target_cal / 4.0)
    meal_budget_target = max(1.0, user_budget / 4.0)

    # 1. Calorie Ratio & Density
    df["calorie_ratio"] = df["calories"] / meal_cal_target
    df["calorie_density"] = df["calories"] / np.maximum(1.0, df["serving_weight_g"])

    # 2. Protein Efficiency
    df["protein_per_calorie"] = df["protein"] / np.maximum(1.0, df["calories"])
    df["protein_per_rupee"] = df["protein"] / np.maximum(1.0, df["cost_inr"])

    # 3. Budget Compatibility
    df["budget_compatibility"] = df["cost_inr"] / meal_budget_target

    # 4. Macro Alignment Score
    # Target macro ratios
    tot_target_g = target_prot + target_carbs + target_fat
    p_ratio_target = target_prot / max(1.0, tot_target_g)
    c_ratio_target = target_carbs / max(1.0, tot_target_g)
    f_ratio_target = target_fat / max(1.0, tot_target_g)

    tot_food_g = df["protein"] + df["carbohydrates"] + df["fat"]
    p_ratio_food = df["protein"] / np.maximum(1.0, tot_food_g)
    c_ratio_food = df["carbohydrates"] / np.maximum(1.0, tot_food_g)
    f_ratio_food = df["fat"] / np.maximum(1.0, tot_food_g)

    diff = np.abs(p_ratio_food - p_ratio_target) + np.abs(c_ratio_food - c_ratio_target) + np.abs(f_ratio_food - f_ratio_target)
    df["macro_alignment_score"] = np.clip(1.0 - (diff / 2.0), 0.0, 1.0)

    # 5. Goal Compatibility Score
    if goal == "Weight Loss":
        # Favors high protein/calorie ratio and low calorie density
        score = (df["protein_per_calorie"] * 150.0) + (df["fiber"] * 2.0) - (df["calorie_density"] * 10.0)
    elif goal == "Muscle Gain":
        # Favors total protein content and moderate/high protein per rupee
        score = (df["protein"] * 2.5) + (df["protein_per_rupee"] * 15.0) + (df["calories"] * 0.05)
    else:  # Maintenance
        # Favors high macro alignment, fiber, and moderate cost
        score = (df["macro_alignment_score"] * 50.0) + (df["fiber"] * 3.0) + (df["protein_per_calorie"] * 80.0)

    df["goal_compatibility_score"] = np.clip(score, 0.0, 100.0)
    return df

def build_user_food_features(user_profile, food_df):
    """
    Constructs complete feature matrix for ML model prediction.
    """
    df = compute_derived_features(user_profile, food_df)

    # Assign User Features as columns
    features_df = pd.DataFrame(index=df.index)

    features_df["user_age"] = float(user_profile.get("age", 25))
    features_df["user_height"] = float(user_profile.get("height", 170))
    features_df["user_weight"] = float(user_profile.get("weight", 70))
    features_df["user_bmi"] = float(user_profile.get("bmi", 24.2))
    features_df["target_calories"] = float(user_profile.get("target_calories", 2000))
    features_df["target_protein"] = float(user_profile.get("target_protein", 100))
    features_df["target_carbs"] = float(user_profile.get("target_carbs", 250))
    features_df["target_fat"] = float(user_profile.get("target_fat", 60))
    features_df["user_budget"] = float(user_profile.get("budget", 200))

    features_df["food_calories"] = df["calories"].values
    features_df["food_protein"] = df["protein"].values
    features_df["food_carbs"] = df["carbohydrates"].values
    features_df["food_fat"] = df["fat"].values
    features_df["food_fiber"] = df["fiber"].values
    features_df["food_sugar"] = df["sugar"].values
    features_df["food_cost"] = df["cost_inr"].values

    features_df["calorie_ratio"] = df["calorie_ratio"].values
    features_df["protein_per_calorie"] = df["protein_per_calorie"].values
    features_df["protein_per_rupee"] = df["protein_per_rupee"].values
    features_df["calorie_density"] = df["calorie_density"].values
    features_df["budget_compatibility"] = df["budget_compatibility"].values
    features_df["goal_compatibility_score"] = df["goal_compatibility_score"].values
    features_df["macro_alignment_score"] = df["macro_alignment_score"].values

    features_df["user_gender"] = str(user_profile.get("gender", "Male")).strip().title()
    features_df["user_goal"] = str(user_profile.get("fitness_goal", "Maintenance")).strip().title()
    features_df["user_diet_preference"] = str(user_profile.get("dietary_preference", "Vegetarian")).strip().title()
    features_df["user_activity"] = str(user_profile.get("activity_level", "Moderately Active")).strip().title()

    features_df["meal_category"] = df["meal_category"].values
    features_df["food_category"] = df["food_category"].values
    features_df["diet_type"] = df["diet_type"].values

    return features_df[ALL_FEATURE_COLUMNS]
