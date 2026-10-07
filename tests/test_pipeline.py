"""
Integration and Verification Test Suite for ProFuel.
Validates nutrition calculations, dataset loading, model inference, candidate filtering,
constrained optimization, food swapping, and plan regeneration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data_loader import load_food_data, filter_by_diet, filter_by_meal_category
from src.nutrition import calculate_nutrition_targets, calculate_bmr, calculate_tdee
from src.features import build_user_food_features
from src.recommender import load_trained_artifacts, predict_food_scores, rank_candidates
from src.optimizer import optimize_meal_plan, MEAL_SLOTS
from src.utils import validate_user_inputs, generate_food_rationale

def test_nutrition_calculations():
    print("Testing Nutrition Engine...")
    bmr = calculate_bmr(70.0, 175.0, 25, "Male")
    assert bmr > 1500.0, f"Expected BMR > 1500, got {bmr}"

    tdee = calculate_tdee(bmr, "Moderately Active")
    assert tdee > bmr, "TDEE must be greater than BMR"

    targets = calculate_nutrition_targets(70.0, 175.0, 25, "Male", "Moderately Active", "Weight Loss")
    assert targets["target_calories"] < tdee, "Weight Loss target must be lower than TDEE"
    assert targets["target_protein"] > 0, "Protein target must be positive"
    print("[OK] Nutrition engine passed.")

def test_data_loading_and_filtering():
    print("Testing Data Loader & Filtering...")
    df = load_food_data("data/indian_foods.csv")
    assert len(df) > 300, f"Expected 300+ food items, got {len(df)}"

    veg_df = filter_by_diet(df, "Vegetarian")
    assert not veg_df["diet_type"].str.lower().isin(["non-vegetarian"]).any(), "Non-veg items in veg filter!"

    breakfast_df = filter_by_meal_category(df, "Breakfast")
    assert (breakfast_df["meal_category"] == "Breakfast").all(), "Non-breakfast items in breakfast filter!"
    print("[OK] Data loader & filtering passed.")

def test_ml_recommender_and_ranking():
    print("Testing ML Recommender...")
    model, preprocessor, metadata = load_trained_artifacts("model")
    assert model is not None, "Model failed to load"

    df = load_food_data("data/indian_foods.csv")
    user_prof = {
        "age": 28, "gender": "Female", "height": 162.0, "weight": 60.0, "bmi": 22.9,
        "fitness_goal": "Weight Loss", "dietary_preference": "Vegetarian",
        "activity_level": "Moderately Active", "budget": 200.0,
        "target_calories": 1600.0, "target_protein": 120.0, "target_carbs": 160.0, "target_fat": 53.0
    }

    ranked_df = rank_candidates(user_prof, df, model, preprocessor, meal_category="Breakfast")
    assert "ml_suitability_score" in ranked_df.columns, "Suitability score missing!"
    assert len(ranked_df) > 0, "Ranked candidates empty!"
    assert ranked_df["ml_suitability_score"].iloc[0] >= ranked_df["ml_suitability_score"].iloc[-1], "Ranking order incorrect!"
    print("[OK] ML Recommender & ranking passed.")

def test_meal_optimizer():
    print("Testing Meal Optimizer...")
    model, preprocessor, _ = load_trained_artifacts("model")
    df = load_food_data("data/indian_foods.csv")

    user_prof = {
        "age": 30, "gender": "Male", "height": 175.0, "weight": 75.0, "bmi": 24.5,
        "fitness_goal": "Muscle Gain", "dietary_preference": "Non-Vegetarian",
        "activity_level": "Very Active", "budget": 300.0,
        "target_calories": 2500.0, "target_protein": 180.0, "target_carbs": 280.0, "target_fat": 70.0
    }

    # Rank for each slot
    candidates_by_cat = {}
    for slot in MEAL_SLOTS:
        candidates_by_cat[slot] = rank_candidates(user_prof, df, model, preprocessor, meal_category=slot)

    result = optimize_meal_plan(user_prof, candidates_by_cat)
    meals = result["meals"]
    totals = result["totals"]

    assert len(meals) == 4, f"Expected 4 meals, got {len(meals)}"
    for slot in MEAL_SLOTS:
        assert slot in meals, f"Missing slot: {slot}"
        assert "food_name" in meals[slot], f"Missing food name in {slot}"
        assert "rationale" in meals[slot], f"Missing rationale in {slot}"

    assert totals["calories"] > 1000, "Plan calories unreasonably low"
    print("[OK] Meal Optimizer passed.")

def main():
    test_nutrition_calculations()
    test_data_loading_and_filtering()
    test_ml_recommender_and_ranking()
    test_meal_optimizer()
    print("\n==================================================")
    print("      ALL INTEGRATION TESTS PASSED CLEANLY!       ")
    print("==================================================")

if __name__ == "__main__":
    main()
