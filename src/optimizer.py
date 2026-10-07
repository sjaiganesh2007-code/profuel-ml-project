"""
Constrained Meal Plan Optimization Engine for ProFuel.
Performs budget-aware, macro-balanced multi-objective search to generate exactly four meals
(Breakfast, Lunch, Snack, Dinner) with sensible portion sizing and controlled variation.
"""

import random
import numpy as np
import pandas as pd

from src.utils import generate_food_rationale

MEAL_SLOTS = ["Breakfast", "Lunch", "Snack", "Dinner"]

# Sensible portion multiplier search grid
PORTION_GRID = [0.75, 1.0, 1.25, 1.5]

def evaluate_meal_combination(combo_items, target_cal, target_prot, target_carbs, target_fat, budget):
    """
    Evaluates candidate 4-meal plan combination for cost penalty, macro deviation, and ML quality.
    """
    tot_cal = sum(item["calories"] for item in combo_items)
    tot_prot = sum(item["protein"] for item in combo_items)
    tot_carbs = sum(item["carbohydrates"] for item in combo_items)
    tot_fat = sum(item["fat"] for item in combo_items)
    tot_cost = sum(item["cost_inr"] for item in combo_items)
    avg_ml_score = sum(item["ml_suitability_score"] for item in combo_items) / 4.0

    # Macro Relative Deviations
    cal_dev = abs(tot_cal - target_cal) / max(1.0, target_cal)
    prot_dev = abs(tot_prot - target_prot) / max(1.0, target_prot)
    carbs_dev = abs(tot_carbs - target_carbs) / max(1.0, target_carbs)
    fat_dev = abs(tot_fat - target_fat) / max(1.0, target_fat)

    macro_penalty = (1.2 * cal_dev) + (1.5 * prot_dev) + (0.8 * carbs_dev) + (0.8 * fat_dev)

    # Cost / Budget Penalty
    cost_overrun = max(0.0, tot_cost - budget)
    cost_penalty = cost_overrun * 15.0

    # Variety Penalty (penalize similar food names)
    food_names = [item["food_name"] for item in combo_items]
    unique_names = len(set(food_names))
    variety_penalty = (4 - unique_names) * 5.0

    # Total Multi-objective Loss (Lower is better)
    loss = (macro_penalty * 100.0) + cost_penalty - (0.4 * avg_ml_score) + variety_penalty

    return {
        "loss": loss,
        "tot_cal": tot_cal,
        "tot_prot": tot_prot,
        "tot_carbs": tot_carbs,
        "tot_fat": tot_fat,
        "tot_cost": tot_cost,
        "avg_ml_score": avg_ml_score,
        "cost_overrun": cost_overrun
    }

def optimize_meal_plan(user_profile, candidates_by_category, random_seed=None):
    """
    Finds optimal 4-meal combination respecting daily budget, macro targets, and ML suitability.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    target_cal = float(user_profile.get("target_calories", 2000))
    target_prot = float(user_profile.get("target_protein", 100))
    target_carbs = float(user_profile.get("target_carbs", 250))
    target_fat = float(user_profile.get("target_fat", 60))
    budget = float(user_profile.get("budget", 200))

    # Verify candidates exist for all 4 slots
    for slot in MEAL_SLOTS:
        if slot not in candidates_by_category or candidates_by_category[slot].empty:
            raise ValueError(f"No eligible food candidates available for meal slot: {slot}")

    # Top candidate pools for search
    top_candidates = {}
    for slot in MEAL_SLOTS:
        cat_df = candidates_by_category[slot]
        # Take top 10 ranked candidates
        pool = cat_df.head(12).copy().reset_index(drop=True)
        # If random_seed is given (e.g., Regenerate Plan), add slight perturbation / shuffle top items
        if random_seed is not None:
            pool = pool.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        top_candidates[slot] = pool

    best_plan = None
    best_loss = float("inf")
    feasible_plans = []
    unfeasible_plans = []

    # Combinatorial candidate search across Breakfast, Lunch, Snack, Dinner
    # To maintain sub-second latency, sample search space intelligently
    b_pool = top_candidates["Breakfast"].to_dict("records")[:8]
    l_pool = top_candidates["Lunch"].to_dict("records")[:8]
    s_pool = top_candidates["Snack"].to_dict("records")[:8]
    d_pool = top_candidates["Dinner"].to_dict("records")[:8]

    # Sample reasonable portion combinations
    search_count = 0
    max_search = 3500

    for b_food in b_pool:
        for l_food in l_pool:
            for s_food in s_pool:
                for d_food in d_pool:
                    # Test standard portion multiplier first (1.0x)
                    # And portion adjustments if needed
                    portion_combos = [
                        (1.0, 1.0, 1.0, 1.0),
                        (1.0, 1.25, 0.75, 1.0),
                        (0.75, 1.0, 1.0, 1.25),
                        (1.25, 1.0, 1.0, 0.75)
                    ]
                    for p_b, p_l, p_s, p_d in portion_combos:
                        search_count += 1
                        if search_count > max_search:
                            break

                        item_b = _scale_food_item(b_food, p_b)
                        item_l = _scale_food_item(l_food, p_l)
                        item_s = _scale_food_item(s_food, p_s)
                        item_d = _scale_food_item(d_food, p_d)

                        combo_items = [item_b, item_l, item_s, item_d]
                        eval_res = evaluate_meal_combination(
                            combo_items, target_cal, target_prot, target_carbs, target_fat, budget
                        )

                        record = {
                            "items": {
                                "Breakfast": item_b,
                                "Lunch": item_l,
                                "Snack": item_s,
                                "Dinner": item_d
                            },
                            "eval": eval_res
                        }

                        if eval_res["cost_overrun"] <= 0.0:
                            feasible_plans.append(record)
                        else:
                            unfeasible_plans.append(record)

    # Select best plan
    feasibility_warning = False
    warning_message = ""

    if feasible_plans:
        # Sort by loss
        feasible_plans.sort(key=lambda x: x["eval"]["loss"])
        selected_record = feasible_plans[0]
    else:
        # Budget constraint impossible; take minimum cost overrun plan
        unfeasible_plans.sort(key=lambda x: x["eval"]["loss"])
        selected_record = unfeasible_plans[0]
        feasibility_warning = True
        actual_cost = selected_record["eval"]["tot_cost"]
        warning_message = (
            f"Daily food budget of ₹{budget:.0f} is tight for target calories ({target_cal:.0f} kcal). "
            f"Closest feasible plan costs estimated ₹{actual_cost:.0f}/day."
        )

    # Format output dictionary
    meals_output = {}
    for slot in MEAL_SLOTS:
        item = selected_record["items"][slot]
        # Attach factual rationale
        item["rationale"] = generate_food_rationale(
            item, user_profile.get("fitness_goal", "Maintenance"), budget
        )
        meals_output[slot] = item

    eval_meta = selected_record["eval"]
    totals = {
        "calories": round(eval_meta["tot_cal"], 1),
        "protein": round(eval_meta["tot_prot"], 1),
        "carbohydrates": round(eval_meta["tot_carbs"], 1),
        "fat": round(eval_meta["tot_fat"], 1),
        "cost_inr": round(eval_meta["tot_cost"], 1),
        "avg_ml_score": round(eval_meta["avg_ml_score"], 1)
    }

    return {
        "meals": meals_output,
        "totals": totals,
        "feasibility_warning": feasibility_warning,
        "warning_message": warning_message
    }

def _scale_food_item(food_dict, multiplier):
    """Scales food item attributes by portion size multiplier."""
    scaled = food_dict.copy()
    m = float(multiplier)
    
    orig_weight = float(scaled.get("serving_weight_g", 200))
    orig_size = str(scaled.get("serving_size", "1 serving"))
    
    scaled["portion_multiplier"] = m
    scaled["serving_weight_g"] = round(orig_weight * m, 0)
    
    if m == 1.0:
        scaled["portion_label"] = orig_size
    else:
        scaled["portion_label"] = f"{m:.2f}x ({int(scaled['serving_weight_g'])}g)"

    scaled["calories"] = round(float(scaled["calories"]) * m, 1)
    scaled["protein"] = round(float(scaled["protein"]) * m, 1)
    scaled["carbohydrates"] = round(float(scaled["carbohydrates"]) * m, 1)
    scaled["fat"] = round(float(scaled["fat"]) * m, 1)
    scaled["fiber"] = round(float(scaled.get("fiber", 0)) * m, 1)
    scaled["sugar"] = round(float(scaled.get("sugar", 0)) * m, 1)
    scaled["cost_inr"] = round(float(scaled["cost_inr"]) * m, 1)

    return scaled
