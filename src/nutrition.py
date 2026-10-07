"""
Nutrition Engine for ProFuel.
Calculates BMR, TDEE, goal-adjusted calorie targets, and macro distributions
using validated physiological formulas (Mifflin-St Jeor equation).
"""

ACTIVITY_MULTIPLIERS = {
    "Sedentary (Little or no exercise)": 1.2,
    "Lightly Active (1-3 days/week)": 1.375,
    "Moderately Active (3-5 days/week)": 1.55,
    "Very Active (6-7 days/week)": 1.725,
    "Super Active (Hard exercise & physical job)": 1.9
}

# Simplified key mapping for UI dropdowns
ACTIVITY_LEVEL_MAP = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
    "Super Active": 1.9
}

def calculate_bmi(weight_kg, height_cm):
    """Calculates Body Mass Index (BMI)."""
    if height_cm <= 0:
        return 0.0
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)

def calculate_bmr(weight_kg, height_cm, age, gender):
    """
    Calculates Basal Metabolic Rate (BMR) using the Mifflin-St Jeor Equation.
    """
    g = str(gender).strip().lower()
    if g == "male":
        bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) + 5.0
    else:  # female
        bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) - 161.0
    return max(800.0, round(bmr, 1))

def calculate_tdee(bmr, activity_level):
    """
    Calculates Total Daily Energy Expenditure (TDEE).
    """
    mult = 1.375
    if isinstance(activity_level, (int, float)):
        mult = float(activity_level)
    else:
        # Match string key
        act_str = str(activity_level).strip()
        for key, val in ACTIVITY_LEVEL_MAP.items():
            if key.lower() in act_str.lower():
                mult = val
                break
        else:
            for key, val in ACTIVITY_MULTIPLIERS.items():
                if key.lower() in act_str.lower():
                    mult = val
                    break
                    
    return round(bmr * mult, 1)

def calculate_nutrition_targets(weight_kg, height_cm, age, gender, activity_level, fitness_goal):
    """
    Calculates daily nutritional targets (Calories, Protein, Carbs, Fat)
    based on BMR, TDEE, and user fitness goal.
    """
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    bmi = calculate_bmi(weight_kg, height_cm)

    goal = str(fitness_goal).strip().title()

    # Calorie Adjustment
    if goal == "Weight Loss":
        target_calories = tdee - 500.0
        # Safety floor check
        min_floor = 1500.0 if str(gender).strip().lower() == "male" else 1200.0
        target_calories = max(min_floor, target_calories)
    elif goal == "Muscle Gain":
        target_calories = tdee + 300.0
    else:  # Maintenance
        target_calories = tdee

    target_calories = round(target_calories, 0)

    # Macro Distributions (% of calories)
    # 1g Protein = 4 kcal, 1g Carb = 4 kcal, 1g Fat = 9 kcal
    if goal == "Weight Loss":
        # High Protein ratio: 30% P, 40% C, 30% F
        p_pct, c_pct, f_pct = 0.30, 0.40, 0.30
    elif goal == "Muscle Gain":
        # Protein/Carb heavy: 30% P, 45% C, 25% F
        p_pct, c_pct, f_pct = 0.30, 0.45, 0.25
    else:  # Maintenance
        # Balanced: 25% P, 50% C, 25% F
        p_pct, c_pct, f_pct = 0.25, 0.50, 0.25

    protein_g = round((target_calories * p_pct) / 4.0, 1)
    carbs_g = round((target_calories * c_pct) / 4.0, 1)
    fat_g = round((target_calories * f_pct) / 9.0, 1)

    return {
        "bmi": bmi,
        "bmr": bmr,
        "tdee": tdee,
        "target_calories": target_calories,
        "target_protein": protein_g,
        "target_carbs": carbs_g,
        "target_fat": fat_g,
        "fitness_goal": goal
    }
