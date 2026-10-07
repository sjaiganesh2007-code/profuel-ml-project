"""
Utility Functions and Helpers for ProFuel Application.
Provides input validation, food rationale generation, formatting, and safety checks.
"""

def generate_food_rationale(food_item, fitness_goal, budget):
    """
    Generates concise, factual rationale for food recommendation without hype or unsupported medical claims.
    """
    prot = float(food_item.get("protein", 0))
    cals = float(food_item.get("calories", 0))
    cost = float(food_item.get("cost_inr", 0))
    ml_score = float(food_item.get("ml_suitability_score", 0))
    diet = str(food_item.get("diet_type", "Vegetarian"))
    goal = str(fitness_goal).strip().title()

    p_per_cal = prot / max(1.0, cals)

    reasons = []

    if p_per_cal > 0.06:
        reasons.append(f"high protein efficiency ({prot:.1f}g protein)")
    elif prot >= 15.0:
        reasons.append(f"rich protein source ({prot:.1f}g protein)")

    if cost <= (budget / 4.0):
        reasons.append(f"cost-effective (₹{cost:.0f})")
    else:
        reasons.append(f"budget allocation (₹{cost:.0f})")

    if goal == "Weight Loss" and p_per_cal > 0.05:
        reasons.append("favorable calorie-to-protein ratio for fat loss")
    elif goal == "Muscle Gain" and prot > 15.0:
        reasons.append("supports hyper-trophy macro target")

    if ml_score >= 80.0:
        reasons.append(f"high ML suitability score ({ml_score:.0f}/100)")

    if not reasons:
        reasons.append("balanced nutritional profile")

    rationale_str = f"Selected because it offers {', '.join(reasons)}, aligning with your {goal} goal and {diet} diet preference."
    return rationale_str

def validate_user_inputs(age, height, weight, budget):
    """
    Validates user numeric inputs and returns (is_valid, list_of_error_messages).
    """
    errors = []

    if age < 10 or age > 100:
        errors.append("Age must be between 10 and 100 years.")

    if height < 80 or height > 250:
        errors.append("Height must be between 80 cm and 250 cm.")

    if weight < 25 or weight > 250:
        errors.append("Weight must be between 25 kg and 250 kg.")

    if budget < 30 or budget > 5000:
        errors.append("Daily food budget must be between ₹30 and ₹5,000.")

    return (len(errors) == 0), errors

def format_currency(amount):
    """Formats numeric amount as Indian Rupees (₹)."""
    return f"₹{amount:,.0f}"
