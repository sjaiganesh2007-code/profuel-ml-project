"""
Evaluation and Metrics Visualization Helpers for ProFuel.
Provides calculation and layout methods for model metrics and target-vs-actual plan comparisons.
"""

import pandas as pd
import numpy as np

def calculate_plan_deviations(target_nutrition, actual_totals):
    """
    Computes absolute and percentage deviations between target and actual plan totals.
    """
    metrics = ["calories", "protein", "carbohydrates", "fat"]
    results = {}

    for m in metrics:
        target_val = float(target_nutrition.get(f"target_{m}" if m != "calories" else "target_calories", 0))
        actual_val = float(actual_totals.get(m, 0))
        
        diff = actual_val - target_val
        pct_diff = (diff / target_val * 100.0) if target_val > 0 else 0.0

        results[m] = {
            "target": round(target_val, 1),
            "actual": round(actual_val, 1),
            "diff": round(diff, 1),
            "pct_diff": round(pct_diff, 1)
        }

    return results

def get_metric_status_color(pct_diff):
    """Returns CSS color code based on deviation percentage."""
    abs_pct = abs(pct_diff)
    if abs_pct <= 5.0:
        return "#0D9488"  # Teal / Excellent
    elif abs_pct <= 12.0:
        return "#2563EB"  # Blue / Good
    else:
        return "#E11D48"  # Rose / Warning
