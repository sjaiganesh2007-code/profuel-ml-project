"""
PROFUEL: AI-Powered Personalized Indian Diet & Fitness Planner
Streamlit Web Application
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_food_data, filter_by_diet, filter_by_meal_category
from src.nutrition import calculate_nutrition_targets, calculate_bmi
from src.recommender import load_trained_artifacts, rank_candidates
from src.optimizer import optimize_meal_plan, MEAL_SLOTS
from src.utils import validate_user_inputs, generate_food_rationale, format_currency
from src.evaluation import calculate_plan_deviations, get_metric_status_color

# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

FAVICON_PATH = PROJECT_ROOT / "assets" / "favicon.png"
page_icon = str(FAVICON_PATH) if FAVICON_PATH.exists() else "🥗"

st.set_page_config(
    page_title="ProFuel - Personalized Indian Diet & Fitness Planner",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLING (HEALTHIFYME-INSPIRED CLEAN UX)
# =========================================================

st.markdown("""
<style>
    /* Global Reset & Typography */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Hide Default Header Decoration */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* ProFuel Brand Title */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .brand-name {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0D9488; /* Primary Teal */
        margin: 0;
    }
    .brand-tagline {
        font-size: 13px;
        color: #64748B;
        margin: 0;
    }

    /* Cards */
    .profuel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .profuel-card-header {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 8px;
    }
    
    /* Meal Cards */
    .meal-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0D9488;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .meal-slot-title {
        font-size: 14px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #0D9488;
        margin-bottom: 4px;
    }
    .meal-food-title {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .meal-meta-pill {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .meal-rationale {
        font-size: 13px;
        color: #475569;
        font-style: italic;
        background: #F8FAFC;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 10px;
        border: 1px dashed #CBD5E1;
    }

    /* Metrics Display */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    
    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 12px;
        color: #92400E;
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton button {
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CACHED DATA & MODEL LOADERS
# =========================================================

@st.cache_data
def get_cached_food_data():
    return load_food_data("data/indian_foods.csv")

@st.cache_resource
def get_cached_model():
    return load_trained_artifacts("model")

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "current_plan" not in st.session_state:
    st.session_state.current_plan = None

if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

if "explorer_page" not in st.session_state:
    st.session_state.explorer_page = 1

# Load resources
try:
    food_df = get_cached_food_data()
    model, preprocessor, model_meta = get_cached_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error_msg = str(e)

# =========================================================
# SIDEBAR NAVIGATION & USER PROFILE FORM
# =========================================================

with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-name">PROFUEL</h1>
            <p class="brand-tagline">AI Diet & Fitness Planner</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_option = st.radio(
        "Navigation",
        ["Planner", "Food Explorer", "Model / Method", "Privacy Policy", "Terms & Conditions"],
        index=0
    )

    st.markdown("---")
    st.caption("PBL Academic ML Project • Production Prototype")

# =========================================================
# PAGE 1: PLANNER (PRIMARY SCREEN)
# =========================================================

if nav_option == "Planner":
    st.title("Personalized Diet & Meal Planner")
    st.markdown("Configure your profile, fitness goals, and daily budget to generate a machine-learning optimized 4-meal plan.")

    # Medical / Safety Disclaimer Banner
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>HEALTH DISCLAIMER:</strong> ProFuel provides general nutrition planning based on user input for educational and prototype demonstration purposes. It is not medical advice. Users with medical conditions, pregnancy, allergies, or clinical dietary requirements should consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Failed to load ML Model artifacts. Please run `python training/train_model.py`. Error details: {model_error_msg}")
        st.stop()

    # Step 1: User Profile Form
    with st.expander("1. Enter Your Physical Profile & Goals", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age (years)", min_value=12, max_value=90, value=25, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col2:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=172.0, step=0.5)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

        with col3:
            activity = st.selectbox("Activity Level", [
                "Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Super Active"
            ], index=2)
            fitness_goal = st.selectbox("Fitness Goal", ["Weight Loss", "Maintenance", "Muscle Gain"], index=0)

        with col4:
            diet_pref = st.selectbox("Dietary Preference", ["Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian"], index=0)
            budget = st.number_input("Daily Food Budget (₹)", min_value=50.0, max_value=2000.0, value=200.0, step=10.0)

    # Input Validation
    is_valid, val_errors = validate_user_inputs(age, height, weight, budget)
    if not is_valid:
        for err in val_errors:
            st.error(err)
        st.stop()

    # Step 2: Target Calculation
    targets = calculate_nutrition_targets(weight, height, age, gender, activity, fitness_goal)
    user_prof = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "bmi": targets["bmi"],
        "activity_level": activity,
        "fitness_goal": fitness_goal,
        "dietary_preference": diet_pref,
        "budget": budget,
        "target_calories": targets["target_calories"],
        "target_protein": targets["target_protein"],
        "target_carbs": targets["target_carbs"],
        "target_fat": targets["target_fat"]
    }

    # Display Macro & Energy Target Cards
    st.markdown("### 2. Calculated Daily Targets")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    with m_col1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{targets["target_calories"]:.0f}</div><div class="metric-label">Calories (kcal)</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{targets["target_protein"]:.0f}g</div><div class="metric-label">Protein</div></div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{targets["target_carbs"]:.0f}g</div><div class="metric-label">Carbs</div></div>', unsafe_allow_html=True)
    with m_col4:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{targets["target_fat"]:.0f}g</div><div class="metric-label">Fat</div></div>', unsafe_allow_html=True)
    with m_col5:
        st.markdown(f'<div class="metric-box"><div class="metric-value">₹{budget:.0f}</div><div class="metric-label">Budget / Day</div></div>', unsafe_allow_html=True)
    with m_col6:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{targets["bmi"]:.1f}</div><div class="metric-label">BMI</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Action Buttons
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        generate_clicked = st.button("Generate Personalized 4-Meal Plan", type="primary", use_container_width=True)
    with btn_col2:
        regenerate_clicked = st.button("Regenerate Plan (New Variation)", use_container_width=True)

    # Generate or Regenerate logic
    if generate_clicked or st.session_state.current_plan is None or regenerate_clicked:
        if regenerate_clicked:
            st.session_state.random_seed += 1

        with st.spinner("Filtering candidates, running ML scoring model, and optimizing 4-meal budget plan..."):
            candidates_by_cat = {}
            for slot in MEAL_SLOTS:
                candidates_by_cat[slot] = rank_candidates(
                    user_prof, food_df, model, preprocessor, meal_category=slot
                )

            opt_res = optimize_meal_plan(
                user_prof, candidates_by_cat, random_seed=st.session_state.random_seed
            )
            st.session_state.current_plan = opt_res
            st.session_state.user_profile = user_prof

    plan_data = st.session_state.current_plan
    active_profile = st.session_state.user_profile

    # Display Feasibility Warning if budget constraint is impossible
    if plan_data.get("feasibility_warning"):
        st.warning(f"⚠️ {plan_data['warning_message']}")

    st.markdown("---")
    st.markdown("## 3. Your Personalized 4-Meal Plan")

    # Display Meal Cards (Breakfast, Lunch, Snack, Dinner)
    meals = plan_data["meals"]
    totals = plan_data["totals"]

    meal_cols = st.columns(4)

    for idx, slot in enumerate(MEAL_SLOTS):
        item = meals[slot]
        with meal_cols[idx]:
            st.markdown(f"""
            <div class="meal-card">
                <div class="meal-slot-title">{slot}</div>
                <div class="meal-food-title">{item['food_name']}</div>
                <div>
                    <span class="meal-meta-pill">Portion: {item.get('portion_label', item['serving_size'])}</span>
                    <span class="meal-meta-pill">₹{item['cost_inr']:.0f}</span>
                </div>
                <div style="margin-top: 8px; font-size: 13px; color: #334155;">
                    <strong>{item['calories']:.0f} kcal</strong> • P: {item['protein']:.1f}g • C: {item['carbohydrates']:.1f}g • F: {item['fat']:.1f}g
                </div>
                <div style="font-size: 11px; color: #64748B; margin-top: 4px;">
                    ML Suitability: <strong>{item['ml_suitability_score']:.1f}/100</strong>
                </div>
                <div class="meal-rationale">
                    "{item['rationale']}"
                </div>
            </div>
            """, unsafe_allow_html=True)

            # SWAP ITEM FUNCTIONALITY
            with st.expander(f"Swap {slot} Item"):
                cat_candidates = rank_candidates(
                    active_profile, food_df, model, preprocessor, meal_category=slot
                )
                # Filter out currently selected food
                alt_options = cat_candidates[cat_candidates["food_name"] != item["food_name"]].head(6)
                
                selected_swap_name = st.selectbox(
                    f"Choose replacement for {slot}:",
                    options=["-- Keep Current --"] + alt_options["food_name"].tolist(),
                    key=f"swap_select_{slot}"
                )

                if selected_swap_name != "-- Keep Current --":
                    swap_row = alt_options[alt_options["food_name"] == selected_swap_name].iloc[0].to_dict()
                    # Preserve portion multiplier
                    m_mult = item.get("portion_multiplier", 1.0)
                    from src.optimizer import _scale_food_item
                    new_scaled = _scale_food_item(swap_row, m_mult)
                    new_scaled["rationale"] = generate_food_rationale(
                        new_scaled, active_profile["fitness_goal"], active_profile["budget"]
                    )
                    # Update plan state
                    st.session_state.current_plan["meals"][slot] = new_scaled
                    
                    # Recalculate plan totals
                    new_meals = st.session_state.current_plan["meals"]
                    st.session_state.current_plan["totals"] = {
                        "calories": round(sum(m["calories"] for m in new_meals.values()), 1),
                        "protein": round(sum(m["protein"] for m in new_meals.values()), 1),
                        "carbohydrates": round(sum(m["carbohydrates"] for m in new_meals.values()), 1),
                        "fat": round(sum(m["fat"] for m in new_meals.values()), 1),
                        "cost_inr": round(sum(m["cost_inr"] for m in new_meals.values()), 1),
                        "avg_ml_score": round(sum(m["ml_suitability_score"] for m in new_meals.values()) / 4.0, 1)
                    }
                    st.rerun()

    # Target vs Actual Comparison Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 4. Target vs. Actual Nutritional Comparison")

    deviations = calculate_plan_deviations(active_profile, totals)

    comp_df = pd.DataFrame([
        {
            "Nutrient Metric": "Calories (kcal)",
            "Target": f"{deviations['calories']['target']:.0f}",
            "Actual Plan": f"{deviations['calories']['actual']:.0f}",
            "Difference": f"{deviations['calories']['diff']:+.0f} kcal",
            "Deviation %": f"{deviations['calories']['pct_diff']:+.1f}%"
        },
        {
            "Nutrient Metric": "Protein (g)",
            "Target": f"{deviations['protein']['target']:.1f}g",
            "Actual Plan": f"{deviations['protein']['actual']:.1f}g",
            "Difference": f"{deviations['protein']['diff']:+.1f}g",
            "Deviation %": f"{deviations['protein']['pct_diff']:+.1f}%"
        },
        {
            "Nutrient Metric": "Carbohydrates (g)",
            "Target": f"{deviations['carbohydrates']['target']:.1f}g",
            "Actual Plan": f"{deviations['carbohydrates']['actual']:.1f}g",
            "Difference": f"{deviations['carbohydrates']['diff']:+.1f}g",
            "Deviation %": f"{deviations['carbohydrates']['pct_diff']:+.1f}%"
        },
        {
            "Nutrient Metric": "Fat (g)",
            "Target": f"{deviations['fat']['target']:.1f}g",
            "Actual Plan": f"{deviations['fat']['actual']:.1f}g",
            "Difference": f"{deviations['fat']['diff']:+.1f}g",
            "Deviation %": f"{deviations['fat']['pct_diff']:+.1f}%"
        },
        {
            "Nutrient Metric": "Daily Budget (₹)",
            "Target": f"₹{active_profile['budget']:.0f}",
            "Actual Plan": f"₹{totals['cost_inr']:.0f}",
            "Difference": f"₹{totals['cost_inr'] - active_profile['budget']:+.0f}",
            "Deviation %": f"{(totals['cost_inr'] - active_profile['budget'])/max(1, active_profile['budget'])*100:+.1f}%"
        }
    ])

    st.table(comp_df)

# =========================================================
# PAGE 2: FOOD EXPLORER
# =========================================================

elif nav_option == "Food Explorer":
    st.title("Indian Food Database Explorer")
    st.markdown("Search, filter, and inspect nutritional & cost metrics across the complete catalog of authentic Indian dishes.")

    # Search and Filter Controls
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        search_q = st.text_input("Search Dish Name", value="", placeholder="e.g. Dosa, Paneer, Chicken, Dal...")
    with f_col2:
        meal_filter = st.selectbox("Meal Category", ["All Categories"] + MEAL_SLOTS)
    with f_col3:
        diet_filter = st.selectbox("Dietary Preference", ["All Diets", "Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian"])

    # Range Sliders
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        max_cal = st.slider("Max Calories (kcal)", 50, 1000, 800, step=25)
    with r_col2:
        min_prot = st.slider("Min Protein (g)", 0, 50, 0, step=2)
    with r_col3:
        max_cost = st.slider("Max Cost (₹)", 10, 300, 250, step=5)

    # Filter Execution
    filtered_df = food_df.copy()

    if search_q.strip():
        filtered_df = filtered_df[filtered_df["food_name"].str.lower().str.contains(search_q.strip().lower())]

    if meal_filter != "All Categories":
        filtered_df = filter_by_meal_category(filtered_df, meal_filter)

    if diet_filter != "All Diets":
        filtered_df = filter_by_diet(filtered_df, diet_filter)

    filtered_df = filtered_df[
        (filtered_df["calories"] <= max_cal) &
        (filtered_df["protein"] >= min_prot) &
        (filtered_df["cost_inr"] <= max_cost)
    ]

    st.markdown(f"Showing **{len(filtered_df)}** matching food items.")

    # Pagination
    PAGE_SIZE = 15
    total_pages = max(1, (len(filtered_df) + PAGE_SIZE - 1) // PAGE_SIZE)
    
    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
    with p_col1:
        if st.button("Previous Page") and st.session_state.explorer_page > 1:
            st.session_state.explorer_page -= 1
    with p_col2:
        st.markdown(f"<p style='text-align:center; margin-top:8px;'>Page <strong>{st.session_state.explorer_page}</strong> of <strong>{total_pages}</strong></p>", unsafe_allow_html=True)
    with p_col3:
        if st.button("Next Page") and st.session_state.explorer_page < total_pages:
            st.session_state.explorer_page += 1

    start_idx = (st.session_state.explorer_page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = filtered_df.iloc[start_idx:end_idx]

    display_cols = [
        "food_id", "food_name", "meal_category", "diet_type", "food_category",
        "serving_size", "calories", "protein", "carbohydrates", "fat", "fiber", "cost_inr"
    ]

    st.dataframe(
        page_df[display_cols].rename(columns={
            "food_id": "ID", "food_name": "Dish Name", "meal_category": "Meal", "diet_type": "Diet",
            "food_category": "Category", "serving_size": "Serving Size", "calories": "Calories (kcal)",
            "protein": "Protein (g)", "carbohydrates": "Carbs (g)", "fat": "Fat (g)", "fiber": "Fiber (g)",
            "cost_inr": "Cost (₹)"
        }),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# PAGE 3: MODEL / METHOD (TECHNICAL BREAKDOWN)
# =========================================================

elif nav_option == "Model / Method":
    st.title("System Architecture & ML Methodology")
    st.markdown("Comprehensive technical documentation of the end-to-end ProFuel pipeline, training methodology, and honest empirical model evaluation.")

    # Architecture Overview Flowchart
    st.markdown("### 1. End-to-End Pipeline Architecture")
    st.markdown("""
    ```
    User Profile Inputs (Age, Weight, Height, Goal, Budget, Diet)
                          │
                          ▼
            [ Physiological Nutrition Engine ]
            Calculates BMR & TDEE (Mifflin-St Jeor)
            Determines Target Calories & Macros
                          │
                          ▼
             [ Candidate Filtering Layer ]
       Filters Food Dataset by Diet & Meal Category
                          │
                          ▼
             [ Feature Engineering Vector ]
      Constructs User-Food Features + Derived Metrics
    (Protein/Calorie, Calorie Density, Macro Alignment)
                          │
                          ▼
             [ ML Suitability Model ]
     Scikit-Learn HistGradientBoostingRegressor (Saved Artifact)
          Predicts Food Suitability Score (0-100)
                          │
                          ▼
         [ Constrained Multi-Objective Optimizer ]
       Evaluates 4-Meal Combinations (Breakfast, Lunch, Snack, Dinner)
      Balances Macro Target Deviation, Budget & Variety
                          │
                          ▼
               Personalized 4-Meal Plan
    ```
    """)

    st.markdown("---")

    # Model Evaluation Metrics (Empirical Data)
    st.markdown("### 2. Empirical Model Evaluation Metrics")
    
    if model_meta and "metrics" in model_meta:
        m = model_meta["metrics"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test MAE (Mean Abs Error)", f"{m['test_mae']:.4f}")
            st.caption(f"Train MAE: {m['train_mae']:.4f}")
        with col2:
            st.metric("Test RMSE (Root Mean Sq Error)", f"{m['test_rmse']:.4f}")
            st.caption(f"Train RMSE: {m['train_rmse']:.4f}")
        with col3:
            st.metric("Test R² Score", f"{m['test_r2']:.4f}")
            st.caption(f"Train R²: {m['train_r2']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        **Model Metadata Summary:**
        - **Algorithm**: `{model_meta.get('model_type', 'HistGradientBoostingRegressor')}`
        - **Training Dataset Size**: `{model_meta.get('total_pairs_trained', 0):,}` user-food pair records
        - **Catalog Food Items**: `{model_meta.get('num_food_items', 0)}` authentic Indian dishes
        - **Feature Vector Dimensions**: `{model_meta.get('feature_count', 0)}` tabular features
        - **Split Ratio**: 80% Training (`{model_meta.get('train_samples', 0):,}` samples), 20% Holdout Test (`{model_meta.get('test_samples', 0):,}` samples)
        - **Target Description**: {model_meta.get('target_description', '')}
        """)
    else:
        st.info("Metadata unavailable. Please train model using `python training/train_model.py`.")

    st.markdown("---")
    st.markdown("### 3. Key Feature Groups Used in Model Inference")
    
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown("**User Demographic Features:**\n- Age, Gender, Height, Weight\n- Calculated BMI\n- Activity Level & Multiplier\n- Selected Fitness Goal\n- Dietary Preference\n- Calorie & Macro Targets\n- Daily Budget (₹)")
    with f_col2:
        st.markdown("**Food Item Features:**\n- Serving Weight (g) & Size\n- Calories (kcal)\n- Protein, Carbs, Fat (g)\n- Fiber & Sugar (g)\n- Estimated Cost per Serving (₹)\n- Meal Category & Food Group\n- Diet Type Classification")
    with f_col3:
        st.markdown("**Derived Domain Interactions:**\n- Calorie Ratio to Meal Allocation\n- Protein per Calorie Efficiency\n- Protein per Rupee Value\n- Calorie Density (kcal/g)\n- Budget Compatibility Ratio\n- Macro Alignment Index Score")

# =========================================================
# PAGE 4: PRIVACY POLICY
# =========================================================

elif nav_option == "Privacy Policy":
    st.title("Privacy Policy")
    st.markdown("#### Effective Date: October 2026")
    
    st.markdown("""
    ### 1. Information Processing Scope
    ProFuel is a localized academic machine-learning application built for diet and fitness planning. All personal parameters entered into the planner interface—including age, height, weight, gender, daily budget, and dietary choices—are processed **entirely within your current local session**.

    ### 2. Data Storage & Transmission
    - **No External API Transmissions**: ProFuel does not send your personal parameters or diet preferences to any third-party external AI APIs, advertising trackers, or external databases.
    - **No Permanent Profile Storage**: Physical profile inputs are kept transiently in browser session state (`st.session_state`) during your active session and are not stored in any permanent server database.
    - **No Account / Registration Required**: The application operates without user registration, credentials, or personal identification.

    ### 3. Dataset & Model Usage
    The machine learning recommendation models (`model/model.joblib`) operate offline using pre-trained regression parameters on standard nutritional schemas. No individual user interaction is logged or transmitted to retrain shared public models.

    ### 4. Security Measures
    ProFuel runs as a stateless client application. Standard browser TLS/HTTPS encryption applies when deployed on Streamlit Cloud infrastructure.

    ### 5. Contact Placeholder
    For academic queries or implementation details regarding ProFuel, refer to the project repository documentation.
    """)

# =========================================================
# PAGE 5: TERMS & CONDITIONS
# =========================================================

elif nav_option == "Terms & Conditions":
    st.title("Terms & Conditions")
    st.markdown("#### Educational Prototype Notice")

    st.markdown("""
    ### 1. Educational & PBL Prototype Nature
    ProFuel is an academic Machine Learning Problem-Based Learning (PBL) software prototype designed to demonstrate constrained budget optimization, feature engineering, and tabular ML recommendation systems. It is intended solely for educational and portfolio evaluation purposes.

    ### 2. Non-Medical Advice Disclaimer
    **ProFuel is NOT a medical diagnosis or clinical dietetics system.**
    - Recommendations generated by the application are calculated using standard physiological formulas and cost-aware ML ranking algorithms.
    - They do not constitute clinical dietary prescriptions, medical diagnoses, or treatment plans for any health conditions.
    - Individuals with clinical conditions, pregnancy, food allergies, eating disorders, diabetes, renal diseases, or specific therapeutic dietary needs must consult a qualified healthcare professional or licensed dietitian before altering their diet.

    ### 3. Data Accuracy & Limitations
    - Food costs and nutritional values are based on standard Indian compositional databases and market estimates; actual local market prices and food compositions may vary by brand, region, and preparation method.
    - Users assume full responsibility for verifying food suitability and allergen safety prior to consumption.

    ### 4. Limitation of Liability
    The developers of ProFuel shall not be held liable for any direct, indirect, or consequential decisions made based on meal plans generated by the software prototype.
    """)