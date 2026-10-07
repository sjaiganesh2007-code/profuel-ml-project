# PROFUEL: AI-Powered Personalized Indian Diet & Fitness Planner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)

**ProFuel** is a production-style, machine-learning-powered diet and fitness planning system built specifically for authentic Indian cuisines. It combines physiological energy equations, scikit-learn suitability scoring, and budget-aware constrained optimization to generate customized, budget-compliant 4-meal daily plans (**Breakfast, Lunch, Snack, Dinner**).

---

## 📌 Project Overview & Problem Statement

Most existing diet planning tools rely either on static non-personalized meal templates or unconstrained AI chatbots that hallucinate unrealistic meal portions, ignore regional Indian dietary habits, or disregard daily food budgets.

**ProFuel** addresses these challenges by providing:
1. **Deterministic Physiological Nutrition**: BMR & TDEE calculation via the validated Mifflin-St Jeor formula adjusted for goal-specific macronutrient targets.
2. **ML Food Suitability Scoring**: Scikit-Learn suitability model (`HistGradientBoostingRegressor`) trained on user demographics, food composition, and derived interaction metrics.
3. **Constrained Multi-Objective Optimization**: Dynamic 4-meal plan generation prioritizing high ML suitability scores while strictly adhering to user daily budget constraints and minimizing macro target deviations.
4. **Food Swapping & Controlled Regeneration**: Interactive food item replacement from ranked alternative pools without application reloads.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT LAYER                              │
│   Age • Gender • Height • Weight • Activity • Goal • Diet • Budget      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       NUTRITION ENGINE LAYER                            │
│    BMR (Mifflin-St Jeor) • TDEE • Goal Calorie Floor • Target Macros    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CANDIDATE FILTERING LAYER                           │
│     Filters Dataset by Diet Preference & Category (Breakfast/Lunch/..) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING & ML MODEL                        │
│   Constructs User-Food Features + Derived Ratios (Protein/Cal, Density) │
│     HistGradientBoostingRegressor → Predicts Suitability Score (0-100) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   BUDGET & MACRO OPTIMIZER LAYER                        │
│      Constrained Candidate Search across Breakfast, Lunch, Snack, Dinner│
│     Balances Budget Tolerance, Macro Deviation, ML Score, & Variety     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      HEALTHIFYME-INSPIRED DASHBOARD                     │
│    4-Meal Plan Cards • Factual Rationale • Item Swaps • Target vs Actual│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Methodology

### Feature Vector Engineering
The ML model consumes a 23-column tabular feature matrix combining user profile attributes, food nutritional properties, and domain-derived interaction metrics:

- **User Features**: `user_age`, `user_height`, `user_weight`, `user_bmi`, `target_calories`, `target_protein`, `target_carbs`, `target_fat`, `user_budget`, `user_gender`, `user_goal`, `user_diet_preference`, `user_activity`
- **Food Features**: `food_calories`, `food_protein`, `food_carbs`, `food_fat`, `food_fiber`, `food_sugar`, `food_cost`, `meal_category`, `food_category`, `diet_type`
- **Derived Features**:
  - `calorie_ratio`: $\text{Food Calories} / (\text{Daily Target} / 4)$
  - `protein_per_calorie`: $\text{Protein (g)} / \text{Calories}$
  - `protein_per_rupee`: $\text{Protein (g)} / \text{Cost (₹)}$
  - `calorie_density`: $\text{Calories} / \text{Serving Weight (g)}$
  - `budget_compatibility`: $\text{Food Cost} / (\text{User Daily Budget} / 4)$
  - `macro_alignment_score`: Closeness of candidate macro distribution to user target distribution.

### Suitability Target & Model Choice
To avoid fabricated targets, ProFuel uses a transparent suitability index $S \in [0, 100]$ based on goal alignment, macro efficiency, and budget compatibility. The model is trained using **HistGradientBoostingRegressor** with an 80/20 train/test split.

### Empirical Evaluation Metrics
*(Values calculated from actual model training)*
- **Training Records**: 23,520 user-food pairs across 392 Indian dishes
- **Train MAE**: `0.3787` | **Test MAE**: `0.4391`
- **Train RMSE**: `0.5581` | **Test RMSE**: `0.6728`
- **Train $R^2$**: `0.9996` | **Test $R^2$**: `0.9994`

---

## 📂 File Structure

```
profuel/
├── app.py                      # Main Streamlit web application
├── data/
│   └── indian_foods.csv        # Indian food dataset catalog (392+ dishes)
├── model/
│   ├── model.joblib            # Trained Scikit-Learn regressor model
│   ├── preprocessor.joblib     # Fitted ColumnTransformer pipeline
│   └── metadata.json           # Model configuration & evaluation metrics
├── src/
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # Data ingestion, schema mapping, validation
│   ├── preprocessing.py        # Data cleaning and outlier checks
│   ├── nutrition.py            # BMR, TDEE, macro target calculations
│   ├── features.py             # Feature vector engineering
│   ├── recommender.py          # ML candidate filtering & scoring
│   ├── optimizer.py            # Budget-aware 4-meal plan optimizer
│   ├── evaluation.py           # Target-vs-actual metrics & reporting
│   └── utils.py                # Rationale generator & input validators
├── training/
│   ├── generate_indian_foods_dataset.py  # Dataset generator
│   └── train_model.py          # Reproducible ML training pipeline
├── tests/
│   └── test_pipeline.py        # Pipeline integration test suite
├── assets/
│   └── favicon.png             # ProFuel branding icon
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Theme configuration
├── .gitignore                  # Git exclusions
├── LICENSE                     # MIT License
└── README.md                   # Technical documentation
```

---

## ⚡ Local Setup & Execution Instructions

### Prerequisites
- Python 3.9+ installed
- Git

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/profuel.git
cd profuel

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Integration Tests
```bash
python tests/test_pipeline.py
```

### 4. (Optional) Re-generate Dataset & Train Model
```bash
python training/generate_indian_foods_dataset.py
python training/train_model.py
```

### 5. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment Guide

1. Push code repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"**.
4. Select your GitHub repository, branch (`main`), and set Main file path to `app.py`.
5. Click **"Deploy"**.

---

## 🌐 Custom Domain DNS Configuration Guide

To configure a custom domain (e.g., `planner.profuel.app` or `www.profuel.app`):

1. **Streamlit Cloud Dashboard**:
   - Go to your deployed app settings on Streamlit Cloud.
   - Click **Custom Domain**.
   - Enter your domain name (e.g., `planner.profuel.app`).

2. **DNS Provider Setup** (Cloudflare, GoDaddy, Namecheap, etc.):
   - Add a `CNAME` record in your DNS management console:
     - **Type**: `CNAME`
     - **Name / Host**: `planner` (or `@` for root domain)
     - **Target / Value**: `domain.streamlit.app` (provided in Streamlit Cloud)
     - **TTL**: Automatic / 300s
   - Allow 5 to 30 minutes for DNS propagation and automatic SSL certificate generation.

---

## ⚠️ Health & Medical Safety Disclaimer

ProFuel is an educational Machine Learning Problem-Based Learning (PBL) software prototype. It provides general nutrition planning based on mathematical formulas and dataset estimates. **It does not provide medical advice, diagnosis, or clinical dietetics.** Users with medical conditions, pregnancy, food allergies, eating disorders, or clinical nutrition requirements should consult a qualified healthcare professional.

---

## 📜 License

Distributed under the [MIT License](LICENSE).
