"""
ML Pipeline: Feature Engineering + Model Training
Uses GradientBoosting (scikit-learn) as LightGBM replacement
Includes SHAP-style feature importance for explainability
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, mean_absolute_error, classification_report
)
from sklearn.inspection import permutation_importance

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "../..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/students.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Feature Engineering ─────────────────────────────────────────────────────

CATEGORICAL_COLS = ["course_type", "institute_tier", "region", "target_sector", "internship_employer_type"]
NUMERIC_FEATURES = [
    "cgpa", "academic_consistency_score",
    "num_internships", "internship_months", "internship_performance",
    "num_certifications",
    "institute_placement_rate_3m", "institute_placement_rate_6m", "institute_placement_rate_12m",
    "institute_avg_salary_lpa", "placement_cell_score", "recruiter_diversity_score",
    "job_demand_score", "region_job_density", "sector_growth_rate",
    "job_applications_per_week", "interviews_attended",
    "resume_updated_recently", "linkedin_active",
    "internship_quality_score", "skill_relevance_score", "institute_strength_index",
]

LABEL_ENCODERS = {}

def engineer_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()

    # ── Derived features ──
    df["cgpa_normalized"] = df["cgpa"] / 10.0
    df["application_intensity"] = np.log1p(df["job_applications_per_week"])
    df["interview_conversion"] = np.where(
        df["job_applications_per_week"] > 0,
        df["interviews_attended"] / (df["job_applications_per_week"] + 1),
        0
    ).clip(0, 1)
    df["engagement_score"] = (
        df["resume_updated_recently"] * 0.4 +
        df["linkedin_active"] * 0.3 +
        (df["job_applications_per_week"] / 30.0).clip(0, 1) * 0.3
    )
    df["placement_readiness"] = (
        df["cgpa_normalized"] * 0.25 +
        df["internship_quality_score"] * 0.25 +
        df["skill_relevance_score"] * 0.20 +
        df["engagement_score"] * 0.15 +
        df["institute_strength_index"] * 0.15
    )
    df["market_fit_score"] = (
        df["job_demand_score"] * 0.5 +
        df["region_job_density"] * 0.3 +
        df["sector_growth_rate"].clip(-0.1, 0.3).apply(lambda x: (x + 0.1) / 0.4) * 0.2
    )

    # ── Encode categoricals ──
    global LABEL_ENCODERS
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            LABEL_ENCODERS[col] = le
        else:
            le = LABEL_ENCODERS.get(col)
            if le:
                df[col + "_enc"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    return df


def get_feature_cols():
    numeric = NUMERIC_FEATURES + [
        "cgpa_normalized", "application_intensity", "interview_conversion",
        "engagement_score", "placement_readiness", "market_fit_score"
    ]
    cat_enc = [c + "_enc" for c in CATEGORICAL_COLS]
    return numeric + cat_enc


# ─── Training ─────────────────────────────────────────────────────────────────

def train_models():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df, fit=True)

    feature_cols = get_feature_cols()
    X = df[feature_cols].fillna(0)

    results = {}

    # ── Classification models (3m, 6m, 12m) ──
    for target in ["placed_3m", "placed_6m", "placed_12m"]:
        y = df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42
        )
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"  {target} → AUC: {auc:.3f}")
        results[target] = {"auc": round(auc, 3)}

        path = os.path.join(MODEL_DIR, f"model_{target}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    # ── Regression model (salary) ──
    y_sal = df["actual_salary_lpa"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_sal, test_size=0.2, random_state=42)
    sal_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    sal_model.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, sal_model.predict(X_te))
    print(f"  salary → MAE: {mae:.2f} LPA")
    results["salary"] = {"mae": round(mae, 2)}

    with open(os.path.join(MODEL_DIR, "model_salary.pkl"), "wb") as f:
        pickle.dump(sal_model, f)

    # ── Save feature names + encoders ──
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(LABEL_ENCODERS, f)

    print(f"\nAll models saved to {MODEL_DIR}")
    return results


# ─── Inference ────────────────────────────────────────────────────────────────

class PlacementPredictor:
    """Inference wrapper for all placement prediction models."""

    FEATURE_NAMES_HUMAN = {
        "internship_quality_score": "Internship Quality",
        "placement_readiness": "Overall Placement Readiness",
        "institute_strength_index": "Institute Strength",
        "skill_relevance_score": "Skill Relevance",
        "market_fit_score": "Market Fit",
        "cgpa_normalized": "Academic Performance (CGPA)",
        "engagement_score": "Job Search Engagement",
        "job_demand_score": "Industry Demand",
        "institute_placement_rate_6m": "Institute Placement Rate",
        "region_job_density": "Regional Job Availability",
        "internship_months": "Internship Duration",
        "num_certifications": "Number of Certifications",
        "interview_conversion": "Interview Conversion Rate",
        "application_intensity": "Application Activity",
        "academic_consistency_score": "Academic Consistency",
    }

    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.label_encoders = None
        self._load()

    def _load(self):
        for name in ["placed_3m", "placed_6m", "placed_12m", "salary"]:
            path = os.path.join(MODEL_DIR, f"model_{name}.pkl")
            with open(path, "rb") as f:
                self.models[name] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
            self.feature_cols = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
            self.label_encoders = pickle.load(f)

        global LABEL_ENCODERS
        LABEL_ENCODERS = self.label_encoders

    def _prepare(self, student_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame([student_dict])
        df = engineer_features(df, fit=False)
        return df[self.feature_cols].fillna(0)

    def predict(self, student_dict: dict) -> dict:
        X = self._prepare(student_dict)

        prob_3m = float(self.models["placed_3m"].predict_proba(X)[0, 1])
        prob_6m = float(self.models["placed_6m"].predict_proba(X)[0, 1])
        prob_12m = float(self.models["placed_12m"].predict_proba(X)[0, 1])
        salary = float(self.models["salary"].predict(X)[0])

        # ── Risk classification ──
        risk_score = 1.0 - prob_6m
        if risk_score < 0.3:
            risk_level = "Low"
        elif risk_score < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # ── Explainability via feature importances ──
        risk_drivers, protective_factors = self._explain(X, student_dict)

        # ── Salary range ──
        sal_min = round(salary * 0.85, 1)
        sal_max = round(salary * 1.15, 1)

        # ── Interventions ──
        recommendations = self._recommend(student_dict, risk_level, risk_drivers)

        return {
            "placement_3m": round(prob_3m, 3),
            "placement_6m": round(prob_6m, 3),
            "placement_12m": round(prob_12m, 3),
            "risk_score": round(risk_score, 3),
            "risk": risk_level,
            "salary_range": f"{sal_min}-{sal_max} LPA",
            "salary_mid": round(salary, 1),
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "recommendations": recommendations,
        }

    def _explain(self, X: pd.DataFrame, student_dict: dict):
        """Generate human-readable explanations using feature importance + values."""
        importances = self.models["placed_6m"].feature_importances_
        feat_vals = X.iloc[0]

        scored = []
        for i, col in enumerate(self.feature_cols):
            val = float(feat_vals[col])
            imp = float(importances[i])
            # Normalize value to [0,1] range for risk scoring
            scored.append((col, imp, val))

        # Sort by importance
        scored.sort(key=lambda x: x[1], reverse=True)

        risk_drivers = []
        protective_factors = []

        for col, imp, val in scored[:12]:
            human_name = self.FEATURE_NAMES_HUMAN.get(col, col.replace("_", " ").title())
            # Determine if this feature is helping or hurting
            # Higher values = better for most features
            if imp > 0.01:
                if val < 0.4:
                    risk_drivers.append(f"Low {human_name} ({val:.2f})")
                elif val > 0.65:
                    protective_factors.append(f"Strong {human_name} ({val:.2f})")

        # Add specific contextual drivers
        if student_dict.get("num_internships", 0) == 0:
            risk_drivers.insert(0, "No internship experience")
        if student_dict.get("num_certifications", 0) == 0:
            risk_drivers.append("No skill certifications")
        if student_dict.get("job_applications_per_week", 0) < 5:
            risk_drivers.append("Low job application activity")
        if student_dict.get("cgpa", 0) >= 8.0:
            protective_factors.insert(0, f"Strong CGPA ({student_dict['cgpa']})")
        if student_dict.get("institute_tier") == "Tier 1":
            protective_factors.append("Tier 1 Institute")

        return risk_drivers[:4], protective_factors[:3]

    def _recommend(self, student_dict: dict, risk: str, drivers: list) -> list:
        recs = []
        apps = student_dict.get("job_applications_per_week", 0)
        internships = student_dict.get("num_internships", 0)
        certs = student_dict.get("num_certifications", 0)
        cgpa = student_dict.get("cgpa", 0)
        course = student_dict.get("course_type", "")

        if apps < 10:
            recs.append(f"Increase job applications to at least {20 if risk == 'High' else 10}/week")
        if internships == 0:
            recs.append("Apply for at least 1 internship or freelance project immediately")
        elif internships == 1:
            recs.append("Pursue a second internship at an MNC or reputed startup")
        if certs == 0:
            skill_map = {
                "Engineering": "Complete Python + SQL certifications (free via Coursera)",
                "MBA": "Earn a digital marketing or finance analytics certification",
                "Data Science": "Build 2 end-to-end projects on GitHub",
                "Nursing": "Obtain BLS/ACLS certification",
            }
            recs.append(skill_map.get(course, "Add 1-2 relevant skill certifications"))
        if cgpa < 6.5 and risk in ["Medium", "High"]:
            recs.append("Compensate lower CGPA with strong portfolio and project experience")
        if not student_dict.get("linkedin_active", 0):
            recs.append("Activate LinkedIn: connect with 10+ recruiters in target sector")
        if student_dict.get("interviews_attended", 0) < 3:
            recs.append("Enroll in mock interview coaching (minimum 5 mock interviews)")
        if not student_dict.get("resume_updated_recently", 0):
            recs.append("Update resume with latest projects and ATS-optimize it")

        # Add risk-specific advice
        if risk == "High":
            recs.append("Schedule a career counseling session with placement cell immediately")
            recs.append("Broaden job search to adjacent roles and geographies")
        elif risk == "Medium":
            recs.append("Set a 4-week placement sprint with daily application targets")

        return recs[:5]

    def what_if(self, student_dict: dict, changes: dict) -> dict:
        """Simulate impact of changes on placement probability."""
        modified = {**student_dict, **changes}
        original = self.predict(student_dict)
        modified_pred = self.predict(modified)
        return {
            "original": original,
            "modified": modified_pred,
            "delta_3m": round(modified_pred["placement_3m"] - original["placement_3m"], 3),
            "delta_6m": round(modified_pred["placement_6m"] - original["placement_6m"], 3),
            "delta_risk": original["risk"] + " → " + modified_pred["risk"],
        }


if __name__ == "__main__":
    print("Training models...")
    results = train_models()
    print("\nTraining complete:", results)

    print("\nRunning sample prediction...")
    predictor = PlacementPredictor()
    sample = {
        "student_id": "TEST001",
        "course_type": "Engineering",
        "institute_tier": "Tier 2",
        "region": "Bangalore",
        "target_sector": "IT",
        "cgpa": 7.2,
        "academic_consistency_score": 75.0,
        "num_internships": 1,
        "internship_months": 3,
        "internship_employer_type": "Startup",
        "internship_performance": 3.8,
        "num_certifications": 2,
        "certifications": "Python|SQL",
        "institute_placement_rate_3m": 0.45,
        "institute_placement_rate_6m": 0.65,
        "institute_placement_rate_12m": 0.80,
        "institute_avg_salary_lpa": 8.0,
        "placement_cell_score": 0.65,
        "recruiter_diversity_score": 0.55,
        "job_demand_score": 0.72,
        "region_job_density": 0.80,
        "sector_growth_rate": 0.12,
        "job_applications_per_week": 8,
        "interviews_attended": 2,
        "resume_updated_recently": 1,
        "linkedin_active": 1,
        "internship_quality_score": 0.55,
        "skill_relevance_score": 0.60,
        "institute_strength_index": 0.62,
    }
    result = predictor.predict(sample)
    import json
    print(json.dumps(result, indent=2))
