# 🎓 Placement Risk Intelligence & Intervention Platform

> AI-powered employability prediction and intervention system for education loan lenders

---

## 📌 Overview

This system predicts student placement probability at 3, 6, and 12 months post-graduation, estimates expected salary ranges, classifies placement risk (Low / Medium / High), explains *why* risk exists using interpretable AI, and recommends personalized interventions — giving lenders early visibility into loan repayment risk.

**It is NOT a simple ML model.** It is a complete, modular, explainable decision-support system built for real-world fintech use.

---

## 🗂 Project Structure

```
placement_risk_platform/
│
├── data/
│   ├── generate_dataset.py        # Synthetic data generator (1000 students)
│   └── students.csv               # Generated training dataset
│
├── backend/
│   ├── ml/
│   │   ├── pipeline.py            # Feature engineering + model training + inference
│   │   └── models/                # Saved model artifacts (.pkl)
│   │       ├── model_placed_3m.pkl
│   │       ├── model_placed_6m.pkl
│   │       ├── model_placed_12m.pkl
│   │       ├── model_salary.pkl
│   │       ├── feature_cols.pkl
│   │       └── label_encoders.pkl
│   └── api/
│       └── main.py                # FastAPI REST API
│
├── frontend/
│   └── app.py                     # Streamlit multi-page UI
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic dataset

```bash
python data/generate_dataset.py
```

### 3. Train ML models

```bash
python backend/ml/pipeline.py
```

This will:
- Engineer features (internship quality, skill score, market fit, etc.)
- Train 4 models: 3 classifiers (3m/6m/12m placement) + 1 salary regressor
- Save all models to `backend/ml/models/`

### 4. Start the FastAPI backend

```bash
cd backend/api
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 5. Launch the Streamlit frontend

```bash
cd frontend
streamlit run app.py
```

---

## 🔌 REST API Reference

### POST /api/student
Register a student and get immediate risk prediction.

**Request body (JSON):**
```json
{
  "name": "Rahul Sharma",
  "course_type": "Engineering",
  "institute_tier": "Tier 2",
  "region": "Bangalore",
  "target_sector": "IT",
  "cgpa": 7.2,
  "num_internships": 1,
  "num_certifications": 2,
  "job_applications_per_week": 8,
  "loan_amount_lpa": 10.0,
  ...
}
```

**Response:**
```json
{
  "student_id": "STU-ABC123",
  "prediction": {
    "placement_3m": 0.42,
    "placement_6m": 0.72,
    "placement_12m": 0.91,
    "risk": "Medium",
    "risk_score": 0.28,
    "salary_range": "7.7-10.5 LPA",
    "risk_drivers": ["Low job application activity", "No internship experience"],
    "protective_factors": ["Strong CGPA (7.2)", "Regional Job Availability"],
    "recommendations": [
      "Apply to at least 10 jobs/week",
      "Complete Python + SQL certifications"
    ]
  },
  "loan_risk": {
    "repayment_probability": 0.73,
    "emi_affordability": "Moderate",
    "lender_risk_flag": false,
    "suggested_action": "Schedule 3-month check-in"
  }
}
```

### GET /api/prediction/{student_id}
Retrieve latest prediction for a registered student.

### GET /api/portfolio
Portfolio-level risk overview for lenders.

### POST /api/whatif
Simulate impact of student improvements.

**Request:**
```json
{
  "student_id": "STU-ABC123",
  "changes": {
    "num_internships": 2,
    "num_certifications": 4,
    "job_applications_per_week": 20
  }
}
```

### POST /api/monitor
Update behavioral signals and refresh risk.

### GET /api/monitor/alerts
Get all deterioration alerts.

---

## 🤖 ML Architecture

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `internship_quality_score` | Weighted: employer type (MNC>Startup>SME) × performance × duration |
| `skill_relevance_score` | Certifications count + LinkedIn activity |
| `institute_strength_index` | Placement rate + cell quality + recruiter diversity |
| `market_fit_score` | Job demand + regional density + sector growth |
| `placement_readiness` | Composite: CGPA + internship + skills + engagement + institute |
| `engagement_score` | Applications + resume update + LinkedIn activity |
| `interview_conversion` | Interviews / applications ratio |

### Models
- **Classification (3m/6m/12m):** GradientBoostingClassifier (200 estimators, depth 4)
  - Production upgrade: replace with LightGBM for 3-5× speed
- **Regression (salary):** GradientBoostingRegressor
- **Explainability:** Feature importance scores + contextual rule-based enrichment

### Risk Scoring
```
risk_score = 1 - placement_probability_6m

Low:    risk_score < 0.30  (>70% chance of placement by 6m)
Medium: risk_score < 0.60  (40-70% chance)
High:   risk_score >= 0.60 (<40% chance)
```

### Output Schema
```json
{
  "placement_3m": 0.40,
  "placement_6m": 0.70,
  "placement_12m": 0.90,
  "risk": "Medium",
  "salary_range": "5-7 LPA",
  "risk_drivers": ["Low internship exposure", "Weak industry demand"],
  "recommendations": ["Apply to 20 jobs/week", "Learn SQL"]
}
```

---

## 📊 Sample Dataset Features

1. **Student Profile:** Course, CGPA, Internships, Certifications
2. **Institute Data:** Tier, Placement rates, Cell quality, Recruiter diversity
3. **Market Data:** Job demand, Regional density, Sector growth
4. **Behavioral Signals:** Applications/week, Interviews, LinkedIn, Resume
5. **Outcome Labels:** Placed at 3/6/12 months, Actual salary

---

## 🏗 Production Upgrade Roadmap

| Component | Current | Production |
|-----------|---------|------------|
| ML Framework | scikit-learn GBM | LightGBM / XGBoost |
| Explainability | Feature importance | SHAP TreeExplainer |
| Database | In-memory dict | PostgreSQL |
| Monitoring | Manual API calls | Apache Kafka + real-time triggers |
| Auth | None | JWT + role-based (Student / Institute / Lender) |
| Deployment | Local | Docker + Kubernetes |
| Retraining | Manual | MLflow + automated weekly retraining |

---

## 🔐 Security Notes (Production)
- All endpoints should require JWT authentication
- Student data must comply with DPDP Act (India) / GDPR
- Model outputs are decision-support only — NOT automated credit decisions
- Audit logs for all lender access to student risk data

---

## 📈 Judging Criteria Mapping

| Criteria | Implementation |
|----------|---------------|
| Accuracy of 3/6/12m predictions | Separate GBM classifiers per horizon |
| Clarity & explainability | Feature importance → human-readable drivers |
| Usefulness for lenders | Portfolio view + repayment probability + EMI affordability |
| Scalability | Modular pipeline, stateless API, PostgreSQL-ready |
| Impact potential | Early alerts reduce delinquency; interventions improve outcomes |
| Robustness | Handles varied courses, tiers, regions, and behavioral patterns |
"# Placement-Risk-Platform" 
