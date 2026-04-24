"""
Placement Risk Intelligence Platform - FastAPI Backend
Production-grade REST API for lenders and students
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List

# ── Handle imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import pandas as pd
import numpy as np

# Import ML pipeline
ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
sys.path.insert(0, ML_DIR)
from pipeline import PlacementPredictor, engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── In-memory storage (replace with PostgreSQL in production) ───────────────
STUDENT_DB: dict = {}
INSTITUTE_DB: dict = {}
MONITORING_LOG: list = []

# ─── Load predictor ──────────────────────────────────────────────────────────
predictor = PlacementPredictor()
logger.info("ML models loaded successfully")


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    student_id: Optional[str] = None
    name: Optional[str] = "Anonymous"
    course_type: str = Field(..., example="Engineering")
    institute_tier: str = Field(..., example="Tier 2")
    region: str = Field(..., example="Bangalore")
    target_sector: str = Field(..., example="IT")
    cgpa: float = Field(..., ge=0, le=10)
    academic_consistency_score: float = Field(75.0, ge=0, le=100)
    num_internships: int = Field(0, ge=0)
    internship_months: int = Field(0, ge=0)
    internship_employer_type: str = Field("None")
    internship_performance: float = Field(0.0, ge=0, le=5)
    num_certifications: int = Field(0, ge=0)
    certifications: str = Field("None")
    institute_placement_rate_3m: float = Field(0.5, ge=0, le=1)
    institute_placement_rate_6m: float = Field(0.7, ge=0, le=1)
    institute_placement_rate_12m: float = Field(0.85, ge=0, le=1)
    institute_avg_salary_lpa: float = Field(6.0, ge=0)
    placement_cell_score: float = Field(0.6, ge=0, le=1)
    recruiter_diversity_score: float = Field(0.5, ge=0, le=1)
    job_demand_score: float = Field(0.6, ge=0, le=1)
    region_job_density: float = Field(0.5, ge=0, le=1)
    sector_growth_rate: float = Field(0.1, ge=-0.5, le=1)
    job_applications_per_week: int = Field(0, ge=0)
    interviews_attended: int = Field(0, ge=0)
    resume_updated_recently: int = Field(0, ge=0, le=1)
    linkedin_active: int = Field(0, ge=0, le=1)
    internship_quality_score: float = Field(0.3, ge=0, le=1)
    skill_relevance_score: float = Field(0.4, ge=0, le=1)
    institute_strength_index: float = Field(0.5, ge=0, le=1)
    loan_amount_lpa: Optional[float] = None


class InstituteInput(BaseModel):
    institute_id: Optional[str] = None
    institute_name: str
    tier: str
    location: str
    total_students: int
    placement_rate_3m: float
    placement_rate_6m: float
    placement_rate_12m: float
    avg_salary_lpa: float
    placement_cell_score: float
    recruiter_diversity: float
    courses_offered: List[str]


class WhatIfInput(BaseModel):
    student_id: str
    changes: dict


class MonitoringUpdate(BaseModel):
    student_id: str
    job_applications_per_week: Optional[int] = None
    interviews_attended: Optional[int] = None
    resume_updated_recently: Optional[int] = None
    linkedin_active: Optional[int] = None
    num_certifications: Optional[int] = None
    num_internships: Optional[int] = None


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Placement Risk Intelligence Platform",
    description="AI-powered placement risk prediction for education loan lenders",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def compute_risk_color(risk: str) -> str:
    return {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}.get(risk, "#7f8c8d")


def compute_loan_risk(prediction: dict, loan_amount: float = None) -> dict:
    """Estimate repayment risk for the lender."""
    prob_6m = prediction["placement_6m"]
    salary_mid = prediction["salary_mid"]
    risk = prediction["risk"]

    repayment_probability = round(prob_6m * 0.9 + (1 - prob_6m) * 0.4, 3)
    emi_affordability = "Comfortable" if salary_mid > 8 else ("Moderate" if salary_mid > 5 else "Stressed")

    result = {
        "repayment_probability": repayment_probability,
        "emi_affordability": emi_affordability,
        "lender_risk_flag": risk in ["High"],
        "suggested_action": (
            "Monitor closely, trigger intervention" if risk == "High"
            else "Schedule 3-month check-in" if risk == "Medium"
            else "Standard monitoring"
        )
    }
    if loan_amount:
        expected_emi = round(loan_amount * 0.09 / 12, 2)
        result["expected_monthly_emi_lpa"] = expected_emi
        result["emi_to_salary_ratio"] = round(expected_emi / (salary_mid / 12), 2) if salary_mid > 0 else None

    return result


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Placement Risk Intelligence Platform",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/api/student", "/api/institute", "/api/prediction/{id}",
                      "/api/portfolio", "/api/whatif", "/api/monitor"]
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "models_loaded": True}


@app.post("/api/student", tags=["Student"])
def register_student(data: StudentInput):
    """Register a new student and get immediate risk prediction."""
    if not data.student_id:
        data.student_id = f"STU-{str(uuid.uuid4())[:8].upper()}"

    student_dict = data.model_dump()

    # Run prediction
    prediction = predictor.predict(student_dict)

    # Augment with loan risk
    loan_risk = compute_loan_risk(prediction, data.loan_amount_lpa)

    # Store
    record = {
        "student": student_dict,
        "prediction": prediction,
        "loan_risk": loan_risk,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    STUDENT_DB[data.student_id] = record

    return {
        "student_id": data.student_id,
        "status": "registered",
        "prediction": prediction,
        "loan_risk": loan_risk,
    }


@app.get("/api/prediction/{student_id}", tags=["Prediction"])
def get_prediction(student_id: str):
    """Get latest prediction for a registered student."""
    if student_id not in STUDENT_DB:
        raise HTTPException(404, f"Student {student_id} not found")

    record = STUDENT_DB[student_id]
    return {
        "student_id": student_id,
        "name": record["student"].get("name", "Anonymous"),
        "prediction": record["prediction"],
        "loan_risk": record["loan_risk"],
        "last_updated": record["updated_at"],
    }


@app.get("/api/students", tags=["Student"])
def list_students(
    risk: Optional[str] = Query(None, description="Filter by risk: Low/Medium/High"),
    course: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """List all registered students with optional filters."""
    students = []
    for sid, rec in list(STUDENT_DB.items())[:limit]:
        pred = rec["prediction"]
        stu = rec["student"]
        if risk and pred["risk"] != risk:
            continue
        if course and stu.get("course_type") != course:
            continue
        students.append({
            "student_id": sid,
            "name": stu.get("name", "Anonymous"),
            "course_type": stu.get("course_type"),
            "institute_tier": stu.get("institute_tier"),
            "risk": pred["risk"],
            "placement_6m": pred["placement_6m"],
            "salary_range": pred["salary_range"],
        })
    return {"total": len(students), "students": students}


@app.post("/api/institute", tags=["Institute"])
def register_institute(data: InstituteInput):
    """Register an institute and compute cohort-level stats."""
    if not data.institute_id:
        data.institute_id = f"INST-{str(uuid.uuid4())[:8].upper()}"

    INSTITUTE_DB[data.institute_id] = data.model_dump()

    return {
        "institute_id": data.institute_id,
        "status": "registered",
        "summary": {
            "tier": data.tier,
            "placement_rate_6m": data.placement_rate_6m,
            "avg_salary_lpa": data.avg_salary_lpa,
            "strength_index": round(
                data.placement_rate_6m * 0.4 +
                data.placement_cell_score * 0.3 +
                data.recruiter_diversity * 0.3, 3
            )
        }
    }


@app.get("/api/portfolio", tags=["Lender"])
def get_portfolio_summary():
    """Portfolio-level risk overview for lenders."""
    if not STUDENT_DB:
        return {"message": "No students registered yet", "total": 0}

    preds = [rec["prediction"] for rec in STUDENT_DB.values()]
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    for p in preds:
        risk_counts[p["risk"]] += 1

    total = len(preds)
    avg_prob_6m = round(sum(p["placement_6m"] for p in preds) / total, 3)
    high_risk_students = [
        {"student_id": sid, "risk": rec["prediction"]["risk"],
         "placement_6m": rec["prediction"]["placement_6m"],
         "recommendations": rec["prediction"]["recommendations"][:2]}
        for sid, rec in STUDENT_DB.items()
        if rec["prediction"]["risk"] == "High"
    ]

    return {
        "portfolio_summary": {
            "total_students": total,
            "risk_distribution": risk_counts,
            "risk_distribution_pct": {k: round(v/total*100, 1) for k, v in risk_counts.items()},
            "avg_placement_probability_6m": avg_prob_6m,
            "portfolio_health": "Healthy" if risk_counts["High"]/total < 0.15 else "At Risk",
        },
        "high_risk_alerts": high_risk_students[:10],
        "intervention_priority_count": len(high_risk_students),
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.post("/api/whatif", tags=["Simulation"])
def what_if_simulation(data: WhatIfInput):
    """Simulate what happens if a student makes specific improvements."""
    if data.student_id not in STUDENT_DB:
        raise HTTPException(404, f"Student {data.student_id} not found")

    student_dict = STUDENT_DB[data.student_id]["student"]
    result = predictor.what_if(student_dict, data.changes)

    return {
        "student_id": data.student_id,
        "simulation_result": result,
        "insight": f"Making these changes could move risk from {result['delta_risk']} "
                   f"and improve 6-month placement by {result['delta_6m']*100:+.1f}%"
    }


@app.post("/api/monitor", tags=["Monitoring"])
def update_monitoring(data: MonitoringUpdate):
    """Update student behavioral signals and refresh risk prediction."""
    if data.student_id not in STUDENT_DB:
        raise HTTPException(404, f"Student {data.student_id} not found")

    record = STUDENT_DB[data.student_id]
    old_risk = record["prediction"]["risk"]
    old_pred = record["prediction"].copy()

    # Apply updates
    updates = {k: v for k, v in data.model_dump().items() if v is not None and k != "student_id"}
    record["student"].update(updates)

    # Recompute
    new_prediction = predictor.predict(record["student"])
    record["prediction"] = new_prediction
    record["updated_at"] = datetime.utcnow().isoformat()

    # Log change
    MONITORING_LOG.append({
        "student_id": data.student_id,
        "timestamp": datetime.utcnow().isoformat(),
        "old_risk": old_risk,
        "new_risk": new_prediction["risk"],
        "old_prob_6m": old_pred["placement_6m"],
        "new_prob_6m": new_prediction["placement_6m"],
    })

    return {
        "student_id": data.student_id,
        "status": "updated",
        "risk_change": f"{old_risk} → {new_prediction['risk']}",
        "placement_6m_change": round(new_prediction["placement_6m"] - old_pred["placement_6m"], 3),
        "new_prediction": new_prediction,
    }


@app.get("/api/monitor/alerts", tags=["Monitoring"])
def get_alerts():
    """Get all recent risk change alerts."""
    deteriorated = [
        log for log in MONITORING_LOG
        if log["new_risk"] in ["High"] or log["new_prob_6m"] < log["old_prob_6m"] - 0.1
    ]
    return {
        "total_updates": len(MONITORING_LOG),
        "deterioration_alerts": len(deteriorated),
        "recent_alerts": deteriorated[-20:],
    }


@app.get("/api/sample-prediction", tags=["Demo"])
def sample_prediction():
    """Returns a demo prediction for quick testing."""
    sample = {
        "student_id": "DEMO001",
        "name": "Demo Student",
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
    prediction = predictor.predict(sample)
    loan_risk = compute_loan_risk(prediction, loan_amount=10.0)
    return {"student": sample, "prediction": prediction, "loan_risk": loan_risk}


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except ImportError:
        print("uvicorn not available. API defined successfully.")
        print("Install with: pip install fastapi uvicorn")
