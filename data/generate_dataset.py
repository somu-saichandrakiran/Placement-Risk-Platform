"""
Synthetic Dataset Generator for Placement Risk Intelligence Platform
Generates realistic student, institute, and placement outcome data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

np.random.seed(42)
random.seed(42)

# ─── Constants ───────────────────────────────────────────────────────────────

COURSE_TYPES = ["Engineering", "MBA", "Data Science", "Nursing", "Law", "Commerce", "Arts"]
INSTITUTE_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
SECTORS = ["IT", "BFSI", "Healthcare", "Manufacturing", "Consulting", "FMCG", "Pharma", "EdTech"]
REGIONS = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]
EMPLOYER_TYPES = ["MNC", "Startup", "PSU", "SME", "NGO"]
CERTIFICATIONS = [
    "Python", "SQL", "AWS", "Java", "Machine Learning", "Power BI",
    "CFA", "PMP", "Digital Marketing", "Cybersecurity", "React", "Tableau"
]

SALARY_BENCHMARKS = {
    ("Engineering", "Tier 1"): (12, 30),
    ("Engineering", "Tier 2"): (6, 15),
    ("Engineering", "Tier 3"): (3, 8),
    ("MBA", "Tier 1"): (15, 40),
    ("MBA", "Tier 2"): (8, 20),
    ("MBA", "Tier 3"): (4, 10),
    ("Data Science", "Tier 1"): (14, 35),
    ("Data Science", "Tier 2"): (7, 18),
    ("Data Science", "Tier 3"): (4, 10),
    ("Nursing", "Tier 1"): (4, 8),
    ("Nursing", "Tier 2"): (3, 6),
    ("Nursing", "Tier 3"): (2, 5),
    ("Law", "Tier 1"): (10, 25),
    ("Law", "Tier 2"): (5, 12),
    ("Law", "Tier 3"): (3, 7),
    ("Commerce", "Tier 1"): (6, 15),
    ("Commerce", "Tier 2"): (4, 10),
    ("Commerce", "Tier 3"): (2, 6),
    ("Arts", "Tier 1"): (4, 10),
    ("Arts", "Tier 2"): (3, 7),
    ("Arts", "Tier 3"): (2, 5),
}


def generate_student_data(n=1000):
    records = []
    for i in range(n):
        course = random.choice(COURSE_TYPES)
        tier = random.choice(INSTITUTE_TIERS)
        region = random.choice(REGIONS)
        sector = random.choice(SECTORS)

        # Academic features
        cgpa = round(np.random.beta(7, 3) * 10, 2)  # Skew toward 7-9
        cgpa = min(max(cgpa, 4.0), 10.0)
        academic_consistency = round(np.random.beta(6, 3) * 100, 1)

        # Internship features
        num_internships = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
        internship_months = num_internships * np.random.randint(1, 7) if num_internships > 0 else 0
        internship_employer = random.choice(EMPLOYER_TYPES) if num_internships > 0 else "None"
        internship_performance = round(np.random.uniform(3, 5), 1) if num_internships > 0 else 0.0

        # Skills
        num_certs = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.1, 0.05])
        certs = random.sample(CERTIFICATIONS, num_certs) if num_certs > 0 else []

        # Institute features
        placement_rate_3m = round(np.random.beta(
            5 if tier == "Tier 1" else (3 if tier == "Tier 2" else 2),
            5 if tier == "Tier 3" else (3 if tier == "Tier 2" else 2)
        ), 2)
        placement_rate_6m = min(placement_rate_3m + round(np.random.uniform(0.05, 0.2), 2), 1.0)
        placement_rate_12m = min(placement_rate_6m + round(np.random.uniform(0.05, 0.15), 2), 1.0)

        avg_salary_lpa = round(np.random.uniform(*SALARY_BENCHMARKS.get((course, tier), (3, 8))), 1)
        placement_cell_score = round(np.random.uniform(
            0.7 if tier == "Tier 1" else (0.4 if tier == "Tier 2" else 0.2), 1.0
        ), 2)
        recruiter_diversity = round(np.random.uniform(
            0.6 if tier == "Tier 1" else (0.3 if tier == "Tier 2" else 0.1), 1.0
        ), 2)

        # Industry features
        job_demand_score = round(np.random.uniform(0.3, 1.0), 2)
        region_job_density = round(np.random.uniform(0.2, 1.0), 2)
        sector_growth = round(np.random.uniform(-0.1, 0.3), 2)

        # Behavioral signals
        job_applications_per_week = np.random.randint(0, 30)
        interviews_attended = np.random.randint(0, 15)
        resume_updated = random.choice([True, False])
        linkedin_active = random.choice([True, False])

        # ── Derive composite scores ──
        internship_quality = round(
            (internship_performance / 5.0) * 0.4 +
            (1.0 if internship_employer == "MNC" else 0.7 if internship_employer == "Startup" else 0.4) * 0.4 +
            min(internship_months / 12, 1.0) * 0.2
        , 2)

        skill_score = round(min(num_certs / 4.0, 1.0) * 0.7 + (0.3 if linkedin_active else 0.0), 2)

        institute_strength = round(
            placement_rate_6m * 0.3 +
            placement_cell_score * 0.4 +
            recruiter_diversity * 0.3
        , 2)

        # ── Compute placement probability (ground truth simulation) ──
        base_prob = (
            (cgpa / 10.0) * 0.20 +
            internship_quality * 0.25 +
            institute_strength * 0.20 +
            job_demand_score * 0.15 +
            skill_score * 0.10 +
            (job_applications_per_week / 30.0) * 0.10
        )

        noise = np.random.normal(0, 0.05)
        prob_3m = min(max(base_prob * 0.6 + noise, 0.0), 1.0)
        prob_6m = min(max(base_prob * 0.8 + noise, 0.0), 1.0)
        prob_12m = min(max(base_prob * 1.0 + noise, 0.0), 1.0)

        # ── Outcome labels ──
        placed_3m = 1 if np.random.random() < prob_3m else 0
        placed_6m = 1 if (placed_3m == 1 or np.random.random() < prob_6m) else 0
        placed_12m = 1 if (placed_6m == 1 or np.random.random() < prob_12m) else 0

        # ── Salary outcome ──
        sal_min, sal_max = SALARY_BENCHMARKS.get((course, tier), (3, 8))
        salary_adjustment = (cgpa / 10.0) * 0.3 + internship_quality * 0.3 + skill_score * 0.2
        predicted_salary = round(sal_min + (sal_max - sal_min) * salary_adjustment + np.random.normal(0, 0.5), 1)
        predicted_salary = max(sal_min * 0.7, min(sal_max * 1.2, predicted_salary))

        records.append({
            "student_id": f"STU{i+1:04d}",
            "course_type": course,
            "institute_tier": tier,
            "region": region,
            "target_sector": sector,
            "cgpa": cgpa,
            "academic_consistency_score": academic_consistency,
            "num_internships": num_internships,
            "internship_months": internship_months,
            "internship_employer_type": internship_employer,
            "internship_performance": internship_performance,
            "num_certifications": num_certs,
            "certifications": "|".join(certs) if certs else "None",
            "institute_placement_rate_3m": placement_rate_3m,
            "institute_placement_rate_6m": placement_rate_6m,
            "institute_placement_rate_12m": placement_rate_12m,
            "institute_avg_salary_lpa": avg_salary_lpa,
            "placement_cell_score": placement_cell_score,
            "recruiter_diversity_score": recruiter_diversity,
            "job_demand_score": job_demand_score,
            "region_job_density": region_job_density,
            "sector_growth_rate": sector_growth,
            "job_applications_per_week": job_applications_per_week,
            "interviews_attended": interviews_attended,
            "resume_updated_recently": int(resume_updated),
            "linkedin_active": int(linkedin_active),
            # Composite scores
            "internship_quality_score": internship_quality,
            "skill_relevance_score": skill_score,
            "institute_strength_index": institute_strength,
            # Outcome labels
            "placed_3m": placed_3m,
            "placed_6m": placed_6m,
            "placed_12m": placed_12m,
            "actual_salary_lpa": round(predicted_salary, 1),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset = generate_student_data(1000)
    df.to_csv("data/students.csv", index=False)
    print(f"Dataset generated: {len(df)} rows, {len(df.columns)} columns")
    print(df.head(3).to_string())
    print("\nPlacement rates:")
    print(f"  3m: {df['placed_3m'].mean():.1%}")
    print(f"  6m: {df['placed_6m'].mean():.1%}")
    print(f"  12m: {df['placed_12m'].mean():.1%}")
