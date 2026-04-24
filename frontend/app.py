"""
Placement Risk Intelligence Platform
Professional redesign — clean, minimal, fintech-grade UI
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend/ml"))

import pandas as pd
import numpy as np
import streamlit as st
from pipeline import PlacementPredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlaceIQ — Placement Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return PlacementPredictor()

predictor = load_predictor()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar branding ── */
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.15); }

/* ── Stat cards ── */
.stat-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 2px;
}
.stat-label { font-size: 12px; font-weight: 600; letter-spacing: 0.06em;
              text-transform: uppercase; opacity: 0.55; margin-bottom: 4px; }
.stat-value { font-size: 26px; font-weight: 700; line-height: 1.1; }
.stat-sub   { font-size: 12px; opacity: 0.5; margin-top: 3px; }

/* ── Risk pill ── */
.pill-Low    { display:inline-block; background:#16a34a; color:#fff;
               padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600; }
.pill-Medium { display:inline-block; background:#d97706; color:#fff;
               padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600; }
.pill-High   { display:inline-block; background:#dc2626; color:#fff;
               padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600; }

/* ── Section header ── */
.section-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.45; margin: 18px 0 8px;
}

/* ── List items ── */
.item-risk  { border-left: 3px solid #dc2626; padding: 9px 13px; margin: 5px 0;
              border-radius: 0 6px 6px 0; background: rgba(220,38,38,0.08);
              font-size: 13.5px; color: inherit; }
.item-good  { border-left: 3px solid #16a34a; padding: 9px 13px; margin: 5px 0;
              border-radius: 0 6px 6px 0; background: rgba(22,163,74,0.08);
              font-size: 13.5px; color: inherit; }
.item-rec   { border-left: 3px solid #2563eb; padding: 9px 13px; margin: 5px 0;
              border-radius: 0 6px 6px 0; background: rgba(37,99,235,0.08);
              font-size: 13.5px; color: inherit; }

/* ── Step badge ── */
.step-badge {
    display: inline-block; width: 24px; height: 24px; line-height: 24px;
    border-radius: 50%; background: rgba(37,99,235,0.15);
    color: #2563eb; font-size: 12px; font-weight: 700;
    text-align: center; margin-right: 8px;
}

/* ── Alert card ── */
.alert-card {
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    background: rgba(220,38,38,0.05);
}
.alert-name  { font-size: 14px; font-weight: 600; }
.alert-meta  { font-size: 12px; opacity: 0.6; margin-top: 2px; }
.alert-badge { float: right; background: #dc2626; color: #fff;
               padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }

/* ── Home feature tiles ── */
.feat-tile {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 10px; padding: 20px 18px;
    height: 100%; margin-bottom: 4px;
}
.feat-icon  { font-size: 22px; margin-bottom: 10px; }
.feat-title { font-size: 14px; font-weight: 700; margin-bottom: 5px; }
.feat-desc  { font-size: 13px; opacity: 0.65; line-height: 1.55; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid rgba(128,128,128,0.15); margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = []
loan_amount = 0.0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 PlaceIQ")
    st.caption("Placement Risk Intelligence Platform")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Overview",
         "👤  Student Analysis",
         "🏦  Lender Dashboard",
         "🔬  What-If Simulator",
         "📊  Portfolio Analytics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("v1.0  ·  GradientBoosting ML")
    st.caption("For education loan lenders")


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("## Placement Risk Intelligence Platform")
    st.markdown(
        "An AI-powered decision-support system that helps education loan lenders "
        "identify students at risk of delayed placement — before it becomes a repayment problem."
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Feature tiles
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("🎯", "Placement Forecast",
         "Predicts probability of job placement at 3, 6, and 12 months after graduation."),
        ("💰", "Salary Estimate",
         "Estimates expected starting salary range based on course, institute, and skills."),
        ("🔍", "Risk Explanation",
         "Identifies the exact factors driving high or low placement risk — no black box."),
        ("💡", "Intervention Engine",
         "Generates personalized action items to improve each student's placement outcome."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], tiles):
        with col:
            st.markdown(
                f"<div class='feat-tile'>"
                f"<div class='feat-icon'>{icon}</div>"
                f"<div class='feat-title'>{title}</div>"
                f"<div class='feat-desc'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # How it works
    st.markdown("#### How it works")
    col_a, col_b = st.columns(2)
    with col_a:
        steps = [
            ("Enter student data", "Academic profile, internship history, certifications, and job search activity."),
            ("Run AI analysis",    "GradientBoosting models score placement probability across three time horizons."),
            ("Review risk report", "See risk level, top drivers, protective factors, and salary estimate."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(
                f"<span class='step-badge'>{i}</span>"
                f"<strong>{title}</strong><br>"
                f"<span style='font-size:13px;opacity:0.65;padding-left:32px'>{desc}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")

    with col_b:
        steps2 = [
            ("Get lender insights",  "Repayment probability, EMI affordability, and suggested action for the loan officer."),
            ("Simulate improvements","Use the What-If Simulator to model the impact of adding internships or certifications."),
            ("Monitor portfolio",    "Track all students in the Lender Dashboard with risk distribution and alerts."),
        ]
        for i, (title, desc) in enumerate(steps2, 4):
            st.markdown(
                f"<span class='step-badge'>{i}</span>"
                f"<strong>{title}</strong><br>"
                f"<span style='font-size:13px;opacity:0.65;padding-left:32px'>{desc}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.info("👤 Start by going to **Student Analysis** in the sidebar to analyze your first student.")


# ═══════════════════════════════════════════════════════════════════════════════
#  STUDENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Student Analysis":
    st.markdown("## Student Placement Risk Analysis")
    st.caption("Fill in the student profile below and click Analyze to generate a risk report.")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.form("student_form", border=False):

        # ── Section 1 ──
        st.markdown("<div class='section-hdr'>Academic Profile</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: name        = st.text_input("Student Name", "Rahul Sharma")
        with c2: student_id  = st.text_input("Student ID",   "STU-001")
        with c3: course_type = st.selectbox("Course", ["Engineering","MBA","Data Science","Nursing","Law","Commerce","Arts"])
        with c4: cgpa        = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.2, 0.1)

        c5, c6, c7, c8 = st.columns(4)
        with c5: institute_tier       = st.selectbox("Institute Tier", ["Tier 1","Tier 2","Tier 3"])
        with c6: region               = st.selectbox("Region", ["Bangalore","Mumbai","Delhi","Hyderabad","Pune","Chennai","Kolkata","Ahmedabad"])
        with c7: academic_consistency = st.slider("Academic Consistency (%)", 0, 100, 75)
        with c8: target_sector        = st.selectbox("Target Sector", ["IT","BFSI","Healthcare","Manufacturing","Consulting","FMCG","Pharma","EdTech"])

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 2 ──
        st.markdown("<div class='section-hdr'>Internship & Skills</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: num_internships        = st.number_input("No. of Internships", 0, 5, 1)
        with c2: internship_months      = st.number_input("Internship Duration (months)", 0, 24, 3)
        with c3: internship_employer    = st.selectbox("Employer Type", ["None","MNC","Startup","PSU","SME","NGO"])
        with c4: internship_performance = st.select_slider("Internship Rating", [0.0,1.0,2.0,3.0,3.5,4.0,4.5,5.0], value=3.5)

        c5, c6, c7, c8 = st.columns(4)
        with c5: num_certifications = st.number_input("No. of Certifications", 0, 10, 2)
        with c6: certifications     = st.text_input("Certifications (comma-separated)", "Python, SQL")
        with c7: resume_updated     = st.selectbox("Resume Updated?", ["Yes","No"])
        with c8: linkedin_active    = st.selectbox("LinkedIn Active?", ["Yes","No"])

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 3 ──
        st.markdown("<div class='section-hdr'>Institute & Market</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: placement_rate_6m  = st.slider("Institute Placement Rate (6m)", 0.0, 1.0, 0.65, 0.01)
        with c2: placement_cell     = st.slider("Placement Cell Quality",        0.0, 1.0, 0.65, 0.01)
        with c3: recruiter_diversity= st.slider("Recruiter Diversity",           0.0, 1.0, 0.55, 0.01)
        with c4: institute_salary   = st.number_input("Institute Avg Salary (LPA)", 1.0, 50.0, 8.0, 0.5)

        c5, c6, c7, c8 = st.columns(4)
        with c5: job_demand      = st.slider("Industry Job Demand",  0.0, 1.0, 0.72, 0.01)
        with c6: region_density  = st.slider("Regional Job Density", 0.0, 1.0, 0.80, 0.01)
        with c7: sector_growth   = st.slider("Sector Growth Rate",  -0.10, 0.30, 0.12, 0.01)
        with c8: loan_amount_in  = st.number_input("Loan Amount (LPA)", 0.0, 100.0, 10.0, 0.5)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Section 4 ──
        st.markdown("<div class='section-hdr'>Job Search Activity</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: applications_week = st.number_input("Job Applications per Week", 0, 50, 8)
        with c2: interviews_done   = st.number_input("Interviews Attended",       0, 30, 2)

        st.markdown("")
        submitted = st.form_submit_button(
            "  🔍  Analyze Placement Risk  ",
            type="primary",
            use_container_width=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    if submitted:
        loan_amount = loan_amount_in
        res_updated  = 1 if resume_updated  == "Yes" else 0
        lin_active   = 1 if linkedin_active == "Yes" else 0

        inq = 0.0
        if num_internships > 0:
            emp = {"MNC":1.0,"Startup":0.7,"PSU":0.6,"SME":0.4,"NGO":0.3}.get(internship_employer, 0.4)
            inq = round((internship_performance/5.0)*0.4 + emp*0.4 + min(internship_months/12,1.0)*0.2, 2)

        skill_sc  = round(min(num_certifications/4.0,1.0)*0.7 + (0.3 if lin_active else 0.0), 2)
        inst_str  = round(placement_rate_6m*0.3 + placement_cell*0.4 + recruiter_diversity*0.3, 2)

        student_data = {
            "student_id": student_id, "name": name,
            "course_type": course_type, "institute_tier": institute_tier,
            "region": region, "target_sector": target_sector,
            "cgpa": cgpa, "academic_consistency_score": float(academic_consistency),
            "num_internships": int(num_internships), "internship_months": int(internship_months),
            "internship_employer_type": internship_employer,
            "internship_performance": float(internship_performance),
            "num_certifications": int(num_certifications),
            "certifications": certifications.replace(", ","|"),
            "institute_placement_rate_3m":  max(placement_rate_6m - 0.15, 0),
            "institute_placement_rate_6m":  placement_rate_6m,
            "institute_placement_rate_12m": min(placement_rate_6m + 0.1, 1.0),
            "institute_avg_salary_lpa": institute_salary,
            "placement_cell_score": placement_cell,
            "recruiter_diversity_score": recruiter_diversity,
            "job_demand_score": job_demand,
            "region_job_density": region_density,
            "sector_growth_rate": sector_growth,
            "job_applications_per_week": int(applications_week),
            "interviews_attended": int(interviews_done),
            "resume_updated_recently": res_updated,
            "linkedin_active": lin_active,
            "internship_quality_score": inq,
            "skill_relevance_score": skill_sc,
            "institute_strength_index": inst_str,
        }

        with st.spinner("Running AI analysis…"):
            pred = predictor.predict(student_data)

        st.session_state.predictions.append({"student": student_data, "prediction": pred})

        risk = pred["risk"]
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Header row ──
        hc1, hc2 = st.columns([3, 1])
        with hc1:
            st.markdown(f"### Risk Report — {name}")
            st.markdown(
                f"<span class='pill-{risk}'>{risk} Risk</span>"
                f"&nbsp;&nbsp;<span style='font-size:13px;opacity:0.55'>{course_type} · {institute_tier} · {region}</span>",
                unsafe_allow_html=True,
            )
        with hc2:
            repay = round(pred["placement_6m"]*0.9 + (1-pred["placement_6m"])*0.4, 2)
            emi_lbl = "Comfortable" if pred["salary_mid"]>8 else ("Moderate" if pred["salary_mid"]>5 else "Stressed")
            st.markdown(
                f"<div class='stat-card' style='text-align:center'>"
                f"<div class='stat-label'>Repayment probability</div>"
                f"<div class='stat-value'>{repay*100:.0f}%</div>"
                f"<div class='stat-sub'>EMI: {emi_lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Placement metrics ──
        m1, m2, m3, m4 = st.columns(4)
        for col, label, val in [
            (m1, "Placement — 3 months",  f"{pred['placement_3m']*100:.0f}%"),
            (m2, "Placement — 6 months",  f"{pred['placement_6m']*100:.0f}%"),
            (m3, "Placement — 12 months", f"{pred['placement_12m']*100:.0f}%"),
            (m4, "Expected Salary",        pred["salary_range"]),
        ]:
            with col:
                st.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-label'>{label}</div>"
                    f"<div class='stat-value'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # ── Placement timeline chart ──
        prob_df = pd.DataFrame({
            "Month": ["3 months", "6 months", "12 months"],
            "Placement probability (%)": [
                round(pred["placement_3m"]*100, 1),
                round(pred["placement_6m"]*100, 1),
                round(pred["placement_12m"]*100, 1),
            ]
        })
        st.markdown("**Placement probability timeline**")
        st.bar_chart(prob_df.set_index("Month"), height=220)

        st.markdown("")

        # ── Risk drivers / Protective / Recommendations ──
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.markdown("**Risk drivers**")
            if pred["risk_drivers"]:
                for d in pred["risk_drivers"]:
                    st.markdown(f"<div class='item-risk'>⚠ {d}</div>", unsafe_allow_html=True)
            else:
                st.success("No major risk drivers found")

        with rc2:
            st.markdown("**Protective factors**")
            if pred["protective_factors"]:
                for f in pred["protective_factors"]:
                    st.markdown(f"<div class='item-good'>✓ {f}</div>", unsafe_allow_html=True)
            else:
                st.info("No strong protective factors detected")

        with rc3:
            st.markdown("**Recommended actions**")
            for i, r in enumerate(pred["recommendations"], 1):
                st.markdown(f"<div class='item-rec'><strong>{i}.</strong> {r}</div>", unsafe_allow_html=True)

        st.markdown("")

        # ── Lender action banner ──
        action_map = {
            "High":   ("🚨", "Trigger immediate counseling intervention",  "#dc2626"),
            "Medium": ("⚠️", "Schedule a 3-month placement check-in",      "#d97706"),
            "Low":    ("✅", "Standard monitoring — student is on track",  "#16a34a"),
        }
        icon, action_txt, color = action_map[risk]
        expected_emi = round(loan_amount * 0.09 / 12, 2) if loan_amount > 0 else 0
        emi_str = f"  ·  Est. monthly EMI: ₹{expected_emi:.1f}L" if expected_emi > 0 else ""
        st.markdown(
            f"<div style='border:1px solid {color}40; border-radius:8px; padding:14px 18px;"
            f"background:{color}10; margin-top:4px;'>"
            f"<strong>{icon} Lender Action:</strong> {action_txt}{emi_str}"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Raw model output (JSON)"):
            st.json(pred)


# ═══════════════════════════════════════════════════════════════════════════════
#  LENDER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏦  Lender Dashboard":
    st.markdown("## Lender Portfolio Dashboard")
    st.caption("Portfolio-level placement risk overview across all analyzed students.")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if not st.session_state.predictions:
        st.info("No students analyzed yet. Go to **Student Analysis** to add students to the portfolio.")
    else:
        records  = st.session_state.predictions
        risks    = [r["prediction"]["risk"] for r in records]
        probs    = [r["prediction"]["placement_6m"] for r in records]
        salaries = [r["prediction"]["salary_mid"] for r in records]
        rc = {"Low": risks.count("Low"), "Medium": risks.count("Medium"), "High": risks.count("High")}
        total = len(records)
        health = "Healthy" if rc["High"]/total < 0.15 else "Needs Attention"

        # ── Top KPIs ──
        k1, k2, k3, k4, k5 = st.columns(5)
        kpis = [
            ("Total students",        str(total),                        "In portfolio"),
            ("High risk",             str(rc["High"]),                   f"{rc['High']/total*100:.0f}% of portfolio"),
            ("Medium risk",           str(rc["Medium"]),                 f"{rc['Medium']/total*100:.0f}% of portfolio"),
            ("Avg placement (6m)",    f"{np.mean(probs)*100:.0f}%",      "Across all students"),
            ("Portfolio health",      health,                            f"Avg salary {np.mean(salaries):.1f} LPA"),
        ]
        for col, (lbl, val, sub) in zip([k1,k2,k3,k4,k5], kpis):
            with col:
                color = ""
                if lbl == "High risk" and rc["High"] > 0:   color = "color:#dc2626;"
                if lbl == "Portfolio health" and health != "Healthy": color = "color:#d97706;"
                if lbl == "Portfolio health" and health == "Healthy": color = "color:#16a34a;"
                st.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-label'>{lbl}</div>"
                    f"<div class='stat-value' style='{color}'>{val}</div>"
                    f"<div class='stat-sub'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # ── Charts ──
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Risk distribution**")
            risk_df = pd.DataFrame({"Risk Level": ["Low","Medium","High"],
                                    "Students": [rc["Low"], rc["Medium"], rc["High"]]})
            st.bar_chart(risk_df.set_index("Risk Level"), height=220)

        with ch2:
            st.markdown("**Placement probability spread (6m)**")
            prob_series = pd.Series(probs)
            prob_bins = pd.cut(prob_series, bins=5).value_counts().sort_index()
            prob_bins.index = prob_bins.index.astype(str)
            st.bar_chart(prob_bins, height=220)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── High risk alerts ──
        high = [(r["student"].get("name","?"), r["student"].get("student_id","?"),
                 r["student"].get("course_type","?"), r["student"].get("institute_tier","?"),
                 r["prediction"]) for r in records if r["prediction"]["risk"] == "High"]

        if high:
            st.markdown(f"**🚨 High-risk alerts — {len(high)} student(s) require immediate attention**")
            for nm, sid, course, tier, p in high:
                st.markdown(
                    f"<div class='alert-card'>"
                    f"<span class='alert-badge'>High Risk</span>"
                    f"<div class='alert-name'>{nm} <span style='opacity:0.45;font-weight:400'>({sid})</span></div>"
                    f"<div class='alert-meta'>{course} · {tier} · 6m placement: {p['placement_6m']*100:.0f}% · {p['salary_range']}</div>"
                    f"<div style='margin-top:8px;font-size:12px;opacity:0.75'>"
                    f"Drivers: {' · '.join(p['risk_drivers'][:2]) if p['risk_drivers'] else 'None identified'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No high-risk students in the current portfolio.")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── All students table ──
        st.markdown("**All students**")
        table = [{
            "Name":         r["student"].get("name","?"),
            "ID":           r["student"].get("student_id","?"),
            "Course":       r["student"].get("course_type","?"),
            "Tier":         r["student"].get("institute_tier","?"),
            "Risk":         r["prediction"]["risk"],
            "Placement 6m": f"{r['prediction']['placement_6m']*100:.0f}%",
            "Salary Range": r["prediction"]["salary_range"],
            "Lender Action": (
                "Intervene now" if r["prediction"]["risk"]=="High"
                else "Check-in at 3m" if r["prediction"]["risk"]=="Medium"
                else "Monitor"
            ),
        } for r in records]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  What-If Simulator":
    st.markdown("## What-If Simulation Engine")
    st.caption("Select a student and adjust parameters to see how improvements change their placement risk.")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if not st.session_state.predictions:
        st.info("No students analyzed yet. Go to **Student Analysis** first.")
    else:
        names = [f"{r['student'].get('name','?')}  ({r['student'].get('student_id','?')})"
                 for r in st.session_state.predictions]
        idx = st.selectbox("Select student to simulate", range(len(names)),
                           format_func=lambda i: names[i])
        sel       = st.session_state.predictions[idx]
        orig      = sel["prediction"]
        student   = sel["student"]
        risk      = orig["risk"]

        # Current state banner
        st.markdown(
            f"<div style='border:1px solid rgba(128,128,128,0.2);border-radius:8px;"
            f"padding:12px 18px;margin:10px 0;'>"
            f"<strong>Current state</strong> &nbsp;·&nbsp; "
            f"<span class='pill-{risk}'>{risk} Risk</span>"
            f"&nbsp;&nbsp; 6m probability: <strong>{orig['placement_6m']*100:.0f}%</strong>"
            f"&nbsp;&nbsp; Salary: <strong>{orig['salary_range']}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("**Adjust parameters to simulate improvement**")

        cur_intern = int(student.get("num_internships", 0))
        cur_certs  = int(student.get("num_certifications", 0))
        cur_apps   = int(student.get("job_applications_per_week", 0))
        cur_cgpa   = float(student.get("cgpa", 7.0))

        c1, c2, c3, c4 = st.columns(4)
        with c1: new_intern  = st.number_input("Internships",
                    min_value=cur_intern, max_value=max(cur_intern+3,5), value=cur_intern)
        with c2: new_certs   = st.number_input("Certifications",
                    min_value=cur_certs,  max_value=max(cur_certs+5,10), value=cur_certs)
        with c3: new_apps    = st.number_input("Applications / week",
                    min_value=0, max_value=50, value=cur_apps)
        with c4: new_cgpa    = st.number_input("CGPA", 0.0, 10.0, cur_cgpa, 0.1)

        c5, c6, _, _ = st.columns(4)
        with c5: new_linkedin = st.selectbox("LinkedIn Active", ["Yes","No"],
                    index=0 if student.get("linkedin_active",0) else 1)
        with c6: new_resume   = st.selectbox("Resume Updated",  ["Yes","No"],
                    index=0 if student.get("resume_updated_recently",0) else 1)

        st.markdown("")
        if st.button("  ▶  Run Simulation  ", type="primary"):
            lin = 1 if new_linkedin == "Yes" else 0
            res = 1 if new_resume   == "Yes" else 0
            inq = student.get("internship_quality_score", 0)
            if new_intern > cur_intern:
                inq = min(inq + 0.15*(new_intern - cur_intern), 1.0)

            changes = {
                "num_internships": int(new_intern),
                "num_certifications": int(new_certs),
                "job_applications_per_week": int(new_apps),
                "cgpa": new_cgpa,
                "linkedin_active": lin,
                "resume_updated_recently": res,
                "internship_quality_score": inq,
                "skill_relevance_score": round(min(new_certs/4.0,1.0)*0.7 + (0.3 if lin else 0.0), 2),
            }
            result = predictor.what_if(student, changes)
            mod    = result["modified"]

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("**Simulation results**")

            r1, r2, r3, r4 = st.columns(4)
            deltas = [
                ("Risk level",        mod["risk"],                    result["delta_risk"]),
                ("3m probability",    f"{mod['placement_3m']*100:.0f}%", f"{result['delta_3m']*100:+.1f}%"),
                ("6m probability",    f"{mod['placement_6m']*100:.0f}%", f"{result['delta_6m']*100:+.1f}%"),
                ("Expected salary",   mod["salary_range"],            "After improvements"),
            ]
            for col, (lbl, val, sub) in zip([r1,r2,r3,r4], deltas):
                with col:
                    st.markdown(
                        f"<div class='stat-card'>"
                        f"<div class='stat-label'>{lbl}</div>"
                        f"<div class='stat-value'>{val}</div>"
                        f"<div class='stat-sub'>{sub}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("")
            comp_df = pd.DataFrame({
                "Scenario": ["Before 3m","After 3m","Before 6m","After 6m","Before 12m","After 12m"],
                "Probability (%)": [
                    round(orig["placement_3m"]*100,1),  round(mod["placement_3m"]*100,1),
                    round(orig["placement_6m"]*100,1),  round(mod["placement_6m"]*100,1),
                    round(orig["placement_12m"]*100,1), round(mod["placement_12m"]*100,1),
                ]
            })
            st.bar_chart(comp_df.set_index("Scenario"), height=240)

            d6 = result["delta_6m"]
            if d6 > 0.05:
                st.success(f"✅ These changes improve 6-month placement probability by {d6*100:.0f} percentage points.")
            elif d6 > 0:
                st.info(f"📈 Modest improvement of {d6*100:.1f} percentage points at 6 months.")
            else:
                st.warning("These changes have minimal impact. Consider more significant interventions.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Portfolio Analytics":
    st.markdown("## Portfolio Analytics")
    st.caption("Deep-dive analysis across the full simulated cohort of 1,000 students.")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        csv = os.path.join(os.path.dirname(__file__), "../data/students.csv")
        return pd.read_csv(csv)

    try:
        df = load_data()

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            ("Total students",      f"{len(df):,}",                          "Full cohort"),
            ("6m placement rate",   f"{df['placed_6m'].mean()*100:.0f}%",    "Across all courses"),
            ("Avg salary",          f"{df['actual_salary_lpa'].mean():.1f} LPA", "Predicted"),
            ("Avg CGPA",            f"{df['cgpa'].mean():.1f}",              "Out of 10"),
        ]
        for col, (lbl, val, sub) in zip([k1,k2,k3,k4], kpis):
            with col:
                st.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-label'>{lbl}</div>"
                    f"<div class='stat-value'>{val}</div>"
                    f"<div class='stat-sub'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        tab1, tab2, tab3 = st.tabs(["  📈  Placement  ", "  💰  Salary  ", "  🎓  Academic Profile  "])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Placement rate by course (6m)**")
                cs = df.groupby("course_type")["placed_6m"].mean().mul(100).round(1).sort_values()
                st.bar_chart(cs, height=260)
            with c2:
                st.markdown("**Placement rate by institute tier**")
                ts = df.groupby("institute_tier")[["placed_3m","placed_6m","placed_12m"]].mean().mul(100).round(1)
                st.bar_chart(ts, height=260)

            st.markdown("**3m vs 6m vs 12m placement by course**")
            all_cs = df.groupby("course_type")[["placed_3m","placed_6m","placed_12m"]].mean().mul(100).round(1)
            all_cs.columns = ["3 months","6 months","12 months"]
            st.bar_chart(all_cs, height=280)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Average salary by institute tier (LPA)**")
                st.bar_chart(df.groupby("institute_tier")["actual_salary_lpa"].mean().round(1), height=260)
            with c2:
                st.markdown("**Average salary by course type (LPA)**")
                st.bar_chart(df.groupby("course_type")["actual_salary_lpa"].mean().round(1).sort_values(), height=260)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Average salary by region (LPA)**")
                st.bar_chart(df.groupby("region")["actual_salary_lpa"].mean().round(1).sort_values(), height=260)
            with c4:
                st.markdown("**Average salary by sector (LPA)**")
                st.bar_chart(df.groupby("target_sector")["actual_salary_lpa"].mean().round(1).sort_values(), height=260)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**CGPA distribution**")
                cgpa_bins = pd.cut(df["cgpa"], bins=[0,5,6,7,8,9,10]).value_counts().sort_index()
                cgpa_bins.index = cgpa_bins.index.astype(str)
                st.bar_chart(cgpa_bins, height=240)

            with c2:
                st.markdown("**Number of internships**")
                ic = df["num_internships"].value_counts().sort_index()
                ic.index = ic.index.astype(str)
                st.bar_chart(ic, height=240)

            st.markdown("**Feature importance — what drives placement success**")
            fi = pd.Series({
                "Internship Quality":   0.28,
                "Institute Strength":   0.22,
                "Skill Relevance":      0.18,
                "CGPA":                 0.15,
                "Market Fit":           0.12,
                "Job Search Activity":  0.05,
            }).sort_values()
            st.bar_chart(fi, height=240)

    except FileNotFoundError:
        st.error("students.csv not found. Run `python data/generate_dataset.py` first.")