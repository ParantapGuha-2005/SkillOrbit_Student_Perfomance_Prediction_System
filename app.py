"""
Student Performance Prediction System — Streamlit Dashboard
Capstone Project (Module 4: Performance Dashboard, Module 5: Recommendation System)

Expects the following artifacts (produced by the companion notebook) in an
`artifacts/` folder next to this file:
    - student_performance_model.pkl   (trained RandomForestClassifier)
    - scaler.pkl                      (fitted StandardScaler)
    - encoders.pkl                    (dict: gender_encoder, parental_support_map)
    - cleaned_student_data.csv        (cleaned dataset + PredictedLevel column)
    - model_comparison.csv            (accuracy / F1 comparison across models)
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sns.set_style("whitegrid")

ARTIFACTS_DIR = "artifacts"
FEATURE_COLUMNS = [
    "AttendanceRate", "StudyHoursPerWeek", "PreviousGrade",
    "ExtracurricularActivities", "Gender_enc", "ParentalSupport_enc", "OnlineClasses_enc",
]
LEVEL_ORDER = ["Low", "Average", "High"]
LEVEL_COLORS = {"Low": "#d62728", "Average": "#ff7f0e", "High": "#2ca02c"}

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data / model loading
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "student_performance_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "encoders.pkl"))
    return model, scaler, encoders


@st.cache_data
def load_student_data():
    df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "cleaned_student_data.csv"))
    df["PerformanceLevel"] = pd.Categorical(df["PerformanceLevel"], categories=LEVEL_ORDER, ordered=True)
    df["PredictedLevel"] = pd.Categorical(df["PredictedLevel"], categories=LEVEL_ORDER, ordered=True)
    return df


@st.cache_data
def load_model_comparison():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"), index_col=0)


def artifacts_available():
    required = [
        "student_performance_model.pkl", "scaler.pkl", "encoders.pkl",
        "cleaned_student_data.csv", "model_comparison.csv",
    ]
    return all(os.path.exists(os.path.join(ARTIFACTS_DIR, f)) for f in required)


# --------------------------------------------------------------------------- #
# Recommendation engine (Module 5)
# --------------------------------------------------------------------------- #
def generate_recommendations(attendance_rate, study_hours, previous_grade,
                              extracurricular, parental_support, predicted_level):
    """Generate personalized recommendations using a weighted scoring system.

    Each factor is scored 0-10, where higher scores indicate greater improvement
    opportunity. Factors are weighted by their relative impact on student performance.
    A composite score determines the recommendation tier (High/Medium/Low priority).
    """

    tips = []

    # -- Factor scoring (0 = no concern, 10 = maximum concern) -----------------
    # Attendance: <75 is at-risk; linearly scale so 0% -> score 10, 75% -> score 0
    if attendance_rate < 75:
        attendance_score = min(10, (75 - attendance_rate) * 10 / 75)
    else:
        attendance_score = 0

    # Study hours: <12 is at-risk; scale 0 h -> score 10, 12 h -> score 0
    if study_hours < 12:
        study_score = min(10, (12 - study_hours) * 10 / 12)
    else:
        study_score = 0

    # Previous grade: <70 is at-risk; scale 0 -> score 10, 70 -> score 0
    if previous_grade < 70:
        grade_score = min(10, (70 - previous_grade) * 10 / 70)
    else:
        grade_score = 0

    # Parental support: categorical mapping
    parent_score = {"Low": 8, "Moderate": 4, "High": 0}.get(parental_support, 0)

    # Predicted level: categorical mapping
    predicted_score = {"Low": 8, "Average": 4, "High": 0}.get(predicted_level, 0)

    # Extracurricular activities: 0 is at-risk
    extracurricular_score = 6 if extracurricular == 0 else 0

    # -- Weighted composite score ------------------------------------------------
    # Weights reflect relative impact based on educational research
    weights = {
        "attendance": 0.4,
        "study_hours": 0.3,
        "grade": 0.2,
        "parental_support": 0.05,
        "predicted_level": 0.03,
        "extracurricular": 0.02,
    }
    total_score = (
        attendance_score * weights["attendance"]
        + study_score * weights["study_hours"]
        + grade_score * weights["grade"]
        + parent_score * weights["parental_support"]
        + predicted_score * weights["predicted_level"]
        + extracurricular_score * weights["extracurricular"]
    )

    # -- Tier assignment based on composite score --------------------------------
    # Tier 1: High-priority interventions needed (score >= 6)
    # Tier 2: Medium-priority recommendations (score >= 3)
    # Tier 3: Maintenance / positive reinforcement (score < 3)
    if total_score >= 6:
        priority = "High"

        if attendance_rate < 75:
            tips.append("Improve attendance — regular class attendance is strongly linked to better academic outcomes.")
        if study_hours < 12:
            tips.append("Increase weekly study hours, aiming for consistent daily study blocks rather than cramming.")
        if previous_grade < 70:
            tips.append("Focus on revisiting foundational topics from previous coursework where scores were weaker.")
        if parental_support == "Low":
            tips.append("Involve parents/guardians more actively in academic planning and progress check-ins.")
        if predicted_level == "Low":
            tips.append("Attend additional practice sessions or tutoring to reinforce weaker subject areas.")
        if extracurricular == 0:
            tips.append("Consider a well-balanced extracurricular activity — participation is linked to steadier engagement.")

        if not tips:
            tips.append("Keep up the current habits — attendance, study routine, and support levels all look solid.")

        # Tier-1 priority opener highlighting the most impactful combined factors
        if attendance_rate < 75 and study_hours < 12:
            tier1_tips = ["Priority: Focus on attendance and study habits together — they are your biggest leverage points for improvement."]
        elif attendance_rate < 75:
            tier1_tips = ["Priority: Attendance is your biggest leverage point right now."]
        elif study_hours < 12:
            tier1_tips = ["Priority: Increasing study time consistently will significantly improve your performance."]
        else:
            tier1_tips = ["Priority: Address the foundational factors above for the biggest impact."]

        tips = tier1_tips + tips

    elif total_score >= 3:
        priority = "Medium"

        if attendance_rate < 75:
            tips.insert(0, "Maintaining or improving your attendance rate will strengthen your academic foundation.")
        if study_hours < 12:
            tips.insert(0, "Aim for at least 12 study hours per week to solidify your knowledge.")
        if parental_support == "Low":
            tips.insert(0, "Talk with parents/guardians about creating a supportive study environment.")
        if predicted_level == "Low":
            tips.insert(0, "Additional practice sessions could help prevent falling behind.")
        if extracurricular == 0:
            tips.insert(0, "Joining an extracurricular activity could boost your engagement.")

        if not tips:
            tips.append("Your current habits look balanced; continue with your effective routines.")

    else:
        priority = "Low"
        # Tier 3 recommendations (maintenance or minor improvements)
        tips = tips + [
            "Your attendance and study habits are strong — keep it up!",
            "Maintain your current level of parental support and extracurricular balance.",
            "Your performance level appears stable; a few additional study hours could help maintain this.",
        ]

    # -- Add level-specific encouragement ----------------------------------------
    if predicted_level == "High":
        tips.append("Maintain your excellent performance with continued focus on all areas.")
    elif predicted_level == "Average":
        tips.append("You're doing okay — small improvements in attendance or study habits could push you to the next level.")
    elif predicted_level == "Low":
        tips.append("Your predicted level shows potential for improvement — targeted efforts will make a difference.")

    return tips


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
def page_overview(df, comparison_df):

    # --- Responsive spacing that scales with viewport height ---
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 0.5rem;
                overflow: visible;
            }
            /* Prevent the header emoji/text from being clipped */
            h1, div[data-testid="stHeader"], div[data-testid="stMarkdownContainer"] h1 {
                line-height: 1.3 !important;
                overflow: visible !important;
                padding-top: 0.2rem;
                margin-top: 0;
            }
            div[data-testid="stMetric"] {
                padding: 0.3rem 0.5rem;
            }
            div[data-testid="stMetric"] + div,
            div[data-testid="stHorizontalBlock"] {
                margin-bottom: 0 !important;
            }
            hr {
                margin: 0.2rem 0 0.6rem 0 !important;
            }
            div[data-testid="stImage"] img {
                max-height: 38vh;
                width: auto;
                margin: 0 auto;
                display: block;
            }
            div[data-testid="stVerticalBlock"] > div:has(h3) {
                margin-bottom: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("📊 Overview")

    n_students = len(df)
    avg_grade = df["FinalGrade"].mean()
    avg_attendance = df["AttendanceRate"].mean()
    high_pct = (df["PerformanceLevel"] == "High").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{n_students:,}", help="Number of students in dataset")
    c2.metric("Average Final Grade", f"{avg_grade:.1f}", help="Mean score across all students")
    c3.metric("Average Attendance Rate", f"{avg_attendance:.1f}%", help="Mean attendance percentage")
    c4.metric("High Performers", f"{high_pct:.1f}%", help="Percentage of students in 'High' performance level")

    st.divider()

    view = st.selectbox(
        "Select a view",
        ["Performance Level Distribution", "Model Comparison (Test Set)"],
    )

    # Row height for a dataframe sized exactly to its content (no blank trailing rows)
    ROW_PX = 35
    HEADER_PX = 38

    if view == "Performance Level Distribution":
        st.subheader("Performance Level Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(
            data=df, x="PerformanceLevel", order=LEVEL_ORDER,
            palette=[LEVEL_COLORS[l] for l in LEVEL_ORDER], ax=ax
        )
        ax.set_xlabel("Performance Level")
        ax.set_ylabel("Number of Students")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    else:
        st.subheader("Model Comparison (Test Set)")
        st.caption(
            "Accuracy / Weighted F1 / Macro F1 across all trained models — "
        )

        col1, col2 = st.columns([1, 1.2])

        with col1:
            table_height = HEADER_PX + ROW_PX * len(comparison_df)
            st.dataframe(
                comparison_df.style.format("{:.2f}"),
                use_container_width=True,
                height=table_height,
            )

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            (comparison_df / 100).plot(kind="bar", ax=ax, colormap="viridis")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

# --------------------------------------------------------------------------- #
# Page: Student Explorer (comparison)
# --------------------------------------------------------------------------- #
def page_student_explorer(df):

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 0.5rem;
                overflow: visible;
            }
            h1, div[data-testid="stMarkdownContainer"] h1 {
                line-height: 1.3 !important;
                overflow: visible !important;
                padding-top: 0.2rem;
                margin-top: 0;
            }
            div[data-testid="stWidgetLabel"] p {
                line-height: 1.4 !important;
                overflow: visible !important;
                padding-top: 0.15rem;
            }
            div[data-testid="stImage"] img {
                max-height: 46vh;
                width: auto;
                margin: 0 auto;
                display: block;
            }
            hr {
                margin: 1.2rem 0 1.5rem 0 !important;
            }
            /* Give the bottom metrics section more visual weight so it doesn't feel like empty space */
            div[data-testid="stMetric"] {
                padding: 1rem 1.2rem;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            div[data-testid="stMetric"] label p {
                font-size: 1rem !important;
            }
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
                font-size: 2.1rem !important;
                padding-top: 0.3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.3])

    # ----------------------------- LEFT COLUMN ----------------------------- #
    with left:
        st.header("\U0001F4DA Student Explorer")  # 📚
        st.caption(
            "Look up an individual student and compare them against their "
            "predicted performance and class averages."
        )

        perf_filter = st.selectbox("Filter by Performance Level", ["All"] + LEVEL_ORDER)
        filtered_df = df
        if perf_filter != "All":
            filtered_df = filtered_df[filtered_df["PerformanceLevel"] == perf_filter]

        search = st.text_input("Search by Student ID or Name")
        if search:
            mask = (
                filtered_df["StudentID"].astype(str).str.contains(search, case=False, na=False)
                | filtered_df["Name"].astype(str).str.contains(search, case=False, na=False)
            )
            filtered_df = filtered_df[mask]

        if filtered_df.empty:
            st.warning("No matching students found.")
            student = None
        else:
            options = filtered_df.apply(lambda r: f"{r['StudentID']} - {r['Name']}", axis=1).tolist()
            choice = st.selectbox("Select a student", options)
            student = filtered_df.iloc[options.index(choice)]

    if student is None:
        return

    # ----------------------------- RIGHT COLUMN ----------------------------- #
    with right:
        st.write("")

        section = st.selectbox(
            "Select a section",
            ["Student vs. Class Average", "Recommendations"],
        )

        compare_cols = ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "FinalGrade"]

        if section == "Student vs. Class Average":
            class_avg = df[compare_cols].mean()
            compare_df = pd.DataFrame({
                "Student": student[compare_cols].astype(float),
                "Class Average": class_avg,
            })
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            compare_df.plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52"])
            ax.set_ylabel("Value")
            ax.set_title(f"{student['Name']} vs. Class Average", fontsize=11)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        else:
            st.markdown("**\U0001F4A1 Recommendations**")  # 💡
            tips = generate_recommendations(
                student["AttendanceRate"], student["StudyHoursPerWeek"], student["PreviousGrade"],
                student["ExtracurricularActivities"], student["ParentalSupport"], student["PredictedLevel"],
            )
            for tip in tips:
                st.write(f"- {tip}")

# ----------------------------- BOTTOM ROW ----------------------------- #
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Grade", f"{student['FinalGrade']:.1f}")
    c2.metric("Actual Level", str(student["PerformanceLevel"]))

    is_match = student["PerformanceLevel"] == student["PredictedLevel"]
    badge_color = "rgba(76, 175, 80, 0.25)" if is_match else "rgba(244, 67, 54, 0.25)"
    badge_text_color = "#8fd694" if is_match else "#f39a94"
    badge_label = "Match" if is_match else "Mismatch"

    with c3:
        st.markdown(
            f"""
            <div style="
                padding: 1rem 1.2rem;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            ">
                <p style="margin:0; font-size:1rem; color: rgba(255,255,255,0.6);">Predicted Level</p>
                <div style="display:flex; align-items:center; gap:0.6rem; margin-top:0.3rem;">
                    <span style="font-size:2.1rem; font-weight:400;">{student['PredictedLevel']}</span>
                    <span style="
                        background:{badge_color};
                        color:{badge_text_color};
                        padding:0.2rem 0.6rem;
                        border-radius:999px;
                        font-size:0.85rem;
                        font-weight:500;
                        white-space:nowrap;
                    ">{badge_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4.metric("Attendance", f"{student['AttendanceRate']:.1f}%")

# --------------------------------------------------------------------------- #
# Page: Factor Analysis
# --------------------------------------------------------------------------- #
def page_factor_analysis(df):

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 0.5rem;
                overflow: visible;
            }
            h1, h3, div[data-testid="stMarkdownContainer"] h1, div[data-testid="stMarkdownContainer"] h3 {
                line-height: 1.3 !important;
                overflow: visible !important;
                padding-top: 0.2rem;
                margin-top: 0;
            }
            div[data-testid="stWidgetLabel"] p {
                line-height: 1.4 !important;
                overflow: visible !important;
                padding-top: 0.15rem;
            }
            div[data-testid="stImage"] img {
                max-height: 46vh;
                width: auto;
                margin: 0 auto;
                display: block;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("\U0001F4C8 Factor Analysis")  # 📈
    st.caption(
        "How each contributing factor relates to performance level "
        "(no per-subject breakdown exists in this dataset)."
    )

    factor = st.selectbox(
        "Choose a factor",
        ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "ExtracurricularActivities", "ParentalSupport"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{factor} by Performance Level")
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        is_categorical = (
            df[factor].dtype == object
            or (str(df[factor].dtype).startswith("category") is False and df[factor].nunique() <= 5)
        )
        if is_categorical:
            pd.crosstab(df[factor], df["PerformanceLevel"], normalize="index")[LEVEL_ORDER].plot(
                kind="bar", stacked=True, colormap="viridis", ax=ax
            )
            ax.set_ylabel("Proportion")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        else:
            sns.boxplot(
                data=df, x="PerformanceLevel", y=factor, order=LEVEL_ORDER,
                palette=[LEVEL_COLORS[l] for l in LEVEL_ORDER], ax=ax
            )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col2:
        st.subheader("Correlation with Final Grade")
        numeric_factors = ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "ExtracurricularActivities", "FinalGrade"]
        corr = df[numeric_factors].corr()
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax, annot_kws={"size": 8})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# --------------------------------------------------------------------------- #
# Page: Performance Trends
# --------------------------------------------------------------------------- #
def page_trends(df):

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 0.5rem;
                overflow: visible;
            }
            h1, h3, div[data-testid="stMarkdownContainer"] h1, div[data-testid="stMarkdownContainer"] h3 {
                line-height: 1.3 !important;
                overflow: visible !important;
                padding-top: 0.2rem;
                margin-top: 0;
            }
            div[data-testid="stWidgetLabel"] p {
                line-height: 1.4 !important;
                overflow: visible !important;
                padding-top: 0.15rem;
            }
            div[data-testid="stImage"] img {
                max-height: 46vh;
                width: auto;
                margin: 0 auto;
                display: block;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("\U0001F4C9 Performance Trends")  # 📉

    # Persist the bucket choice across reruns so the plot (drawn first) can use it
    # even though the radio widget itself is rendered further down the page.
    if "trend_bucket_col" not in st.session_state:
        st.session_state.trend_bucket_col = "AttendanceRate"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Previous Grade vs. Final Grade")
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        sns.scatterplot(
            data=df, x="PreviousGrade", y="FinalGrade", hue="PerformanceLevel",
            hue_order=LEVEL_ORDER, palette=LEVEL_COLORS, alpha=0.6, ax=ax,
        )
        ax.set_xlabel("Previous Grade")
        ax.set_ylabel("Final Grade")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col2:
        st.subheader("Attendance & Study Hours Trend")

        bucket_col = st.session_state.trend_bucket_col
        bucketed = pd.cut(df[bucket_col], bins=8)
        trend = df.groupby(bucketed, observed=True)["FinalGrade"].mean().reset_index()
        trend[bucket_col] = trend[bucket_col].astype(str)

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        sns.lineplot(data=trend, x=bucket_col, y="FinalGrade", marker="o", ax=ax)
        ax.set_xlabel(bucket_col)
        ax.set_ylabel("Average Final Grade")
        ax.tick_params(axis="x", rotation=40)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        # Radio now appears below the plot; updates session_state and triggers a rerun
        st.radio(
            "Bucket by",
            ["AttendanceRate", "StudyHoursPerWeek"],
            horizontal=True,
            key="trend_bucket_col",
        )

# --------------------------------------------------------------------------- #
# Page: Predict a New Student (live model + recommendations)
# --------------------------------------------------------------------------- #
def page_predict(model, scaler, encoders):
    st.header("ð¦ Predict a New Student")
    st.caption("Enter a student's details to get a live, out-of-sample performance prediction and tailored recommendations.")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            attendance_rate = st.slider(
                "Attendance Rate (%)", 
                0.0, 100.0, 80.0,
                help="Higher attendance rates are strongly correlated with better academic performance."
            )
            study_hours = st.slider(
                "Study Hours per Week", 
                0.0, 40.0, 12.0,
                help="Aim for consistent study habits; 12+ hours per week is often beneficial."
            )
            previous_grade = st.slider(
                "Previous Grade", 
                0.0, 100.0, 75.0,
                help="Strong foundational knowledge from previous coursework supports current performance."
            )
            extracurricular = st.selectbox(
                "Extracurricular Activities (count)", 
                [0, 1, 2, 3, 4],
                help="Balanced extracurricular participation can improve engagement and time management."
            )
        with c2:
            gender = st.selectbox(
                "Gender", 
                list(encoders["gender_encoder"].classes_),
                help="Select the student's gender."
            )
            parental_support = st.selectbox(
                "Parental Support", 
                list(encoders["parental_support_map"].keys()),
                help="Level of parental/guardian involvement in academic planning."
            )
            online_classes = st.checkbox(
                "Takes Online Classes", 
                value=False,
                help="Check if the student primarily attends classes online."
            )

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    gender_enc = encoders["gender_encoder"].transform([gender])[0]
    parental_enc = encoders["parental_support_map"][parental_support]
    online_enc = int(online_classes)

    row = pd.DataFrame([{
        "AttendanceRate": attendance_rate,
        "StudyHoursPerWeek": study_hours,
        "PreviousGrade": previous_grade,
        "ExtracurricularActivities": extracurricular,
        "Gender_enc": gender_enc,
        "ParentalSupport_enc": parental_enc,
        "OnlineClasses_enc": online_enc,
    }])[FEATURE_COLUMNS]

    # Random Forest was trained on unscaled features (see notebook, Module 3.6-3.7)
    predicted_level = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    prob_df = pd.DataFrame({"Performance Level": model.classes_, "Probability": probabilities}).set_index(
        "Performance Level"
    ).reindex(LEVEL_ORDER)

    st.divider()
    st.subheader("Prediction")
    st.metric("Predicted Performance Level", predicted_level)
    st.bar_chart(prob_df)

    # Explanation section
    st.subheader("Why this prediction?")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **Attendance Rate**: {attendance_rate:.1f}%")
        st.write(f"- **Study Hours/Week**: {study_hours:.1f}")
        st.write(f"- **Previous Grade**: {previous_grade:.1f}")
        st.write(f"- **Extracurricular Activities**: {extracurricular}")
    with col2:
        st.write(f"- **Gender**: {gender}")
        st.write(f"- **Parental Support**: {parental_support}")
        st.write(f"- **Online Classes**: {'Yes' if online_classes else 'No'}")
        st.write(f"- **Model Confidence**: {prob_df['Probability'].max():.2%}")

    st.subheader("Recommendations")
    tips = generate_recommendations(attendance_rate, study_hours, previous_grade,
                                     extracurricular, parental_support, predicted_level)
    for tip in tips:
        st.write(f"- {tip}")

    gender_enc = encoders["gender_encoder"].transform([gender])[0]
    parental_enc = encoders["parental_support_map"][parental_support]
    online_enc = int(online_classes)

    row = pd.DataFrame([{
        "AttendanceRate": attendance_rate,
        "StudyHoursPerWeek": study_hours,
        "PreviousGrade": previous_grade,
        "ExtracurricularActivities": extracurricular,
        "Gender_enc": gender_enc,
        "ParentalSupport_enc": parental_enc,
        "OnlineClasses_enc": online_enc,
    }])[FEATURE_COLUMNS]

    # Random Forest was trained on unscaled features (see notebook, Module 3.6-3.7)
    predicted_level = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    prob_df = pd.DataFrame({"Performance Level": model.classes_, "Probability": probabilities}).set_index(
        "Performance Level"
    ).reindex(LEVEL_ORDER)

    st.divider()
    st.subheader("Prediction")
    st.metric("Predicted Performance Level", predicted_level)
    st.bar_chart(prob_df)

    st.subheader("Recommendations")
    tips = generate_recommendations(attendance_rate, study_hours, previous_grade,
                                     extracurricular, parental_support, predicted_level)
    for tip in tips:
        st.write(f"- {tip}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.sidebar.title("🎓 Student Performance Dashboard")

    if not artifacts_available():
        st.error(
            "Required artifacts were not found in the `artifacts/` folder. "
            "Run the companion notebook's 'Save Artifacts & Deployment Prep' cell first, "
            "and make sure the `artifacts/` folder sits next to this app.py."
        )
        return

    # Add loading spinner for better UX
    with st.spinner("Loading dashboard data..."):
        model, scaler, encoders = load_artifacts()
        df = load_student_data()
        comparison_df = load_model_comparison()


    model, scaler, encoders = load_artifacts()
    df = load_student_data()
    comparison_df = load_model_comparison()

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Student Explorer", "Factor Analysis", "Performance Trends", "Predict a New Student"],
    )

    if page == "Overview":
        page_overview(df, comparison_df)
    elif page == "Student Explorer":
        page_student_explorer(df)
    elif page == "Factor Analysis":
        page_factor_analysis(df)
    elif page == "Performance Trends":
        page_trends(df)
    elif page == "Predict a New Student":
        page_predict(model, scaler, encoders)


if __name__ == "__main__":
    main()
