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
    tips = []

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

    return tips


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
def page_overview(df, comparison_df):
    st.header("📊 Overview")

    n_students = len(df)
    avg_grade = df["FinalGrade"].mean()
    avg_attendance = df["AttendanceRate"].mean()
    high_pct = (df["PerformanceLevel"] == "High").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{n_students:,}")
    c2.metric("Avg. Final Grade", f"{avg_grade:.1f}")
    c3.metric("Avg. Attendance", f"{avg_attendance:.1f}%")
    c4.metric("Share in 'High'", f"{high_pct:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Level Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="PerformanceLevel", order=LEVEL_ORDER,
                      palette=[LEVEL_COLORS[l] for l in LEVEL_ORDER], ax=ax)
        ax.set_xlabel("Performance Level")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)

    with col2:
        st.subheader("Model Comparison (Test Set)")
        st.caption("Accuracy / Weighted F1 / Macro F1 across all trained models — "
                   "see the notebook (Module 3-4) for the full evaluation.")
        st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        (comparison_df / 100).plot(kind="bar", ax=ax, colormap="viridis", rot=20)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        st.pyplot(fig)

    st.info(
        "Note: this dataset records one overall `FinalGrade` per student rather than "
        "per-subject scores, so the analysis below is organized by contributing factor "
        "(attendance, study hours, previous grade, etc.) rather than by subject."
    )


# --------------------------------------------------------------------------- #
# Page: Student Explorer (comparison)
# --------------------------------------------------------------------------- #
def page_student_explorer(df):
    st.header("ð Student Explorer")
    st.caption("Look up an individual student and compare them against their predicted performance and class averages.")

    # Add performance level filter
    perf_filter = st.selectbox("Filter by Performance Level", ["All"] + LEVEL_ORDER)
    if perf_filter != "All":
        df = df[df["PerformanceLevel"] == perf_filter]

    search = st.text_input("Search by Student ID or Name")
    filtered = df
    if search:
        mask = (
            df["StudentID"].astype(str).str.contains(search, case=False, na=False)
            | df["Name"].astype(str).str.contains(search, case=False, na=False)
        )
        filtered = df[mask]

    if filtered.empty:
        st.warning("No matching students found.")
        return

    options = filtered.apply(lambda r: f"{r['StudentID']} â {r['Name']}", axis=1).tolist()
    choice = st.selectbox("Select a student", options)
    student = filtered.iloc[options.index(choice)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Grade", f"{student['FinalGrade']:.1f}")
    c2.metric("Actual Level", str(student["PerformanceLevel"]))
    c3.metric("Predicted Level", str(student["PredictedLevel"]),
              delta="Match" if student["PerformanceLevel"] == student["PredictedLevel"] else "Mismatch",
              delta_color="off")
    c4.metric("Attendance", f"{student['AttendanceRate']:.1f}%")

    st.divider()
    st.subheader("Student vs. Class Average")

    compare_cols = ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "FinalGrade"]
    class_avg = df[compare_cols].mean()
    compare_df = pd.DataFrame({
        "Student": student[compare_cols].astype(float),
        "Class Average": class_avg,
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    compare_df.plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52"], rot=15)
    ax.set_ylabel("Value")
    ax.set_title(f"{student['Name']} vs. Class Average")
    st.pyplot(fig)

    st.subheader("Recommendations")
    tips = generate_recommendations(
        student["AttendanceRate"], student["StudyHoursPerWeek"], student["PreviousGrade"],
        student["ExtracurricularActivities"], student["ParentalSupport"], student["PredictedLevel"],
    )
    for tip in tips:
        st.write(f"- {tip}")


# --------------------------------------------------------------------------- #
# Page: Factor Analysis
# --------------------------------------------------------------------------- #
def page_factor_analysis(df):
    st.header("📈 Factor Analysis")
    st.caption("How each contributing factor relates to performance level (no per-subject breakdown exists in this dataset).")

    factor = st.selectbox(
        "Choose a factor",
        ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "ExtracurricularActivities", "ParentalSupport"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{factor} by Performance Level")
        fig, ax = plt.subplots(figsize=(6, 4))
        if df[factor].dtype == object or str(df[factor].dtype).startswith("category") is False and df[factor].nunique() <= 5:
            pd.crosstab(df[factor], df["PerformanceLevel"], normalize="index")[LEVEL_ORDER].plot(
                kind="bar", stacked=True, colormap="viridis", ax=ax
            )
            ax.set_ylabel("Proportion")
        else:
            sns.boxplot(data=df, x="PerformanceLevel", y=factor, order=LEVEL_ORDER,
                        palette=[LEVEL_COLORS[l] for l in LEVEL_ORDER], ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation with Final Grade")
        numeric_factors = ["AttendanceRate", "StudyHoursPerWeek", "PreviousGrade", "ExtracurricularActivities", "FinalGrade"]
        corr = df[numeric_factors].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        st.pyplot(fig)


# --------------------------------------------------------------------------- #
# Page: Performance Trends
# --------------------------------------------------------------------------- #
def page_trends(df):
    st.header("📉 Performance Trends")

    st.subheader("Previous Grade vs. Final Grade")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="PreviousGrade", y="FinalGrade", hue="PerformanceLevel",
        hue_order=LEVEL_ORDER, palette=LEVEL_COLORS, alpha=0.6, ax=ax,
    )
    ax.set_xlabel("Previous Grade")
    ax.set_ylabel("Final Grade")
    st.pyplot(fig)

    st.subheader("Attendance & Study Hours Trend")
    bucket_col = st.radio("Bucket by", ["AttendanceRate", "StudyHoursPerWeek"], horizontal=True)
    bucketed = pd.cut(df[bucket_col], bins=8)
    trend = df.groupby(bucketed, observed=True)["FinalGrade"].mean().reset_index()
    trend[bucket_col] = trend[bucket_col].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=trend, x=bucket_col, y="FinalGrade", marker="o", ax=ax)
    ax.set_xlabel(bucket_col)
    ax.set_ylabel("Average Final Grade")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


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
