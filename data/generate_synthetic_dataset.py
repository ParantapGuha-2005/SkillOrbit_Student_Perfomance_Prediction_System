"""
Synthetic Student Performance Dataset Generator
SkillOrbit Machine Learning Capstone Project

Generates a student performance dataset that is deliberately built for the
capstone in two ways at once:

1. REAL SIGNAL — FinalGrade is generated as an actual (noisy) function of
   AttendanceRate, StudyHoursPerWeek, PreviousGrade, ExtracurricularActivities,
   ParentalSupport, and OnlineClassesTaken. So a model trained on clean data
   should genuinely be able to predict performance well above a baseline guess.

2. REALISTIC MESSINESS — on top of that signal, the script injects the kind
   of problems a real dataset (and this capstone brief) expects you to clean:
       - Missing values (MCAR) across most columns
       - Duplicate rows
       - Inconsistent categorical text casing/spacing ("male", " High ", "MEDIUM")
       - Two "look-alike" noise columns (Attendance (%), Study Hours) that
         resemble real features by name but are uncorrelated / out-of-range
         junk — the same "don't trust the column name" lesson as before,
         but now alongside columns that DO carry real signal.

Run:
    python generate_synthetic_dataset.py

Output:
    student_performance_synthetic.csv
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
N_STUDENTS = 1000
RANDOM_SEED = 42
OUTPUT_FILE = "student_performance_synthetic.csv"

rng = np.random.default_rng(RANDOM_SEED)

FIRST_NAMES = ["John", "Sarah", "Alex", "Michael", "Emma", "Olivia", "Liam", "Noah",
               "Ava", "Sophia", "James", "Isabella", "Benjamin", "Mia", "Lucas", "Amelia",
               "Henry", "Harper", "Ethan", "Evelyn", "Daniel", "Abigail", "Matthew", "Emily",
               "Jackson", "Elizabeth", "Sebastian", "Sofia", "David", "Charlotte"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore",
              "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
              "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen"]


def generate_names(n):
    first = rng.choice(FIRST_NAMES, size=n)
    last = rng.choice(LAST_NAMES, size=n)
    return [f"{f} {l}" for f, l in zip(first, last)]


# --------------------------------------------------------------------------
# Step 1: Generate clean, realistic base features
# --------------------------------------------------------------------------
def generate_base_data(n):
    student_id = np.arange(1, n + 1)
    name = generate_names(n)
    gender = rng.choice(["Male", "Female"], size=n)

    attendance_rate = np.clip(rng.normal(82, 10, n), 45, 100).round(1)
    study_hours_per_week = np.clip(rng.normal(16, 6, n), 0, 40).round(1)
    previous_grade = np.clip(rng.normal(75, 12, n), 35, 100).round(1)
    extracurricular_activities = np.clip(rng.poisson(1.2, n), 0, 4)

    parental_support = rng.choice(
        ["Low", "Medium", "High"], size=n, p=[0.25, 0.45, 0.30]
    )
    online_classes_taken = rng.choice([True, False], size=n, p=[0.4, 0.6])

    return pd.DataFrame({
        "StudentID": student_id,
        "Name": name,
        "Gender": gender,
        "AttendanceRate": attendance_rate,
        "StudyHoursPerWeek": study_hours_per_week,
        "PreviousGrade": previous_grade,
        "ExtracurricularActivities": extracurricular_activities,
        "ParentalSupport": parental_support,
        "OnlineClassesTaken": online_classes_taken,
    })


# --------------------------------------------------------------------------
# Step 2: Generate FinalGrade as a genuine (noisy) function of the features
# --------------------------------------------------------------------------
def generate_final_grade(df):
    parental_bonus_map = {"Low": 0.0, "Medium": 3.0, "High": 6.0}
    parental_bonus = df["ParentalSupport"].map(parental_bonus_map).values

    # Standardize a couple of inputs so their contribution is on a comparable scale
    attendance_term = 0.28 * df["AttendanceRate"].values
    study_term = 0.55 * df["StudyHoursPerWeek"].values
    previous_term = 0.35 * df["PreviousGrade"].values
    extracurricular_term = 1.8 * df["ExtracurricularActivities"].values
    online_term = np.where(df["OnlineClassesTaken"].values, 1.5, 0.0)

    noise = rng.normal(0, 5.5, len(df))  # realistic randomness: two similar students can still land differently

    final_grade = (
        16.0
        + attendance_term
        + study_term
        + previous_term
        + extracurricular_term
        + parental_bonus
        + online_term
        + noise
    )
    return np.clip(final_grade, 0, 100).round(1)


# --------------------------------------------------------------------------
# Step 3: Inject realistic messiness for the cleaning exercise
# --------------------------------------------------------------------------
def inject_missing_values(df, columns, frac=0.045):
    df = df.copy()
    for col in columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(object)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


def inject_duplicate_rows(df, n_duplicates=18):
    dup_rows = df.sample(n=n_duplicates, random_state=RANDOM_SEED).copy()
    return pd.concat([df, dup_rows], ignore_index=True)


def messify_categorical_text(series, kind):
    """Randomly re-cases / pads category text to simulate inconsistent manual entry."""
    def mess(val):
        if pd.isna(val):
            return val
        r = rng.random()
        if r < 0.15:
            return str(val).upper()
        elif r < 0.30:
            return str(val).lower()
        elif r < 0.38:
            return f" {val} "  # stray whitespace
        return val
    return series.apply(mess)


def add_lookalike_noise_columns(df):
    """
    Adds two columns that resemble real features by name, but are NOT the same
    signal-carrying data — they're independent noise with some out-of-range
    "corrupted" values thrown in. Mirrors the messy pattern from the original
    dataset so the same cleaning technique (checking correlation + ranges
    before trusting a column) is still a required, teachable step.
    """
    n = len(df)
    attendance_noise = np.clip(rng.normal(75, 20, n), None, None)
    # inject some out-of-range corrupted values
    corrupt_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    attendance_noise[corrupt_idx] = rng.uniform(101, 200, size=len(corrupt_idx))
    df["Attendance (%)"] = attendance_noise.round(1)

    study_noise = rng.normal(2.5, 1.5, n)
    corrupt_idx2 = rng.choice(n, size=int(n * 0.04), replace=False)
    study_noise[corrupt_idx2] = rng.uniform(-8, -1, size=len(corrupt_idx2))
    df["Study Hours"] = study_noise.round(1)

    return df


# --------------------------------------------------------------------------
# Build the full dataset
# --------------------------------------------------------------------------
def build_dataset(n=N_STUDENTS):
    df = generate_base_data(n)
    df["FinalGrade"] = generate_final_grade(df)

    # Inject missingness into features AND target (target missingness models
    # "we never recorded this student's final grade" -- a real scenario)
    feature_cols_for_missing = [
        "Gender", "AttendanceRate", "StudyHoursPerWeek", "PreviousGrade",
        "ExtracurricularActivities", "ParentalSupport", "OnlineClassesTaken",
        "Name", "StudentID",
    ]
    df = inject_missing_values(df, feature_cols_for_missing, frac=0.04)
    df = inject_missing_values(df, ["FinalGrade"], frac=0.04)

    # Messy categorical text (applied after missingness injection so NaNs stay NaN)
    df["Gender"] = messify_categorical_text(df["Gender"], "gender")
    df["ParentalSupport"] = messify_categorical_text(df["ParentalSupport"], "support")

    # Look-alike noisy duplicate-named columns
    df = add_lookalike_noise_columns(df)

    # Rename to match the original capstone dataset's column naming
    df = df.rename(columns={"OnlineClassesTaken": "Online Classes Taken"})

    # Reorder to match original schema
    df = df[[
        "StudentID", "Name", "Gender", "AttendanceRate", "StudyHoursPerWeek",
        "PreviousGrade", "ExtracurricularActivities", "ParentalSupport",
        "FinalGrade", "Study Hours", "Attendance (%)", "Online Classes Taken",
    ]]

    # Duplicate rows (added last, after everything else is finalized)
    df = inject_duplicate_rows(df, n_duplicates=18)

    # Shuffle row order so duplicates aren't obviously stacked at the bottom
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(dataset)} rows -> {OUTPUT_FILE}")
    print("\nMissing values per column:")
    print(dataset.isnull().sum())
    print("\nDuplicated rows:", dataset.duplicated().sum())
    print("\nSample:")
    print(dataset.head())
