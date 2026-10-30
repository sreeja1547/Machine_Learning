# ============================================================
# LOAN RISK ASSESSMENT API
# FastAPI backend for the trained Loan Risk model
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path

import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# SMOTE
# ============================================================

def smote(X, y, k=5, seed=42):

    try:

        from imblearn.over_sampling import SMOTE

        return SMOTE(
            k_neighbors=k,
            random_state=seed
        ).fit_resample(X, y)

    except ImportError:

        rng = np.random.RandomState(seed)

        X = np.asarray(X)
        y = np.asarray(y)

        minority = X[y == 1]

        need = int(
            (y == 0).sum()
            -
            (y == 1).sum()
        )

        neighbours = (
            NearestNeighbors(
                n_neighbors=k + 1
            )
            .fit(minority)
            .kneighbors(
                minority,
                return_distance=False
            )[:, 1:]
        )

        i = rng.randint(
            0,
            len(minority),
            need
        )

        j = neighbours[
            i,
            rng.randint(0, k, need)
        ]

        synthetic = (
            minority[i]
            +
            rng.rand(need, 1)
            *
            (
                minority[j]
                -
                minority[i]
            )
        )

        return (
            np.vstack([
                X,
                synthetic
            ]),
            np.concatenate([
                y,
                np.ones(
                    need,
                    dtype=int
                )
            ])
        )


# ============================================================
# BALANCED CLASSIFIER
# Required to load the saved Loan Risk model
# ============================================================

class BalancedClassifier(
    ClassifierMixin,
    BaseEstimator
):

    def __init__(
        self,
        base_model=None,
        strategy="none"
    ):

        self.base_model = base_model
        self.strategy = strategy

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(y)

        self.scaler_ = StandardScaler().fit(X)

        Xs = self.scaler_.transform(X)

        if self.strategy == "smote":

            Xs, y = smote(
                Xs,
                y
            )

        self.model_ = clone(
            self.base_model
        ).fit(
            Xs,
            y
        )

        self.classes_ = np.array([
            0,
            1
        ])

        return self

    def predict_proba(self, X):

        X = np.asarray(
            X,
            dtype=float
        )

        Xs = self.scaler_.transform(X)

        return self.model_.predict_proba(Xs)

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return (
            probabilities[:, 1] >= 0.5
        ).astype(int)


# ============================================================
# JOBLIB COMPATIBILITY
# ============================================================

# Your trained model was saved when BalancedClassifier
# belonged to the __main__ module.

sys.modules["__main__"].BalancedClassifier = BalancedClassifier


# ============================================================
# LOAD LOAN RISK MODEL
# ============================================================

# This is the exact location of your trained model.
MODEL_FILE = Path(
    r"C:\Users\Dell\OneDrive\ドキュメント\python\credit_risk_model.joblib"
)


print()
print("=" * 60)
print("LOADING LOAN RISK MODEL")
print("=" * 60)

print("Model file:")
print(MODEL_FILE)


# Check whether model exists
if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nLoan Risk model not found.\n"
        f"Expected file:\n{MODEL_FILE}\n\n"
        f"Please check that credit_risk_model.joblib "
        f"exists at this location."
    )


# Load trained model
bundle = joblib.load(MODEL_FILE)


# ============================================================
# EXTRACT MODEL INFORMATION
# ============================================================

model = bundle["model"]

prep = bundle["prep"]

features = bundle["features"]

threshold = bundle["threshold"]

model_name = bundle.get(
    "model_name",
    "XGBoost"
)


print()
print("Loan Risk model loaded successfully")
print("Model       :", model_name)
print("Threshold   :", threshold)
print("Features    :", len(features))
print("=" * 60)
print()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Intelligent Loan Risk Assessment API",
    description=(
        "API for assessing loan default risk "
        "using a trained machine learning model."
    ),
    version="1.0.0"
)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class LoanApplication(BaseModel):

    person_age: float = Field(
        ...,
        ge=18,
        le=100
    )

    person_income: float = Field(
        ...,
        gt=0
    )

    person_home_ownership: str

    person_emp_length: float = Field(
        ...,
        ge=0,
        le=60
    )

    loan_intent: str

    loan_grade: str

    loan_amnt: float = Field(
        ...,
        gt=0
    )

    loan_int_rate: float = Field(
        ...,
        gt=0
    )

    loan_percent_income: float = Field(
        ...,
        gt=0
    )

    cb_person_default_on_file: str

    cb_person_cred_hist_length: float = Field(
        ...,
        ge=0
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Intelligent Loan Risk Assessment API",
        "status": "running",
        "model": model_name,
        "threshold": threshold
    }




# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": model_name,
        "threshold": threshold,
        "number_of_features": len(features),
        "features": features
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(applicant):

    df = pd.DataFrame([applicant])

    input_columns = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length"
    ]

    X = df[input_columns].copy()


    # --------------------------------------------------------
    # MISSING VALUE INDICATORS
    # --------------------------------------------------------

    X["rate_missing"] = (
        X["loan_int_rate"]
        .isna()
        .astype(int)
    )

    X["emp_missing"] = (
        X["person_emp_length"]
        .isna()
        .astype(int)
    )


    # --------------------------------------------------------
    # MISSING VALUE HANDLING
    # Same logic as training
    # --------------------------------------------------------

    X["loan_int_rate"] = (
        X["loan_int_rate"]
        .fillna(
            X["loan_grade"].map(
                prep["rate_by_grade"]
            )
        )
        .fillna(
            prep["rate_overall"]
        )
    )

    X["person_emp_length"] = (
        X["person_emp_length"]
        .fillna(
            prep["emp_median"]
        )
    )


    # --------------------------------------------------------
    # LOAN GRADE ENCODING
    # --------------------------------------------------------

    grade_mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    }

    X["loan_grade"] = (
        X["loan_grade"]
        .map(grade_mapping)
    )


    # --------------------------------------------------------
    # PREVIOUS DEFAULT
    # --------------------------------------------------------

    X["default_on_file"] = (
        X["cb_person_default_on_file"] == "Y"
    ).astype(int)


    # --------------------------------------------------------
    # HOME OWNERSHIP
    # --------------------------------------------------------

    home_options = [
        "MORTGAGE",
        "OTHER",
        "OWN",
        "RENT"
    ]

    for home in home_options:

        X["home_" + home] = (
            X["person_home_ownership"] == home
        ).astype(int)


    # --------------------------------------------------------
    # LOAN INTENT
    # --------------------------------------------------------

    intent_options = [
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE"
    ]

    for intent in intent_options:

        X["intent_" + intent] = (
            X["loan_intent"] == intent
        ).astype(int)


    # --------------------------------------------------------
    # REMOVE ORIGINAL TEXT COLUMNS
    # --------------------------------------------------------

    X = X.drop(
        columns=[
            "person_home_ownership",
            "loan_intent",
            "cb_person_default_on_file"
        ]
    )


    # --------------------------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------------------------

    X["income_log"] = np.log1p(
        X["person_income"]
    )

    X["loan_to_income"] = (
        X["loan_amnt"]
        /
        X["person_income"]
    )

    X["rate_x_pct"] = (
        X["loan_int_rate"]
        *
        X["loan_percent_income"]
    )

    X["grade_x_pct"] = (
        X["loan_grade"]
        *
        X["loan_percent_income"]
    )

    X["credit_hist_ratio"] = (
        X["cb_person_cred_hist_length"]
        /
        X["person_age"]
    )

    X["emp_to_age"] = (
        X["person_emp_length"]
        /
        X["person_age"]
    )


    # --------------------------------------------------------
    # SAME FEATURE ORDER AS TRAINING
    # --------------------------------------------------------

    X = X[features]


    # --------------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------------

    if X.isna().any().any():

        raise ValueError(
            "Missing values remain after preprocessing."
        )

    return X


# ============================================================
# LOAN RISK PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    application: LoanApplication
):

    try:

        # Convert Pydantic object to dictionary
        applicant = application.model_dump()


        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        X = make_features(
            applicant
        )


        # ----------------------------------------------------
        # PREDICT DEFAULT PROBABILITY
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(X)[:, 1][0]
        )


        # ----------------------------------------------------
        # RISK CLASSIFICATION
        # ----------------------------------------------------

        low_cut = min(
            0.10,
            threshold / 2
        )


        if probability >= threshold:

            risk_level = "HIGH"

            decision = "REJECT / MANUAL REVIEW"


        elif probability >= low_cut:

            risk_level = "MEDIUM"

            decision = "APPROVE WITH CARE"


        else:

            risk_level = "LOW"

            decision = "APPROVE"


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "default_probability": round(
                probability,
                4
            ),

            "default_probability_percent": round(
                probability * 100,
                2
            ),

            "risk_level": risk_level,

            "decision": decision,

            "threshold": threshold,

            "model": model_name

        }


    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# END
# ============================================================