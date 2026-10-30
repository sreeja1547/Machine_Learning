# ============================================================
# INTELLIGENT LOAN RISK ASSESSMENT
# SINGLE FILE VERSION
#
# DATASET:
# 12 ORIGINAL COLUMNS
# + 6 FEATURE-ENGINEERED COLUMNS
# = 18 DATASET COLUMNS
#
# PREPROCESSING / ENCODING COLUMNS ARE NOT ADDED TO DATASET
# They are created internally only during model training.
# ============================================================


import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from sklearn.base import BaseEstimator, ClassifierMixin, clone

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

    HAS_XGB = True

except ImportError:

    HAS_XGB = False

    print("WARNING: XGBoost is not installed.")
    print("Run: pip install xgboost")


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "credit_risk_dataset.csv"

SEED = 42

N_TREES = 300


# ============================================================
# DATASET INFORMATION
# ============================================================

# These are the ORIGINAL dataset columns
ORIGINAL_COLUMNS = [

    "person_age",

    "person_income",

    "person_home_ownership",

    "person_emp_length",

    "loan_intent",

    "loan_grade",

    "loan_amnt",

    "loan_int_rate",

    "loan_status",

    "loan_percent_income",

    "cb_person_default_on_file",

    "cb_person_cred_hist_length"

]


# Target column
TARGET_COLUMN = "loan_status"


# ============================================================
# ORIGINAL INPUT COLUMNS
# ============================================================

INPUT_COLUMNS = [

    column

    for column in ORIGINAL_COLUMNS

    if column != TARGET_COLUMN

]


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CATEGORICAL_COLUMNS = [

    "person_home_ownership",

    "loan_intent",

    "loan_grade",

    "cb_person_default_on_file"

]


# ============================================================
# NUMERICAL ORIGINAL COLUMNS
# ============================================================

NUMERICAL_COLUMNS = [

    "person_age",

    "person_income",

    "person_emp_length",

    "loan_amnt",

    "loan_int_rate",

    "loan_percent_income",

    "cb_person_cred_hist_length"

]


# ============================================================
# SIX FEATURE-ENGINEERED COLUMNS
# ============================================================

ENGINEERED_COLUMNS = [

    "income_log",

    "loan_to_income",

    "rate_x_pct",

    "grade_x_pct",

    "credit_hist_ratio",

    "emp_to_age"

]


# ============================================================
# FINAL DATASET FEATURE COLUMNS
#
# 11 original input columns
# + 6 engineered columns
# = 17 input features
#
# loan_status is the target.
# Therefore the complete dataset including target has
# 18 columns.
# ============================================================

FINAL_FEATURE_COLUMNS = (

    INPUT_COLUMNS

    + ENGINEERED_COLUMNS

)


# ============================================================
# GRADE INFORMATION
# ============================================================

GRADES = list("ABCDEFG")


# ============================================================
# HOME OWNERSHIP OPTIONS
# ============================================================

HOME_OPTIONS = [

    "MORTGAGE",

    "OTHER",

    "OWN",

    "RENT"

]


# ============================================================
# LOAN INTENT OPTIONS
# ============================================================

INTENT_OPTIONS = [

    "DEBTCONSOLIDATION",

    "EDUCATION",

    "HOMEIMPROVEMENT",

    "MEDICAL",

    "PERSONAL",

    "VENTURE"

]


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_name):

    data = pd.read_csv(file_name)

    # Make numeric columns numeric
    numeric_columns = [

        "person_age",

        "person_income",

        "person_emp_length",

        "loan_amnt",

        "loan_int_rate",

        "loan_status",

        "loan_percent_income",

        "cb_person_cred_hist_length"

    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(

            data[column],

            errors="coerce"

        )

    return data


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove impossible age
    df = df[
        df["person_age"] <= 100
    ]

    # Remove impossible employment length
    df = df[
        df["person_emp_length"].fillna(0) <= 60
    ]

    return df.reset_index(drop=True)


# ============================================================
# 3. PREPROCESSING INFORMATION
#
# This information is used internally.
#
# It is NOT added as dataset columns.
# ============================================================

def fit_prep(train_df):

    rate = train_df["loan_int_rate"]

    return {

        "rate_by_grade":
            rate.groupby(
                train_df["loan_grade"]
            ).median().to_dict(),

        "rate_overall":
            float(rate.median()),

        "emp_median":
            float(
                train_df[
                    "person_emp_length"
                ].median()
            )

    }


# ============================================================
# 4. FEATURE ENGINEERING
#
# IMPORTANT:
#
# Only the six engineered columns are added.
#
# NO:
# rate_missing
# emp_missing
# home_MORTGAGE
# home_OTHER
# home_OWN
# home_RENT
# intent_*
# default_on_file
#
# are added here.
# ============================================================

def make_features(df, prep):

    # Start with ORIGINAL input columns
    X = df[
        INPUT_COLUMNS
    ].copy()


    # --------------------------------------------------------
    # TEMPORARY VALUES FOR CALCULATING FEATURES
    #
    # These are NOT added as columns.
    # --------------------------------------------------------

    # Interest rate used for feature calculation
    rate_for_features = (

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


    # Employment length used for feature calculation
    emp_for_features = (

        X["person_emp_length"]

        .fillna(
            prep["emp_median"]
        )

    )


    # Numerical grade value used only for
    # calculating grade_x_pct
    grade_numeric = (

        X["loan_grade"]

        .map(
            {
                grade: number

                for number, grade

                in enumerate(GRADES)
            }
        )

    )


    # --------------------------------------------------------
    # SIX FEATURE-ENGINEERED COLUMNS ONLY
    # --------------------------------------------------------

    # 1. Log transformed income
    X["income_log"] = np.log1p(
        X["person_income"]
    )


    # 2. Loan amount / income
    X["loan_to_income"] = (

        X["loan_amnt"]

        /

        X["person_income"]

    )


    # 3. Interest rate × loan percentage of income
    X["rate_x_pct"] = (

        rate_for_features

        *

        X["loan_percent_income"]

    )


    # 4. Grade × loan percentage of income
    X["grade_x_pct"] = (

        grade_numeric

        *

        X["loan_percent_income"]

    )


    # 5. Credit history / age
    X["credit_hist_ratio"] = (

        X["cb_person_cred_hist_length"]

        /

        X["person_age"]

    )


    # 6. Employment length / age
    X["emp_to_age"] = (

        emp_for_features

        /

        X["person_age"]

    )


    # Return 17 input columns:
    # 11 original input columns + 6 engineered
    return X


# ============================================================
# 5. INTERNAL PREPROCESSING
#
# IMPORTANT:
#
# This does NOT modify the dataset.
#
# It converts categorical information into numerical form
# internally before model training.
# ============================================================

def create_preprocessor():

    # Numerical columns:
    #
    # 7 original numerical columns
    # + 6 engineered columns
    #
    numeric_columns = (

        NUMERICAL_COLUMNS

        +

        ENGINEERED_COLUMNS

    )


    # Numerical preprocessing
    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )

            ),

            (
                "scaler",

                StandardScaler()

            )

        ]

    )


    # Categorical preprocessing
    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )

            ),

            (
                "onehot",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )

            )

        ]

    )


    # Combine numerical + categorical preprocessing
    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",

                numeric_pipeline,

                numeric_columns

            ),

            (
                "categorical",

                categorical_pipeline,

                CATEGORICAL_COLUMNS

            )

        ]

    )


    return preprocessor


# ============================================================
# 6. SMOTE
# ============================================================

def smote(X, y, k=5, seed=42):

    try:

        from imblearn.over_sampling import SMOTE

        return SMOTE(

            k_neighbors=k,

            random_state=seed

        ).fit_resample(

            X,

            y

        )


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


        if need <= 0:

            return X, y


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

            rng.randint(

                0,

                k,

                need

            )

        ]


        synthetic = (

            minority[i]

            +

            rng.rand(

                need,

                1

            )

            *

            (

                minority[j]

                -

                minority[i]

            )

        )


        return (

            np.vstack(

                [

                    X,

                    synthetic

                ]

            ),

            np.concatenate(

                [

                    y,

                    np.ones(

                        need,

                        dtype=int

                    )

                ]

            )

        )


# ============================================================
# 7. BALANCED CLASSIFIER
#
# Preprocessing happens INSIDE this class.
#
# Therefore preprocessing columns never become part of
# the original dataset.
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

        # Create preprocessing pipeline
        self.preprocessor_ = create_preprocessor()


        # Fit preprocessing ONLY on training data
        X_processed = (

            self.preprocessor_

            .fit_transform(X)

        )


        # Apply SMOTE only when requested
        if self.strategy == "smote":

            X_processed, y = smote(

                X_processed,

                y

            )


        # Train model
        self.model_ = clone(

            self.base_model

        ).fit(

            X_processed,

            y

        )


        self.classes_ = np.array(

            [0, 1]

        )


        return self


    def predict_proba(self, X):

        X_processed = (

            self.preprocessor_

            .transform(X)

        )


        return (

            self.model_

            .predict_proba(

                X_processed

            )

        )


    def predict(self, X):

        probabilities = (

            self.predict_proba(X)[:, 1]

        )


        return (

            probabilities >= 0.5

        ).astype(int)


# ============================================================
# 8. LOAD + CLEAN
# ============================================================

raw = load_data(

    CSV_FILE

)


df = clean_data(

    raw

)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    "\nRows loaded:",
    len(raw)
)

print(
    "Rows after cleaning:",
    len(df)
)

print(
    "Original dataset columns:",
    len(ORIGINAL_COLUMNS)
)


print(
    "\nOriginal columns:"
)

for column in ORIGINAL_COLUMNS:

    print(
        " -",
        column
    )


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(

    df,

    test_size=0.20,

    stratify=df.loan_status,

    random_state=SEED

)


# ============================================================
# 10. PREPARE FEATURE INFORMATION
# ============================================================

prep = fit_prep(

    train_df

)


X_train = make_features(

    train_df,

    prep

)


X_test = make_features(

    test_df,

    prep

)


y_train = (

    train_df

    .loan_status

    .values

)


y_test = (

    test_df

    .loan_status

    .values

)


FEATURES = list(

    X_train.columns

)


# ============================================================
# VERIFY DATASET FEATURE STRUCTURE
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE STRUCTURE")
print("=" * 70)

print(
    "\nOriginal input columns:",
    len(INPUT_COLUMNS)
)

print(
    "Feature-engineered columns:",
    len(ENGINEERED_COLUMNS)
)

print(
    "Total model input columns before internal preprocessing:",
    len(FEATURES)
)


print("\nFeature-engineered columns:")

for column in ENGINEERED_COLUMNS:

    print(
        " -",
        column
    )


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

POS_WEIGHT = (

    (y_train == 0).sum()

    /

    (y_train == 1).sum()

)


# ============================================================
# 11. CREATE BASE MODEL
# ============================================================

def base_model(

    name,

    class_weights=False

):


    if name == "Logistic Regression":

        return LogisticRegression(

            C=1.0,

            max_iter=3000,

            class_weight=(

                "balanced"

                if class_weights

                else None

            )

        )


    if name == "Random Forest":

        return RandomForestClassifier(

            n_estimators=N_TREES,

            min_samples_leaf=3,

            max_features="sqrt",

            n_jobs=-1,

            random_state=SEED,

            class_weight=(

                "balanced"

                if class_weights

                else None

            )

        )


    if HAS_XGB:

        return XGBClassifier(

            n_estimators=400,

            learning_rate=0.05,

            max_depth=6,

            subsample=0.8,

            colsample_bytree=0.8,

            eval_metric="logloss",

            n_jobs=-1,

            random_state=SEED,

            scale_pos_weight=(

                POS_WEIGHT

                if class_weights

                else 1

            )

        )


    return HistGradientBoostingClassifier(

        learning_rate=0.05,

        max_iter=300,

        max_leaf_nodes=20,

        min_samples_leaf=20,

        l2_regularization=1.0,

        random_state=SEED

    )


# ============================================================
# 12. BUILD MODEL
# ============================================================

def build(

    name,

    strategy

):


    if strategy == "Class weights":

        return BalancedClassifier(

            base_model(

                name,

                True

            ),

            "none"

        )


    if strategy == "SMOTE":

        return BalancedClassifier(

            base_model(

                name

            ),

            "smote"

        )


    return BalancedClassifier(

        base_model(

            name

        ),

        "none"

    )


# ============================================================
# 13. METRIC FUNCTION
# ============================================================

def score(

    y,

    probabilities,

    threshold=0.5

):

    predictions = (

        probabilities >= threshold

    )


    return {

        "ROC_AUC":

            roc_auc_score(

                y,

                probabilities

            ),

        "PR_AUC":

            average_precision_score(

                y,

                probabilities

            ),

        "Accuracy":

            accuracy_score(

                y,

                predictions

            ),

        "Precision":

            precision_score(

                y,

                predictions,

                zero_division=0

            ),

        "Recall":

            recall_score(

                y,

                predictions,

                zero_division=0

            ),

        "F1":

            f1_score(

                y,

                predictions,

                zero_division=0

            )

    }


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

MODELS = [

    "Logistic Regression",

    "Random Forest",

    "XGBoost"

]


STRATEGIES = [

    "No balancing",

    "Class weights",

    "SMOTE"

]


cv5 = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=SEED

)


rows = []

oof_store = {}


for model_name in MODELS:

    for strategy in STRATEGIES:

        model = build(

            model_name,

            strategy

        )


        probabilities = (

            cross_val_predict(

                model,

                X_train,

                y_train,

                cv=cv5,

                method="predict_proba"

            )[:, 1]

        )


        oof_store[

            (

                model_name,

                strategy

            )

        ] = probabilities


        metrics = score(

            y_train,

            probabilities

        )


        metrics["Model"] = (

            model_name

        )


        metrics["Balancing"] = (

            strategy

        )


        rows.append(

            metrics

        )


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(

    rows

)[

    [

        "Model",

        "Balancing",

        "ROC_AUC",

        "PR_AUC",

        "Accuracy",

        "Precision",

        "Recall",

        "F1"

    ]

].round(4)


comparison.to_csv(

    "model_comparison.csv",

    index=False

)


# ============================================================
# 15. CROSS-VALIDATION METRICS
# ============================================================

cv_rows = []


for model_name in MODELS:

    for strategy in STRATEGIES:

        model = build(

            model_name,

            strategy

        )


        fold_metrics = {

            "ROC_AUC": [],

            "PR_AUC": [],

            "Accuracy": [],

            "Precision": [],

            "Recall": [],

            "F1": []

        }


        for train_index, val_index in cv5.split(

            X_train,

            y_train

        ):


            X_cv_train = X_train.iloc[

                train_index

            ]


            X_cv_val = X_train.iloc[

                val_index

            ]


            y_cv_train = y_train[

                train_index

            ]


            y_cv_val = y_train[

                val_index

            ]


            cv_model = clone(

                model

            )


            cv_model.fit(

                X_cv_train,

                y_cv_train

            )


            cv_probability = (

                cv_model

                .predict_proba(

                    X_cv_val

                )[:, 1]

            )


            cv_prediction = (

                cv_probability >= 0.5

            )


            fold_metrics[

                "ROC_AUC"

            ].append(

                roc_auc_score(

                    y_cv_val,

                    cv_probability

                )

            )


            fold_metrics[

                "PR_AUC"

            ].append(

                average_precision_score(

                    y_cv_val,

                    cv_probability

                )

            )


            fold_metrics[

                "Accuracy"

            ].append(

                accuracy_score(

                    y_cv_val,

                    cv_prediction

                )

            )


            fold_metrics[

                "Precision"

            ].append(

                precision_score(

                    y_cv_val,

                    cv_prediction,

                    zero_division=0

                )

            )


            fold_metrics[

                "Recall"

            ].append(

                recall_score(

                    y_cv_val,

                    cv_prediction,

                    zero_division=0

                )

            )


            fold_metrics[

                "F1"

            ].append(

                f1_score(

                    y_cv_val,

                    cv_prediction,

                    zero_division=0

                )

            )


        cv_rows.append({

            "Model":

                model_name,

            "Balancing":

                strategy,

            "ROC_AUC_Mean":

                np.mean(

                    fold_metrics["ROC_AUC"]

                ),

            "ROC_AUC_STD":

                np.std(

                    fold_metrics["ROC_AUC"]

                ),

            "PR_AUC_Mean":

                np.mean(

                    fold_metrics["PR_AUC"]

                ),

            "PR_AUC_STD":

                np.std(

                    fold_metrics["PR_AUC"]

                ),

            "Accuracy_Mean":

                np.mean(

                    fold_metrics["Accuracy"]

                ),

            "Accuracy_STD":

                np.std(

                    fold_metrics["Accuracy"]

                ),

            "Precision_Mean":

                np.mean(

                    fold_metrics["Precision"]

                ),

            "Precision_STD":

                np.std(

                    fold_metrics["Precision"]

                ),

            "Recall_Mean":

                np.mean(

                    fold_metrics["Recall"]

                ),

            "Recall_STD":

                np.std(

                    fold_metrics["Recall"]

                ),

            "F1_Mean":

                np.mean(

                    fold_metrics["F1"]

                ),

            "F1_STD":

                np.std(

                    fold_metrics["F1"]

                )

        })


cv_summary = pd.DataFrame(

    cv_rows

)


cv_summary.to_csv(

    "cross_validation_metrics.csv",

    index=False

)


# ============================================================
# 16. SELECT BEST MODEL
#
# First ROC-AUC
# Then F1
# ============================================================

best = (

    comparison

    .sort_values(

        [

            "ROC_AUC",

            "F1"

        ],

        ascending=False

    )

    .iloc[0]

)


best_model = best.Model

best_strategy = best.Balancing


selected_cv = cv_summary[

    (cv_summary["Model"] == best_model)

    &

    (cv_summary["Balancing"] == best_strategy)

].iloc[0]


# ============================================================
# 17. ROC CURVE
# SELECTED MODEL ONLY
# ============================================================

selected_oof = oof_store[

    (

        best_model,

        best_strategy

    )

]


false_positive_rate, true_positive_rate, roc_thresholds = (

    roc_curve(

        y_train,

        selected_oof

    )

)


selected_roc_auc = roc_auc_score(

    y_train,

    selected_oof

)


plt.figure(

    figsize=(8, 6)

)


plt.plot(

    false_positive_rate,

    true_positive_rate,

    label=(

        f"{best_model} + "

        f"{best_strategy} "

        f"(ROC-AUC = "

        f"{selected_roc_auc:.4f})"

    )

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"

)


plt.xlabel(

    "False Positive Rate"

)

plt.ylabel(

    "True Positive Rate"

)

plt.title(

    "ROC Curve - Selected Model"

)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    "selected_model_roc_curve.png",

    dpi=300

)

plt.close()


# ============================================================
# 18. PRECISION-RECALL CURVE
# ============================================================

precision, recall, pr_thresholds = (

    precision_recall_curve(

        y_train,

        selected_oof

    )

)


selected_pr_auc = average_precision_score(

    y_train,

    selected_oof

)


plt.figure(

    figsize=(8, 6)

)


plt.plot(

    recall,

    precision,

    label=(

        f"{best_model} + "

        f"{best_strategy} "

        f"(PR-AUC = "

        f"{selected_pr_auc:.4f})"

    )

)


plt.xlabel(

    "Recall"

)

plt.ylabel(

    "Precision"

)

plt.title(

    "Precision-Recall Curve - Selected Model"

)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    "selected_model_precision_recall_curve.png",

    dpi=300

)

plt.close()


# ============================================================
# 19. RANDOMIZED SEARCH
# ============================================================

if best_model == "Logistic Regression":

    grid = {

        "C": [

            0.01,

            0.1,

            1,

            10,

            100

        ]

    }


elif best_model == "Random Forest":

    grid = {

        "n_estimators": [

            200,

            300,

            500

        ],

        "max_depth": [

            8,

            12,

            16,

            None

        ],

        "min_samples_leaf": [

            1,

            3,

            5,

            10

        ],

        "max_features": [

            "sqrt",

            0.5

        ]

    }


elif HAS_XGB:

    grid = {

        "n_estimators": [

            300,

            500,

            800

        ],

        "learning_rate": [

            0.03,

            0.05,

            0.1

        ],

        "max_depth": [

            3,

            4,

            6,

            8

        ],

        "subsample": [

            0.7,

            0.9

        ],

        "colsample_bytree": [

            0.6,

            0.8,

            1.0

        ]

    }


else:

    grid = {

        "learning_rate": [

            0.03,

            0.05,

            0.1

        ],

        "max_leaf_nodes": [

            15,

            20,

            31,

            40

        ],

        "min_samples_leaf": [

            10,

            20,

            40

        ],

        "max_iter": [

            200,

            300,

            500

        ]

    }


search = RandomizedSearchCV(

    build(

        best_model,

        best_strategy

    ),

    {

        "base_model__" + key: value

        for key, value in grid.items()

    },

    n_iter=8,

    scoring="roc_auc",

    cv=StratifiedKFold(

        3,

        shuffle=True,

        random_state=SEED

    ),

    random_state=SEED,

    n_jobs=1

)


search.fit(

    X_train,

    y_train

)


tuned = search.best_estimator_


# ============================================================
# 20. FIND DECISION THRESHOLD
# ============================================================

oof = (

    cross_val_predict(

        tuned,

        X_train,

        y_train,

        cv=cv5,

        method="predict_proba"

    )[:, 1]

)


grid_t = np.arange(

    0.05,

    0.90,

    0.01

)


weakest = []


for threshold in grid_t:

    predictions = (

        oof >= threshold

    )


    accuracy = accuracy_score(

        y_train,

        predictions

    )


    precision_value = precision_score(

        y_train,

        predictions,

        zero_division=0

    )


    recall_value = recall_score(

        y_train,

        predictions,

        zero_division=0

    )


    f1_value = f1_score(

        y_train,

        predictions,

        zero_division=0

    )


    # Maximize the weakest metric
    weakest.append(

        min(

            accuracy,

            precision_value,

            recall_value,

            f1_value

        )

    )


THRESHOLD = float(

    grid_t[

        int(

            np.argmax(

                weakest

            )

        )

    ]

)


# ============================================================
# 21. FINAL MODEL
# ============================================================

final = tuned.fit(

    X_train,

    y_train

)


# ============================================================
# 22. FINAL TEST
# ============================================================

test_probabilities = (

    final.predict_proba(

        X_test

    )[:, 1]

)


# Selected threshold
final_results = score(

    y_test,

    test_probabilities,

    THRESHOLD

)


# Standard threshold
default_results = score(

    y_test,

    test_probabilities,

    0.50

)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

predictions = (

    test_probabilities >= THRESHOLD

)


cm = confusion_matrix(

    y_test,

    predictions

)


# ============================================================
# 24. FINAL SELECTED MODEL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SELECTED MODEL")
print("=" * 70)


print(

    "\nModel       :",

    best_model

)


print(

    "Balancing   :",

    best_strategy

)


print(

    "Threshold   :",

    round(

        THRESHOLD,

        2

    )

)


# ============================================================
# FINAL TEST METRICS
# ============================================================

print("\n")
print("TEST DATA METRICS")
print("-" * 70)


print(

    f"Accuracy    : "

    f"{final_results['Accuracy']:.4f}"

)


print(

    f"Precision   : "

    f"{final_results['Precision']:.4f}"

)


print(

    f"Recall      : "

    f"{final_results['Recall']:.4f}"

)


print(

    f"F1-Score    : "

    f"{final_results['F1']:.4f}"

)


print(

    f"ROC-AUC     : "

    f"{final_results['ROC_AUC']:.4f}"

)


print(

    f"PR-AUC      : "

    f"{final_results['PR_AUC']:.4f}"

)


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print("\n")
print("5-FOLD CROSS VALIDATION")
print("-" * 70)


print(

    f"ROC-AUC     : "

    f"{selected_cv['ROC_AUC_Mean']:.4f} ± "

    f"{selected_cv['ROC_AUC_STD']:.4f}"

)


print(

    f"Precision   : "

    f"{selected_cv['Precision_Mean']:.4f} ± "

    f"{selected_cv['Precision_STD']:.4f}"

)


print(

    f"Recall      : "

    f"{selected_cv['Recall_Mean']:.4f} ± "

    f"{selected_cv['Recall_STD']:.4f}"

)


print(

    f"F1-Score    : "

    f"{selected_cv['F1_Mean']:.4f} ± "

    f"{selected_cv['F1_STD']:.4f}"

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("CONFUSION MATRIX")
print("-" * 70)

print(cm)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

print("\n")
print("PRECISION-RECALL CURVE")
print("-" * 70)


print(

    f"PR-AUC      : "

    f"{final_results['PR_AUC']:.4f}"

)


print(

    "Saved       : "

    "selected_model_precision_recall_curve.png"

)


# ============================================================
# ROC CURVE
# ============================================================

print("\n")
print("ROC CURVE")
print("-" * 70)


print(

    "Saved       : "

    "selected_model_roc_curve.png"

)


# ============================================================
# 25. SAVE MODEL
# ============================================================

# Typical values for API
typical = {

    column: (

        train_df[column].mode()[0]

        if column in CATEGORICAL_COLUMNS

        else float(

            train_df[column].median()

        )

    )

    for column in INPUT_COLUMNS

}


bundle = {

    "model": final,

    "prep": prep,

    "features": FEATURES,

    "threshold": THRESHOLD,

    "typical": typical,

    "model_name":

        f"{best_model} + {best_strategy}",

    "metrics_test":

        final_results,

    "metrics_test_threshold_0.5":

        default_results,

    # Store the exact six engineered features
    "engineered_features":

        ENGINEERED_COLUMNS,

    # Store the original dataset columns
    "original_columns":

        ORIGINAL_COLUMNS

}


joblib.dump(

    bundle,

    "credit_risk_model.joblib"

)


# ============================================================
# 26. SAVE FINAL METRICS
# ============================================================

with open(

    "final_metrics.json",

    "w"

) as file:

    json.dump(

        {

            "model":

                bundle["model_name"],

            "threshold":

                THRESHOLD,

            "original_dataset_columns":

                ORIGINAL_COLUMNS,

            "engineered_features":

                ENGINEERED_COLUMNS,

            "dataset_feature_count":

                len(FEATURES),

            "test_balanced":

                final_results,

            "test_standard":

                default_results

        },

        file,

        indent=2

    )


# ============================================================
# 27. DONE
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)


print(

    "Saved: credit_risk_model.joblib"

)


print(

    "Saved: final_metrics.json"

)


print(

    "Saved: model_comparison.csv"

)


print(

    "Saved: cross_validation_metrics.csv"

)


print(

    "Saved: selected_model_roc_curve.png"

)


print(

    "Saved: selected_model_precision_recall_curve.png"

)


print("\n")
print("=" * 70)
print("DATASET STRUCTURE")
print("=" * 70)


print(

    "Original columns      : 12"

)


print(

    "Engineered columns    : 6"

)


print(

    "Total dataset columns : 18"

)


print("\nFeature-engineered columns:")

for column in ENGINEERED_COLUMNS:

    print(

        " -",

        column

    )


print("\nPreprocessing columns were NOT added to the dataset.")