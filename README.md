# Intelligent Loan Risk Assessment System

A machine-learning-based application that estimates loan default risk from applicant and loan details. The project uses a Streamlit user interface and a FastAPI backend.

> 

## Features

- Predicts loan default risk from applicant and loan information.
- Supports classification models such as Logistic Regression, Random Forest, and XGBoost, depending on which models are trained and saved in the project.
- Includes data cleaning and feature engineering.
- Supports evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Provides a Streamlit interface and FastAPI prediction endpoint.
- Can evaluate class-imbalance approaches such as class weighting and SMOTE if implemented in the training workflow.

## Dataset

**Source:** [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) by Lao Tse.

The original CSV contains **32,581 records and 12 columns**. Important columns include:

| Column | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | Home-ownership status |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Intended purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Credit-history length |
| `loan_status` | Target label: `0` = non-default, `1` = default |

Project notes indicate that cleaning reduced the dataset from **32,581 to 32,409 rows**. The project also describes 18 features after preprocessing and feature engineering. Verify these values against the actual training code and saved model before presenting them as final results.

## Project Architecture

1. **Dataset and preprocessing:** Load the CSV, inspect missing values, clean data, encode categorical columns, and prepare features.
2. **Feature engineering:** The project may derive variables such as:
   - `income_log = log1p(person_income)`
   - `loan_to_income = loan_amnt / person_income`
   - `rate_x_pct = loan_int_rate * loan_percent_income`
   - `grade_x_pct = loan_grade * loan_percent_income` (after encoding `loan_grade` numerically)
   - `credit_hist_ratio = cb_person_cred_hist_length / person_age`
   - `emp_to_age = person_emp_length / person_age`
3. **Model training and evaluation:** Compare the models implemented in the training code using a held-out test set and suitable metrics.
4. **FastAPI backend:** Validate input data, prepare model features, load the saved model, and return predictions.
5. **Streamlit frontend:** Collect user input, call the backend, and display predicted risk.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost (if used by the training pipeline)
- Imbalanced-learn (if SMOTE is used)
- FastAPI and Uvicorn
- Streamlit
- Requests
- Joblib

## Setup

### 1. Download or clone the project

If the project is hosted in a Git repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

Replace the placeholders with your actual repository URL and folder name. If you already have the project locally, open a terminal in that folder.

### 2. Create and activate a virtual environment (optional)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the project includes `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the core packages:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn fastapi "uvicorn[standard]" streamlit requests joblib
```

Use the dependency versions tested by your project if a requirements file is available.

### 4. Download the dataset

Download `credit_risk_dataset.csv` from the Kaggle source above and place it where your preprocessing or training script expects it.

## Running the Application

The exact filenames may differ. The examples below assume the backend file is `api.py` and the Streamlit frontend file is `app.py`. Replace these names with your actual filenames.

### Start the FastAPI backend

```bash
uvicorn api:app --reload
```

The API is typically available at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` to inspect and test available endpoints.

### Start the Streamlit frontend

Open a **second terminal**, activate the same virtual environment, and run:

```bash
streamlit run app.py
```

If your frontend file has a different name, replace `app.py`.

### Model files

Ensure the saved model artifact(s), such as `.joblib` files, exist at the paths expected by the backend. Predictions will fail if the artifacts are missing or incompatible with the input feature order.

## Evaluation Metrics

- **Precision:** Of the applicants predicted as defaults, how many were actual defaults?
- **Recall:** Of the actual defaults, how many did the model identify?
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across thresholds.
- **PR-AUC:** Useful for evaluating performance on the minority/default class.
- **Confusion matrix:** Shows true positives, true negatives, false positives, and false negatives.

Do not claim one model is best until you compare actual test results. If SMOTE is used, apply it **only to training data inside the training pipeline**, never to the held-out test set.

## API Usage

The backend may provide an endpoint such as `POST /predict`. Confirm the actual route and request schema in the FastAPI code before using the example below.

Example request shape (adjust field names and values to match the actual API schema):

```json
{
  "person_age": 25,
  "person_income": 50000,
  "person_emp_length": 3,
  "loan_amnt": 10000,
  "loan_int_rate": 12.5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "person_home_ownership": "RENT",
  "loan_percent_income": 0.2,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

## Limitations

- The Kaggle dataset is a public learning dataset, not evidence of live bank data.
- The project is not connected to a live bank API or credit bureau unless such integration is explicitly implemented.
- Dataset patterns may not generalize to real applicants, lenders, or future economic conditions.
- Predictions can reflect bias and should be reviewed for fairness, calibration, and reliability.
- Risk categories and thresholds should match the model actually loaded by the API.
- Never commit API keys, credentials, or private applicant information to source control.

## Future Improvements

- Add explainability with SHAP or feature importance.
- Calibrate probabilities and document threshold selection.
- Add automated tests for input validation and prediction responses.
- Monitor data drift and model performance.
- Improve security, logging, deployment, and accessibility.
- Validate the model on additional representative datasets before real-world use.

## License and Attribution

The dataset license is shown on its Kaggle page. Review the license before redistributing the dataset. Specify a separate license for your project code if needed.

## Acknowledgements

- Lao Tse for publishing the [Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).
- The Python, scikit-learn, FastAPI, and Streamlit communities.
