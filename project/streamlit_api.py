# ============================================================
# INTELLIGENT LOAN RISK ASSESSMENT SYSTEM
# Streamlit Frontend
# ============================================================

import streamlit as st
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .probability {
        font-size: 42px;
        font-weight: 700;
    }

    .risk {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">💰 Intelligent Loan Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered loan default risk assessment using XGBoost'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FASTAPI URL
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This system evaluates loan default risk
        using a trained XGBoost machine learning model.

        The model returns:

        - Default probability
        - Risk level
        - Assessment decision
        """
    )

    st.divider()

    st.write("**Model:** XGBoost + No balancing")
    st.write("**Decision threshold:** 24%")


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )


with col2:

    person_income = st.number_input(
        "Annual Income",
        min_value=1.0,
        value=50000.0,
        step=1000.0
    )


with col3:

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "MORTGAGE",
            "OWN",
            "OTHER"
        ]
    )


col4, col5 = st.columns(2)

with col4:

    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0,
        max_value=60.0,
        value=3.0,
        step=1.0
    )


with col5:

    cb_person_default_on_file = st.selectbox(
        "Previous Default on File",
        [
            "N",
            "Y"
        ]
    )


# ============================================================
# LOAN INFORMATION
# ============================================================

st.header("🏦 Loan Information")

col1, col2, col3 = st.columns(3)

with col1:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=1.0,
        value=10000.0,
        step=1000.0
    )


with col2:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.01,
        value=10.5,
        step=0.1
    )


with col3:

    loan_percent_income = st.number_input(
        "Loan / Income Ratio",
        min_value=0.001,
        max_value=1.0,
        value=0.20,
        step=0.01
    )


col4, col5 = st.columns(2)

with col4:

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE"
        ]
    )


with col5:

    loan_grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )


# ============================================================
# CREDIT INFORMATION
# ============================================================

st.header("📊 Credit Information")

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0.0,
    max_value=100.0,
    value=4.0,
    step=1.0
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Assess Loan Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE REQUEST DATA
    # --------------------------------------------------------

    payload = {

        "person_age": person_age,

        "person_income": person_income,

        "person_home_ownership":
            person_home_ownership,

        "person_emp_length":
            person_emp_length,

        "loan_intent":
            loan_intent,

        "loan_grade":
            loan_grade,

        "loan_amnt":
            loan_amnt,

        "loan_int_rate":
            loan_int_rate,

        "loan_percent_income":
            loan_percent_income,

        "cb_person_default_on_file":
            cb_person_default_on_file,

        "cb_person_cred_hist_length":
            cb_person_cred_hist_length
    }


    # --------------------------------------------------------
    # CALL FASTAPI
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Assessing Loan Risk..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()


            # ------------------------------------------------
            # CHECK API ERROR
            # ------------------------------------------------

            if "error" in result:

                st.error(
                    result["error"]
                )

            else:

                probability = (
                    result[
                        "default_probability"
                    ]
                )

                probability_percent = (
                    result[
                        "default_probability_percent"
                    ]
                )

                risk_level = (
                    result[
                        "risk_level"
                    ]
                )

                decision = (
                    result[
                        "decision"
                    ]
                )

                threshold = (
                    result[
                        "threshold"
                    ]
                )

                model = (
                    result[
                        "model"
                    ]
                )


                # ==========================================
                # RESULTS
                # ==========================================

                st.divider()

                st.header(
                    "📋 Loan Risk Assessment Result"
                )


                # ------------------------------------------------
                # RESULT COLUMNS
                # ------------------------------------------------

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Default Probability",
                        f"{probability_percent}%"
                    )


                with col2:

                    st.metric(
                        "Risk Level",
                        risk_level
                    )


                with col3:

                    st.metric(
                        "Assessment",
                        decision
                    )


                # ------------------------------------------------
                # PROBABILITY BAR
                # ------------------------------------------------

                st.subheader(
                    "Default Risk Probability"
                )

                st.progress(
                    min(
                        probability,
                        1.0
                    )
                )


                st.write(
                    f"Predicted default probability: "
                    f"**{probability_percent}%**"
                )

                st.write(
                    f"Decision threshold: "
                    f"**{threshold * 100:.0f}%**"
                )


                # ------------------------------------------------
                # RISK MESSAGE
                # ------------------------------------------------

                if risk_level == "LOW":

                    st.success(
                        "🟢 LOW LOAN RISK — "
                        "The predicted default probability "
                        "is below the decision threshold."
                    )


                elif risk_level == "MEDIUM":

                    st.warning(
                        "🟡 MEDIUM LOAN RISK — "
                        "The application should be "
                        "reviewed carefully."
                    )


                else:

                    st.error(
                        "🔴 HIGH LOAN RISK — "
                        "The predicted default probability "
                        "is above the decision threshold."
                    )


                # ------------------------------------------------
                # MODEL INFORMATION
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "Model Information"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"**Model:** {model}"
                    )

                with col2:

                    st.write(
                        f"**Threshold:** "
                        f"{threshold * 100:.0f}%"
                    )


        else:

            st.error(
                f"FastAPI returned status code "
                f"{response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the Loan Risk API."
        )

        st.info(
            "Make sure FastAPI is running with:\n\n"
            "uvicorn api:app --reload"
        )


    except requests.exceptions.Timeout:

        st.error(
            "The Loan Risk API took too long to respond."
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Loan Risk Assessment System | "
    "XGBoost + FastAPI + Streamlit"
)