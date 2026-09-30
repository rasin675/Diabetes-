
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOGIN BACKGROUND IMAGE
# ============================================================
LOGIN_IMAGE = "diabetes image.jpg"

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

login_image_base64 = get_image_base64(LOGIN_IMAGE)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>

/* ============================================================
   GENERAL APPLICATION STYLING
   ============================================================ */
.stApp {{
    background-color: #222831;
    color: #eeeeee;
}}

/* ============================================================
   MAIN APPLICATION
   ============================================================ */
.main-header {{
    background: linear-gradient(135deg, #0e4e6b, #1e7093);
    padding: 25px 30px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}}

.main-header h1 {{
    margin: 0;
    font-size: 32px;
}}

.main-header p {{
    margin-top: 8px;
    font-size: 16px;
}}

.metric-card {{
    background-color: #393e46;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #4a4f5b;
    text-align: center;
}}

.metric-title {{
    font-size: 14px;
    color: #a0a0a0;
}}

.metric-value {{
    font-size: 28px;
    font-weight: bold;
    color: #00adb5;
}}

.section-title {{
    color: #00adb5;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 15px;
}}

.footer {{
    text-align: center;
    color: #a0a0a0;
    font-size: 13px;
    padding: 30px 0 10px 0;
}}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL FILES
# ============================================================
MODEL_FILENAME = "random_forest_smotetomek_model.joblib"
FEATURE_COLUMNS_FILENAME = "feature_columns_rf.pkl"

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_and_features(model_file, features_file):
    try:
        model = joblib.load(model_file)
        features = joblib.load(features_file)
        return model, features
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

loaded_rf_model, loaded_feature_columns = load_model_and_features(
    MODEL_FILENAME, FEATURE_COLUMNS_FILENAME
)

# ============================================================
# SESSION STATE
# ============================================================
# Login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================
# LOGIN FUNCTION
# ============================================================
def login_page():
    # Apply the uploaded diabetes image as the login-page background.
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(0, 0, 0, 0.45),
                    rgba(0, 0, 0, 0.45)
                ),
                url("data:image/jpeg;base64,{login_image_base64 or ''}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stForm"] {{
            background: rgba(255, 255, 255, 0.94);
            padding: 30px;
            border-radius: 18px;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
        }}

        .login-title {{
            text-align: center;
            color: white;
            margin-top: 70px;
            margin-bottom: 25px;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7);
        }}

        .login-title h1 {{
            margin: 0;
            font-size: 32px;
        }}

        .login-title p {{
            margin-top: 8px;
            font-size: 16px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="login-title">
            <h1>🩺 Diabetes Prediction System</h1>
            <p>Machine Learning-Based Diabetes Assessment</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col2:
        st.markdown(
            "<h3 style='text-align:center; color:#0e4e6b;'>Login</h3>",
            unsafe_allow_html=True
        )

        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="Enter username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )

            login_button = st.form_submit_button(
                "Login",
                use_container_width=True
            )

            if login_button:
                # Demo credentials
                if (
                    username == "admin" and
                    password == "admin123"
                ):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    st.info("Demo account: admin / admin123")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_diabetic_status(new_patient_data):
    if (
        loaded_rf_model is None or
        loaded_feature_columns is None
    ):
        st.error(
            "Model or feature columns could not be loaded."
        )
        return None

    # Find missing columns
    missing_columns = [
        col for col in loaded_feature_columns if col not in new_patient_data.columns
    ]
    if missing_columns:
        st.error(
            f"Missing required features: {missing_columns}"
        )
        return None

    # Reorder columns
    ordered_data = new_patient_data[
        loaded_feature_columns
    ]

    # Make prediction
    prediction = loaded_rf_model.predict(
        ordered_data
    )
    return prediction

# ============================================================
# PATIENT ASSESSMENT PAGE
# ============================================================
def patient_assessment():
    st.markdown(
        """
        <div class="section-title">
            🧑‍⚕️ Patient Assessment
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(
        "Enter the patient's information below to generate a prediction."
    )
    st.markdown("---")

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        # Patient ID
        patient_id = st.text_input(
            "Patient ID",
            placeholder="Enter patient ID"
        )
        sex = st.radio(
            "Sex",
            [
                ("Female", 1),
                ("Male", 0)
            ],
            format_func=lambda x: x[0],
            horizontal=True
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50
        )
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )
    with col2:
        highbp = st.radio(
            "High Blood Pressure",
            [
                ("No", 0),
                ("Yes", 1)
            ],
            format_func=lambda x: x[0],
            horizontal=True
        )
        knownhtn = st.radio(
            "Known Hypertension",
            [
                ("No", 0),
                ("Yes", 1)
            ],
            format_func=lambda x: x[0],
            horizontal=True
        )
        smoker = st.radio(
            "Smoker",
            [
                ("No", 0),
                ("Yes", 1)
            ],
            format_func=lambda x: x[0],
            horizontal=True
        )
        alcohol = st.radio(
            "Alcohol Consumer",
            [
                ("No", 0),
                ("Yes", 1)
            ],
            format_func=lambda x: x[0],
            horizontal=True
        )
    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------
    if st.button(
        "🔍 Predict Diabetic Status",
        type="primary",
        use_container_width=True
    ):
        # Validate patient ID
        if not patient_id.strip():
            st.warning(
                "Please enter a Patient ID before making a prediction."
            )
            return

        # Create dataframe
        patient_data = pd.DataFrame({
            "sex": [sex[1]],
            "HighBP": [highbp[1]],
            "KnownHTN": [knownhtn[1]],
            "Smoker": [smoker[1]],
            "alcohol": [alcohol[1]],
            "age": [age],
            "BMI": [bmi]
        })

        # Make prediction
        prediction = predict_diabetic_status(
            patient_data
        )
        if prediction is None:
            return

        # Convert prediction
        if prediction[0] == 1:
            result = "Diabetic"
        elif prediction[0] == 0:
            result = "Non-Diabetic"
        else:
            result = "Unknown"

        # ----------------------------------------------------
        # SAVE PREDICTION TO HISTORY
        # ----------------------------------------------------
        prediction_record = {
            "Date & Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Patient ID": patient_id,
            "Sex": "Female" if sex[1] == 1 else "Male",
            "Age": age,
            "BMI": bmi,
            "High BP": "Yes" if highbp[1] == 1 else "No",
            "Known Hypertension": "Yes" if knownhtn[1] == 1 else "No",
            "Smoker": "Yes" if smoker[1] == 1 else "No",
            "Alcohol": "Yes" if alcohol[1] == 1 else "No",
            "Prediction": result,
            "User": st.session_state.get(
                "username", "Unknown"
            )
        }
        st.session_state.prediction_history.append(
            prediction_record
        )

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("Prediction Result")
        if result == "Diabetic":
            st.error(
                "⚠️ Predicted Status: DIABETIC"
            )
            st.warning(
                """
                The model predicted a diabetic status based on the information entered. This is a machine-learning prediction and should not replace clinical diagnosis or appropriate diagnostic testing.
                """
            )
        elif result == "Non-Diabetic":
            st.success(
                "✅ Predicted Status: NON-DIABETIC"
            )
            st.info(
                """
                The model predicted a non-diabetic status based on the information entered. A non-diabetic prediction does not rule out diabetes and should be interpreted alongside appropriate clinical assessment.
                """
            )

        # Show patient summary
        st.markdown("### Patient Summary")
        summary = pd.DataFrame({
            "Patient ID": [patient_id],
            "Sex": [
                "Female" if sex[1] == 1 else "Male"
            ],
            "Age": [age],
            "BMI": [bmi],
            "High BP": [
                "Yes" if highbp[1] == 1 else "No"
            ],
            "Hypertension": [
                "Yes" if knownhtn[1] == 1 else "No"
            ],
            "Smoker": [
                "Yes" if smoker[1] == 1 else "No"
            ],
            "Alcohol": [
                "Yes" if alcohol[1] == 1 else "No"
            ],
            "Prediction": [result]
        })
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# DASHBOARD PAGE
# ============================================================
def dashboard():
    st.markdown(
        """
        <div class="section-title">
            🏠 Prediction Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )
    history = st.session_state.prediction_history

    # --------------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------------
    total_predictions = len(history)
    diabetic_predictions = sum(
        1 for record in history if record["Prediction"] == "Diabetic"
    )
    non_diabetic_predictions = sum(
        1 for record in history if record["Prediction"] == "Non-Diabetic"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title"> Total Predictions </div>
                <div class="metric-value"> {total_predictions} </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title"> Predicted Diabetic </div>
                <div class="metric-value"> {diabetic_predictions} </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title"> Predicted Non-Diabetic </div>
                <div class="metric-value"> {non_diabetic_predictions} </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------------
    st.subheader("📋 Prediction History")
    if not history:
        st.info(
            """
            No predictions have been made yet. Go to **Patient Assessment** to make the first prediction.
            """
        )
        return

    history_df = pd.DataFrame(history)
    # Display history
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
    st.markdown("---")

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------
    st.subheader("🔎 Filter Predictions")
    filter_option = st.selectbox(
        "Prediction Result",
        [
            "All",
            "Diabetic",
            "Non-Diabetic"
        ]
    )

    if filter_option == "All":
        filtered_df = history_df
    else:
        filtered_df = history_df[
            history_df["Prediction"] == filter_option
        ]
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD HISTORY
    # --------------------------------------------------------
    csv_data = history_df.to_csv(
        index=False
    )
    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# MAIN APPLICATION
# ============================================================
if not st.session_state.logged_in:
    login_page()
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 Diabetes AI")
    st.markdown("---")
    st.write(
        f"👤 **Logged in as:** "
        f"{st.session_state.get('username', 'User')}"
    )
    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🧑‍⚕️ Patient Assessment"
        ]
    )
    st.markdown("---")
    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):
        st.session_state.logged_in = False
        st.session_state.pop(
            "username", None
        )
        st.rerun()

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>Diabetes Prediction System</h1>
        <p> Machine Learning-Based Patient Assessment Dashboard </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# NAVIGATION
# ============================================================
if navigation == "🏠 Dashboard":
    dashboard()
elif navigation == "🧑‍⚕️ Patient Assessment":
    patient_assessment()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Diabetes Prediction System | Random Forest Machine Learning Model
    </div>
    """,
    unsafe_allow_html=True
)