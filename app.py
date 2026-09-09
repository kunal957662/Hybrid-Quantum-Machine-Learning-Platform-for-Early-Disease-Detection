import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

from sklearn.utils.validation import check_is_fitted


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #dbeafe 0%,
            #e0f2fe 45%,
            #f0f9ff 100%
        );
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 750;
    }

    p, label {
        color: #0f172a !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        border: 1px solid #93c5fd !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        color: #0f172a !important;
    }

    .stTextInput input::placeholder {
        color: #64748b !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1rem;
        background: linear-gradient(
            90deg,
            #2563eb,
            #0ea5e9
        );
        color: white !important;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #1d4ed8,
            #0284c7
        );
    }

    .stAlert {
        border-radius: 14px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f2742,
            #172554,
            #0f172a
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .medical-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(147,197,253,0.7);
        border-radius: 18px;
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(15,23,42,0.08);
    }

    .result-card {
        background: white;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        border: 1px solid #bfdbfe;
        box-shadow: 0 8px 25px rgba(15,23,42,0.10);
    }

    .small-note {
        color: #475569 !important;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🏥 MediPredict AI")

st.markdown(
    "**Multi-Disease AI Screening Platform**"
)

st.caption(
    "AI-assisted screening using six independent machine-learning models."
)

st.divider()


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {

    "Heart Disease":
        os.path.join(BASE_DIR, "models", "heart_disease"),

    "Parkinson's Disease":
        os.path.join(BASE_DIR, "models", "parkinsons"),

    "Diabetes":
        os.path.join(BASE_DIR, "models", "diabetes"),

    "Chronic Kidney Disease":
        os.path.join(BASE_DIR, "models", "kidney"),

    "Heart Failure":
        os.path.join(BASE_DIR, "models", "heart_failure"),

    "Breast Cancer":
        os.path.join(BASE_DIR, "models", "breast_cancer")
}


# ============================================================
# MODEL ACCURACY
# ============================================================
#
# These values must be TEST-SET accuracies from your training.
#
# Heart Disease = confirmed from your completed training:
# 0.901639 = 90.16%
#
# Do NOT put prediction confidence here.
# ============================================================

MODEL_ACCURACY = {

    "Heart Disease": 90.16,

    # Add the actual test accuracy from your training output
    # when available.
    "Parkinson's Disease": None,

    "Diabetes": None,

    "Chronic Kidney Disease": None,

    "Heart Failure": None,

    "Breast Cancer": None
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    # ========================================================
    # HEART DISEASE
    # ========================================================

    "Heart Disease": {

        "icon": "❤️",

        "risk": [
            "High blood pressure",
            "High cholesterol",
            "Smoking",
            "Diabetes or high blood sugar",
            "Low physical activity",
            "Increasing age",
            "Family history",
            "Certain patterns of chest pain"
        ],

        "nutrients": [
            "Omega-3 fatty acids",
            "Dietary fiber",
            "Potassium",
            "Magnesium",
            "Vitamin D",
            "Folate"
        ],

        "precautions": [
            "Monitor blood pressure regularly",
            "Monitor cholesterol",
            "Maintain a balanced diet",
            "Exercise according to medical advice",
            "Avoid smoking",
            "Maintain a healthy weight",
            "Consult a healthcare professional"
        ]
    },


    # ========================================================
    # PARKINSON'S DISEASE
    # ========================================================

    "Parkinson's Disease": {

        "icon": "🧠",

        "risk": [
            "Increasing age",
            "Family history",
            "Certain genetic factors",
            "Environmental exposures",
            "Some occupational exposures"
        ],

        "nutrients": [
            "Vitamin D",
            "Vitamin B12",
            "Folate",
            "Magnesium",
            "Omega-3 fatty acids",
            "Calcium"
        ],

        "precautions": [
            "Maintain regular physical activity",
            "Maintain balanced nutrition",
            "Get adequate sleep",
            "Discuss new neurological symptoms with a doctor",
            "Attend regular medical follow-up"
        ]
    },


    # ========================================================
    # DIABETES
    # ========================================================

    "Diabetes": {

        "icon": "🩸",

        "risk": [
            "High blood glucose",
            "Overweight or obesity",
            "Low physical activity",
            "Family history",
            "High blood pressure",
            "Unhealthy dietary patterns",
            "Increasing age"
        ],

        "nutrients": [
            "Dietary fiber",
            "Vitamin D",
            "Magnesium",
            "Vitamin B12",
            "Chromium",
            "Folate"
        ],

        "precautions": [
            "Monitor blood glucose",
            "Maintain a balanced diet",
            "Exercise regularly when appropriate",
            "Maintain a healthy weight",
            "Limit excess added sugar",
            "Follow medical advice"
        ]
    },


    # ========================================================
    # CHRONIC KIDNEY DISEASE
    # ========================================================

    "Chronic Kidney Disease": {

        "icon": "🫘",

        "risk": [
            "Diabetes",
            "High blood pressure",
            "Kidney disease family history",
            "Cardiovascular disease",
            "Smoking",
            "Older age"
        ],

        "nutrients": [
            "Iron",
            "Vitamin B12",
            "Folate",
            "Vitamin D",
            "Calcium"
        ],

        "precautions": [
            "Monitor blood pressure",
            "Monitor blood glucose if applicable",
            "Take medicines only as prescribed",
            "Follow an appropriate diet recommended by a healthcare professional",
            "Attend kidney-function follow-ups"
        ]
    },


    # ========================================================
    # HEART FAILURE
    # ========================================================

    "Heart Failure": {

        "icon": "❤️",

        "risk": [
            "High blood pressure",
            "Coronary artery disease",
            "Previous heart attack",
            "Diabetes",
            "Obesity",
            "Smoking",
            "Kidney disease"
        ],

        "nutrients": [
            "Vitamin D",
            "Magnesium",
            "Potassium",
            "Thiamine",
            "Iron",
            "Folate"
        ],

        "precautions": [
            "Monitor blood pressure",
            "Follow prescribed medication",
            "Monitor weight changes",
            "Follow appropriate dietary advice",
            "Avoid smoking",
            "Attend regular medical follow-up"
        ]
    },


    # ========================================================
    # BREAST CANCER
    # ========================================================

    "Breast Cancer": {

        "icon": "🎗️",

        "risk": [
            "Increasing age",
            "Family history",
            "Certain genetic factors",
            "Hormonal and reproductive factors",
            "Alcohol exposure",
            "Obesity after menopause",
            "Previous radiation exposure"
        ],

        "nutrients": [
            "Vitamin D",
            "Folate",
            "Vitamin B12",
            "Calcium",
            "Vitamin C",
            "Vitamin E"
        ],

        "precautions": [
            "Follow recommended screening",
            "Discuss unusual breast changes with a doctor",
            "Maintain a healthy weight",
            "Stay physically active",
            "Avoid smoking",
            "Follow professional medical advice"
        ]
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(disease):

    if disease not in MODEL_PATHS:
        raise ValueError(
            f"Unknown disease: {disease}"
        )

    folder = MODEL_PATHS[disease]

    model_path = os.path.join(
        folder,
        "best_model.pkl"
    )

    scaler_path = os.path.join(
        folder,
        "scaler.pkl"
    )

    features_path = os.path.join(
        folder,
        "features.pkl"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    for path, label in [
        (model_path, "model"),
        (scaler_path, "scaler"),
        (features_path, "features")
    ]:

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    model = joblib.load(model_path)

    scaler = joblib.load(scaler_path)

    features = list(
        joblib.load(features_path)
    )

    # --------------------------------------------------------
    # CHECK MODEL FIT
    # --------------------------------------------------------

    try:

        check_is_fitted(model)

    except Exception as e:

        raise ValueError(
            f"The saved model for {disease} is not fitted.\n"
            f"File: {model_path}"
        ) from e

    # --------------------------------------------------------
    # CHECK SCALER FIT
    # --------------------------------------------------------

    try:

        check_is_fitted(scaler)

    except Exception as e:

        raise ValueError(
            f"The saved scaler for {disease} is not fitted.\n"
            f"File: {scaler_path}"
        ) from e

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    if len(features) == 0:

        raise ValueError(
            f"No features found for {disease}."
        )

    # --------------------------------------------------------
    # CHECK SCALER FEATURE COUNT
    # --------------------------------------------------------

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        None
    )

    if (
        scaler_features is not None
        and scaler_features != len(features)
    ):

        raise ValueError(
            f"Feature mismatch for {disease}. "
            f"Scaler expects {scaler_features} features "
            f"but features.pkl contains {len(features)}."
        )

    return model, scaler, features


# ============================================================
# SAFE STREAMLIT KEY
# ============================================================

def widget_key(prefix, index, feature):

    clean_feature = (
        str(feature)
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("/", "_")
    )

    return (
        f"{prefix}_{index}_{clean_feature}"
    )


# ============================================================
# DISEASE INPUT FORM
# ============================================================

def create_disease_inputs(
    disease,
    features,
    prefix
):

    values = []


    # ========================================================
    # HEART DISEASE
    # ========================================================

    if disease == "Heart Disease":

        st.subheader(
            "❤️ Heart Disease Medical Information"
        )

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with c1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=40,
                key=f"{prefix}_heart_age"
            )

            sex = st.selectbox(
                "Sex",
                ["Male", "Female"],
                key=f"{prefix}_heart_sex"
            )

            cp = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                key=f"{prefix}_heart_cp"
            )

            trestbps = st.number_input(
                "Resting Blood Pressure",
                min_value=50,
                max_value=300,
                value=120,
                key=f"{prefix}_heart_bp"
            )

        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with c2:

            chol = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=700,
                value=200,
                key=f"{prefix}_heart_chol"
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                key=f"{prefix}_heart_fbs"
            )

            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                key=f"{prefix}_heart_restecg"
            )

            thalach = st.number_input(
                "Maximum Heart Rate",
                min_value=30,
                max_value=300,
                value=150,
                key=f"{prefix}_heart_thalach"
            )

        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with c3:

            exang = st.selectbox(
                "Exercise Induced Angina",
                [0, 1],
                key=f"{prefix}_heart_exang"
            )

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key=f"{prefix}_heart_oldpeak"
            )

            slope = st.selectbox(
                "Slope",
                [0, 1, 2],
                key=f"{prefix}_heart_slope"
            )

            ca = st.number_input(
                "Number of Major Vessels (CA)",
                min_value=0,
                max_value=4,
                value=0,
                key=f"{prefix}_heart_ca"
            )

            thal = st.selectbox(
                "Thal",
                [0, 1, 2, 3],
                key=f"{prefix}_heart_thal"
            )

        sex_value = (
            1
            if sex == "Male"
            else 0
        )

        values = [
            age,
            sex_value,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]


    # ========================================================
    # PARKINSON'S
    # ========================================================

    elif disease == "Parkinson's Disease":

        st.subheader(
            "🧠 Parkinson's Disease Medical Information"
        )

        st.info(
            "Enter the voice-analysis measurements "
            "used by the trained Parkinson's model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            with columns[i % 3]:

                value = st.number_input(
                    str(feature),
                    value=0.0,
                    format="%.8f",
                    key=widget_key(
                        prefix,
                        i,
                        feature
                    )
                )

                values.append(value)


    # ========================================================
    # DIABETES
    # ========================================================

    elif disease == "Diabetes":

        st.subheader(
            "🩸 Diabetes Medical Information"
        )

        st.info(
            "Enter the clinical information used "
            "by the diabetes model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            name = (
                str(feature)
                .strip()
                .lower()
            )

            with columns[i % 3]:

                # AGE
                if name == "age":

                    value = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=40,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # GENDER
                elif (
                    "gender" in name
                    or name == "sex"
                ):

                    selected = st.selectbox(
                        "Gender",
                        ["Male", "Female"],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "Male"
                        else 0
                    )

                # OTHER BINARY FEATURES
                else:

                    selected = st.selectbox(
                        str(feature),
                        ["No", "Yes"],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "Yes"
                        else 0
                    )

                values.append(value)


    # ========================================================
    # CHRONIC KIDNEY DISEASE
    # ========================================================

    elif disease == "Chronic Kidney Disease":

        st.subheader(
            "🫘 Chronic Kidney Disease Information"
        )

        st.info(
            "Enter the laboratory and clinical "
            "measurements used by the kidney model."
        )

        categorical = [
            "rbc",
            "pc",
            "pcc",
            "ba",
            "htn",
            "dm",
            "cad",
            "appet",
            "pe",
            "ane"
        ]

        columns = st.columns(3)

        for i, feature in enumerate(features):

            name = (
                str(feature)
                .strip()
                .lower()
            )

            with columns[i % 3]:

                # AGE
                if name == "age":

                    value = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=45,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # NORMAL / ABNORMAL
                elif name in [
                    "rbc",
                    "pc"
                ]:

                    selected = st.selectbox(
                        str(feature),
                        [
                            "normal",
                            "abnormal"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "normal"
                        else 0
                    )

                # PRESENT / NOT PRESENT
                elif name in [
                    "pcc",
                    "ba"
                ]:

                    selected = st.selectbox(
                        str(feature),
                        [
                            "notpresent",
                            "present"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "present"
                        else 0
                    )

                # YES / NO
                elif name in [
                    "htn",
                    "dm",
                    "cad",
                    "pe",
                    "ane"
                ]:

                    selected = st.selectbox(
                        str(feature),
                        [
                            "no",
                            "yes"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "yes"
                        else 0
                    )

                # APPETITE
                elif name == "appet":

                    selected = st.selectbox(
                        "Appetite",
                        [
                            "good",
                            "poor"
                        ],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                    value = (
                        1
                        if selected == "good"
                        else 0
                    )

                # NUMERIC
                else:

                    value = st.number_input(
                        str(feature),
                        value=0.0,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                values.append(value)


    # ========================================================
    # HEART FAILURE
    # ========================================================

    elif disease == "Heart Failure":

        st.subheader(
            "❤️ Heart Failure Medical Information"
        )

        st.info(
            "Enter the clinical measurements "
            "used by the heart-failure model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            name = (
                str(feature)
                .strip()
                .lower()
            )

            with columns[i % 3]:

                # AGE
                if name == "age":

                    value = st.number_input(
                        "Age",
                        min_value=1,
                        max_value=120,
                        value=50,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # BINARY FEATURES
                elif (
                    "anaemia" in name
                    or "diabetes" in name
                    or "high_blood_pressure" in name
                    or "smoking" in name
                    or name == "sex"
                ):

                    value = st.selectbox(
                        str(feature),
                        [0, 1],
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                # NUMERIC FEATURES
                else:

                    value = st.number_input(
                        str(feature),
                        value=0.0,
                        key=widget_key(
                            prefix,
                            i,
                            feature
                        )
                    )

                values.append(value)


    # ========================================================
    # BREAST CANCER
    # ========================================================

    elif disease == "Breast Cancer":

        st.subheader(
            "🎗️ Breast Cancer Medical Information"
        )

        st.info(
            "Enter the numerical measurements "
            "used by the breast-cancer model."
        )

        columns = st.columns(3)

        for i, feature in enumerate(features):

            with columns[i % 3]:

                value = st.number_input(
                    str(feature),
                    value=0.0,
                    format="%.6f",
                    key=widget_key(
                        prefix,
                        i,
                        feature
                    )
                )

                values.append(value)


    # ========================================================
    # RETURN
    # ========================================================

    return values


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(
    model,
    scaler,
    features,
    values
):

    # --------------------------------------------------------
    # FEATURE COUNT
    # --------------------------------------------------------

    if len(values) != len(features):

        raise ValueError(
            f"Feature mismatch. "
            f"Model expects {len(features)} values "
            f"but received {len(values)}."
        )

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    try:

        check_is_fitted(model)

    except Exception as e:

        raise ValueError(
            "The loaded model is not fitted."
        ) from e

    # --------------------------------------------------------
    # CHECK SCALER
    # --------------------------------------------------------

    try:

        check_is_fitted(scaler)

    except Exception as e:

        raise ValueError(
            "The loaded scaler is not fitted."
        ) from e

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [values],
        columns=features
    )

    # --------------------------------------------------------
    # CONVERT TO NUMERIC
    # --------------------------------------------------------

    input_df = input_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # CHECK INVALID VALUES
    # --------------------------------------------------------

    if input_df.isnull().any().any():

        bad_columns = (
            input_df.columns[
                input_df.isnull().any()
            ].tolist()
        )

        raise ValueError(
            f"Invalid values detected in: "
            f"{bad_columns}"
        )

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_df
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        input_scaled
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = (
                model.predict_proba(
                    input_scaled
                )
            )

            probability = float(
                np.max(
                    probabilities[0]
                ) * 100
            )

        except Exception:

            probability = None

    return (
        int(prediction[0]),
        probability
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    disease,
    prediction,
    probability,
    user_type
):

    info = DISEASE_INFO.get(
        disease,
        {}
    )

    # --------------------------------------------------------
    # SAFE FALLBACKS
    # --------------------------------------------------------

    risk_items = (
        info.get("risk")
        or [
            "Risk may depend on age, family history, lifestyle, medical history, and other clinical factors."
        ]
    )

    nutrient_items = (
        info.get("nutrients")
        or [
            "No disease-specific nutrient information has been configured."
        ]
    )

    precaution_items = (
        info.get("precautions")
        or [
            "Follow appropriate medical advice and recommended follow-up."
        ]
    )

    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔍 AI Screening Result"
    )

    # --------------------------------------------------------
    # RISK RESULT
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            f"⚠️ Higher Risk Indicated for {disease}"
        )

    else:

        st.success(
            f"✅ Lower Risk Indicated for {disease}"
        )

    # --------------------------------------------------------
    # ACCURACY + CONFIDENCE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # MODEL ACCURACY
    with col1:

        accuracy = MODEL_ACCURACY.get(
            disease
        )

        if accuracy is not None:

            st.metric(
                "Model Accuracy",
                f"{accuracy:.2f}%"
            )

        else:

            st.metric(
                "Model Accuracy",
                "Not stored"
            )

    # PREDICTION CONFIDENCE
    with col2:

        if probability is not None:

            st.metric(
                "Prediction Confidence",
                f"{probability:.2f}%"
            )

        else:

            st.metric(
                "Prediction Confidence",
                "N/A"
            )

    st.caption(
        "Model Accuracy represents performance measured during model evaluation. "
        "Prediction Confidence represents the model's probability for this individual input. "
        "They are different measures."
    )

    # --------------------------------------------------------
    # MEDICAL DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "This is an AI-based screening result, not a medical diagnosis. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )

    # ========================================================
    # WHY RISK MAY INCREASE
    # ========================================================

    st.subheader(
        "📌 Why Risk May Increase"
    )

    st.caption(
        "These are general factors associated with the condition. "
        "They are not a direct explanation of why this individual received this prediction."
    )

    for item in risk_items:

        st.write(
            f"• {item}"
        )

    # ========================================================
    # NUTRIENTS / DEFICIENCIES
    # ========================================================

    st.subheader(
        "🥗 Nutrients / Deficiencies That May Be Associated"
    )

    st.caption(
        "Nutrient deficiencies or dietary factors may be associated "
        "with health conditions, but they should not be assumed to "
        "be the cause of an individual's prediction."
    )

    for item in nutrient_items:

        st.write(
            f"• {item}"
        )

    # ========================================================
    # PATIENT PRECAUTIONS ONLY
    # ========================================================

    if user_type == "Patient":

        st.subheader(
            "🛡️ General Precautions"
        )

        st.caption(
            "General precautions only. Follow advice from a qualified healthcare professional."
        )

        for item in precaution_items:

            st.write(
                f"• {item}"
            )


# ============================================================
# USER TYPE
# ============================================================

user_type = st.radio(
    "Select User Type",
    [
        "Patient",
        "Doctor"
    ],
    horizontal=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🏥 MediPredict AI"
    )

    st.caption(
        "Multi-Disease Screening Platform"
    )

    st.divider()

    st.markdown(
        "### 🧭 Navigation"
    )

    page = st.radio(
        "Go to",
        [
            "Prediction",
            "About System"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "### 🤖 AI Screening"
    )

    st.write(
        "Six independent machine-learning models are available."
    )

    st.markdown(
        """
        **Supported diseases**

        ❤️ Heart Disease

        🧠 Parkinson's Disease

        🩸 Diabetes

        🫘 Chronic Kidney Disease

        ❤️ Heart Failure

        🎗️ Breast Cancer
        """
    )


# ============================================================
# ABOUT SYSTEM
# ============================================================

if page == "About System":

    st.header(
        "ℹ️ About MediPredict AI"
    )

    st.markdown(
        """
        ### What is MediPredict AI?

        MediPredict AI is a multi-disease screening platform
        that uses independent machine-learning models for
        different medical conditions.

        ### Available Models

        - Heart Disease
        - Parkinson's Disease
        - Diabetes
        - Chronic Kidney Disease
        - Heart Failure
        - Breast Cancer

        ### Machine Learning Workflow

        Dataset
        → Data Cleaning
        → EDA
        → Feature Selection
        → Train/Test Split
        → Standardization
        → Model Training
        → Hyperparameter Tuning
        → Evaluation
        → Model Saving
        → Streamlit Deployment

        ### Important

        The system is designed for screening support and
        educational/project purposes.

        Predictions should not be treated as a medical diagnosis.
        """
    )

    st.stop()


# ============================================================
# PATIENT PAGE
# ============================================================

if user_type == "Patient":

    st.header(
        "👤 Patient Information"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # PATIENT NAME
    # --------------------------------------------------------

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name"
        )

    # --------------------------------------------------------
    # PATIENT GENDER
    # --------------------------------------------------------

    with col2:

        patient_gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    # --------------------------------------------------------
    # PATIENT AGE
    # --------------------------------------------------------

    with col3:

        patient_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    st.divider()

    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    st.header(
        "🩺 Select Disease"
    )

    disease = st.selectbox(
        "Choose a disease for screening",
        list(MODEL_PATHS.keys()),
        format_func=lambda x:
            f"{DISEASE_INFO[x]['icon']} {x}"
    )

    st.success(
        f"✅ Selected: {disease}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model, scaler, features = load_model(
            disease
        )

        st.success(
            f"🤖 {disease} model loaded successfully"
        )

    except Exception as e:

        st.error(
            f"❌ Could not load {disease} model"
        )

        st.exception(e)

        st.stop()

    st.divider()

    # --------------------------------------------------------
    # MEDICAL INPUTS
    # --------------------------------------------------------

    values = create_disease_inputs(
        disease,
        features,
        "patient"
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        f"🔍 Predict {disease}",
        type="primary",
        key="patient_predict_button"
    ):

        try:

            prediction, probability = make_prediction(
                model,
                scaler,
                features,
                values
            )

            display_result(
                disease,
                prediction,
                probability,
                "Patient"
            )

        except Exception as e:

            st.error(
                "❌ Prediction Error"
            )

            st.exception(e)


# ============================================================
# DOCTOR PAGE
# ============================================================

else:

    st.header(
        "👨‍⚕️ Doctor Information"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # DOCTOR NAME
    # --------------------------------------------------------

    with col1:

        doctor_name = st.text_input(
            "Doctor Name",
            placeholder="Enter doctor name"
        )

    # --------------------------------------------------------
    # DOCTOR GENDER
    # --------------------------------------------------------

    with col2:

        doctor_gender = st.selectbox(
            "Doctor Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    # --------------------------------------------------------
    # MEDICAL DOMAIN
    # --------------------------------------------------------

    with col3:

        doctor_domain = st.selectbox(
            "Medical Domain",
            [
                "Cardiology",
                "Neurology",
                "Diabetology",
                "Nephrology",
                "Oncology",
                "General Medicine"
            ]
        )

    st.divider()

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    st.header(
        "👤 Patient Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        doctor_patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name",
            key="doctor_patient_name"
        )

    with col2:

        doctor_patient_gender = st.selectbox(
            "Patient Gender",
            [
                "Male",
                "Female",
                "Other"
            ],
            key="doctor_patient_gender"
        )

    with col3:

        doctor_patient_age = st.number_input(
            "Patient Age",
            min_value=1,
            max_value=120,
            value=25,
            key="doctor_patient_age"
        )

    st.divider()

    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    st.header(
        "🩺 Select Disease"
    )

    doctor_disease = st.selectbox(
        "Choose a disease for screening",
        list(MODEL_PATHS.keys()),
        format_func=lambda x:
            f"{DISEASE_INFO[x]['icon']} {x}",
        key="doctor_disease"
    )

    st.success(
        f"✅ Selected: {doctor_disease}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        doctor_model, doctor_scaler, doctor_features = load_model(
            doctor_disease
        )

        st.success(
            f"🤖 {doctor_disease} model loaded successfully"
        )

    except Exception as e:

        st.error(
            f"❌ Could not load {doctor_disease} model"
        )

        st.exception(e)

        st.stop()

    st.divider()

    # --------------------------------------------------------
    # MEDICAL INPUTS
    # --------------------------------------------------------

    doctor_values = create_disease_inputs(
        doctor_disease,
        doctor_features,
        "doctor"
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        f"🔍 Predict {doctor_disease}",
        type="primary",
        key="doctor_predict_button"
    ):

        try:

            prediction, probability = make_prediction(
                doctor_model,
                doctor_scaler,
                doctor_features,
                doctor_values
            )

            display_result(
                doctor_disease,
                prediction,
                probability,
                "Doctor"
            )

        except Exception as e:

            st.error(
                "❌ Prediction Error"
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 MediPredict AI • Multi-Disease AI Screening • "
    "For screening support only, not medical diagnosis."
)