import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import re


from sklearn.utils.validation import check_is_fitted
from report_ai.ocr.reader import extract_text_from_image
from report_ai.chatbot import ask_nirmaya_ai
from patient_history import (
    create_database,
    save_patient_record,
    get_patient_history,
    get_patient_names
)
create_database()

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 180, 255, 0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(90, 60, 255, 0.10), transparent 30%),
        linear-gradient(135deg, #07111f 0%, #0b1628 50%, #07111f 100%);
    color: #f5f7fa;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4 { color: #f4f8ff !important; }
p, label { color: #c9d6e8 !important; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
    background-color: rgba(12, 25, 43, 0.9) !important;
    color: #f5f7fa !important;
    border-radius: 12px !important;
    border: 1px solid rgba(110, 180, 255, 0.20) !important;
}
.stSelectbox div[data-baseweb="select"] * { color: #f5f7fa !important; }
.stButton > button {
    width: 100%; border: none; border-radius: 13px; padding: 12px 20px;
    font-size: 16px; font-weight: 650; color: white !important;
    background: linear-gradient(90deg, #168cff, #6366f1);
    box-shadow: 0 8px 22px rgba(40, 120, 255, 0.25);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(60, 150, 255, 0.40); }
[data-testid="stFileUploader"] { background: rgba(15, 29, 48, 0.70); border: 1px dashed rgba(90, 190, 255, 0.35); border-radius: 18px; padding: 15px; }
.stRadio > div { background: rgba(15, 29, 48, 0.65); border-radius: 14px; padding: 10px 15px; }
[data-testid="stMetric"] { background: rgba(17, 32, 53, 0.78); border: 1px solid rgba(100, 180, 255, 0.15); border-radius: 18px; padding: 18px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f2742, #172554, #0f172a); }
section[data-testid="stSidebar"] * { color: white !important; }
hr { border-color: rgba(120, 170, 220, 0.15); }
.stCaption { color: #91a4bc !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("🩺 NIRMAYA")

st.markdown(
    "**Multi-Disease AI Screening Platform**"
)

st.caption(
    "AI-Powered Multi-Disease Screening Platform"
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
# OCR PREFILL HELPER
# ============================================================

def _normalise_ocr_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _ocr_number(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def apply_ocr_to_inputs(disease, features, ocr_data, prefix):
    """Safely prefill Streamlit input state from OCR values.

    Only clear feature matches are applied. Missing/ambiguous values are
    left for manual entry; the app never invents a value.
    """
    if not isinstance(ocr_data, dict):
        return

    # OCR responses can contain nested dictionaries such as
    # {"medical_values": {"Cholesterol": {"result": "236"}}}.
    # Flatten them so matching can use the actual field names without
    # ever showing the raw OCR/JSON response to the user.
    source = {}

    def collect_values(obj, parent_key=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{parent_key} {key}".strip()
                if isinstance(value, dict):
                    collect_values(value, full_key)
                else:
                    source[_normalise_ocr_key(full_key)] = value
                    source[_normalise_ocr_key(key)] = value
        elif parent_key:
            source[_normalise_ocr_key(parent_key)] = obj

    collect_values(ocr_data)

    aliases = {
        "age": ["age", "patientage"],
        "sex": ["sex", "gender"],
        "cp": ["cp", "chestpaintype", "chestpain"],
        "trestbps": ["trestbps", "restingbloodpressure", "bloodpressure", "bp"],
        "chol": ["chol", "cholesterol", "totalcholesterol"],
        "fbs": ["fbs", "fastingbloodsugar", "fastingglucose", "glucose"],
        "restecg": ["restecg", "restingecg", "ecg"],
        "thalach": ["thalach", "maximumheartrate", "maxheartrate", "heartrate"],
        "exang": ["exang", "exerciseinducedangina", "angina"],
        "oldpeak": ["oldpeak", "stdepression", "stdepressionoldpeak"],
        "slope": ["slope"],
        "ca": ["ca", "numberofmajorvessels", "majorvessels"],
        "thal": ["thal"]
    }

    def find_value(names):
        for name in names:
            key = _normalise_ocr_key(name)
            if key in source:
                return source[key]
        return None

    if disease == "Heart Disease":
        mapping = {
            "age": find_value(aliases["age"]),
            "sex": find_value(aliases["sex"]),
            "cp": find_value(aliases["cp"]),
            "bp": find_value(aliases["trestbps"]),
            "chol": find_value(aliases["chol"]),
            "fbs": find_value(aliases["fbs"]),
            "restecg": find_value(aliases["restecg"]),
            "thalach": find_value(aliases["thalach"]),
            "exang": find_value(aliases["exang"]),
            "oldpeak": find_value(aliases["oldpeak"]),
            "slope": find_value(aliases["slope"]),
            "ca": find_value(aliases["ca"]),
            "thal": find_value(aliases["thal"]),
        }

        for field, raw in mapping.items():
            if raw is None:
                continue

            if field == "sex":
                text = str(raw).strip().lower()
                if text in ("male", "m", "man"):
                    st.session_state[f"{prefix}_heart_sex"] = "Male"
                elif text in ("female", "f", "woman"):
                    st.session_state[f"{prefix}_heart_sex"] = "Female"
                continue

            number = _ocr_number(raw)
            if number is None:
                continue

            state_keys = {
                "age": f"{prefix}_heart_age",
                "cp": f"{prefix}_heart_cp",
                "bp": f"{prefix}_heart_bp",
                "chol": f"{prefix}_heart_chol",
                "fbs": f"{prefix}_heart_fbs",
                "restecg": f"{prefix}_heart_restecg",
                "thalach": f"{prefix}_heart_thalach",
                "exang": f"{prefix}_heart_exang",
                "oldpeak": f"{prefix}_heart_oldpeak",
                "slope": f"{prefix}_heart_slope",
                "ca": f"{prefix}_heart_ca",
                "thal": f"{prefix}_heart_thal",
            }
            key = state_keys[field]

            # Keep values inside the actual widget choices/ranges.
            if field in ("cp", "restecg", "slope", "thal") and number not in [0, 1, 2, 3]:
                continue
            if field in ("fbs", "exang") and number not in [0, 1]:
                continue
            if field == "ca" and not 0 <= number <= 4:
                continue
            if field == "age" and not 1 <= number <= 120:
                continue
            if field == "bp" and not 50 <= number <= 300:
                continue
            if field == "chol" and not 50 <= number <= 700:
                continue
            if field == "thalach" and not 30 <= number <= 300:
                continue
            if field == "oldpeak" and not 0 <= number <= 10:
                continue

            st.session_state[key] = int(number) if field != "oldpeak" else float(number)

    else:
        # Generic numeric prefill for diseases whose feature names match
        # OCR keys. Categorical fields remain manual unless safely matched.
        for i, feature in enumerate(features):
            feature_key = _normalise_ocr_key(feature)
            raw = source.get(feature_key)
            if raw is None:
                continue
            number = _ocr_number(raw)
            if number is not None:
                st.session_state[widget_key(prefix, i, feature)] = number

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
       
    input_df = pd.DataFrame([values], columns=features)

    input_df = input_df.apply(pd.to_numeric, errors="coerce")
   
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
    user_type,
    patient_name=None,
    patient_gender=None,
    patient_age=None,
    input_data=None
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
        # --------------------------------------------------------
    # 💾 SAVE PATIENT SCREENING RECORD
    # --------------------------------------------------------

    if user_type == "Patient":

        if st.button(
            "💾 Save Screening Record",
            key="save_patient_screening_record"
        ):

            save_patient_record(
                patient_name=patient_name,
                gender=patient_gender,
                age=patient_age,
                disease=disease,
                prediction=(
                    "Higher Risk"
                    if prediction == 1
                    else "Lower Risk"
                ),
                confidence=probability,
                input_data=str(input_data if input_data is not None else {})
            )

            st.success(
                "✅ Screening record saved successfully!"
            )

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

    st.header("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        existing_patient_names = get_patient_names()
        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name",
            key="patient_name_input"
        )

        if patient_name.strip():
            matching_names = [
                name for name in existing_patient_names
                if patient_name.lower() in name.lower()
            ]
            if matching_names:
                st.caption("🔎 Existing patient names:")
                for name in matching_names[:5]:
                    if st.button(
                        f"👤 {name}",
                        key=f"patient_suggestion_{name}"
                    ):
                        st.session_state["patient_name_input"] = name
                        st.rerun()

    with col2:
        patient_gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"],
            key="patient_gender_input"
        )

    with col3:
        patient_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
            key="patient_age_input"
        )

    st.divider()

    st.header("🩺 Select Disease")

    disease = st.selectbox(
        "Choose a disease for screening",
        list(MODEL_PATHS.keys()),
        format_func=lambda x: f"{DISEASE_INFO[x]['icon']} {x}",
        key="patient_disease"
    )

    st.success(f"✅ Selected: {disease}")

    # ========================================================
    # AI MEDICAL REPORT OCR
    # ========================================================

    st.subheader("📄 Medical Data")

    input_method = st.radio(
        "Choose how you want to provide medical data:",
        ["Enter Data Manually", "Upload Medical Report"],
        horizontal=True,
        key="patient_input_method"
    )

    if input_method == "Upload Medical Report":
        uploaded_report = st.file_uploader(
            "Upload your medical report",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a clear medical report image.",
            key="patient_medical_report"
        )

        if uploaded_report is not None:
            st.image(
                uploaded_report,
                caption="Uploaded Medical Report",
                width="stretch"
            )

            if st.button(
                "🔍 Extract Report Data",
                key="extract_report_data"
            ):
                try:
                    temp_path = os.path.join(
                        os.getcwd(),
                        "temp_medical_report.png"
                    )

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_report.getbuffer())

                    with st.spinner("AI is reading the medical report..."):
                        extracted = extract_text_from_image(temp_path)

                    st.session_state["ocr_data"] = extracted
                    st.success("✅ Report data extracted successfully!")

                except Exception as e:
                    st.error(f"❌ Could not extract report data: {e}")

    # Keep OCR data hidden. Matching values are applied directly to the
    # disease input widgets after the model features are loaded.
    ocr_data = st.session_state.get("ocr_data")

    if ocr_data is not None and input_method == "Upload Medical Report":
        st.info(
            "🤖 Report processed. Matching values will be pre-filled in the "
            "medical parameters below. Please verify them; missing or unclear "
            "values must be entered manually."
        )

    st.divider()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:
        model, scaler, features = load_model(disease)
        st.success(f"🤖 {disease} model loaded successfully")
    except Exception as e:
        st.error(f"❌ Could not load {disease} model")
        st.exception(e)
        st.stop()

    st.divider()

    # ========================================================
    # MEDICAL INPUTS
    # ========================================================

    st.caption(
        "Enter the values below using the same clinical/measurement units "
        "used during model training. Do not guess missing medical values."
    )

    if ocr_data is not None and input_method == "Upload Medical Report":
        apply_ocr_to_inputs(
            disease,
            features,
            ocr_data,
            "patient"
        )

    values = create_disease_inputs(
        disease,
        features,
        "patient"
    )

    st.divider()

    # ========================================================
    # PREDICT
    # ========================================================

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
                "Patient",
                patient_name=patient_name,
                patient_gender=patient_gender,
                patient_age=patient_age,
                input_data=dict(zip(features, values))
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# DOCTOR PAGE
# ============================================================

if user_type == "Doctor":

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
    # MEDICAL REPORT UPLOAD / OCR
    # --------------------------------------------------------

    st.subheader("📄 Patient Medical Report")

    doctor_input_method = st.radio(
        "Choose how you want to provide patient medical data:",
        ["Enter Data Manually", "Upload Medical Report"],
        horizontal=True,
        key="doctor_input_method"
    )

    if doctor_input_method == "Upload Medical Report":

        doctor_uploaded_report = st.file_uploader(
            "Upload patient's medical report",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a clear image of the patient's medical report.",
            key="doctor_medical_report"
        )

        if doctor_uploaded_report is not None:

            st.image(
                doctor_uploaded_report,
                caption="Uploaded Patient Medical Report",
                width="stretch"
            )

            if st.button(
                "🔍 Extract & Fill Patient Data",
                key="doctor_extract_report_data"
            ):
                try:
                    temp_path = os.path.join(
                        BASE_DIR,
                        "temp_doctor_medical_report.png"
                    )

                    with open(temp_path, "wb") as f:
                        f.write(doctor_uploaded_report.getbuffer())

                    with st.spinner(
                        "AI is reading the patient's medical report..."
                    ):
                        doctor_ocr = extract_text_from_image(temp_path)

                    st.session_state["doctor_ocr_data"] = doctor_ocr
                    st.session_state["doctor_ocr_disease"] = doctor_disease
                    st.session_state["ocr_data"] = doctor_ocr

                    st.success(
                        "✅ Report processed. Matching values have been filled into the parameters below."
                    )
                    st.rerun()

                except Exception as e:
                    st.error(
                        f"❌ Could not extract patient report data: {e}"
                    )

    doctor_ocr_data = st.session_state.get("doctor_ocr_data")
    doctor_ocr_disease = st.session_state.get("doctor_ocr_disease")

    if (
        doctor_ocr_data is not None
        and doctor_ocr_disease == doctor_disease
    ):
        st.info(
            "🤖 Extracted values are being used to pre-fill matching parameters. "
            "Please verify them. Any missing or unclear value must be entered manually."
        )

        # Values are intentionally NOT displayed as raw OCR JSON.
        # The helper only fills clear, safe matches in the input widgets.
        apply_ocr_to_inputs(
            doctor_disease,
            st.session_state.get("doctor_features_for_ocr", []),
            doctor_ocr_data,
            "doctor"
        )

    st.divider()

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

    if doctor_ocr_data is not None and doctor_ocr_disease == doctor_disease:
        apply_ocr_to_inputs(
            doctor_disease,
            doctor_features,
            doctor_ocr_data,
            "doctor"
        )

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

st.divider()

st.caption(
    "🏥 MediPredict AI • Multi-Disease AI Screening • "
    "For screening support only, not medical diagnosis."
)
# ==========================================
# ==========================================
# 🤖 NIRMAYA AI ASSISTANT — ATTRACTIVE CHAT UI
# ==========================================

st.divider()

st.markdown("""
<style>
/* NIRMAYA AI chatbot container */
.nirmaya-chat-box {
    background: linear-gradient(135deg, #101827 0%, #172554 55%, #1e1b4b 100%);
    border: 1px solid rgba(129, 140, 248, 0.45);
    border-radius: 22px;
    padding: 24px;
    margin: 10px 0 20px 0;
    box-shadow: 0 12px 35px rgba(0,0,0,0.35);
}

.nirmaya-chat-title {
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 4px;
    color: #f8fafc;
}

.nirmaya-chat-subtitle {
    color: #c7d2fe;
    font-size: 14px;
    margin-bottom: 18px;
}

.nirmaya-ai-message {
    background: linear-gradient(135deg, #312e81, #4338ca);
    border-left: 5px solid #a5b4fc;
    border-radius: 16px;
    padding: 16px 18px;
    margin-top: 12px;
    color: #ffffff;
    line-height: 1.65;
    box-shadow: 0 8px 20px rgba(49,46,129,0.35);
}

.nirmaya-user-message {
    background: rgba(30, 41, 59, 0.95);
    border-left: 5px solid #38bdf8;
    border-radius: 16px;
    padding: 14px 18px;
    margin-top: 12px;
    color: #e2e8f0;
}

.nirmaya-highlight {
    background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(16,185,129,0.08));
    border: 1px solid rgba(74,222,128,0.35);
    border-radius: 13px;
    padding: 12px 15px;
    margin-top: 12px;
    color: #dcfce7;
}

.nirmaya-warning {
    background: linear-gradient(90deg, rgba(245,158,11,0.20), rgba(234,88,12,0.08));
    border: 1px solid rgba(251,191,36,0.40);
    border-radius: 13px;
    padding: 12px 15px;
    margin-top: 12px;
    color: #fef3c7;
}

/* Make chatbot input attractive */
div[data-testid="stTextArea"] textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(129,140,248,0.45) !important;
    background: rgba(15,23,42,0.90) !important;
    color: #f8fafc !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border: 2px solid #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="nirmaya-chat-box">', unsafe_allow_html=True)
    st.markdown('<div class="nirmaya-chat-title">🤖 NIRMAYA AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nirmaya-chat-subtitle">✨ Your intelligent medical screening assistant</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="nirmaya-highlight">💡 Ask about medical parameters, your screening result, uploaded reports, or how NIRMAYA works.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    chatbot_question = st.text_area(
        "💬 Ask NIRMAYA",
        placeholder=(
            "Example: What does cholesterol mean?\n"
            "Example: What does my screening result mean?\n"
            "Example: How does report OCR work?"
        ),
        key="nirmaya_chat_question",
        height=120
    )

    if st.button(
        "🚀 Ask NIRMAYA AI",
        type="primary",
        key="nirmaya_chat_button"
    ):

        if chatbot_question.strip():

            st.markdown(
                f'<div class="nirmaya-user-message"><b>👤 You</b><br>{chatbot_question}</div>',
                unsafe_allow_html=True
            )

            with st.spinner("🤖 NIRMAYA AI is thinking..."):

                try:

                    # Use OCR report data if available
                    report_context = st.session_state.get(
                        "ocr_data",
                        None
                    )

                    answer = ask_nirmaya_ai(
                        chatbot_question,
                        report_context
                    )

                    st.markdown(
                        '<div class="nirmaya-ai-message"><b>🤖 NIRMAYA AI</b></div>',
                        unsafe_allow_html=True
                    )

                    # Keep AI response as normal Streamlit text/markdown
                    # so that formatting from the chatbot is preserved.
                    st.markdown(answer)

                    st.markdown(
                        '<div class="nirmaya-warning">⚠️ NIRMAYA AI provides informational screening support only. It does not replace a qualified medical professional.</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"Unable to get an AI response: {e}"
                    )

        else:

            st.warning(
                "💬 Please enter a question first."
            )


st.divider()

st.caption(
    "🏥 MediPredict AI • Multi-Disease AI Screening • "
    "For screening support only, not medical diagnosis."
)
