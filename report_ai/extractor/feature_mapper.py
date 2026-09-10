import re


# Features required by the Heart Disease ML model
HEART_DISEASE_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# Possible names Gemini may extract from a medical report
FEATURE_ALIASES = {

    "chol": [
        "chol",
        "total cholesterol",
        "total cholesterol (tc)",
        "cholesterol",
        "total chol"
    ],

    "fbs": [
        "fbs",
        "fasting blood sugar",
        "fasting blood sugar (fbs)",
        "fasting glucose",
        "fasting blood glucose"
    ],

    "trestbps": [
        "trestbps",
        "resting blood pressure",
        "resting bp",
        "blood pressure"
    ],

    "thalach": [
        "thalach",
        "maximum heart rate",
        "max heart rate",
        "heart rate"
    ],

    "oldpeak": [
        "oldpeak",
        "st depression",
        "st depression induced by exercise"
    ],

    "sex": [
        "sex",
        "gender"
    ],

    "cp": [
        "cp",
        "chest pain type",
        "chest pain"
    ],

    "restecg": [
        "restecg",
        "resting ecg",
        "resting electrocardiogram"
    ],

    "exang": [
        "exang",
        "exercise induced angina",
        "exercise-induced angina"
    ],

    "slope": [
        "slope",
        "st slope"
    ],

    "ca": [
        "ca",
        "number of major vessels",
        "major vessels"
    ],

    "thal": [
        "thal",
        "thalassemia",
        "thal test"
    ]
}


def find_value(medical_values, aliases):
    """
    Find a value from the OCR medical_values
    using possible aliases.
    """

    normalized_values = {
        str(key).strip().lower(): value
        for key, value in medical_values.items()
    }

    for alias in aliases:

        alias = alias.strip().lower()

        if alias in normalized_values:
            return normalized_values[alias]

    return None


def clean_numeric_value(value):
    """
    Convert OCR values such as:

    '236 mg/dL' -> 236.0
    '168 mg/dL' -> 168.0
    '1.2 mg/dL' -> 1.2

    If no number is found, return None.
    """

    if value is None or value == "":
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip()

    # Find the first number, including decimals and negative numbers
    match = re.search(r"-?\d+(?:\.\d+)?", value)

    if match:
        return float(match.group())

    return None


def map_heart_disease_features(ocr_data):
    """
    Convert Gemini OCR output into the
    feature format required by the Heart Disease model.
    """

    medical_values = ocr_data.get("medical_values", {})

    mapped_data = {}

    # Age comes from the main OCR field
    mapped_data["age"] = clean_numeric_value(
        ocr_data.get("age")
    )

    # Map medical report names to ML model features
    for feature in HEART_DISEASE_FEATURES:

        if feature == "age":
            continue

        aliases = FEATURE_ALIASES.get(
            feature,
            [feature]
        )

        # Get original OCR value
        raw_value = find_value(
            medical_values,
            aliases
        )

        # Convert value to numeric format
        mapped_data[feature] = clean_numeric_value(
            raw_value
        )

    return mapped_data


def check_missing_features(mapped_data):
    """
    Check which Heart Disease model features
    are still missing.
    """

    missing_features = []

    for feature in HEART_DISEASE_FEATURES:

        value = mapped_data.get(feature)

        if value is None or value == "":
            missing_features.append(feature)

    return missing_features