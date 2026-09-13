🏥 NIRAMAYA AI

AI-powered platform for multi-disease screening and healthcare assistance

🚀 Live Demo:
https://hybrid-quantum-machine-learning-platform-for-early-disease-det.streamlit.app/


📌 About the Project

NIRAMAYA AI is a healthcare-focused Machine Learning project that brings multiple disease screening models into one web application.

Instead of building a separate application for every disease, NIRAMAYA provides one platform where users can select a disease, enter the required medical information, and get a preliminary screening result.

The project also includes medical report reading using Google Gemini Vision, separate Patient and Doctor accounts, patient screening history, and an AI chat assistant.

The main idea is simple:

Make Machine Learning-based health screening easier to use and understand.

🚀 Live Demo

Try the deployed application:

👉 https://hybrid-quantum-machine-learning-platform-for-early-disease-det.streamlit.app/

The application is deployed using Streamlit Community Cloud.

✨ Main Features

🏥 Multi-disease screening

👤 Patient registration and login

👨‍⚕️ Doctor registration and login

🔎 Doctor can search patients using Patient ID or name

📄 Medical report/image upload

🤖 Medical information extraction using Gemini Vision

🧠 Machine Learning-based screening

📊 Accuracy, Precision, Recall, F1 and ROC-AUC evaluation

💾 Patient screening history using SQLite

💬 Floating AI chat assistant

🥦 Nutrient, vitamin and mineral information

⚠️ Risk-factor information

🛡️ Patient precautions

🎨 Modern healthcare-focused Streamlit interface

🧠 Diseases Supported

NIRAMAYA currently includes six disease screening models:

Disease

Dataset

❤️ Heart Disease

UCI Heart Disease

🧠 Parkinson's Disease

UCI Parkinson's

🩸 Diabetes

UCI Diabetes

🫘 Chronic Kidney Disease

UCI CKD

❤️ Heart Failure

UCI Heart Failure

🎗️ Breast Cancer

UCI Breast Cancer Wisconsin

Each disease has its own preprocessing and trained model because the required medical features are different.

🤖 Machine Learning

The project uses several classical Machine Learning classification algorithms:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

The models are trained and compared before selecting a suitable model for each disease.

Techniques Used

Data cleaning

Missing-value handling

Duplicate detection/removal

Categorical encoding

Feature selection

Train/Test split

Feature scaling

GridSearchCV

Hyperparameter tuning

Model serialization with Joblib

Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC-AUC

🔄 Machine Learning Workflow

UCI Dataset
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Duplicate Detection
     ↓
Categorical Encoding
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Train Multiple Models
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Save Model + Scaler + Features
     ↓
Streamlit Application

📄 Medical Report OCR

One of the main features of NIRAMAYA is the ability to extract medical information from uploaded reports/images.

The application uses Google Gemini Vision to identify available medical parameters.

Workflow

Upload Medical Report
        ↓
Gemini Vision
        ↓
Extract Available Values
        ↓
Match Values With Model Features
        ↓
Check Required Features
        ↓
All Required Values Available?
       / \
     YES   NO
      ↓     ↓
   Predict  Ask for Missing Values

NIRAMAYA does not guess missing medical values.

If the uploaded report does not contain all the information required by the selected model, the missing values need to be provided separately.

Supported Image Formats

PNG

JPG

JPEG

WEBP

👤 Patient Module

Patients can create an account and receive a unique Patient ID.

Example:

NIR-P-XXXXXXXX

Patient Registration

A patient provides:

Name

Gender

Age

Password

After registration, the application generates a unique Patient ID.

Patient Login

Patients can log in using:

Patient ID
+
Password

After login, they can access their account and previous screening information.

🩺 Patient Screening

Patients can:

Select a disease.

Enter medical parameters manually.

Upload a medical report/image.

Extract available information using Gemini Vision.

Provide missing information when required.

Run the screening model.

View the screening result.

View possible risk factors.

View associated nutrient/vitamin/mineral information.

View precautions.

Screening records can be stored in the patient history database.

👨‍⚕️ Doctor Module

Doctors have a separate registration and login system.

A registered doctor receives a unique Doctor ID such as:

NIR-D-XXXXXXXX

Doctor registration includes:

Doctor name

Gender

Medical domain

Password

🔎 Patient Search

Doctors can search for patients using:

Patient ID

or:

Patient Name

Available information can include:

Patient ID

Patient name

Gender

Age

Previous screening records

Disease screened

Screening result

Confidence

Date and time

Stored medical input data

Associated Doctor ID

This makes it easier to review previous screening activity.

👨‍⚕️ Doctor Screening Workflow

Doctor Login
      ↓
Search Patient
      ↓
Select Disease
      ↓
Manual Parameters / Medical Report
      ↓
OCR & Feature Extraction
      ↓
Machine Learning Model
      ↓
Screening Result
      ↓
Risk Factors
      ↓
Nutrient / Vitamin / Mineral Information

The doctor interface does not show the patient precaution section.

💬 AI Chat Assistant

NIRAMAYA includes a floating AI chat button on the right side of the application.

The button is designed as a circular dark-green icon.

Clicking the button opens the AI chat panel.

The assistant can be used for general healthcare-related questions and information.

The AI assistant is for general informational support and should not replace professional medical advice.

🗄️ Patient History Database

NIRAMAYA uses SQLite to store account and screening information.

Patient information

Patient ID
Name
Gender
Age
Password Hash
Account Creation Time

Doctor information

Doctor ID
Doctor Name
Gender
Medical Domain
Password Hash

Screening history

Patient ID
Doctor ID
Disease
Prediction
Confidence
Date/Time
Medical Input Data

This allows previous screening records to be retrieved later.

📁 Project Structure

Hybrid-Quantum-Machine-Learning-Platform-for-Early-Disease-Detection/
│
├── app.py
├── patient_history.py
├── requirements.txt
├── setup.py
├── .gitignore
├── README.md
│
├── models/
│   ├── heart_disease/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── features.pkl
│   │
│   ├── parkinsons/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── features.pkl
│   │
│   ├── diabetes/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── features.pkl
│   │
│   ├── kidney/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── features.pkl
│   │
│   ├── heart_failure/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── features.pkl
│   │
│   └── breast_cancer/
│       ├── best_model.pkl
│       ├── scaler.pkl
│       └── features.pkl
│
└── report_ai/
    └── ocr/
        └── reader.py

🛠️ Technologies Used

Programming

Python

Machine Learning

Scikit-learn

XGBoost

NumPy

Pandas

Data Visualization

Matplotlib

Seaborn

Web Application

Streamlit

AI / OCR

Google Gemini Vision

Google GenAI SDK

Database

SQLite

Model Saving

Joblib

Configuration

Python-dotenv

Streamlit Secrets

Deployment

GitHub

Streamlit Community Cloud

💻 Run the Project Locally

1. Clone the repository

git clone https://github.com/kunal957662/Hybrid-Quantum-Machine-Learning-Platform-for-Early-Disease-Detection.git

Then:

cd Hybrid-Quantum-Machine-Learning-Platform-for-Early-Disease-Detection

2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

3. Install dependencies

python -m pip install --upgrade pip

Then:

pip install -r requirements.txt

4. Configure Gemini API

Create a local .env file:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

Do not commit this file to GitHub.

5. Run the application

streamlit run app.py

The application will normally open at:

http://localhost:8501

☁️ Streamlit Cloud Deployment

NIRAMAYA is deployed through Streamlit Community Cloud.

The deployment flow is:

Local Development
       ↓
Git
       ↓
GitHub
       ↓
Streamlit Community Cloud
       ↓
Live NIRAMAYA AI

Live Demo

👉 https://hybrid-quantum-machine-learning-platform-for-early-disease-det.streamlit.app/

For the Gemini API key, use Streamlit Cloud Secrets.

Example:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

🔐 Security & Privacy

Because NIRAMAYA works with patient-related information, security is important.

Never commit:

.env

Gemini API keys

patient_history.db

Private medical reports

Temporary OCR files

Real patient information

For a production healthcare application, additional security would be required, including:

Strong password hashing such as Argon2 or bcrypt

Database encryption

Secure sessions

Role-based access control

Patient consent

Audit logs

Secure data storage

Appropriate healthcare privacy and regulatory compliance

⚠️ Medical Disclaimer

NIRAMAYA AI is a screening and educational project.

The predictions generated by the Machine Learning models are not medical diagnoses.

For example, a result such as:

Higher Risk

does not mean that the person definitely has the disease.

The result should be treated as preliminary screening information and should be discussed with a qualified healthcare professional.

Do not use this application for emergency medical decisions or as a replacement for professional medical care.

🔮 Future Improvements

Some features I would like to work on in future versions include:

🤖 Machine Learning

Explainable AI using SHAP

Feature-importance visualization

Better model calibration

Ensemble approaches

Hybrid classical-quantum ML experiments

📄 Medical Reports

PDF report support

Multi-page report processing

Better laboratory-value extraction

More medical report formats

Improved missing-value detection

👨‍⚕️ Doctor Features

Patient consent system

Better access control

Appointment management

Prescription management

Detailed medical history

🔐 Security

Stronger authentication

Database encryption

Secure sessions

Role-based access control

Audit logging

🌍 Accessibility

Multiple language support

Local-language healthcare information

Better mobile interface

Voice-based interaction

📊 Project Status

Feature

Status

Multi-disease screening

✅

Heart Disease

✅

Parkinson's Disease

✅

Diabetes

✅

Chronic Kidney Disease

✅

Heart Failure

✅

Breast Cancer

✅

Logistic Regression

✅

SVM

✅

Random Forest

✅

XGBoost

✅

Hyperparameter tuning

✅

Medical Report OCR

✅

Gemini Vision

✅

Patient registration

✅

Patient login

✅

Doctor registration

✅

Doctor login

✅

Patient search

✅

Patient history

✅

SQLite database

✅

AI chat assistant

✅

Streamlit deployment

✅

Live demo

✅

🎯 Why I Built This

I built NIRAMAYA AI to go beyond simply training Machine Learning models and checking their accuracy.

I wanted to understand how a complete ML project works when it is turned into an actual application.

The project helped me work with:

Dataset
   ↓
EDA
   ↓
Data Preprocessing
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit
   ↓
OCR
   ↓
Database
   ↓
Authentication
   ↓
Deployment

It combines the Machine Learning concepts I am learning with a practical healthcare use case.

👨‍💻 Author

Kunal Kumar

B.Tech Electronics & Communication Engineering Student

Interested in:

Data Science

Machine Learning

Artificial Intelligence

Healthcare AI

IoT

Embedded Systems

Research

GitHub

👉 https://github.com/kunal957662

Project Repository

👉 https://github.com/kunal957662/Hybrid-Quantum-Machine-Learning-Platform-for-Early-Disease-Detection

Live Demo

👉 https://hybrid-quantum-machine-learning-platform-for-early-disease-det.streamlit.app/

⭐ Support the Project

If you find NIRAMAYA AI interesting:

⭐ Star the repository
🍴 Fork the project
🐛 Report bugs
💡 Suggest improvements
🤝 Contribute

📜 Disclaimer

This project is created for educational, research, and demonstration purposes.

NIRAMAYA AI is not a certified medical device and should not be used as a replacement for professional medical advice, diagnosis, or treatment.
