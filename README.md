## 🤖 Model Technology Stack

Our platform uses multiple machine learning models to screen for the risk of different diseases. Instead of using one model for everything, each disease has its own trained model along with the required features and preprocessing steps.

### 🔬 Machine Learning Models

We experimented with and evaluated several classification algorithms:

- **Logistic Regression** – A simple and easy-to-understand model that works well for binary classification.
- **Support Vector Machine (SVM)** – Helps classify data by finding a suitable boundary between different classes.
- **Random Forest** – Combines multiple decision trees to make more reliable predictions.
- **XGBoost** – A powerful boosting algorithm that can capture complex patterns in the data.

The best-performing model is selected for each disease based on the evaluation results.

### ⚙️ Technologies Used

| Technology | How We Use It |
|---|---|
| **Python** | Main programming language |
| **NumPy** | Numerical calculations and data handling |
| **Pandas** | Loading, cleaning, and preparing datasets |
| **Scikit-learn** | Machine learning, preprocessing, tuning, and evaluation |
| **XGBoost** | Gradient boosting for classification |
| **Joblib** | Saving and loading trained models |
| **StandardScaler** | Scaling features before model training and prediction |
| **Matplotlib** | Creating graphs during data analysis |
| **Seaborn** | Visualizing patterns and relationships in the datasets |

### 🧠 How the Models Are Built

Each disease model goes through a similar machine learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Selection
   ↓
Train / Test Split
   ↓
Feature Scaling
   ↓
Model Training
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




live demo link :https://hybrid-quantum-machine-learning-platform-for-early-disease-det.streamlit.app/
