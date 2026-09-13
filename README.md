# 🏦 Customer Churn Classification & Salary Regression Using Artificial Neural Networks

## 📌 Project Overview

This project focuses on predicting **bank customer churn** and **estimated customer salary** using **Artificial Neural Networks (ANN)** built with TensorFlow/Keras.

The project is based on the classic `Churn_Modelling.csv` bank customer dataset and demonstrates two complementary deep learning tasks on the same dataset:

- A **binary classification** ANN that predicts whether a customer will leave (churn) the bank
- A **regression** ANN that predicts a customer's estimated salary

The solution supports:

- Customer retention strategy
- Identifying at-risk, high-value customers
- Understanding drivers of customer attrition
- Data-driven salary estimation use cases

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset, containing bank customer records with demographic, account, and activity information.

**Main Features**

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country/region |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years the customer has been with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products the customer uses |
| HasCrCard | Whether the customer has a credit card (0/1) |
| IsActiveMember | Whether the customer is an active member (0/1) |
| EstimatedSalary | Customer's estimated salary |
| Exited | Target for classification — whether the customer churned (0/1) |

**Dataset File**

| File | Description |
|---|---|
| Churn_Modelling.csv | Raw bank customer data used for both classification and regression tasks |

## 🔍 Exploratory Data Analysis & Preprocessing

- Loading and inspecting the dataset
- Checking missing values and data types
- Dropping non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoding the categorical `Gender` column with `LabelEncoder`
- One-hot encoding the categorical `Geography` column with `OneHotEncoder`
- Feature scaling with `StandardScaler`
- Train/test split for model validation

## ⚙️ Feature Engineering

**Encoded Features**
- Gender → Label Encoding (`label_encoder_gender.pkl`)
- Geography → One-Hot Encoding (`onehot_encoder_geo.pkl`)

**Scaling**
- All numerical and encoded features scaled using `StandardScaler` (`scaler.pkl`)

**Task-Specific Target Handling**
- **Classification task:** `Exited` is the target; all other fields (including `EstimatedSalary`) are used as predictors
- **Regression task:** `EstimatedSalary` is the target; `Exited` is used as a predictor instead

## 🔄 Project Workflow

```
Business Understanding
        ↓
Data Understanding
        ↓
Data Cleaning & Encoding
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering & Scaling
        ↓
ANN Model Development (Classification + Regression)
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Streamlit App Deployment
```

## 🤖 Machine Learning

Two separate ANN models are trained on the same base dataset:

| Task | Target Variable | Model File | Notebook |
|---|---|---|---|
| Churn Classification | `Exited` | `model.h5` | `experiments.ipynb` |
| Salary Regression | `EstimatedSalary` | `regression_model.h5` | `salaryregression.ipynb` |
| Hyperparameter Tuning | `Exited` | — | `hyperparametertuningann.ipynb` |

**Model Architecture**
- Fully connected (Dense) feed-forward Artificial Neural Network built with TensorFlow/Keras
- Sigmoid output activation for the churn classification task
- Linear output activation for the salary regression task
- Hyperparameter tuning performed with `scikeras` to search over layers, units, and other network parameters

**Training Monitoring**
- Training runs are logged with TensorBoard
- Classification logs stored under `logs/fit`
- Regression logs stored under `regressionlogs/fit`

## 📈 Model Evaluation

| Task | Evaluation Approach |
|---|---|
| Churn Classification | Binary cross-entropy loss, accuracy, churn probability threshold at 0.5 |
| Salary Regression | Mean squared error / mean absolute error on held-out test data |

## 🏆 Final Deliverables

| Task | Output |
|---|---|
| Churn Classification | Trained ANN (`model.h5`) served through an interactive Streamlit app |
| Salary Regression | Trained ANN (`regression_model.h5`) served through a separate Streamlit app |

> "Where a single dataset trains two different neural networks — one to classify, one to predict."

## 💡 Business Insights

- Which customer profiles are most likely to churn
- How geography, age, and account activity relate to churn risk
- How number of products and account balance influence churn probability
- Estimated salary patterns across customer segments
- Opportunities for targeted retention and engagement strategies

## 📊 Interactive Streamlit Applications

Two interactive Streamlit apps are included, one for each ANN model:

- 🔮 **Customer Churn Prediction** (`app.py`) — enter customer details to get an instant churn probability from the trained classification model
- 💰 **Estimated Salary Prediction** (`streamlit_regression.py`) — enter customer details to get an instant estimated salary from the trained regression model

## 🖥️ Dashboard Preview

### 🔮 Customer Churn Prediction (Classification)

![Customer Churn Prediction Dashboard](https://raw.githubusercontent.com/sgl13/ANN-Classification-Churn/a76cc51fdf6ac2628e9005cc37182553ba64b794/Screenshot_ann_Classification/Customer%20Churn%20Prediction_Dashboard%202026-09-12%20at%204.48.59%E2%80%AFPM.png)

*Input form where a user enters Geography, Gender, Age, Balance, Credit Score, Estimated Salary, Tenure, Number of Products, and account status to get a churn prediction.*

![Customer Churn Prediction Output](https://raw.githubusercontent.com/sgl13/ANN-Classification-Churn/a76cc51fdf6ac2628e9005cc37182553ba64b794/Screenshot_ann_Classification/Customer%20Churn_Prediction%202026-09-12%20at%204.49.21%E2%80%AFPM.png)

*The app returns a churn probability score along with a plain-language verdict on whether the customer is likely to churn.*

### 💰 Estimated Salary Prediction (Regression)

![Estimated Salary Prediction Dashboard](https://raw.githubusercontent.com/sgl13/ANN-Classification-Churn/a76cc51fdf6ac2628e9005cc37182553ba64b794/screenshot_ann_regression/Screenshot_Estimated%20Prediction%20Dashboard%202026-09-12%20at%207.59.22%E2%80%AFPM.png)

*Input form where a user enters Geography, Gender, Age, Balance, Credit Score, Tenure, Number of Products, account status, and churn status to get a salary estimate.*

![Estimated Salary Prediction Output](https://raw.githubusercontent.com/sgl13/ANN-Classification-Churn/a76cc51fdf6ac2628e9005cc37182553ba64b794/screenshot_ann_regression/Estimated_Predictions%20salary%202026-09-12%20at%207.59.34%E2%80%AFPM.png)

*The app returns the model's predicted estimated salary for the entered customer profile.*

## 💻 Technologies Used

Python · TensorFlow / Keras · Scikit-learn · Pandas · NumPy · Matplotlib · Streamlit · TensorBoard · SciKeras · Pickle · Git & GitHub

## 📁 Project Structure

```
ANN-Classification-Churn/
│
├── .devcontainer/
├── .idea/
│
├── logs/
│   └── fit/                          # TensorBoard logs — classification model
│
├── regressionlogs/
│   └── fit/                          # TensorBoard logs — regression model
│
├── Screenshot_ann_Classification/     # Churn app screenshots
├── screenshot_ann_regression/         # Salary regression app screenshots
│
├── Churn_Modelling.csv                # Raw dataset
│
├── experiments.ipynb                  # Churn classification ANN — training notebook
├── hyperparametertuningann.ipynb      # Hyperparameter tuning for classification ANN
├── salaryregression.ipynb             # Salary regression ANN — training notebook
│
├── label_encoder_gender.pkl           # LabelEncoder for Gender
├── onehot_encoder_geo.pkl             # OneHotEncoder for Geography
├── scaler.pkl                         # StandardScaler for features
│
├── model.h5                           # Trained churn classification model
├── regression_model.h5                # Trained salary regression model
│
├── app.py                             # Streamlit app — Churn Prediction
├── streamlit_regression.py            # Streamlit app — Salary Prediction
├── main.py
│
├── requirements.txt
└── README.md
```

Keep your virtual environment (`.venv/` or `venv/`) local and excluded from Git via `.gitignore`.

## 🚀 How to Run the Project

**1. Clone the repository**
```bash
git clone https://github.com/sgl13/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

**2. Create a virtual environment**
```bash
python -m venv .venv
```

**3. Activate the virtual environment**

macOS / Linux
```bash
source .venv/bin/activate
```

Windows
```bash
.venv\Scripts\activate
```

**4. Upgrade pip**
```bash
python -m pip install --upgrade pip
```

**5. Install required libraries**
```bash
pip install -r requirements.txt
```

**6. Verify installation**
```bash
python --version
streamlit --version
```

**7. Run the Churn Prediction app**
```bash
streamlit run app.py
```

**8. Run the Salary Prediction app**
```bash
streamlit run streamlit_regression.py
```

Open the local URL shown in the terminal, usually:
```
http://localhost:8501
```

**9. Stop the application**

Press `Ctrl + C`

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `experiments.ipynb` | Data preprocessing, encoding, scaling, and training the churn classification ANN |
| `hyperparametertuningann.ipynb` | Hyperparameter search for the classification ANN using SciKeras |
| `salaryregression.ipynb` | Data preparation and training of the salary regression ANN |

## 📦 Project Deliverables

- Source Code
- Jupyter Notebooks (Classification, Hyperparameter Tuning, Regression)
- Trained ANN Models (Classification & Regression)
- Fitted Encoders and Scaler
- Two Interactive Streamlit Applications
- Screenshots of Both Applications
- GitHub Repository

## 🎓 Learning Outcomes

After completing this project, the learner should be able to:

- Preprocess and encode structured tabular data for neural networks
- Build and train Artificial Neural Networks for classification and regression
- Apply feature scaling and categorical encoding correctly and consistently
- Perform hyperparameter tuning for deep learning models
- Monitor training with TensorBoard
- Deploy trained models through interactive Streamlit applications
- Present an end-to-end deep learning project using industry best practices

## 🎯 Project Objective

The objective of this project is to build and deploy two Artificial Neural Network models — one for **customer churn classification** and one for **estimated salary regression** — using the same underlying bank customer dataset.

**Skills Demonstrated**

Python · TensorFlow · Keras · Deep Learning · Artificial Neural Networks · Scikit-learn · Data Preprocessing · Feature Encoding · Feature Scaling · Hyperparameter Tuning · Regression · Classification · Streamlit · TensorBoard · Git & GitHub

## 👨‍💻 Author

**Shivakumar G L**

ANN-Based Customer Churn & Salary Prediction

> "Turning a single dataset into two working neural networks."

`Python` | `Deep Learning` | `TensorFlow` | `Streamlit`