import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

model = tf.keras.models.load_model("regression_model.h5")

# Load encoders
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------

st.title("Estimated Salary Prediction")

st.write("Enter customer details to predict the estimated salary.")


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    18,
    92,
    35
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

tenure = st.slider(
    "Tenure",
    0,
    10,
    5
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4,
    2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

exited = st.selectbox(
    "Exited",
    [0, 1]
)


# ---------------------------------------------------------
# Create input DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [
        label_encoder_gender.transform([gender])[0]
    ],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})


# ---------------------------------------------------------
# One-Hot Encode Geography
# ---------------------------------------------------------

geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ["Geography"]
    )
)


# ---------------------------------------------------------
# Combine all features
# ---------------------------------------------------------

input_data = pd.concat(
    [
        input_data,
        geo_encoded_df
    ],
    axis=1
)


# ---------------------------------------------------------
# Check scaler compatibility
# ---------------------------------------------------------

expected_columns = list(scaler.feature_names_in_)

if "EstimatedSalary" in expected_columns:

    st.error(
        """
        The saved scaler.pkl is incompatible with this
        Estimated Salary Prediction model.

        EstimatedSalary should be the target, not an input.
        Please regenerate scaler.pkl using the training code.
        """
    )

    st.stop()


# ---------------------------------------------------------
# Arrange columns in correct order
# ---------------------------------------------------------

input_data = input_data[expected_columns]


# ---------------------------------------------------------
# Scale input
# ---------------------------------------------------------

input_data_scaled = scaler.transform(input_data)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

prediction = model.predict(
    input_data_scaled,
    verbose=0
)

predicted_salary = prediction[0][0]


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

st.success(
    f"Predicted Estimated Salary: ${predicted_salary:,.2f}"
)
