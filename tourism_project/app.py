
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "deployment" / "tourism_model.joblib"


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tourism Product Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Tourism Product Prediction")

st.write(
    "Enter customer details to predict whether the customer "
    "will purchase the tourism package."
)


# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=60,
    value=10
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Unmarried",
        "Single",
        "Divorced"
    ]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=3
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Manager",
        "Executive",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)


# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict", type="primary"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "🎉 Prediction: Customer is likely to purchase the product."
        )
    else:
        st.info(
            "Prediction: Customer is unlikely to purchase the product."
        )

    # Show probability if the model supports it
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

        st.metric(
            "Purchase Probability",
            f"{probability:.2%}"
        )

    st.subheader("Input Data")
    st.dataframe(input_data)
