import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------- Page setup ----------
st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="centered")

st.title("🏥 Medical Insurance Cost Prediction")
st.write(
    "This app predicts estimated medical insurance charges based on personal details, "
    "using a Linear Regression model trained on the Medical Cost Personal Dataset."
)

# ---------- Load data & train model (cached so it only runs once) ----------
@st.cache_resource
def load_model():
    df = pd.read_csv("data/insurance_cleaned.csv")
    X = df.drop("charges", axis=1)
    y = df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

model, feature_columns = load_model()

# ---------- Input form ----------
st.subheader("Enter your details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ---------- Predict ----------
if st.button("Predict Insurance Cost", type="primary"):
    sex_val = 0 if sex == "male" else 1
    smoker_val = 1 if smoker == "yes" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    input_data = pd.DataFrame(
        [[age, sex_val, bmi, children, smoker_val, region_northwest, region_southeast, region_southwest]],
        columns=feature_columns,
    )

    prediction = model.predict(input_data)[0]

    st.success(f"### Predicted Insurance Cost: ${prediction:,.2f}")

    with st.expander("See input summary"):
        st.write(f"**Age:** {age}  |  **Sex:** {sex}  |  **BMI:** {bmi}")
        st.write(f"**Children:** {children}  |  **Smoker:** {smoker}  |  **Region:** {region}")

st.divider()
st.caption("Built with Streamlit · Model: Linear Regression · R² ≈ 0.81")
