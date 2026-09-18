import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Electric Bill Predictor", page_icon="⚡", layout="centered"
)

# Load the saved model pipeline (.pkl) with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("electric_bill_model_acfan.pkl")
        return model, None
    except FileNotFoundError:
        return (
            None,
            "Model file `electric_bill_model_acfan.pkl` not found. Please ensure it is in the same directory as `app.py`.",
        )
    except Exception as e:
        return None, f"An error occurred while loading the model: {e}"


model, load_error = load_model()

# App UI Header
st.title("⚡ Electric Bill Predictor")
st.write(
    "Enter the number of **AC Units** and **Fan Units** consumed to predict your estimated electricity bill using a trained Polynomial Regression model."
)

# Show error if model failed to load
if load_error:
    st.error(f"🚨 **Initialization Error**: {load_error}")
    st.stop()

# User input form
with st.form("prediction_form"):
    ac_units = st.number_input(
        "AC Units Consumed",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0,
        help="Enter the total units consumed by your AC.",
    )

    fan_units = st.number_input(
        "Fan Units Consumed",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0,
        help="Enter the total units consumed by your Fans.",
    )

    submit_button = st.form_submit_button(label="Predict Bill")

# Prediction logic with error handling
if submit_button:
    try:
        # Format input into a DataFrame matching model expectation
        input_data = pd.DataFrame(
            [[ac_units, fan_units]], columns=["AC_Units", "Fan_Units"]
        )

        # Predict using the loaded model
        predicted_bill = model.predict(input_data)[0]

        # Ensure prediction is non-negative
        predicted_bill = max(0.0, predicted_bill)

        # Display result
        st.success(
            f"### Estimated Electric Bill: ₹ {predicted_bill:,.2f}"
        )
        st.info(
            f"Based on **{ac_units}** AC units and **{fan_units}** Fan units consumed."
        )

    except Exception as e:
        st.error(
            f"⚠️ An error occurred during prediction: {e}. Please check your inputs and try again."
        )

# Footer info
st.markdown("---")
st.caption(
    "Developed with Streamlit & Scikit-Learn | Christ College Project"
)
