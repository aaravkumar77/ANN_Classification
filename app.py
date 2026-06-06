import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

st.title("Customer Churn Prediction")
st.markdown("Predict whether a bank customer is likely to leave based on their profile.")

# ---------- Load model & transformers ----------
use_onnx = False
onnx_session = None
model = None
tf = None
if os.path.exists('model.onnx'):
    try:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession('model.onnx')
        use_onnx = True
    except Exception as e:
        st.warning(f"ONNX runtime unavailable or failed to load model.onnx: {e}. Will try TensorFlow if present.")

if not use_onnx:
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model.h5')
    except Exception as e:
        st.error(f"Model load error: TensorFlow not available and ONNX not usable: {e}")
        st.stop()

with open('onehot_encoder_geo.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# ---------- UI: user inputs ----------
with st.sidebar:
    st.header("How to use")
    st.write(
        "Enter the customer information below, then click Predict. "
        "The model will estimate whether the customer is likely to churn."
    )
    st.markdown("---")
    st.write("**Tip:** Use real customer details for the most accurate result.")
    st.write("If you want, change the values and re-run prediction.")

st.subheader("Customer profile")
col1, col2 = st.columns(2)
with col1:
    credit_score = st.number_input(
        "Credit score",
        min_value=0,
        max_value=1000,
        value=600,
        help="Higher credit score means lower risk."
    )
    geography = st.selectbox(
        "Country",
        ["France", "Spain", "Germany"],
        help="Select the customer’s country."
    )
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        help="Customer gender."
    )
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        help="Customer age in years."
    )
    tenure = st.slider(
        "Tenure (years with bank)",
        min_value=0,
        max_value=10,
        value=3,
        help="How many years the customer has been with the bank."
    )
with col2:
    balance = st.number_input(
        "Account balance",
        min_value=0.0,
        value=60000.0,
        step=100.0,
        format="%.2f",
        help="Customer’s current bank balance."
    )
    estimated_salary = st.number_input(
        "Estimated salary",
        min_value=0.0,
        value=50000.0,
        step=100.0,
        format="%.2f",
        help="Customer’s approximate annual income."
    )
    num_of_products = st.slider(
        "Number of bank products",
        min_value=1,
        max_value=4,
        value=2,
        help="How many bank products the customer has."
    )
    has_cr_card = st.selectbox(
        "Has credit card?",
        ["No", "Yes"],
        help="Does the customer have a credit card with the bank?"
    )
    is_active_member = st.selectbox(
        "Is active member?",
        ["No", "Yes"],
        help="Whether the customer is actively using their account."
    )

has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

st.markdown("---")
st.write("### Customer input summary")
st.json(input_data)

# ---------- Prepare DataFrame ----------
input_df = pd.DataFrame([input_data])   # single-row dataframe

# ---------- Encode Geography (OneHotEncoder expected a 2D input) ----------
geo_encoded = label_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_cols = label_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# ---------- Encode Gender (handle LabelEncoder or OneHotEncoder) ----------
gender_encoded_df = None
try:
    # If encoder is OneHotEncoder (has get_feature_names_out)
    gender_encoded = label_encoder_gender.transform(input_df[['Gender']]).toarray()
    gender_cols = label_encoder_gender.get_feature_names_out(['Gender'])
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=gender_cols)
except Exception:
    # Fall back: LabelEncoder (returns 1D array)
    try:
        gender_num = label_encoder_gender.transform(input_df['Gender'])
        # create column name same as original (e.g., 'Gender')
        gender_encoded_df = pd.DataFrame(gender_num, columns=['Gender'])
    except Exception as e:
        st.error(f"Gender encoder error: {e}")
        st.stop()

# ---------- Drop original categorical columns (we'll replace them with encoded) ----------
input_df = input_df.drop(columns=['Geography', 'Gender'])

# ---------- Combine all features in a single DataFrame (order doesn't yet matter) ----------
combined_df = pd.concat(
    [input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True), gender_encoded_df.reset_index(drop=True)],
    axis=1
)

st.write("### Processed features (before scaling)")
st.dataframe(combined_df)

# ---------- Make sure scaler input shape/order matches what scaler expects ----------
# If scaler has feature names recorded, reorder combined_df accordingly
if hasattr(scaler, 'feature_names_in_'):
    scaler_cols = list(getattr(scaler, 'feature_names_in_'))
    # check that all scaler_cols are present
    missing = [c for c in scaler_cols if c not in combined_df.columns]
    if missing:
        st.error(f"Scaler expects these columns but they are missing: {missing}")
        st.stop()
    combined_df = combined_df[scaler_cols]
else:
    # If scaler doesn't store feature names, check numeric match
    if hasattr(scaler, 'n_features_in_'):
        if scaler.n_features_in_ != combined_df.shape[1]:
            st.error(
                f"Scaler expects {scaler.n_features_in_} features but got {combined_df.shape[1]}. "
                "Check column order or how you saved the scaler."
            )
            st.stop()
    # otherwise proceed (best-effort)

# ---------- Scale ----------
try:
    input_data_scaled = scaler.transform(combined_df)
except Exception as e:
    st.error(f"Error when scaling input: {e}")
    st.stop()

# ---------- Predict (button to run) ----------
if st.button("Predict"):
    try:
        if use_onnx and onnx_session is not None:
            input_array = input_data_scaled.astype(np.float32)
            input_name = onnx_session.get_inputs()[0].name
            res = onnx_session.run(None, {input_name: input_array})
            prediction = np.array(res[0])
        else:
            prediction = model.predict(input_data_scaled)

        prediction_proba = float(np.ravel(prediction)[0])
        prediction_pct = prediction_proba * 100

        st.write("### Prediction Probability")
        st.write(f"{prediction_proba:.4f} ({prediction_pct:.1f}% chance)")

        if prediction_proba > 0.5:
            st.error(
                f"🔴 The customer is likely to churn.\n\n"
                f"**Churn Probability:** {prediction_proba:.2f} ({prediction_pct:.1f}%)\n\n"
                f"That means there is a {prediction_pct:.1f}% chance the customer will leave.\n\n"
                f"This customer may need attention or retention efforts."
            )
        else:
            st.success(
                f"🟢 The customer is unlikely to churn.\n\n"
                f"**Churn Probability:** {prediction_proba:.2f} ({prediction_pct:.1f}%)\n\n"
                f"That means there is only a {prediction_pct:.1f}% chance the customer will leave.\n\n"
                f"This customer is most likely to stay with the team."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
