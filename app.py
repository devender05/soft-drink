# app.py - Profitable Product Predictor (works with YOUR CSV)
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Profit Predictor", page_icon="Money")
st.title("Will This Product Be Profitable?")
st.write("Predict if a product/order will make money using your sales data")

# --------------------- Load Model ---------------------
MODEL_PATH = "profit_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Run the training script first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.markdown("---")

# --------------------- Input Form ---------------------
st.subheader("Enter Product/Order Details")

col1, col2 = st.columns(2)

with col1:
    units_sold = st.number_input("Units Sold", min_value=1, value=151)
    revenue = st.number_input("Revenue ($)", min_value=0.0, value=2141.0, step=10.0)
    cogs = st.number_input("Cost of Goods Sold ($)", min_value=0.0, value=4956.0, step=10.0)

with col2:
    category = st.selectbox("Category", ["Coffee", "Tea", "Soft Drink", "Energy Drink", "Juice", "Water"])
    company = st.text_input("Company (e.g. Coca-Cola)", "Coca-Cola")
    product = st.text_input("Product Name", "Coca-Cola Classic")

# Create single sample
sample = pd.DataFrame([{
    "Units Sold": units_sold,
    "Revenue": revenue,
    "Cost of Goods Sold": cogs,
    "Category": len(category),           # trick: use length as numeric proxy
    "is_coke": 1 if "coca" in company.lower() else 0
}])

# --------------------- CSV Upload ---------------------
st.markdown("---")
uploaded_file = st.file_uploader("Or upload your full CSV for batch prediction", type="csv")

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    
    # Auto-create required numeric features
    df_input["is_coke"] = df_input["Company"].str.lower().str.contains("coca", na=False).astype(int)
    df_input["category_length"] = df_input["Category"].str.len()
    
    required = ["Units Sold", "Revenue", "Cost of Goods Sold", "is_coke", "category_length"]
    df_input = df_input[required]
else:
    df_input = sample

# --------------------- Predict ---------------------
if st.button("Predict Profitability", type="primary"):
    predictions = model.predict(df_input)
    probabilities = model.predict_proba(df_input)[:, 1]  # Prob of being Profitable

    result_df = df_input.copy()
    result_df["Prediction"] = ["Profitable" if p == 1 else "Loss-Making" for p in predictions]
    result_df["Confidence"] = probabilities.round(4)

    profitable_count = (predictions == 1).sum()
    total = len(predictions)

    st.markdown("---")
    if len(df_input) == 1:
        if predictions[0] == 1:
            st.success(f"**This will be PROFITABLE!** (Confidence: {probabilities[0]:.1%})")
        else:
            st.error(f"**This will be LOSS-MAKING** (Confidence: {(1-probabilities[0]):.1%})")
    else:
        st.success(f"{profitable_count} out of {total} items are predicted **Profitable**")

    st.dataframe(result_df.style.background_gradient(subset=["Confidence"], cmap="Greens"))

    if len(predictions) > 1:

        st.bar_chart(result_df["Prediction"].value_counts())
