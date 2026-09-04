import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bank Personal Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------- Load saved models ----------
@st.cache_resource
def load_models():
    logistic_model = joblib.load("logistic_regression_model.pkl")
    random_forest_model = joblib.load("random_forest_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return logistic_model, random_forest_model, feature_columns

@st.cache_data
def load_performance():
    return pd.read_csv("model_performance.csv")

@st.cache_data
def load_best_model():
    with open("best_model_name.pkl", "rb") as f:
        return joblib.load(f)

logistic_model, random_forest_model, feature_columns = load_models()
performance = load_performance()
best_model_name = load_best_model()

# ---------- Styling ----------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 25px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}
.small-note {
    color: #666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 Bank Personal Loan Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-based customer prediction using Logistic Regression and Random Forest</div>',
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📌 Project Information")
    st.write("**Domain:** Banking / Machine Learning")
    st.write("**Target:** Personal Loan")
    st.write("**Algorithms:**")
    st.write("• Logistic Regression")
    st.write("• Random Forest")
    st.write("**Purpose:** Predict whether a customer is likely to accept a personal loan.")

    st.divider()
    st.info(
        "Enter the customer's information below. "
        "The exact same customer data is sent to both ML models."
    )

# ---------- Input section ----------
st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    experience = st.number_input("Experience (years)", min_value=0, max_value=60, value=10, step=1)
    income = st.number_input("Annual Income ($000)", min_value=1.0, max_value=1000.0, value=80.0, step=1.0)
    family = st.number_input("Family Size", min_value=1, max_value=10, value=3, step=1)

with col2:
    ccavg = st.number_input(
        "Average Credit Card Spending ($000/month)",
        min_value=0.0, max_value=100.0, value=2.0, step=0.1
    )
    education = st.selectbox(
        "Education Level",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1 - Undergraduate",
            2: "2 - Graduate",
            3: "3 - Advanced/Professional"
        }[x]
    )
    mortgage = st.number_input(
        "Mortgage ($000)",
        min_value=0.0, max_value=1000.0, value=0.0, step=1.0
    )

with col3:
    securities = st.selectbox(
        "Securities Account",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )
    cd_account = st.selectbox(
        "CD Account",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )
    online = st.selectbox(
        "Online Banking",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )
    credit_card = st.selectbox(
        "Credit Card",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

# ---------- Feature engineering ----------
def create_features():
    data = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities,
        "CD Account": cd_account,
        "Online": online,
        "CreditCard": credit_card
    }])

    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["Young", "Adult", "Middle_Aged", "Senior", "Elderly"]
    )

    data["Income_Category"] = pd.cut(
        data["Income"],
        bins=[0, 50, 100, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    data["Experience_Age_Ratio"] = data["Experience"] / data["Age"]

    data["Total_Financial_Products"] = (
        data["Securities Account"]
        + data["CD Account"]
        + data["Online"]
        + data["CreditCard"]
    )

    data["CCAvg_Category"] = pd.cut(
        data["CCAvg"],
        bins=[-1, 1, 2, 5, float("inf")],
        labels=["Low", "Moderate", "High", "Very_High"]
    )

    data["Family_Category"] = pd.cut(
        data["Family"],
        bins=[0, 2, 4, float("inf")],
        labels=["Small", "Medium", "Large"]
    )

    data["Income_Per_Family_Member"] = data["Income"] / data["Family"]
    data["Income_CCAvg_Interaction"] = data["Income"] * data["CCAvg"]

    categorical_columns = [
        "Age_Group",
        "Income_Category",
        "CCAvg_Category",
        "Family_Category"
    ]

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # Match exactly the feature columns used during model training.
    data = data.reindex(columns=feature_columns, fill_value=0)

    return data

# ---------- Prediction ----------
st.divider()

if st.button("🔮 Predict Personal Loan", type="primary", use_container_width=True):

    input_data = create_features()

    logistic_prediction = int(logistic_model.predict(input_data)[0])
    random_forest_prediction = int(random_forest_model.predict(input_data)[0])

    logistic_probability = None
    rf_probability = None

    if hasattr(logistic_model, "predict_proba"):
        logistic_probability = float(logistic_model.predict_proba(input_data)[0][1])

    if hasattr(random_forest_model, "predict_proba"):
        rf_probability = float(random_forest_model.predict_proba(input_data)[0][1])

    logistic_result = "Loan Accepted ✅" if logistic_prediction == 1 else "Loan Not Accepted ❌"
    rf_result = "Loan Accepted ✅" if random_forest_prediction == 1 else "Loan Not Accepted ❌"

    # ---------- Prediction cards ----------
    st.header("📊 Prediction Results")

    r1, r2 = st.columns(2)

    with r1:
        st.subheader("Logistic Regression")
        if logistic_prediction == 1:
            st.success(logistic_result)
        else:
            st.error(logistic_result)

        if logistic_probability is not None:
            st.metric("Probability of Loan Acceptance", f"{logistic_probability * 100:.2f}%")

    with r2:
        st.subheader("Random Forest")
        if random_forest_prediction == 1:
            st.success(rf_result)
        else:
            st.error(rf_result)

        if rf_probability is not None:
            st.metric("Probability of Loan Acceptance", f"{rf_probability * 100:.2f}%")

    # ---------- Comparison ----------
    st.header("⚖️ Algorithm Prediction Comparison")

    comparison = pd.DataFrame({
        "Algorithm": ["Logistic Regression", "Random Forest"],
        "Prediction": [
            "Accepted" if logistic_prediction else "Not Accepted",
            "Accepted" if random_forest_prediction else "Not Accepted"
        ],
        "Prediction Value": [logistic_prediction, random_forest_prediction]
    })

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Prediction chart
    fig, ax = plt.subplots(figsize=(8, 4))
    algorithms = ["Logistic Regression", "Random Forest"]
    values = [logistic_prediction, random_forest_prediction]

    ax.bar(algorithms, values)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Prediction")
    ax.set_title("Loan Prediction Comparison")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not Accepted", "Accepted"])
    for i, value in enumerate(values):
        ax.text(i, value + 0.05, str(value), ha="center", fontweight="bold")

    st.pyplot(fig)

    # ---------- Model performance ----------
    st.header("🏆 Model Performance")

    st.dataframe(performance, use_container_width=True, hide_index=True)

    if "F1 Score" in performance.columns:
        metric_col = "F1 Score"
    elif "F1" in performance.columns:
        metric_col = "F1"
    else:
        metric_col = None

    if metric_col:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(performance["Model"], performance[metric_col])
        ax2.set_title("F1-Score Comparison")
        ax2.set_ylabel("F1 Score")
        ax2.set_ylim(0, 1)
        for i, value in enumerate(performance[metric_col]):
            ax2.text(i, value + 0.02, f"{value:.3f}", ha="center")

        st.pyplot(fig2)

    # ---------- Best algorithm ----------
    st.header("🥇 Best Algorithm")

    st.success(f"Based on the saved model evaluation, the better algorithm is: **{best_model_name}**")

    st.write(
        "The best model is selected using the model evaluation results, "
        "with F1-score being especially useful when the target classes are imbalanced."
    )

    # ---------- Input summary ----------
    with st.expander("🔎 View Entered Customer Details"):
        st.dataframe(
            pd.DataFrame({
                "Feature": [
                    "Age", "Experience", "Income", "Family", "CCAvg",
                    "Education", "Mortgage", "Securities Account",
                    "CD Account", "Online", "Credit Card"
                ],
                "Value": [
                    age, experience, income, family, ccavg,
                    education, mortgage, securities,
                    cd_account, online, credit_card
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

st.divider()
st.caption("MSc IT Machine Learning Project • Bank Personal Loan Prediction")
