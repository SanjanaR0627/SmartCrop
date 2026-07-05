import streamlit as st
import pandas as pd
import joblib
import os

# ===========================
# PAGE CONFIG
# ===========================

st.set_page_config(
    page_title="🌾 Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)

# ===========================
# LOAD MODEL
# ===========================

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
recommendations = joblib.load("recommendations.pkl")

# ===========================
# CUSTOM CSS
# ===========================

st.markdown("""
<style>

.main{
    background-color:#F5FFF5;
}

h1{
    color:#1B5E20;
    text-align:center;
}

h2{
    color:#2E7D32;
}

.stButton>button{
    background:#2E7D32;
    color:white;
    border-radius:12px;
    height:55px;
    width:100%;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1B5E20;
}

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:#E8F5E9;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# SIDEBAR
# ===========================

st.sidebar.title("🌾 Crop Recommendation")

st.sidebar.markdown("---")

st.sidebar.info("""
This application recommends suitable crops using **Machine Learning**.

### Features

✅ Soil Analysis

✅ Weather Analysis

✅ K-Means Clustering

✅ Crop Recommendation

""")

st.sidebar.markdown("---")


# ===========================
# HEADER
# ===========================

st.title("🌾 Smart Crop Recommendation System")

st.markdown(
"""
Predict suitable crops based on:

- 🌱 Soil Nutrients
- 🌦 Weather Conditions
- 🤖 Machine Learning Clustering
"""
)

st.markdown("---")

# ===========================
# INPUTS
# ===========================

left,right = st.columns(2)

with left:

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=150.0,
        value=90.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=150.0,
        value=42.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=250.0,
        value=43.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

with right:

    temperature = st.number_input(
        "Temperature (°C)",
        value=20.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=82.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        value=200.0
    )

st.markdown("---")

# ===========================
# DASHBOARD
# ===========================

st.subheader("📊 Soil Health Dashboard")

soil = pd.DataFrame({

    "Nutrient":[
        "Nitrogen",
        "Phosphorus",
        "Potassium"
    ],

    "Value":[
        N,
        P,
        K
    ]
})

st.bar_chart(
    soil.set_index("Nutrient")
)

st.subheader("🌦 Weather Dashboard")

weather = pd.DataFrame({

    "Parameter":[
        "Temperature",
        "Humidity",
        "Rainfall"
    ],

    "Value":[
        temperature,
        humidity,
        rainfall
    ]
})

st.bar_chart(
    weather.set_index("Parameter")
)

st.markdown("---")

st.subheader("📋 Input Summary")

summary = pd.DataFrame({

    "Parameter":[
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall"
    ],

    "Value":[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]
})

st.table(summary)

st.markdown("---")

predict = st.button("🌱 Recommend Crops")
# =====================================
# PREDICTION
# =====================================

if predict:

    # Create input dataframe
    sample = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    # Scale data
    sample_scaled = scaler.transform(sample)

    # Predict Cluster
    cluster = model.predict(sample_scaled)[0]

    st.markdown("---")

    st.success(f"✅ Soil belongs to Cluster {cluster}")

    # =============================
    # Cluster Information
    # =============================

    cluster_info = {
        0: "🌱 Suitable for fertile regions with balanced nutrients.",
        1: "🌦 Suitable for warm and humid agricultural conditions.",
        2: "🌾 High rainfall agricultural region.",
        3: "☀ Dry to moderately irrigated agricultural region."
    }

    st.info(cluster_info.get(cluster, "Agricultural Cluster"))

    # =============================
    # Crop Recommendations
    # =============================

    st.subheader("🌾 Recommended Crops")

    cols = st.columns(len(recommendations[cluster]))

    for col, crop in zip(cols, recommendations[cluster]):

        with col:

            st.markdown(
                f"""
                <div style="
                background:#E8F5E9;
                padding:10px;
                border-radius:10px;
                text-align:center;
                font-weight:bold;
                color:#1B5E20;">
                {crop.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # =============================
    # Soil Health Progress
    # =============================

    st.subheader("🌱 Soil Health Overview")

    st.write("Nitrogen")
    st.progress(min(int(N), 100))

    st.write("Phosphorus")
    st.progress(min(int(P), 100))

    st.write("Potassium")
    st.progress(min(int(K / 2), 100))

    st.markdown("---")

    # =============================
    # Recommendation Explanation
    # =============================

    st.subheader("📖 Recommendation")

    st.write(
        f"""
The entered soil nutrient values and environmental conditions
are most similar to **Cluster {cluster}**.

The crops shown above are the most common crops found in this
cluster based on the K-Means clustering model.

This recommendation should be considered as a **decision-support tool**
rather than an exact prediction.
"""
    )

    st.markdown("---")

    # =============================
    # Download Report
    # =============================

    report = f"""
SMART CROP RECOMMENDATION REPORT

==================================

Predicted Cluster : {cluster}

Recommended Crops:

{', '.join(recommendations[cluster])}

----------------------------------

INPUT VALUES

Nitrogen : {N}

Phosphorus : {P}

Potassium : {K}

Temperature : {temperature}

Humidity : {humidity}

pH : {ph}

Rainfall : {rainfall}

==================================

Generated using Machine Learning
(K-Means Clustering)

"""

    st.download_button(
        label="📥 Download Recommendation Report",
        data=report,
        file_name="Crop_Recommendation_Report.txt",
        mime="text/plain"
    )

st.markdown("---")

st.caption(
    "🌾 GROW SMART | CHOOSE THE RIGHT CROP EVERYTIME"
)
