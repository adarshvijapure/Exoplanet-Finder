import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoplanet Finder", layout="centered")

st.title("🪐 NASA Exoplanet Finder")
st.markdown("Scanning Kepler Space Telescope light curves for alien worlds using a 1D CNN.")

REQUIRED_FEATURES = 3197

# Cache the model and data so the app runs lightning fast
@st.cache_resource
def load_brain():
    return tf.keras.models.load_model('exoplanet_model.h5')

@st.cache_data
def load_data():
    return pd.read_csv('exoTrain.zip', compression='zip')

try:
    model = load_brain()
    df    = load_data()

    # ── Sidebar UI ────────────────────────────────────────────────────────────
    st.sidebar.header("Mission Control")

    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload a Custom Light Curve (CSV)",
        type=["csv"],
        help=(
            f"CSV must contain exactly {REQUIRED_FEATURES} numeric flux columns "
            "(one row per star). No label column required."
        ),
    )

    # ── MODE SELECTION ────────────────────────────────────────────────────────
    if uploaded_file is None:
        # ── DEFAULT MODE: NASA dataset ────────────────────────────────────────
        st.sidebar.markdown("Select a star system from the NASA database to analyze.")
        star_index = st.sidebar.number_input(
            "Star ID:", min_value=0, max_value=len(df) - 1, value=3
        )

        star_data    = df.iloc[star_index]
        actual_label = "Planet" if star_data['LABEL'] == 2 else "No Planet"
        flux_values  = star_data.drop('LABEL').values
        flux_scaled  = (flux_values - np.mean(flux_values)) / np.std(flux_values)

        source_label = f"NASA Dataset — Star ID {star_index}"
        show_ground_truth = True

    else:
        # ── UPLOAD MODE: custom CSV ───────────────────────────────────────────
        st.sidebar.success("✅ Custom file received. NASA slider is bypassed.")

        try:
            custom_df = pd.read_csv(uploaded_file, header=None)
        except Exception:
            st.error("❌ Could not parse the uploaded file. Please upload a valid CSV.")
            st.stop()

        num_cols = custom_df.shape[1]

        if num_cols != REQUIRED_FEATURES:
            st.error(
                f"❌ **Shape mismatch:** Your CSV has **{num_cols} columns**, "
                f"but the model requires exactly **{REQUIRED_FEATURES} columns** "
                f"(one per flux timestep). Please reformat your data to match the "
                f"required timeline length of {REQUIRED_FEATURES} features and re-upload."
            )
            st.stop()

        # Use the first row if multiple rows are present
        flux_values  = custom_df.iloc[0].values.astype(float)
        flux_scaled  = (flux_values - np.mean(flux_values)) / np.std(flux_values)

        actual_label      = None
        show_ground_truth = False
        source_label      = f"Custom Upload — {uploaded_file.name} (row 1)"

        if custom_df.shape[0] > 1:
            st.info(
                f"ℹ️ Your file contains **{custom_df.shape[0]} rows**. "
                "Only the **first row** is being analyzed."
            )

    # ── LIGHT CURVE VISUALIZATION ─────────────────────────────────────────────
    st.markdown(f"**Source:** {source_label}")

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.plot(flux_values, color='#00FF41', linewidth=0.7)
    ax.set_xlabel("Timestep", color='gray', fontsize=9)
    ax.set_ylabel("Flux", color='gray', fontsize=9)
    ax.tick_params(colors='gray')
    ax.grid(color='gray', linestyle='--', linewidth=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    st.pyplot(fig)

    # ── NEURAL NETWORK PREDICTION ─────────────────────────────────────────────
    if st.button("🔍 Run Neural Network Scan"):
        # Reshape to 3D tensor: (1 Sample, 3197 Timesteps, 1 Feature)
        input_tensor = np.expand_dims(np.expand_dims(flux_scaled, axis=0), axis=2)

        with st.spinner('Neural Network is analyzing the time-series...'):
            prediction = model.predict(input_tensor)[0][0]

        st.markdown("---")
        if prediction > 0.5:
            st.error(
                f"🚨 **ANOMALY DETECTED:** Exoplanet Transit Signature Found! "
                f"(AI Confidence: {prediction * 100:.1f}%)"
            )
        else:
            st.success(
                f"✅ **ALL CLEAR:** Normal Stellar Activity. "
                f"(AI Confidence: {(1 - prediction) * 100:.1f}%)"
            )

        if show_ground_truth:
            st.caption(f"NASA Ground Truth: {actual_label}")
        else:
            st.caption("ℹ️ No ground-truth label available for custom uploads.")

except Exception as e:
    st.warning(
        "⚠️ Loading systems... please ensure **'exoplanet_model.h5'** and "
        "**'exoTrain.zip'** are in the same folder as app.py."
    )
    st.exception(e)
