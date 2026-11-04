import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoplanet Finder", layout="centered")

st.title("🪐 NASA Exoplanet Finder")
st.markdown("Scanning Kepler Space Telescope light curves for alien worlds using a 1D CNN.")

# Cache the model and data so the app runs lightning fast
@st.cache_resource
def load_brain():
    return tf.keras.models.load_model('exoplanet_model.h5')

@st.cache_data
def load_data():
   return pd.read_csv('exoTrain.zip', compression='zip')
try:
    model = load_brain()
    df = load_data()
    
    # Sidebar UI
    st.sidebar.header("Mission Control")
    st.sidebar.markdown("Select a star system from the NASA database to analyze.")
    star_index = st.sidebar.number_input("Star ID:", min_value=0, max_value=len(df)-1, value=3)
    
    # Data prep for the selected star
    star_data = df.iloc[star_index]
    actual_label = "Planet" if star_data['LABEL'] == 2 else "No Planet"
    
    flux_values = star_data.drop('LABEL').values
    # Normalize the single row so the neural network understands it
    flux_scaled = (flux_values - np.mean(flux_values)) / np.std(flux_values)
    
    # Visualization (Cyberpunk Green on Dark Background)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.plot(flux_values, color='#00FF41', linewidth=0.7)
    ax.tick_params(colors='gray')
    ax.grid(color='gray', linestyle='--', linewidth=0.2)
    st.pyplot(fig)
    
    # Neural Network Prediction execution
    if st.button("🔍 Run Neural Network Scan"):
        # Reshape data to 3D tensor: (1 Sample, 3197 Timesteps, 1 Feature)
        input_tensor = np.expand_dims(np.expand_dims(flux_scaled, axis=0), axis=2)
        
        with st.spinner('Neural Network is analyzing the time-series...'):
            prediction = model.predict(input_tensor)[0][0]
        
        st.markdown("---")
        if prediction > 0.5:
            st.error(f"🚨 **ANOMALY DETECTED:** Exoplanet Transit Signature Found! (AI Confidence: {prediction*100:.1f}%)")
        else:
            st.success(f"✅ **ALL CLEAR:** Normal Stellar Activity. (AI Confidence: {(1-prediction)*100:.1f}%)")
            
        st.caption(f"NASA Ground Truth: {actual_label}")

except Exception as e:
    st.warning("Loading systems... please ensure 'exoplanet_model.h5' and 'exoTrain.csv' are in the folder.")