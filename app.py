import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set background color
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title('Laptop Price Predictor')

# Input Fields
Company = st.selectbox('Brand', df['Company'].unique())
TypeName = st.selectbox('Type', df['TypeName'].unique())  # Ensure this matches the actual column name
Ram = st.selectbox('RAM in GB', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the laptop')
Touchscreen = st.selectbox('Touch Screen', ['Yes', 'No'])
ips = st.selectbox('IPS', ['Yes', 'No'])
screen_size = st.number_input('Screen Size')
screen_resolution = st.selectbox('Screen Resolution', [
    "1366 x 768", "1920 x 1080", "2560 x 1440", "3840 x 2160",
    "5120 x 2880", "6016 x 3384", "7680 x 4320", "1280 x 800",
    "1680 x 1050", "2560 x 1600", "2560 x 1080", "3440 x 1440",
    "3840 x 2400", "1920 x 1080", "1920 x 1080", "3840 x 2160"
])
Cpu_Brand = st.selectbox('CPU', df['Cpu Brand'].unique())  # Ensure this matches the actual column name
hdd = st.selectbox('HDD in GB', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD in GB', [0, 128, 256, 512, 1024])
gpu_brand = st.selectbox('GPU', df['Gpu Brand'].unique())  # Ensure this matches the actual column name
Os = st.selectbox('OS', df['os'].unique())

# Prediction
if st.button('Predict Price'):
    # Convert categorical fields to numerical values
    Touchscreen = 1 if Touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(screen_resolution.split('x')[0])
    Y_res = int(screen_resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create the query array
    query = np.array([Company, TypeName, Ram, weight, Touchscreen, ips, ppi, Cpu_Brand, hdd, ssd, gpu_brand, Os])
    query = query.reshape(1, -1)

    # Create a DataFrame with the same columns as the model
    query_df = pd.DataFrame(query, columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi', 'Cpu Brand', 'HDD', 'SSD', 'Gpu Brand', 'os'])

    # Convert to numerical format using the pipeline
    prediction = pipe.predict(query_df)

    # Output the prediction
    st.title(f"Predicted Price: {np.exp(prediction)[0]}")
