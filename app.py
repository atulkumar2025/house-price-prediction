import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- 1. PAGE CONFIG & BACKGROUND STYLE ---
st.set_page_config(page_title="California Real Estate AI", page_icon="🏘️", layout="wide")

st.markdown("""
<style>
    /* 1. Background image and dark overlay */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('https://www.classicist.org/assets/images/articles/_pagebody/CalifornianVillas07.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* 2. FORCE titles, headers, and standard text to be white */
    h1, h2, h3, p {
        color: #ffffff !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.8); /* Adds a drop shadow for extra readability */
    }
    
    /* 3. FORCE the labels above sliders and input boxes to be bright white */
    [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* 4. Style the main Calculate button */
    .stButton > button {
        background-color: #2e7d32 !important; 
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        width: 100% !important;
        border: none !important;
        padding: 15px !important;
    }
    .stButton > button:hover {
        background-color: #1b5e20 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL (WITH CACHING) ---
@st.cache_resource
def load_model():
    model = joblib.load('house_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# --- 3. HEADER SECTION ---
st.title("🏘️ California Real Estate AI")
st.write("Adjust the property details below to get an instant, machine-learning-powered price estimate.")
st.divider() 

# --- 4. MAIN LAYOUT (INPUTS) ---
# We use standard Streamlit columns without the broken HTML wrappers
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📍 Demographics & Location")
    MedInc = st.slider("Median Income (in $10,000s)", 0.0, 15.0, 8.5)
    Population = st.slider("Population in Block", 10, 5000, 1500)
    AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 2.8)
    
    st.write("")
    st.write("**Coordinates**")
    Latitude = st.number_input("Latitude", 32.0, 42.0, 34.1)
    Longitude = st.number_input("Longitude", -125.0, -114.0, -118.2)

with col2:
    st.subheader("🏠 Property Features")
    HouseAge = st.slider("House Age (years)", 0.0, 50.0, 50.0)
    AveRooms = st.slider("Average Rooms", 1.0, 15.0, 7.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 2.0)
    
    st.write("") 
    st.write("")
    st.write("")
    st.write("")
    
    predict_button = st.button("Calculate Property Value 🚀", key="predict_btn")

# --- 5. RESULTS & ANALYSIS LAYOUT ---
if predict_button:
    # Scale user data and predict
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]], 
                              columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    estimated_price = prediction[0] * 100000
    
    # 1. Show Estimate and Map
    st.divider() 
    res_col1, res_col2 = st.columns([1, 2], gap="large") 

    with res_col1:
        st.subheader("📊 Estimate")
        st.metric(label="Predicted Market Value", value=f"${estimated_price:,.2f}")

    with res_col2:
        st.subheader("🗺️ Property Location")
        map_data = pd.DataFrame({'lat': [Latitude], 'lon': [Longitude]})
        st.map(map_data, zoom=6, use_container_width=True)

    # 2. Show Price Trend Chart (SENSITIVITY ANALYSIS)
    st.write("---")
    st.subheader("📈 Price Trend Analysis")
    st.write("See how the estimated value changes as Median Income increases, keeping all other features constant.")

    # Create a range of possible incomes (from $0 to $150k)
    income_range = np.linspace(0, 15, 50)
    
    # Create a "scenario" dataframe where ONLY the income changes
    trend_data = pd.DataFrame([
        [i, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude] 
        for i in income_range
    ], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

    # Scale and Predict for the whole range
    trend_scaled = scaler.transform(trend_data)
    trend_preds = model.predict(trend_scaled) * 100000

    # Create the Chart using Plotly
    fig_df = pd.DataFrame({'Income': income_range * 10000, 'Predicted Price': trend_preds})
    fig = px.line(fig_df, x='Income', y='Predicted Price', 
                  labels={'Income': 'Neighborhood Median Income ($)', 'Predicted Price': 'Estimated House Value ($)'},
                  template="plotly_dark")
    
    fig.update_traces(line_color='#2e7d32', line_width=4)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)')

    st.plotly_chart(fig, use_container_width=True)