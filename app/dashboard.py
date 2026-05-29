import os
import sys
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_fetch import fetch_weather_data
from src.preprocessing import preprocess_weather_data
from src.train_model import train_and_evaluate_models
from src.evaluate import evaluate_best_model


st.set_page_config(
    page_title="Istanbul Rain Prediction",
    layout="wide"
)

st.title("Istanbul Rain Prediction Dashboard")

st.write("""
This project uses historical weather data from Istanbul to predict whether it will rain tomorrow.
The dashboard includes data exploration, model results, and an interactive prediction tool.
""")


@st.cache_data
def load_data():
    if not os.path.exists("data/weather_raw.csv"):
        fetch_weather_data()

    if not os.path.exists("data/weather_processed.csv"):
        preprocess_weather_data()

    return pd.read_csv("data/weather_processed.csv")


df = load_data()
df["time"] = pd.to_datetime(df["time"])


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Data Overview", "Visualizations", "Model Performance", "Prediction Tool"]
)


if page == "Data Overview":
    st.header("Data Overview")

    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.subheader("Preview of processed data")
    st.dataframe(df.head())

    st.subheader("Target variable distribution")
    target_counts = df["rain_tomorrow"].value_counts().reset_index()
    target_counts.columns = ["rain_tomorrow", "count"]

    fig = px.bar(
        target_counts,
        x="rain_tomorrow",
        y="count",
        title="Distribution of Rain Tomorrow Variable"
    )
    st.plotly_chart(fig, use_container_width=True)


elif page == "Visualizations":
    st.header("Exploratory Visualizations")

    st.subheader("Daily Mean Temperature Over Time")
    fig_temp = px.line(
        df,
        x="time",
        y="temperature_2m_mean",
        title="Daily Mean Temperature in Istanbul"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader("Daily Precipitation Over Time")
    fig_rain = px.line(
        df,
        x="time",
        y="precipitation_sum",
        title="Daily Precipitation in Istanbul"
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    st.subheader("Average Precipitation by Month")
    monthly_rain = df.groupby("month")["precipitation_sum"].mean().reset_index()

    fig_month = px.bar(
        monthly_rain,
        x="month",
        y="precipitation_sum",
        title="Average Monthly Precipitation"
    )
    st.plotly_chart(fig_month, use_container_width=True)


elif page == "Model Performance":
    st.header("Model Performance")

    if not os.path.exists("models/model_validation_results.csv"):
        train_and_evaluate_models()

    if not os.path.exists("models/test_results.csv"):
        evaluate_best_model()

    validation_results = pd.read_csv("models/model_validation_results.csv")
    test_results = pd.read_csv("models/test_results.csv")

    st.subheader("Validation Results")
    st.dataframe(validation_results)

    fig_models = px.bar(
        validation_results,
        x="model",
        y="f1_score",
        title="Model Comparison by F1 Score"
    )
    st.plotly_chart(fig_models, use_container_width=True)

    st.subheader("Test Results")
    st.dataframe(test_results)

    if os.path.exists("models/confusion_matrix.csv"):
        cm = pd.read_csv("models/confusion_matrix.csv", index_col=0)
        st.subheader("Confusion Matrix")
        st.dataframe(cm)


elif page == "Prediction Tool":
    st.header("Interactive Rain Prediction Tool")

    if not os.path.exists("models/best_model.pkl"):
        train_and_evaluate_models()

    model = joblib.load("models/best_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    st.write("Enter weather values and estimate the probability of rain tomorrow.")

    col1, col2 = st.columns(2)

    with col1:
        temperature_max = st.number_input("Maximum temperature", value=20.0)
        temperature_min = st.number_input("Minimum temperature", value=12.0)
        temperature_mean = st.number_input("Mean temperature", value=16.0)
        precipitation_sum = st.number_input("Today's precipitation", value=0.0)
        wind_speed = st.number_input("Maximum wind speed", value=15.0)

    with col2:
        month = st.slider("Month", 1, 12, 5)
        day_of_year = st.slider("Day of year", 1, 366, 150)
        precipitation_yesterday = st.number_input("Yesterday's precipitation", value=0.0)
        temp_mean_yesterday = st.number_input("Yesterday's mean temperature", value=16.0)
        wind_speed_yesterday = st.number_input("Yesterday's wind speed", value=15.0)
        precipitation_3day_avg = st.number_input("3-day average precipitation", value=0.0)
        precipitation_7day_avg = st.number_input("7-day average precipitation", value=0.0)
        temperature_7day_avg = st.number_input("7-day average temperature", value=16.0)

    input_data = pd.DataFrame([{
        "temperature_2m_max": temperature_max,
        "temperature_2m_min": temperature_min,
        "temperature_2m_mean": temperature_mean,
        "precipitation_sum": precipitation_sum,
        "wind_speed_10m_max": wind_speed,
        "month": month,
        "day_of_year": day_of_year,
        "precipitation_yesterday": precipitation_yesterday,
        "temp_mean_yesterday": temp_mean_yesterday,
        "wind_speed_yesterday": wind_speed_yesterday,
        "precipitation_3day_avg": precipitation_3day_avg,
        "precipitation_7day_avg": precipitation_7day_avg,
        "temperature_7day_avg": temperature_7day_avg
    }])

    input_data = input_data[feature_columns]

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(f"The model predicts rain tomorrow. Probability: {probability:.2%}")
        else:
            st.success(f"The model predicts no rain tomorrow. Probability of rain: {probability:.2%}")
 
