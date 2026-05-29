import pandas as pd


def preprocess_weather_data(input_path="data/weather_raw.csv",
                            output_path="data/weather_processed.csv"):
    """
    Prepare raw Istanbul weather data for prediction modeling.
    Creates target variable and basic time-based features.
    """

    df = pd.read_csv(input_path)

    # Convert date column
    df["time"] = pd.to_datetime(df["time"])

    # Create target variable: will it rain tomorrow?
    df["rain_tomorrow"] = (df["precipitation_sum"].shift(-1) > 0).astype(int)

    # Date-based features
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    # Lag features: yesterday's weather
    df["precipitation_yesterday"] = df["precipitation_sum"].shift(1)
    df["temp_mean_yesterday"] = df["temperature_2m_mean"].shift(1)
    df["wind_speed_yesterday"] = df["wind_speed_10m_max"].shift(1)

    # Rolling features
    df["precipitation_3day_avg"] = df["precipitation_sum"].rolling(window=3).mean()
    df["precipitation_7day_avg"] = df["precipitation_sum"].rolling(window=7).mean()
    df["temperature_7day_avg"] = df["temperature_2m_mean"].rolling(window=7).mean()

    # Remove rows with missing values caused by shift/rolling
    df = df.dropna()

    df.to_csv(output_path, index=False)

    print("Preprocessing completed.")
    print(f"Saved to: {output_path}")
    print(df.head())
    print(df["rain_tomorrow"].value_counts())

    return df


if __name__ == "__main__":
    preprocess_weather_data()
