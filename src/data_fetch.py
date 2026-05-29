import os
import requests
import pandas as pd


def fetch_weather_data():
    """
    Fetch historical daily weather data for Istanbul from Open-Meteo API.
    The data is saved into the data folder as a CSV file.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 41.0082,
        "longitude": 28.9784,
        "start_date": "2021-01-01",
        "end_date": "2026-05-15",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "Europe/Istanbul"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")

    data = response.json()

    df = pd.DataFrame(data["daily"])

    os.makedirs("data", exist_ok=True)

    output_path = "data/weather_raw.csv"
    df.to_csv(output_path, index=False)

    print("Weather data downloaded successfully.")
    print(f"Saved to: {output_path}")
    print(df.head())

    return df


if __name__ == "__main__":
    fetch_weather_data()
