# src/feature_engineering.py

"""
Feature definitions for Istanbul rain prediction model.

Feature selection rationale:
    Features are grouped into 5 categories based on meteorological relevance
    and their relationship to next-day precipitation in Istanbul's climate.

    Istanbul has a Mediterranean-influenced climate with wet winters (Oct-Mar)
    and dry summers (Jun-Aug). The Bosphorus strait creates local humidity effects.
    These domain characteristics are encoded in 'season', 'is_rainy_season', and
    humidity-based features.
"""


# ---------------------------------------------------------------------------
# Feature column list — single source of truth
# Used by train_model.py, evaluate.py, and dashboard.py
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [

    # --- 1. Current day atmospheric measurements ---
    # Direct observations that influence next-day precipitation
    "temperature_2m_max",           # Max temperature (°C)
    "temperature_2m_min",           # Min temperature (°C)
    "temperature_2m_mean",          # Mean temperature (°C)
    "precipitation_sum",            # Today's total precipitation (mm)
    "wind_speed_10m_max",           # Max wind speed (km/h)
    "relative_humidity_2m_mean",    # Mean relative humidity (%)
    "cloud_cover_mean",             # Mean cloud cover (%)
    "pressure_msl_mean",            # Mean sea-level pressure (hPa)

    # --- 2. Derived atmospheric features ---
    # Engineered from raw measurements to capture instability signals
    "temp_range",                   # Max - Min temperature: high range = atmospheric instability

    # --- 3. Calendar features ---
    # Encode seasonal precipitation patterns
    "month",                        # Month (1-12): captures seasonal cycle
    "day_of_year",                  # Day of year (1-366): smooth seasonal signal
    "season",                       # Meteorological season (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
    "is_rainy_season",              # 1 if Oct-Mar (Istanbul wet season), 0 otherwise

    # --- 4. Lag features (yesterday's values) ---
    # Weather is autocorrelated: today's conditions predict tomorrow's
    "precipitation_yesterday",      # Yesterday's precipitation (mm)
    "temp_mean_yesterday",          # Yesterday's mean temperature (°C)
    "wind_speed_yesterday",         # Yesterday's max wind speed (km/h)
    "humidity_yesterday",           # Yesterday's mean humidity (%)
    "pressure_yesterday",           # Yesterday's mean pressure (hPa): falling pressure = rain signal

    # --- 5. Rolling averages (short-term trends) ---
    # Capture persistence of weather systems passing through Istanbul
    "precipitation_3day_avg",       # 3-day avg precipitation: recent wet spell
    "precipitation_7day_avg",       # 7-day avg precipitation: longer wet period
    "temperature_7day_avg",         # 7-day avg temperature: seasonal baseline
    "humidity_7day_avg",            # 7-day avg humidity: moisture in atmosphere
    "pressure_7day_avg",            # 7-day avg pressure: synoptic trend
]


# ---------------------------------------------------------------------------
# Feature descriptions — used in dashboard tooltips and README
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "temperature_2m_max":           "Maximum daily temperature at 2m height (°C)",
    "temperature_2m_min":           "Minimum daily temperature at 2m height (°C)",
    "temperature_2m_mean":          "Mean daily temperature at 2m height (°C)",
    "precipitation_sum":            "Total precipitation today (mm) — direct rain signal",
    "wind_speed_10m_max":           "Maximum wind speed at 10m height (km/h)",
    "relative_humidity_2m_mean":    "Mean relative humidity (%) — high humidity precedes rain",
    "cloud_cover_mean":             "Mean cloud cover (%) — correlated with precipitation probability",
    "pressure_msl_mean":            "Mean sea-level pressure (hPa) — falling pressure signals rain",
    "temp_range":                   "Diurnal temperature range (°C) — proxy for atmospheric instability",
    "month":                        "Calendar month (1–12)",
    "day_of_year":                  "Day of year (1–366)",
    "season":                       "Meteorological season: 1=Winter, 2=Spring, 3=Summer, 4=Autumn",
    "is_rainy_season":              "1 if October–March (Istanbul wet season), 0 otherwise",
    "precipitation_yesterday":      "Yesterday's precipitation (mm)",
    "temp_mean_yesterday":          "Yesterday's mean temperature (°C)",
    "wind_speed_yesterday":         "Yesterday's max wind speed (km/h)",
    "humidity_yesterday":           "Yesterday's mean relative humidity (%)",
    "pressure_yesterday":           "Yesterday's mean sea-level pressure (hPa)",
    "precipitation_3day_avg":       "3-day rolling average precipitation (mm)",
    "precipitation_7day_avg":       "7-day rolling average precipitation (mm)",
    "temperature_7day_avg":         "7-day rolling average temperature (°C)",
    "humidity_7day_avg":            "7-day rolling average humidity (%)",
    "pressure_7day_avg":            "7-day rolling average pressure (hPa)",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_columns():
    """
    Returns the list of feature columns used for model training and inference.
    Only columns present in the processed dataframe will be used at runtime
    (train_model.py filters for available columns).
    """
    return FEATURE_COLUMNS.copy()


def get_feature_description(feature_name):
    """
    Returns a human-readable description for a given feature name.
    Used in dashboard tooltips and documentation.
    """
    return FEATURE_DESCRIPTIONS.get(feature_name, feature_name)


def get_all_descriptions():
    """
    Returns the full description dictionary.
    """
    return FEATURE_DESCRIPTIONS.copy()


if __name__ == "__main__":
    print(f"Toplam feature sayısı: {len(FEATURE_COLUMNS)}")
    print("\nFeature listesi ve açıklamaları:")
    for col in FEATURE_COLUMNS:
        print(f"  {col:<35} {FEATURE_DESCRIPTIONS.get(col, '')}")