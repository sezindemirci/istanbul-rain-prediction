import pandas as pd


def preprocess_weather_data(input_path="data/weather_raw.csv",
                            output_path="data/weather_processed.csv"):
    """
    Prepare raw Istanbul weather data for prediction modeling.
    Creates target variable, time-based features, lag and rolling features.

    Target definition: rain_tomorrow = 1 if next day precipitation >= 1mm.
    Threshold of 1mm is the standard meteorological definition of a rainy day.
    """

    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)  # zaman sırası garantisi

    # --- Target variable ---
    # Yarının yağışı >= 1mm ise 1, değilse 0
    # 1mm eşiği: meteorolojide anlamlı yağış sınırı (çiy/nem filtreler)
    df["rain_tomorrow"] = (df["precipitation_sum"].shift(-1) >= 1.0).astype(int)

    # --- Tarih özellikleri ---
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    # Meteorolojik mevsim (1=Kış, 2=İlkbahar, 3=Yaz, 4=Sonbahar)
    df["season"] = df["month"].map({
        12: 1, 1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3,
        9: 4, 10: 4, 11: 4
    })

    # İstanbul yağışlı mevsim: Ekim-Mart arası yağış oranı belirgin şekilde yüksek
    df["is_rainy_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)

    # Gün içi sıcaklık farkı: atmosferik dengesizlik göstergesi
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # --- Lag features: dünün değerleri ---
    # Hava durumu ardışık günler arasında güçlü korelasyon gösterir
    df["precipitation_yesterday"] = df["precipitation_sum"].shift(1)
    df["temp_mean_yesterday"] = df["temperature_2m_mean"].shift(1)
    df["wind_speed_yesterday"] = df["wind_speed_10m_max"].shift(1)
    df["humidity_yesterday"] = df["relative_humidity_2m_mean"].shift(1)
    df["pressure_yesterday"] = df["pressure_msl_mean"].shift(1)

    # --- Rolling features: kısa dönem trend ---
    # Nemli hava kütlelerinin geçişini yakalamak için
    df["precipitation_3day_avg"] = df["precipitation_sum"].rolling(window=3).mean()
    df["precipitation_7day_avg"] = df["precipitation_sum"].rolling(window=7).mean()
    df["temperature_7day_avg"] = df["temperature_2m_mean"].rolling(window=7).mean()
    df["humidity_7day_avg"] = df["relative_humidity_2m_mean"].rolling(window=7).mean()
    df["pressure_7day_avg"] = df["pressure_msl_mean"].rolling(window=7).mean()

    # --- Eksik değer raporu (dropna öncesi) ---
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("Eksik değerler (dropna öncesi):")
        print(missing)
    else:
        print("Eksik değer yok.")

    # Shift ve rolling işlemlerinden kaynaklanan NaN'ları at
    df = df.dropna()
    df = df.reset_index(drop=True)

    # --- Class balance raporu ---
    counts = df["rain_tomorrow"].value_counts()
    total = len(df)
    print(f"\nClass dağılımı:")
    print(f"  Yağmur yok (0): {counts[0]} ({counts[0]/total:.1%})")
    print(f"  Yağmur var (1): {counts[1]} ({counts[1]/total:.1%})")

    df.to_csv(output_path, index=False)

    print(f"\nPreprocessing tamamlandı.")
    print(f"Toplam satır: {total}")
    print(f"Kaydedildi: {output_path}")

    return df


if __name__ == "__main__":
    preprocess_weather_data()