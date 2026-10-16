import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Sayfa ayarları
st.set_page_config(
    page_title="Vehicle Energy Predictor",
    page_icon="🚗",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "vehicle_energy_pipeline.joblib")

try:
    model = load_model()
except FileNotFoundError:
    st.error("vehicle_energy_pipeline.joblib dosyasını app.py ile aynı klasöre koy.")
    st.stop()

st.title("🚗 Vehicle Energy Predictor")
st.write(
    "Vehicle telemetry data kullanarak enerji tüketimi tahmini yapan "
    "makine öğrenmesi uygulaması."
)

st.caption(
    "Yerel araştırma prototipi. Hedefin fiziksel birimi ve hesaplanma "
    "yöntemi doğrulanmamıştır. Sonuçlar litre veya kWh olarak yorumlanamaz. "
    "Elle girilen koşullar eğitim verisinin dışında olabilir."
)
st.divider()

st.subheader("Vehicle Telemetry")

col1, col2 = st.columns(2)

with col1:
    speed = st.number_input(
        "Vehicle Speed (km/h)",
        min_value=0.0,
        max_value=200.0,
        value=50.0
    )

    maf = st.number_input(
        "MAF (g/sec)",
        min_value=0.0,
        max_value=200.0,
        value=8.0
    )

    rpm = st.number_input(
        "Engine RPM",
        min_value=0.0,
        max_value=7000.0,
        value=1200.0
    )

    load = st.number_input(
        "Absolute Load (%)",
        min_value=0.0,
        max_value=150.0,
        value=25.0
    )

with col2:
    oat = st.number_input(
        "Outside Air Temperature (°C)",
        min_value=-50.0,
        max_value=70.0,
        value=15.0
    )

    elevation = st.number_input(
        "Elevation (m)",
        min_value=0.0,
        max_value=1000.0,
        value=260.0
    )

    gradient = st.number_input(
        "Gradient",
        min_value=-1.0,
        max_value=1.0,
        value=0.01
    )

    speed_limit = st.number_input(
        "Speed Limit (km/h)",
        min_value=0.0,
        max_value=150.0,
        value=56.0
    )

if st.button("🔮 Predict Energy Consumption"):

    input_data = pd.DataFrame([{
        "Vehicle Speed[km/h]": speed,
        "MAF[g/sec]": maf,
        "Engine RPM[RPM]": rpm,
        "Absolute Load[%]": load,
        "OAT[DegC]": oat,
        "Elevation Smoothed[m]": elevation,
        "Gradient": gradient,
        "Speed Limit[km/h]": speed_limit
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        "Predicted Energy Consumption",
        f"{prediction:.6f}"
    )

    st.caption(
        "Prediction is expressed in the scale of the dataset's "
        "Energy_Consumption target."
    )

# Yolculuk inceleme bölümü
st.divider()
st.header("📊 Yolculuk: Gerçek ve Tahmini Tüketim")

trip_path = Path(__file__).resolve().parent / "trip_review.csv"

if trip_path.exists():
    trip_data = pd.read_csv(trip_path)
    trip_data = trip_data.sort_values("Elapsed_Seconds")
    # CSV'deki eski tahminlere bağlı kalmadan yüklü modelle hesapla
    trip_data["Predicted_Energy"] = model.predict(
        trip_data[list(model.feature_names_in_)]
    )
    trip_data["Absolute_Error"] = (
        trip_data["Energy_Consumption"] - trip_data["Predicted_Energy"]
    ).abs()

    st.caption(
        "Eğitimde bulunmayan bir araçtan, hedef değeri mevcut "
        "yolculuk kayıtları gösterilir. Değerler veri setinin "
        "ölçeğindedir; fiziksel birim doğrulanmamıştır."
    )

    c1, c2 = st.columns(2)

    c1.metric("Kayıt sayısı", len(trip_data))
    c2.metric(
        "Ortalama mutlak hata",
        f"{trip_data['Absolute_Error'].mean():.6f}"
    )

    chart_data = (
        trip_data[
            ["Elapsed_Seconds", "Energy_Consumption", "Predicted_Energy"]
        ]
        .rename(columns={
            "Energy_Consumption": "Gerçek değer",
            "Predicted_Energy": "Model tahmini"
        })
        .set_index("Elapsed_Seconds")
    )

    st.line_chart(chart_data)
    st.caption(
        "Yatay eksen: ilk mevcut kayıttan itibaren saniye. "
        "Çizgiler mevcut noktaları birleştirir; aradaki "
        "zamanlarda ölçüm bulunduğu anlamına gelmez."
    )

    st.subheader("Tahmin farkının en yüksek olduğu 10 kayıt")

    st.dataframe(
        trip_data.nlargest(10, "Absolute_Error")[
            [
                "Elapsed_Seconds",
                "Vehicle Speed[km/h]",
                "MAF[g/sec]",
                "Energy_Consumption",
                "Predicted_Energy",
                "Absolute_Error"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Yüksek farklar inceleme adayıdır; tek başına arıza "
        "veya verimsizlik kanıtı değildir. Bu yolculuk, "
        "test örneklemindeki kayıt sayısına göre seçilmiştir."
    )

else:
    st.info(
        "Yolculuk grafiği için trip_review.csv dosyasını "
        "app.py ile aynı klasöre koy."
    )