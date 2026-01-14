import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="AQI Prediction Demo",
    layout="centered"
)

st.title("Dự đoán mức độ ô nhiễm không khí (AQI)")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("hanoi-aqi-weather-data.csv")

df = load_data()

# ======================
# CREATE AQI CATEGORY (FROM RAW AQI)
# ======================
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

df["AQI_Category"] = df["AQI"].apply(classify_aqi)

# ======================
# PREPROCESSING
# ======================
drop_cols = [
    "Local Time", "UTC Time", "City",
    "Country Code", "Timezone",
    "AQI", "PM25", "PM10"
]

df = df.drop(columns=drop_cols, errors="ignore")
df = df.dropna()

# Encode label
le = LabelEncoder()
df["AQI_Category"] = le.fit_transform(df["AQI_Category"])

X = df.drop("AQI_Category", axis=1)
y = df["AQI_Category"]

# ======================
# TRAIN MODEL
# ======================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model

rf = train_model(X, y)

# ======================
# USER INPUT
# ======================
st.subheader("📥 Nhập thông số môi trường")

cols = st.columns(2)
input_data = {}

for i, col in enumerate(X.columns):
    with cols[i % 2]:
        input_data[col] = st.number_input(
            col,
            value=float(X[col].mean())
        )

input_df = pd.DataFrame([input_data])


# ======================
# PREDICTION
# ======================
if st.button("Dự đoán AQI"):
    pred = rf.predict(input_df)[0]
    label = le.inverse_transform([pred])[0]

    st.success(f"Mức AQI dự đoán: **{label}**")
