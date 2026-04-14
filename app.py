import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, cos, sin, asin, sqrt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Uber ML Dashboard", layout="wide")
st.title("🚖 Uber Fare Prediction Dashboard")

# -------------------------
# HAVERSINE FUNCTION
# -------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

# -------------------------
# SESSION STATE INIT
# -------------------------
for key, default in {
    "raw_data": None,
    "clean_data": None,
    "cleaned": False,
    "results": None,
    "last_file_name": None,   # <-- track file identity
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    file = st.file_uploader("Upload CSV")

    if file:
        # Only reset state if a NEW file is uploaded
        if file.name != st.session_state.last_file_name:
            df = pd.read_csv(file).dropna()
            st.session_state.raw_data = df
            st.session_state.clean_data = None
            st.session_state.cleaned = False      # reset only on new file
            st.session_state.results = None
            st.session_state.last_file_name = file.name
            st.success(f"File uploaded: {file.name}")
        else:
            st.success(f"File loaded: {file.name}")

# -------------------------
# MAIN APP
# -------------------------
if st.session_state.raw_data is not None:

    tabs = st.tabs(["📊 EDA", "⚙ Cleaning", "📈 Feature", "🤖 Model", "📊 Performance"])

    # -------------------------
    # EDA TAB
    # -------------------------
    with tabs[0]:
        df = st.session_state.raw_data.copy()
        st.subheader("Dataset Summary")
        st.dataframe(df.describe().T, use_container_width=True)

        st.subheader("Correlation Heatmap")
        fig = px.imshow(
            df.select_dtypes(include=np.number).corr(),
            color_continuous_scale="RdBu_r",
            text_auto=".2f"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # CLEANING TAB
    # -------------------------
    with tabs[1]:
        st.subheader("Cleaning & Outlier Removal")

        df = st.session_state.raw_data.copy()

        if 'key' in df.columns:
            df = df.drop('key', axis=1)

        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        df = df.dropna(subset=['pickup_datetime'])

        df['hour']  = df['pickup_datetime'].dt.hour
        df['day']   = df['pickup_datetime'].dt.day
        df['month'] = df['pickup_datetime'].dt.month

        df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
        df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 7)]

        df['distance'] = df.apply(lambda row: haversine(
            row['pickup_longitude'], row['pickup_latitude'],
            row['dropoff_longitude'], row['dropoff_latitude']
        ), axis=1)

        st.write("Shape after basic cleaning:", df.shape)

        remove_outliers = st.checkbox("Remove Outliers (IQR)", value=True)

        if remove_outliers:
            Q1  = df['fare_amount'].quantile(0.25)
            Q3  = df['fare_amount'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[
                (df['fare_amount'] < (Q1 - 1.5 * IQR)) |
                (df['fare_amount'] > (Q3 + 1.5 * IQR))
            ]
            st.write("Outliers Detected:", len(outliers))
            df = df.drop(outliers.index)
            st.write("After Outlier Removal Shape:", df.shape)

        if st.button("✅ Apply Cleaning"):
            st.session_state.clean_data = df.copy()
            st.session_state.cleaned = True
            st.session_state.results = None      # reset old model results
            st.success("Cleaning applied! Now go to Feature or Model tab.")

        # Show current status clearly
        if st.session_state.cleaned:
            st.info(f"✔ Cleaned data ready — {st.session_state.clean_data.shape[0]} rows, {st.session_state.clean_data.shape[1]} columns")

    # -------------------------
    # FEATURE TAB
    # -------------------------
    with tabs[2]:
        if not st.session_state.cleaned:
            st.warning("⚠ Please apply cleaning first (go to ⚙ Cleaning tab and click Apply Cleaning)")
        else:
            df = st.session_state.clean_data.copy()
            st.subheader("Distance vs Fare Amount")
            fig = px.scatter(df, x='distance', y='fare_amount',
                             opacity=0.4, trendline="ols",
                             labels={"distance": "Distance (km)", "fare_amount": "Fare ($)"})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Fare Distribution by Hour")
            fig2 = px.box(df, x='hour', y='fare_amount')
            st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # MODEL TAB
    # -------------------------
    with tabs[3]:
        if not st.session_state.cleaned:
            st.warning("⚠ Please apply cleaning first (go to ⚙ Cleaning tab and click Apply Cleaning)")
        else:
            df = st.session_state.clean_data.copy()

            model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
            test_size    = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05)

            if model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 50, 300, 100, step=50)

            X = df[['passenger_count', 'hour', 'day', 'month', 'distance']]
            y = df['fare_amount']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            st.write(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

            if st.button("🚀 Train Model"):
                with st.spinner("Training..."):
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators, random_state=42, n_jobs=-1
                        )

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.session_state.results = {
                        "name":   model_choice,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "model":  model,
                        "features": X.columns.tolist()
                    }

                st.success("✅ Model trained! Check 📊 Performance tab.")

    # -------------------------
    # PERFORMANCE TAB
    # -------------------------
    with tabs[4]:
        if not st.session_state.results:
            st.warning("⚠ Train a model first (go to 🤖 Model tab)")
        else:
            res = st.session_state.results

            st.subheader(f"Results — {res['name']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE",  f"{mean_absolute_error(res['y_test'], res['y_pred']):.2f}")
            col2.metric("RMSE", f"{np.sqrt(mean_squared_error(res['y_test'], res['y_pred'])):.2f}")
            col3.metric("R²",   f"{r2_score(res['y_test'], res['y_pred']):.4f}")

            st.subheader("Actual vs Predicted")
            fig = px.scatter(
                x=res["y_test"], y=res["y_pred"],
                labels={"x": "Actual Fare ($)", "y": "Predicted Fare ($)"},
                opacity=0.4
            )
            # perfect prediction line
            mn = min(res["y_test"].min(), res["y_pred"].min())
            mx = max(res["y_test"].max(), res["y_pred"].max())
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance for RF
            if res["name"] == "Random Forest":
                st.subheader("Feature Importances")
                importances = pd.Series(
                    res["model"].feature_importances_,
                    index=res["features"]
                ).sort_values(ascending=True)
                fig2 = px.bar(importances, orientation='h',
                              labels={"value": "Importance", "index": "Feature"})
                st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("⬅ Upload a CSV file from the sidebar to get started.")