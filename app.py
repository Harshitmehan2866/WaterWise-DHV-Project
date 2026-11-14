import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

st.set_page_config(page_title="WaterWise | Water Network Analytics", page_icon="💧", layout="wide")

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: white; border: 1px solid #e8edf4; padding: 16px; border-radius: 12px;}
[data-testid="stSidebar"] {background: #eef4f8;}
</style>
""", unsafe_allow_html=True)

st.title("💧 WaterWise")
st.subheader("Data-Driven Urban Water Network Analysis")
st.caption("Explore sensor and operational records, examine fault-proximity patterns, and compare machine-learning models.")

with st.sidebar:
    st.header("Project menu")
    page = st.radio("Navigate", ["Overview", "Data Explorer", "Visual Analytics", "Model Results", "Try a Prediction"])
    st.divider()
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload the WaterWise CSV", type=["csv"])
    st.caption("Upload `input_model_potenc_predXfault7_A.csv` to activate the dashboard.")

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def prepare_data(raw):
    df = raw.copy()
    if "timestamp" not in df.columns:
        raise ValueError("The CSV must contain a 'timestamp' column.")
    # Accept the dataset's documented DD-MM-YYYY HH:00 format and common alternatives.
    parsed = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    df["timestamp"] = parsed
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates().reset_index(drop=True)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    sparse = [c for c in numeric if df[c].isna().mean() > 0.40 and c != "fault_d7"]
    df = df.drop(columns=sparse)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    return df, sparse

def get_model_data(df):
    if "fault_d7" not in df.columns:
        raise ValueError("Target column 'fault_d7' was not found in the uploaded CSV.")
    model_df = df.copy()
    dt = model_df["timestamp"]
    model_df["hour"] = dt.dt.hour
    model_df["day"] = dt.dt.day
    model_df["month"] = dt.dt.month
    model_df["day_of_week"] = dt.dt.dayofweek
    model_df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    X = model_df.select_dtypes(include=np.number).drop(columns=["fault_d7"], errors="ignore")
    y = model_df["fault_d7"].astype(int)
    valid = y.isin([0, 1])
    return X.loc[valid], y.loc[valid]

def train_models(df):
    X, y = get_model_data(df)
    if len(X) < 10 or y.nunique() < 2:
        raise ValueError("At least 10 valid rows and both target classes (0 and 1) are required.")
    split = int(len(X) * 0.80)
    split = min(max(split, 1), len(X)-1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("The chronological 80/20 split must contain both classes in train and test. Check the date range or label distribution.")
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }
    preds = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds[name] = pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced accuracy": balanced_accuracy_score(y_test, pred),
            "Precision (label 1)": precision_score(y_test, pred, zero_division=0),
            "Recall (label 1)": recall_score(y_test, pred, zero_division=0),
            "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0)
        })
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    rows.insert(0, {
        "Model": "Most-frequent baseline",
        "Accuracy": accuracy_score(y_test, base_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, base_pred),
        "Precision (label 1)": precision_score(y_test, base_pred, zero_division=0),
        "Recall (label 1)": recall_score(y_test, base_pred, zero_division=0),
        "Macro F1": f1_score(y_test, base_pred, average="macro", zero_division=0)
    })
    return X, y, X_test, y_test, models, preds, pd.DataFrame(rows).set_index("Model")

if uploaded is None:
    st.info("👈 Upload your project CSV from the sidebar to load the live dashboard.")
    a, b, c = st.columns(3)
    a.metric("Project", "WaterWise")
    b.metric("Analysis", "Sensor + fault proximity")
    c.metric("Models", "Logistic Regression & Random Forest")
    st.markdown("### What this app demonstrates")
    st.markdown("""
    - **Data preparation:** timestamp handling, duplicate removal and missing-value treatment.
    - **Exploratory analysis:** fault-proximity distribution, sensor trends and correlations.
    - **Machine learning:** chronological train/test split and comparison of two classifiers.
    - **Interactive exploration:** filter the time period and inspect sensor variables.
    """)
    st.warning("Important: `fault_d7` is a label for records within ±7 days of known fault events. It is not proof that a physical leak is present.")
    st.stop()

try:
    raw_df = load_data(uploaded.getvalue())
    df, removed_cols = prepare_data(raw_df)
    if "fault_d7" not in df.columns:
        st.error("Target column `fault_d7` is missing. Please upload the WaterWise project CSV.")
        st.stop()
    X, y, X_test, y_test, models, predictions, results = train_models(df)
except Exception as e:
    st.error(f"Could not process this file: {e}")
    st.stop()

if page == "Overview":
    st.header("Project overview")
    st.write("A compact view of the uploaded urban water-network dataset and its supplied fault-proximity labels.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Sensor / numeric features", f"{X.shape[1]}")
    c3.metric("Within proximity (1)", f"{(df['fault_d7']==1).sum():,}")
    c4.metric("Outside proximity (0)", f"{(df['fault_d7']==0).sum():,}")
    d1, d2 = st.columns(2)
    d1.metric("First timestamp", df["timestamp"].min().strftime("%d %b %Y"))
    d2.metric("Last timestamp", df["timestamp"].max().strftime("%d %b %Y"))
    st.markdown("### Target distribution")
    counts = df["fault_d7"].value_counts().sort_index().rename(index={0:"Outside proximity (0)", 1:"Within proximity (1)"}).reset_index()
    counts.columns = ["Label", "Records"]
    fig = px.pie(counts, names="Label", values="Records", hole=0.55)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Removed sparse columns (>40% missing): {', '.join(removed_cols) if removed_cols else 'None'}")

elif page == "Data Explorer":
    st.header("Data explorer")
    min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
    chosen = st.date_input("Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    view = df.copy()
    if isinstance(chosen, (tuple, list)) and len(chosen) == 2:
        view = view[view["timestamp"].dt.date.between(chosen[0], chosen[1])]
    st.write(f"Showing **{len(view):,}** records")
    st.dataframe(view, use_container_width=True, height=360)
    st.download_button("Download filtered data as CSV", view.to_csv(index=False).encode("utf-8"), "waterwise_filtered_data.csv", "text/csv")
    st.markdown("### Missing values")
    miss = raw_df.isna().sum().sort_values(ascending=False).head(15)
    miss = miss[miss > 0]
    if len(miss):
        st.plotly_chart(px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x":"Missing values","y":"Column"}), use_container_width=True)
    else:
        st.success("No missing values in the uploaded data.")

elif page == "Visual Analytics":
    st.header("Visual analytics")
    numeric_features = [c for c in df.select_dtypes(include=np.number).columns if c != "fault_d7"]
    selected = st.multiselect("Choose sensor / operational features", numeric_features, default=numeric_features[:min(4, len(numeric_features))])
    if selected:
        st.plotly_chart(px.line(df, x="timestamp", y=selected, title="Sensor and operational trends over time"), use_container_width=True)
        st.plotly_chart(px.box(df, x="fault_d7", y=selected[0], color="fault_d7", title=f"{selected[0]} by fault-proximity label"), use_container_width=True)
    st.markdown("### Correlation with target")
    corr = df.select_dtypes(include=np.number).corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation heatmap"), use_container_width=True)

elif page == "Model Results":
    st.header("Model training and evaluation")
    st.caption("Models are trained when the CSV is uploaded, using the first 80% of records for training and the latest 20% for testing.")
    st.dataframe(results.style.format("{:.3f}"), use_container_width=True)
    st.plotly_chart(px.bar(results.reset_index(), x="Model", y=["Balanced accuracy", "Macro F1"], barmode="group", range_y=[0,1], title="Model comparison"), use_container_width=True)
    model_name = st.selectbox("Confusion matrix", ["Logistic Regression", "Random Forest"])
    cm = confusion_matrix(y_test, predictions[model_name], labels=[0,1])
    fig = px.imshow(cm, text_auto=True, x=["Predicted 0","Predicted 1"], y=["Actual 0","Actual 1"],
                    color_continuous_scale="Blues", title=f"{model_name} — confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    rf = models["Random Forest"]
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).sort_values()
    st.plotly_chart(px.bar(x=importance.values, y=importance.index, orientation="h", title="Top 10 Random Forest feature importances", labels={"x":"Importance","y":"Feature"}), use_container_width=True)
    st.warning("Interpret class-wise results, not accuracy alone. A model can perform poorly on label 0 even when overall accuracy looks high.")

elif page == "Try a Prediction":
    st.header("Try a prediction")
    st.write("Choose one record from the uploaded dataset and see how the trained models classify its supplied sensor readings.")
    model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
    row_number = st.number_input("Record number", min_value=1, max_value=len(X), value=1, step=1)
    record = X.iloc[[int(row_number)-1]]
    prediction = int(models[model_choice].predict(record)[0])
    actual = int(y.iloc[int(row_number)-1])
    p1, p2 = st.columns(2)
    p1.metric("Model prediction", "Within proximity (1)" if prediction == 1 else "Outside proximity (0)")
    p2.metric("Dataset label", "Within proximity (1)" if actual == 1 else "Outside proximity (0)")
    st.caption("This is a classification of the dataset's fault-proximity label, not a real-time leak diagnosis.")
    with st.expander("View selected record's input features"):
        st.dataframe(record.T.rename(columns={int(row_number)-1:"Value"}), use_container_width=True)

st.divider()
st.caption("WaterWise | DHV Project-Based Learning | Educational analysis — not an operational water-network monitoring system.")
