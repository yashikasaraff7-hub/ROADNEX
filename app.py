
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# ============================================================
# ROADNEX AI — Clickable, auto-loaded, full Streamlit dashboard
# Follows the supplied Colab notebook's data and ML workflow.
# ============================================================

st.set_page_config(
    page_title="ROADNEX AI | Mobility Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root{--cyan:#58f0df;--blue:#4e8cff;--ink:#07101c;--muted:#91a7bd;}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.stApp{
 background:radial-gradient(ellipse at 0% 0%,rgba(28,205,190,.14),transparent 30%),
 radial-gradient(ellipse at 100% 10%,rgba(62,105,255,.16),transparent 30%),
 linear-gradient(145deg,#050c16,#09182a 58%,#07101c);
 color:#edf7ff;
}
.block-container{max-width:1540px;padding-top:1.1rem;padding-bottom:3rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1b2d,#07111e);border-right:1px solid rgba(88,240,223,.18);}
[data-testid="stSidebar"] *{color:#e8f5ff;}
.hero{border:1px solid rgba(88,240,223,.30);border-radius:25px;padding:30px 34px;background:linear-gradient(112deg,rgba(8,48,57,.97),rgba(12,27,53,.98) 75%);box-shadow:0 22px 65px rgba(0,0,0,.22);position:relative;overflow:hidden;}
.kicker{font:10px 'DM Mono';letter-spacing:3px;color:#58f0df;}
.brand{font-size:clamp(42px,5vw,66px);font-weight:700;letter-spacing:-3px;line-height:1;margin:12px 0;color:#f5fbff;}
.brand span{color:#58f0df;text-shadow:0 0 25px rgba(88,240,223,.35);}
.hero-sub{font-size:19px;font-weight:700;letter-spacing:2px;color:#dceaf6;margin-bottom:10px;}
.hero-copy{color:#a9bfd1;max-width:900px;font-size:14px;line-height:1.7;}
.hero-pill{display:inline-block;margin-top:16px;border:1px solid rgba(88,240,223,.32);border-radius:30px;padding:7px 12px;color:#baf8ef;font:10px 'DM Mono';letter-spacing:1px;}
.metric{border:1px solid rgba(125,177,210,.19);border-radius:17px;padding:16px 18px;background:linear-gradient(145deg,rgba(17,39,59,.96),rgba(9,24,40,.96));min-height:105px;}
.metric-label{font:10px 'DM Mono';letter-spacing:1.4px;color:#91aabe;}
.metric-value{font-size:25px;font-weight:700;color:#f4fbff;margin-top:8px;line-height:1.15;overflow-wrap:anywhere;}
.metric-foot{font-size:10px;color:#58dccc;margin-top:6px;}
.panel{background:linear-gradient(145deg,rgba(13,31,49,.95),rgba(8,21,36,.95));border:1px solid rgba(125,177,210,.18);border-radius:19px;padding:19px 21px;margin-bottom:14px;}
.panel-title{font-size:18px;font-weight:700;color:#f0f8ff;margin-bottom:3px;}
.panel-copy{font-size:12px;color:#91aabe;line-height:1.6;}
.section-kicker{font:10px 'DM Mono';letter-spacing:2px;color:#58f0df;margin:12px 0 7px;}
div.stButton>button,div.stFormSubmitButton>button{background:linear-gradient(95deg,#58f0df,#4e8cff);color:#06121e;border:0;border-radius:12px;font-weight:700;min-height:44px;}
[data-testid="stRadio"] label{padding:9px 10px;border-radius:10px;margin-bottom:3px;}
[data-testid="stRadio"] label:hover{background:rgba(88,240,223,.10);}
[data-testid="stRadio"] div[role="radiogroup"]{gap:2px;}
div[data-testid="stAlert"]{border-radius:12px;}
[data-testid="stDataFrame"]{border:1px solid rgba(125,177,210,.16);border-radius:12px;overflow:hidden;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
TARGET_CANDIDATES = [
    "accident_severity", "accident_severity_c", "severity",
    "crash_severity", "severity_level"
]
CATEGORY_CANDIDATES = [
    "weather_conditions", "road_surface_conditions", "light_conditions",
    "types_of_junction", "type_of_vehicle", "day_of_week", "road_surface_type"
]
ID_KEYWORDS = ["id", "reference", "case_number"]
OUTCOME_KEYWORDS = [
    "casualty", "casualties", "fatality", "fatalities",
    "number_of_injured", "number_of_deaths", "injury",
    "hospital", "death", "accident_severity"
]

def metric(label, value, foot):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True
    )

def panel(title, copy):
    st.markdown(
        f'<div class="panel"><div class="panel-title">{title}</div>'
        f'<div class="panel-copy">{copy}</div></div>',
        unsafe_allow_html=True
    )

def style_chart(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=12, r=16, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dceaf5", family="Space Grotesk"),
        title_font=dict(color="#f2f8ff", size=16),
        legend_title_font_color="#b9ccdc"
    )
    fig.update_xaxes(gridcolor="rgba(150,185,210,.12)")
    fig.update_yaxes(gridcolor="rgba(150,185,210,.12)")
    return fig

def find_dataset():
    # No uploader is displayed. Put the dataset beside app.py.
    preferred = [
        "RTA Dataset.csv", "rta_dataset.csv", "RTA_Dataset.csv",
        "accident_dataset.csv", "accident_data.csv",
        "cleaned.csv", "cleaned_dataset.csv", "dataset.csv"
    ]
    search_dirs = [APP_DIR, APP_DIR / "data"]
    for folder in search_dirs:
        if folder.exists():
            for name in preferred:
                candidate = folder / name
                if candidate.exists():
                    return candidate
    for folder in search_dirs:
        if folder.exists():
            candidates = sorted(folder.glob("*.csv"))
            if candidates:
                return candidates[0]
    return None

@st.cache_data(show_spinner=False)
def load_dataset(path_string, modified_time):
    return pd.read_csv(path_string)

def prepare_dataset(raw):
    df = raw.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace(["", "?", "NA", "N/A", "nan", "None"], np.nan)

    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        target = next((c for c in df.columns if "severity" in c), None)
    if target is None:
        raise ValueError(
            "The severity target column was not found. Expected a column such as "
            "'Accident_severity'. Check the CSV column names."
        )

    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(str).str.strip()

    drop_cols = []
    for col in df.columns:
        if col == target:
            continue
        if any(key in col for key in ID_KEYWORDS + OUTCOME_KEYWORDS):
            drop_cols.append(col)

    X = df.drop(columns=[target] + drop_cols, errors="ignore").copy()
    y = df[target].copy()

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols, errors="ignore")
    all_missing_cols = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing_cols, errors="ignore")

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(str)

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lower, upper=upper)

    return df, target, X, y, duplicates, drop_cols, constant_cols, all_missing_cols

def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=12
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced",
            max_depth=18, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": LinearSVC(
            class_weight="balanced", random_state=42, max_iter=5000
        )
    }

@st.cache_resource(show_spinner=False)
def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric),
        ("cat", categorical_transformer, categorical)
    ])

    models = make_models()
    trained, predictions, rows = {}, {}, []
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        trained[name] = pipe
        predictions[name] = pred
        rows.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_test, pred, average="macro", zero_division=0)
        })

    results = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return {
        "trained": trained, "predictions": predictions, "results": results,
        "best_name": results.iloc[0]["Algorithm"],
        "X_test": X_test, "y_test": y_test,
        "numeric_features": numeric, "categorical_features": categorical
    }

# ----------------------- SIDEBAR NAVIGATION -----------------------
with st.sidebar:
    st.markdown("## 🛣️ ROADNEX")
    st.markdown("**AI MOBILITY INTELLIGENCE**")
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    st.success("Dashboard interface online")
    st.caption("Your dataset is loaded automatically from the project folder.")
    st.markdown("---")
    st.markdown("### PROJECT MODULES")
    page = st.radio(
        "Select a module",
        [
            "🏠 Mission Control",
            "⚡ Severity Prediction",
            "📊 Exploratory Data Analysis",
            "🧠 Model Comparison",
            "▦ Confusion Matrix & Report",
            "◎ Feature Importance",
            "🗂️ Data Explorer",
            "ℹ️ Project Profile"
        ],
        label_visibility="collapsed",
        key="roadnex_navigation"
    )
    st.markdown("---")
    st.caption("Select any module above to open that working page.")
    st.caption("Academic machine-learning decision-support prototype")

# ----------------------- HERO HEADER -----------------------
st.markdown("""
<div class="hero">
 <div class="kicker">SMART MOBILITY / MACHINE LEARNING / ROAD SAFETY</div>
 <div class="brand">ROAD<span>NEX</span></div>
 <div class="hero-sub">AI MOBILITY INTELLIGENCE</div>
 <div class="hero-copy">Predict Smarter. Travel Safer. Explore historical road-accident patterns,
 compare six machine-learning algorithms, and classify accident severity through an interactive
 prediction laboratory.</div>
 <div class="hero-pill">TRANSPORTATION ACCIDENT SEVERITY PREDICTION &nbsp; • &nbsp; INTERACTIVE ML PLATFORM</div>
</div>
""", unsafe_allow_html=True)

# ----------------------- AUTO DATASET LOAD -----------------------
dataset_path = find_dataset()
if dataset_path is None:
    st.error("ROADNEX cannot find its dataset file.")
    st.markdown("""
    **One-time setup (the CSV upload control has been removed):**
    1. Put your `RTA Dataset.csv` file in the same folder as `app.py`.
    2. Keep the filename as `RTA Dataset.csv` (or update the filename in `find_dataset()`).
    3. Restart Streamlit.

    Your dataset is loaded automatically in the background. No upload button will appear in the UI.
    """)
    st.stop()

try:
    raw = load_dataset(str(dataset_path), dataset_path.stat().st_mtime)
    df, target, X, y, duplicates, dropped_cols, constant_cols, all_missing_cols = prepare_dataset(raw)
except Exception as exc:
    st.error(f"Dataset loading / preparation error: {exc}")
    st.stop()

if X.shape[1] == 0:
    st.error("No usable input features remain after feature selection.")
    st.stop()
if y.value_counts().min() < 2:
    st.error("Each severity class needs at least two records for stratified train/test splitting.")
    st.stop()

# ----------------------- AUTO TRAIN -----------------------
with st.spinner("ROADNEX AI is preparing the data and training six ML algorithms..."):
    try:
        model_pack = train_all_models(X, y)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

results = model_pack["results"]
best_row = results.iloc[0]
best_model = model_pack["trained"][model_pack["best_name"]]
best_predictions = model_pack["predictions"][model_pack["best_name"]]
y_test = model_pack["y_test"]

# ----------------------- GLOBAL STATUS STRIP -----------------------
a, b, c, d = st.columns(4)
with a: metric("DATA RECORDS", f"{len(df):,}", "CLEANED DATASET")
with b: metric("INPUT FEATURES", f"{X.shape[1]}", "MODEL INPUTS")
with c: metric("BEST ALGORITHM", model_pack["best_name"], "HIGHEST MACRO F1")
with d: metric("TEST ACCURACY", f"{best_row['Accuracy']*100:.2f}%", "HELD-OUT TEST SET")

st.markdown(f'<div class="section-kicker">ACTIVE MODULE / {page.upper()}</div>', unsafe_allow_html=True)

# ======================= MISSION CONTROL =======================
if page == "🏠 Mission Control":
    panel("Mission control", "Live dashboard summary generated from your local accident dataset and trained machine-learning models.")
    col1, col2 = st.columns(2)
    with col1:
        sev = y.value_counts().rename_axis("Accident severity").reset_index(name="Records")
        fig = px.pie(sev, names="Accident severity", values="Records", hole=.58,
                     title="Accident severity distribution")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_chart(fig, 370), use_container_width=True)
    with col2:
        fig = px.bar(sev, x="Accident severity", y="Records", color="Accident severity",
                     text="Records", title="Accident records by severity")
        st.plotly_chart(style_chart(fig, 370), use_container_width=True)

    st.markdown("### Model leaderboard")
    st.dataframe(
        results.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score"]}),
        use_container_width=True, hide_index=True
    )
    st.info("Use the clickable Project Modules menu on the left to open prediction, graphs, model evaluation and data analysis.")

# ======================= PREDICTION =======================
elif page == "⚡ Severity Prediction":
    panel("Prediction laboratory", "Enter a road-accident scenario. ROADNEX sends the values through the trained model pipeline and returns the predicted severity class.")
    input_values = {}
    with st.form("roadnex_prediction_form"):
        left, right = st.columns(2)
        for i, col in enumerate(X.columns):
            target_col = left if i % 2 == 0 else right
            with target_col:
                label = col.replace("_", " ").title()
                if pd.api.types.is_numeric_dtype(X[col]):
                    vals = pd.to_numeric(X[col], errors="coerce").dropna()
                    default = float(vals.median()) if len(vals) else 0.0
                    low = float(vals.min()) if len(vals) else 0.0
                    high = float(vals.max()) if len(vals) else 100.0
                    if low == high:
                        input_values[col] = st.number_input(label, value=default)
                    else:
                        input_values[col] = st.number_input(
                            label, min_value=low, max_value=high,
                            value=min(max(default, low), high)
                        )
                else:
                    options = X[col].dropna().astype(str).value_counts().index.tolist()
                    if not options:
                        options = ["Unknown"]
                    input_values[col] = st.selectbox(label, options)
        submitted = st.form_submit_button("⚡ PREDICT ACCIDENT SEVERITY", use_container_width=True)

    if submitted:
        try:
            row = pd.DataFrame([input_values], columns=X.columns)
            prediction = best_model.predict(row)[0]
            st.success(f"Predicted accident severity: **{prediction}**")
            st.caption("This output is the class label learned from historical records. Interpret the label using your dataset's documentation.")
            try:
                probabilities = best_model.predict_proba(row)[0]
                classes = best_model.named_steps["classifier"].classes_.astype(str)
                prob_df = pd.DataFrame({"Severity class": classes, "Probability": probabilities})
                fig = px.bar(prob_df, x="Severity class", y="Probability", color="Severity class",
                             text=prob_df["Probability"].map(lambda v: f"{v:.1%}"),
                             title="Model probability by severity class")
                fig.update_yaxes(range=[0, 1], tickformat=".0%")
                st.plotly_chart(style_chart(fig, 330), use_container_width=True)
            except Exception:
                st.info("The selected SVM model does not provide probability estimates. The predicted class is shown above.")
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

# ======================= EDA =======================
elif page == "📊 Exploratory Data Analysis":
    panel("Exploratory data analysis", "Interactive versions of the six visualizations in your Colab notebook. Each chart uses the loaded dataset.")
    st.markdown("### 01 / Accident severity distribution")
    sev = y.value_counts().rename_axis("Accident severity").reset_index(name="Records")
    fig = px.bar(sev, x="Accident severity", y="Records", color="Accident severity",
                 text="Records", title="Accident Severity Distribution")
    st.plotly_chart(style_chart(fig, 340), use_container_width=True)

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    category_cols = [c for c in CATEGORY_CANDIDATES if c in df.columns and c != target]
    if not category_cols:
        category_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != target and df[c].nunique() <= 30
        ]

    st.markdown("### 02 / Numerical feature histogram")
    if numeric_cols:
        hcol = st.selectbox("Select numerical feature", numeric_cols, key="hist_choice")
        fig = px.histogram(df, x=hcol, nbins=30, marginal="rug", title=f"Distribution of {hcol}")
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)
    else:
        st.info("No numerical feature is available for a histogram in this dataset.")

    st.markdown("### 03 / Categorical accident-feature bar chart")
    if category_cols:
        bcol = st.selectbox("Select accident condition", category_cols, key="bar_choice")
        counts = df[bcol].fillna("Missing").astype(str).value_counts().head(12).rename_axis(bcol).reset_index(name="Records")
        fig = px.bar(counts, x="Records", y=bcol, orientation="h", text="Records",
                     title=f"Records by {bcol.replace('_', ' ').title()}")
        st.plotly_chart(style_chart(fig, 360), use_container_width=True)
    else:
        st.info("No suitable categorical accident feature is available for the bar chart.")

    st.markdown("### 04 / Numerical feature box plot")
    if numeric_cols:
        box_col = st.selectbox("Select numerical feature for box plot", numeric_cols, key="box_choice")
        fig = px.box(df, x=box_col, points="outliers", title=f"Box Plot of {box_col}")
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
    else:
        st.info("No numerical feature is available for a box plot.")

    st.markdown("### 05 / Scatter plot by severity")
    if len(numeric_cols) >= 2:
        c1, c2 = st.columns(2)
        with c1: x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with c2:
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y-axis", y_options, key="scatter_y")
        scatter = df[[x_col, y_col, target]].dropna()
        if len(scatter) > 3000:
            scatter = scatter.sample(3000, random_state=42)
        fig = px.scatter(scatter, x=x_col, y=y_col, color=target, opacity=.7,
                         title=f"{x_col} vs {y_col} by Accident Severity")
        st.plotly_chart(style_chart(fig, 390), use_container_width=True)
    else:
        st.info("At least two numerical features are required for a scatter plot.")

    st.markdown("### 06 / Numerical correlation heatmap")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Numerical Feature Correlation Heatmap")
        st.plotly_chart(style_chart(fig, 450), use_container_width=True)
    else:
        st.info("At least two numerical features are required for a correlation heatmap.")

    st.markdown("### Extra / Accident conditions compared with severity")
    for col in category_cols:
        cross = pd.crosstab(df[col].fillna("Missing").astype(str), df[target])
        cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).head(10).index]
        plot_df = cross.reset_index()
        fig = px.bar(plot_df, x=col, y=cross.columns.astype(str).tolist(),
                     barmode="stack", title=f"Severity by {col.replace('_', ' ').title()}")
        st.plotly_chart(style_chart(fig, 330), use_container_width=True)

# ======================= MODEL COMPARISON =======================
elif page == "🧠 Model Comparison":
    panel("Algorithm comparison", "All six algorithms are trained using the notebook's 80/20 stratified split and the same preprocessing pipeline.")
    st.dataframe(
        results.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score"]}),
        use_container_width=True, hide_index=True
    )
    metric_name = st.selectbox("Choose metric to compare", ["F1-Score", "Accuracy", "Precision", "Recall"])
    fig = px.bar(results.sort_values(metric_name), x=metric_name, y="Algorithm",
                 orientation="h", color=metric_name,
                 color_continuous_scale=["#17445c", "#58f0df"],
                 title=f"Six Algorithm Comparison — {metric_name}")
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(style_chart(fig, 410), use_container_width=True)

# ======================= CONFUSION MATRIX =======================
elif page == "▦ Confusion Matrix & Report":
    panel("Confusion matrix & classification report", f"Evaluation of the best model: {model_pack['best_name']}. Rows are actual classes and columns are predicted classes.")
    labels = sorted(y.unique().tolist(), key=lambda value: str(value))
    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    fig = px.imshow(
        cm, x=[str(v) for v in labels], y=[str(v) for v in labels],
        text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Predicted label", "y": "Actual label", "color": "Records"},
        title=f"Confusion Matrix — {model_pack['best_name']}"
    )
    st.plotly_chart(style_chart(fig, 430), use_container_width=True)
    report = classification_report(y_test, best_predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Class / Metric"})
    st.markdown("### Classification report")
    st.dataframe(report_df, use_container_width=True, hide_index=True)

# ======================= FEATURE IMPORTANCE =======================
elif page == "◎ Feature Importance":
    panel("Feature importance", "Top 15 encoded input features ranked by the Random Forest classifier, matching the feature-importance section in your Colab notebook.")
    rf_pipeline = model_pack["trained"]["Random Forest"]
    rf_model = rf_pipeline.named_steps["classifier"]
    try:
        encoded_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({
            "Feature": encoded_names,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)
        fig = px.bar(
            importance_df.sort_values("Importance"), x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale=["#17445c", "#58f0df"],
            title="Top 15 Important Features — Random Forest"
        )
        st.plotly_chart(style_chart(fig, 470), use_container_width=True)
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Feature importance could not be displayed: {exc}")

# ======================= DATA EXPLORER =======================
elif page == "🗂️ Data Explorer":
    panel("Dataset explorer", "Inspect the dataset that ROADNEX loaded automatically. There is no upload control on this page.")
    c1, c2, c3 = st.columns(3)
    with c1: metric("RAW RECORDS", f"{raw.shape[0]:,}", "SOURCE DATA")
    with c2: metric("CLEAN RECORDS", f"{df.shape[0]:,}", "AFTER CLEANING")
    with c3: metric("MODEL FEATURES", str(X.shape[1]), "AFTER FEATURE SELECTION")
    st.markdown("### Dataset preview")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.markdown("### Column quality report")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Missing values": df.isna().sum().values,
        "Unique values": df.nunique(dropna=False).values
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.markdown("### Feature-selection summary")
    st.write("**Target column:**", target)
    st.write("**Input features:**", X.columns.tolist())
    st.write("**Removed identifier / outcome columns:**", dropped_cols or "None")
    st.write("**Removed constant columns:**", constant_cols or "None")
    st.write("**Removed all-missing columns:**", all_missing_cols or "None")
    st.write("**Numerical features:**", model_pack["numeric_features"])
    st.write("**Categorical features:**", model_pack["categorical_features"])

# ======================= PROJECT PROFILE =======================
elif page == "ℹ️ Project Profile":
    panel("About ROADNEX", "Transportation Accident Severity Prediction Using Machine Learning.")
    st.markdown("""
    **Project objective**  
    Classify accident severity using patterns learned from historical road-accident records.

    **Machine-learning type**  
    Supervised multi-class classification.

    **Algorithms used**
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    - Random Forest
    - Naive Bayes
    - Support Vector Machine

    **Evaluation metrics**
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1-score
    - Confusion Matrix
    - Classification Report

    **Workflow**
    1. Load historical accident dataset.
    2. Clean column names, duplicate rows and missing target labels.
    3. Select input features and remove identifiers / post-accident outcome columns.
    4. Impute missing values, encode categories and scale numeric features.
    5. Train six classification algorithms.
    6. Compare their performance and select the best Macro F1-score model.
    7. Predict the severity class for a new road scenario.

    **Important limitation**  
    ROADNEX classifies severity labels in historical records. It does not predict the future time or location of an accident, monitor live traffic, or replace emergency services.
    """)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#718ba2;font:10px DM Mono;letter-spacing:2px">'
    'ROADNEX AI • PREDICT SMARTER. TRAVEL SAFER. • ACADEMIC ML PROTOTYPE</div>',
    unsafe_allow_html=True
)
