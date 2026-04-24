import os
import re
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "data" / "house_data.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "final_model.pkl"
ENC_PATH   = MODEL_DIR / "location_encoder.pkl"
META_PATH  = MODEL_DIR / "meta.pkl"
_ALT_DATA  = BASE_DIR / "scrapped_data.csv"
_CWD_DATA  = Path.cwd() / "scrapped_data.csv"
_CWD_HOUSE = Path.cwd() / "data" / "house_data.csv"


def load_raw(path=None):
    candidates = [path, DATA_PATH, _ALT_DATA, _CWD_DATA, _CWD_HOUSE] if path \
                 else [DATA_PATH, _ALT_DATA, _CWD_DATA, _CWD_HOUSE]
    for p in candidates:
        if p and Path(p).exists():
            print(f"📂 Loading: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Dataset not found. Place your CSV at:\n  {DATA_PATH}\n"
        "or pass the path explicitly to load_raw(path)."
    )


def _parse_price(raw):
    if pd.isna(raw):
        return np.nan
    m = re.match(r"([\d.]+)\s*(Cr|Lac)", str(raw).strip(), re.IGNORECASE)
    if not m:
        return np.nan
    val, unit = float(m.group(1)), m.group(2).lower()
    return val * 100 if unit == "cr" else val


def _parse_area(raw):
    if pd.isna(raw):
        return np.nan
    m = re.search(r"([\d.]+)", str(raw))
    return float(m.group(1)) if m else np.nan


FURNISHING_MAP = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}


def clean_and_engineer(df):
    df = df.copy()

    df["bhk"] = df["Title"].str.extract(r"(\d+)\s*BHK").astype(float)

    df["location"] = df["Title"].str.extract(
        r"for Sale in .+?,\s*(.+?),\s*Pune", expand=False
    )
    mask = df["location"].isna()
    df.loc[mask, "location"] = df.loc[mask, "Title"].str.extract(
        r"for Sale in ([^,]+)", expand=False
    )
    df["location"] = df["location"].str.strip()

    df["price_lakhs"]    = df["Price"].apply(_parse_price)
    df["area_sqft"]      = df["Carpet_Area"].apply(_parse_area).fillna(df["Super_Area"].apply(_parse_area))
    df["price_per_sqft"] = df["price_lakhs"] * 100_000 / df["area_sqft"]
    df["furnishing_enc"] = df["Furnishing"].map(FURNISHING_MAP).fillna(0).astype(int)
    df["is_resale"]      = (df["Transaction"] == "Resale").astype(int)
    df["bathrooms"]      = df["Bathroom"].fillna(1)
    df["balcony"]        = df["Balcony"].fillna(0)

    df   = df[df["price_lakhs"] < 500]
    freq = df["location"].value_counts()
    df["location_grp"] = df["location"].where(
        df["location"].isin(freq[freq >= 3].index), "Other"
    )

    return df


def print_eda(df):
    print("\n" + "═" * 60)
    print("  📊  EXPLORATORY DATA ANALYSIS")
    print("═" * 60)
    print(f"  Rows: {len(df):,}   |   Columns: {df.shape[1]}")
    print(f"\n  Price (Lakhs):")
    print(f"    Min:    ₹ {df['price_lakhs'].min():.1f}L")
    print(f"    Median: ₹ {df['price_lakhs'].median():.1f}L")
    print(f"    Mean:   ₹ {df['price_lakhs'].mean():.1f}L")
    print(f"    Max:    ₹ {df['price_lakhs'].max():.1f}L")
    print(f"\n  Area (sqft):")
    print(f"    Min:    {df['area_sqft'].min():.0f}")
    print(f"    Median: {df['area_sqft'].median():.0f}")
    print(f"    Max:    {df['area_sqft'].max():.0f}")
    print(f"\n  BHK distribution:")
    for bhk, cnt in df["bhk"].value_counts().sort_index().items():
        bar = "█" * (cnt // 5)
        print(f"    {int(bhk)}-BHK: {bar} ({cnt})")
    print(f"\n  Furnishing:")
    for k, v in df["Furnishing"].value_counts().items():
        print(f"    {k}: {v}")
    print(f"\n  Top-10 Locations by listing count:")
    for loc, cnt in df["location"].value_counts().head(10).items():
        print(f"    {loc}: {cnt}")
    print("═" * 60 + "\n")


FEATURES = ["loc_enc", "bhk", "area_sqft", "bathrooms", "balcony", "furnishing_enc", "is_resale"]
TARGET   = "price_lakhs"


def train(df_clean, verbose=True):
    df = df_clean.dropna(subset=FEATURES[1:] + [TARGET]).copy()

    le = LabelEncoder()
    df["loc_enc"] = le.fit_transform(df["location_grp"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    if verbose:
        print("\n" + "═" * 60)
        print("  🤖  MODEL TRAINING RESULTS")
        print("═" * 60)
        print(f"  Algorithm : GradientBoostingRegressor")
        print(f"  Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")
        print(f"\n  R²  Score : {r2:.4f}  {'✅ Good' if r2 > 0.6 else '⚠️  Moderate'}")
        print(f"  MAE       : ₹ {mae:.1f} Lakhs")
        print(f"  RMSE      : ₹ {rmse:.1f} Lakhs")
        print(f"\n  Feature Importances:")
        for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"    {feat:<20} {bar}  {imp:.3f}")
        print("═" * 60 + "\n")

    meta = {
        "locations":   list(le.classes_),
        "bhk_options": sorted(df["bhk"].dropna().astype(int).unique().tolist()),
        "area_min":    int(df["area_sqft"].min()),
        "area_max":    int(df["area_sqft"].max()),
        "r2":          r2,
        "mae":         mae,
    }

    return model, le, meta


def save_artefacts(model, le, meta):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(le,    open(ENC_PATH,   "wb"))
    pickle.dump(meta,  open(META_PATH,  "wb"))
    print(f"💾 Saved model   → {MODEL_PATH}")
    print(f"💾 Saved encoder → {ENC_PATH}")
    print(f"💾 Saved meta    → {META_PATH}")


def load_artefacts():
    model = pickle.load(open(MODEL_PATH, "rb"))
    le    = pickle.load(open(ENC_PATH,   "rb"))
    meta  = pickle.load(open(META_PATH,  "rb"))
    return model, le, meta


def predict_price(
    location, bhk, area_sqft,
    bathrooms=2, balcony=1,
    furnishing="Semi-Furnished", transaction="Resale",
    model=None, le=None, meta=None,
):
    if model is None:
        model, le, meta = load_artefacts()

    loc_grp        = location if location in meta["locations"] else "Other"
    loc_enc        = le.transform([loc_grp])[0]
    furnishing_enc = FURNISHING_MAP.get(furnishing, 1)
    is_resale      = 1 if transaction == "Resale" else 0

    X           = np.array([[loc_enc, bhk, area_sqft, bathrooms, balcony, furnishing_enc, is_resale]])
    price_lakhs = max(float(model.predict(X)[0]), 0)

    return {
        "price_lakhs":    round(price_lakhs, 2),
        "price_crores":   round(price_lakhs / 100, 3),
        "price_per_sqft": round(price_lakhs * 100_000 / area_sqft, 0),
    }


def run_training_pipeline():
    print("\n🚀 Pune House Price Prediction — Training Pipeline\n")

    raw = load_raw()
    print(f"✅ Raw data loaded: {raw.shape[0]} rows, {raw.shape[1]} columns")

    df = clean_and_engineer(raw)
    print_eda(df)

    model, le, meta = train(df, verbose=True)
    save_artefacts(model, le, meta)

    print("\n🔮 Demo Prediction:")
    result = predict_price(location="Kharadi", bhk=2, area_sqft=900, model=model, le=le, meta=meta)
    print(f"   2 BHK | 900 sqft | Kharadi")
    print(f"   Estimated Price: ₹ {result['price_lakhs']} Lakhs (₹ {result['price_crores']} Cr)")
    print(f"   Rate: ₹ {result['price_per_sqft']:,.0f} / sqft\n")

    print("✅ Pipeline complete!\n")
    print("▶  To launch the web app:")
    print(f"   streamlit run {__file__}\n")


def streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Pune House Price Predictor", page_icon="🏠", layout="centered")

    @st.cache_resource(show_spinner="Loading model…")
    def get_model():
        if MODEL_PATH.exists() and ENC_PATH.exists() and META_PATH.exists():
            return load_artefacts()
        raw = load_raw()
        df  = clean_and_engineer(raw)
        m, le, meta = train(df, verbose=False)
        save_artefacts(m, le, meta)
        return m, le, meta

    model, le, meta = get_model()

    st.title("🏠 Pune House Price Predictor")
    st.markdown(
        "Powered by **Gradient Boosting** trained on real Pune listings. "
        f"Model R² = **{meta['r2']:.2f}** | "
        f"Avg error ≈ **₹ {meta['mae']:.0f} L**"
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("📍 Location", options=sorted(meta["locations"]), index=0)
        bhk      = st.selectbox("🛏 BHK", options=meta["bhk_options"] if meta["bhk_options"] else [1,2,3,4,5])
        area     = st.number_input("📐 Carpet Area (sqft)", min_value=200, max_value=10000, value=900, step=50)

    with col2:
        bathrooms   = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4], index=1)
        balcony     = st.selectbox("🌿 Balcony", [0, 1, 2, 3], index=1)
        furnishing  = st.selectbox("🪑 Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"], index=1)
        transaction = st.selectbox("📋 Transaction Type", ["Resale", "New Property"])

    st.divider()

    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        result = predict_price(
            location=location, bhk=bhk, area_sqft=area,
            bathrooms=bathrooms, balcony=balcony,
            furnishing=furnishing, transaction=transaction,
            model=model, le=le, meta=meta,
        )

        st.success("Estimated Price")
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Price (Lakhs)", f"₹ {result['price_lakhs']:.1f} L")
        m2.metric("💰 Price (Crores)", f"₹ {result['price_crores']:.2f} Cr")
        m3.metric("📊 Rate / sqft", f"₹ {result['price_per_sqft']:,.0f}")

        st.info(
            f"**Summary:** A {bhk}-BHK {furnishing.lower()} flat of {area} sqft "
            f"in **{location}** ({transaction}) is estimated at "
            f"**₹ {result['price_lakhs']:.1f} Lakhs** "
            f"(₹ {result['price_per_sqft']:,.0f}/sqft)."
        )

    st.divider()
    with st.expander("ℹ️ About this model"):
        st.markdown(f"""
**Algorithm:** Gradient Boosting Regressor (scikit-learn)
**Training data:** 894 Pune property listings (scraped)
**Features used:** Location, BHK, Carpet Area, Bathrooms, Balcony, Furnishing, Transaction type

**Model performance on hold-out test set:**
- R² Score: `{meta['r2']:.4f}`
- Mean Absolute Error: `₹ {meta['mae']:.1f} Lakhs`
        """)


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _sr
        _is_streamlit = _sr.get_script_run_ctx() is not None
    except Exception:
        _is_streamlit = False

    if _is_streamlit or "streamlit" in sys.modules:
        streamlit_app()
    else:
        run_training_pipeline()
