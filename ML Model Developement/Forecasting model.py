import pandas as pd
import pickle
from pathlib import Path
from sqlalchemy import create_engine

from read import load_opex_data


# =========================
# CONFIG
# =========================
FORECAST_HORIZON = 36
MODEL_DIR = Path("Trained Models")

# ---------- SQL CONFIG ----------
SERVER = "localhost\SQLEXPRESS"
DATABASE = "master"
TABLE_NAME = "opex_kpi_actual_forecast"

CONN_STR = (
    "mssql+pyodbc://@"
    f"{SERVER}/{DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)


# =========================
# KPI DATA PREP
# =========================
def prepare_kpi_dataframe(df: pd.DataFrame, kpi_name: str) -> pd.DataFrame:
    kpi_df = df[
        (df['dmk_name'] == kpi_name) &
        (df['dsk_name'].isna())
    ].copy()

    kpi_df = kpi_df[
        ['start_date', 'actual_value_pkr',
         'prod_total_boe', 'prod_gas_boe']
    ]

    kpi_df.dropna(inplace=True)
    kpi_df['start_date'] = pd.to_datetime(kpi_df['start_date'])
    kpi_df.sort_values('start_date', inplace=True)
    kpi_df.set_index('start_date', inplace=True)

    return kpi_df


# =========================
# BUILD ACTUALS
# =========================
def build_actuals_df(kpi_name: str, kpi_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "kpi_name": kpi_name,
        "period": kpi_df.index,
        "value_pkr": kpi_df['actual_value_pkr'].values,
        "data_type": "ACTUAL",
        "model_type": "NA",
        "created_on": pd.Timestamp.now()
    })


# =========================
# FUTURE EXOG
# =========================
def build_future_exog(kpi_df: pd.DataFrame) -> pd.DataFrame:
    last_date = kpi_df.index[-1]

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(),
        periods=FORECAST_HORIZON,
        freq="MS"
    )

    last_exog = kpi_df[['prod_total_boe', 'prod_gas_boe']].iloc[-1]

    return pd.DataFrame(
        [last_exog.values] * FORECAST_HORIZON,
        columns=['prod_total_boe', 'prod_gas_boe'],
        index=future_dates
    )


# =========================
# FORECAST KPI
# =========================
def forecast_kpi(kpi_name: str, kpi_df: pd.DataFrame) -> pd.DataFrame | None:
    model_path = MODEL_DIR / f"{kpi_name.replace(' ', '_')}_sarima.pkl"

    if not model_path.exists():
        print(f"⚠ Model not found for KPI: {kpi_name}")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    future_exog = build_future_exog(kpi_df)

    forecast = model.forecast(
        steps=FORECAST_HORIZON,
        exog=future_exog
    )

    return pd.DataFrame({
        "kpi_name": kpi_name,
        "period": forecast.index,
        "value_pkr": forecast.values,
        "data_type": "FORECAST",
        "model_type": "SARIMA",
        "created_on": pd.Timestamp.now()
    })


# =========================
# ACTUAL + FORECAST (ALL KPIs)
# =========================
def build_actual_forecast_dataset(opex_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for kpi in opex_df['dmk_name'].unique():
        print(f"📊 Processing KPI: {kpi}")

        kpi_df = prepare_kpi_dataframe(opex_df, kpi)

        if len(kpi_df) < 12:
            print("⚠ Skipping (insufficient data)")
            continue

        # ACTUALS
        results.append(build_actuals_df(kpi, kpi_df))

        # FORECAST
        forecast_df = forecast_kpi(kpi, kpi_df)
        if forecast_df is not None:
            results.append(forecast_df)

    return pd.concat(results, ignore_index=True)


# =========================
# SAVE TO SQL
# =========================
def save_to_sql(df: pd.DataFrame):
    engine = create_engine(CONN_STR)

    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",  # use replace if rebuilding
        index=False
    )

    print(f"✔ Data saved to SQL table: {TABLE_NAME}")


# =========================
# ENTRY POINT
# =========================
def main():
    opex_data_path = r"C:\Users\Usama Zubair\Projects\Financial Envelope\Data\OPEX\OPEX_data.xlsx"

    opex_df = load_opex_data(opex_data_path)

    final_df = build_actual_forecast_dataset(opex_df)

    save_to_sql(final_df)


if __name__ == "__main__":
    main()
