import os
import pickle
import logging
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from read import load_opex_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (override via environment variables)
# ---------------------------------------------------------------------------
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", 36))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "Trained Models"))
OPEX_DATA_PATH = os.getenv("OPEX_DATA_PATH", "")
SERVER = os.getenv("DB_SERVER", "localhost\\SQLEXPRESS")
DATABASE = os.getenv("DB_NAME", "master")
TABLE_NAME = os.getenv("DB_TABLE", "opex_kpi_actual_forecast")

CONN_STR = (
    f"mssql+pyodbc://{SERVER}/{DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

KPI_COLUMNS = [
    "Maintenance Cost",
    "Production Cost",
    "Transportation Cost",
    "Admin Cost",
    "Security Cost",
    "Logistics Cost",
    "Inventory Cost",
    "Infrastructure Cost",
    "Safety & Compliance Cost",
    "Environmental Cost",
    "Utilities Cost",
    "Training Cost",
    "Communication Cost",
    "Insurance Cost",
    "Equipment Rental Cost",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_kpi_dataframe(df: pd.DataFrame, kpi_name: str) -> pd.DataFrame:
    """Filter and prepare a time-indexed DataFrame for a single KPI.

    Args:
        df: Raw OPEX DataFrame with a date column and KPI columns.
        kpi_name: Name of the KPI column to extract.

    Returns:
        A time-indexed DataFrame containing only the requested KPI series,
        with missing values forward-filled.
    """
    kpi_df = df[["Date", kpi_name]].copy()
    kpi_df["Date"] = pd.to_datetime(kpi_df["Date"])
    kpi_df.set_index("Date", inplace=True)
    kpi_df.index = kpi_df.index.to_period("M")
    kpi_df.fillna(method="ffill", inplace=True)
    return kpi_df


def forecast_kpi(kpi_name: str, kpi_df: pd.DataFrame) -> "pd.DataFrame | None":
    """Load a saved SARIMA model and generate a forecast for one KPI.

    Args:
        kpi_name: Name of the KPI to forecast.
        kpi_df: Time-indexed DataFrame for this KPI.

    Returns:
        DataFrame with forecast values, or None if the model file is missing
        or an error occurs during prediction.
    """
    model_path = MODEL_DIR / f"{kpi_name}.pkl"
    if not model_path.exists():
        logger.warning("Model file not found for KPI '%s': %s", kpi_name, model_path)
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        forecast = model.forecast(steps=FORECAST_HORIZON)
        forecast_index = pd.period_range(
            start=kpi_df.index[-1] + 1, periods=FORECAST_HORIZON, freq="M"
        )
        forecast_df = pd.DataFrame(
            {kpi_name: forecast.values}, index=forecast_index
        )
        logger.info("Forecast generated for KPI '%s'.", kpi_name)
        return forecast_df

    except Exception as exc:
        logger.error("Failed to forecast KPI '%s': %s", kpi_name, exc)
        return None


def build_actual_forecast_dataset(opex_df: pd.DataFrame) -> pd.DataFrame:
    """Combine historical actuals and SARIMA forecasts for all KPIs.

    Args:
        opex_df: Full OPEX DataFrame returned by load_opex_data().

    Returns:
        A single DataFrame whose rows cover both the historical period and
        the forecast horizon, with columns [Date, KPI, Value, Type].
    """
    records = []

    for kpi in KPI_COLUMNS:
        if kpi not in opex_df.columns:
            logger.warning("KPI column '%s' not found in data -- skipping.", kpi)
            continue

        kpi_df = prepare_kpi_dataframe(opex_df, kpi)

        # Historical actuals
        for period, row in kpi_df.iterrows():
            records.append(
                {
                    "Date": str(period),
                    "KPI": kpi,
                    "Value": row[kpi],
                    "Type": "Actual",
                }
            )

        # Forecast
        forecast_df = forecast_kpi(kpi, kpi_df)
        if forecast_df is not None:
            for period, row in forecast_df.iterrows():
                records.append(
                    {
                        "Date": str(period),
                        "KPI": kpi,
                        "Value": row[kpi],
                        "Type": "Forecast",
                    }
                )

    combined = pd.DataFrame(records)
    logger.info("Combined dataset shape: %s", combined.shape)
    return combined


def save_to_sql(df: pd.DataFrame) -> None:
    """Persist a DataFrame to the configured SQL Server table.

    Args:
        df: DataFrame to write.

    Raises:
        SQLAlchemyError: Re-raised after logging if the database write fails.
    """
    try:
        engine = create_engine(CONN_STR)
        df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
        logger.info("Data saved to SQL table '%s' (%d rows).", TABLE_NAME, len(df))
    except SQLAlchemyError as exc:
        logger.error("Database error while saving to '%s': %s", TABLE_NAME, exc)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data loading, forecasting, and persistence.

    Raises:
        EnvironmentError: If OPEX_DATA_PATH environment variable is not set.
    """
    if not OPEX_DATA_PATH:
        raise EnvironmentError(
            "OPEX_DATA_PATH environment variable is not set. "
            "Export it before running, e.g.:\n"
            "  export OPEX_DATA_PATH=/data/OPEX_data.xlsx"
        )

    logger.info("Loading OPEX data from: %s", OPEX_DATA_PATH)
    opex_df = load_opex_data(OPEX_DATA_PATH)

    logger.info("Building actual-vs-forecast dataset...")
    final_df = build_actual_forecast_dataset(opex_df)

    logger.info("Saving results to SQL Server...")
    save_to_sql(final_df)

    logger.info("Pipeline complete. %d rows written.", len(final_df))


if __name__ == "__main__":
    main()
