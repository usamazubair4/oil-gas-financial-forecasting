import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error


# =========================
# CONFIG
# =========================
FORECAST_HORIZON = 24  # months
MODEL_DIR = Path("Trained Models")
MODEL_DIR.mkdir(exist_ok=True)


# =========================
# DATA PREP
# =========================
def prepare_kpi_dataframe(df: pd.DataFrame, kpi_name: str) -> pd.DataFrame:
    """
    1. Filter KPI (main category)
    2. Keep only rows where dsk_name is NA
    3. Drop remaining NA rows
    4. Select required columns
    """
    kpi_df = df[
        (df['dmk_name'] == kpi_name) &
        (df['dsk_name'].isna())
    ].copy()

    kpi_df = kpi_df[
        ['start_date', 'actual_value_pkr', 'prod_total_boe', 'prod_gas_boe']
    ]

    kpi_df.dropna(inplace=True)

    kpi_df['start_date'] = pd.to_datetime(kpi_df['start_date'])
    kpi_df.sort_values('start_date', inplace=True)
    kpi_df.set_index('start_date', inplace=True)

    return kpi_df


# =========================
# SARIMA TRAINING
# =========================
def train_sarima(
    y: pd.Series,
    exog: pd.DataFrame
):
    """
    Train SARIMA model with predefined grid
    """
    param_grid = {
        "order": [(1, 1, 1), (2, 1, 1), (1, 1, 2)],
        "seasonal_order": [(1, 1, 1, 12), (0, 1, 1, 12)]
    }

    best_model = None
    best_mape = np.inf
    best_cfg = None

    # Train-validation split
    train_size = int(len(y) * 0.8)

    y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
    exog_train, exog_val = exog.iloc[:train_size], exog.iloc[train_size:]

    for order in param_grid["order"]:
        for seasonal_order in param_grid["seasonal_order"]:
            try:
                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                result = model.fit(disp=False)

                forecast = result.forecast(
                    steps=len(y_val),
                    exog=exog_val
                )

                mape = mean_absolute_percentage_error(y_val, forecast)

                if mape < best_mape:
                    best_mape = mape
                    best_model = result
                    best_cfg = {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "mape": round(mape, 4)
                    }

            except Exception as e:
                continue

    return best_model, best_cfg


# =========================
# SAVE MODEL
# =========================
def save_model(model, config: dict, kpi_name: str):
    """
    Save trained SARIMA model and config
    """
    model_path = MODEL_DIR / f"{kpi_name.replace(' ', '_')}_sarima.pkl"
    meta_path = MODEL_DIR / f"{kpi_name.replace(' ', '_')}_config.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(meta_path, "wb") as f:
        pickle.dump(config, f)

    print(f"✔ Model saved for KPI: {kpi_name}")


# =========================
# MAIN TRAINING LOOP
# =========================
def train_all_kpis(opex_df: pd.DataFrame):
    kpi_names = opex_df['dmk_name'].unique()

    for kpi in kpi_names:
        print(f"\n🚀 Training SARIMA for KPI: {kpi}")

        kpi_df = prepare_kpi_dataframe(opex_df, kpi)

        if len(kpi_df) < 36:
            print("⚠ Skipping (not enough data)")
            continue

        y = kpi_df['actual_value_pkr']
        exog = kpi_df[['prod_total_boe', 'prod_gas_boe']]

        model, config = train_sarima(y, exog)

        if model is None:
            print("❌ No valid SARIMA model found")
            continue

        save_model(model, config, kpi)

        print(f"📊 Best MAPE: {config['mape']}")


# =========================
# ENTRY POINT
# =========================
def main():
    opex_data_path = r"C:\Users\Usama Zubair\Projects\Financial Envelope\Data\OPEX\OPEX_data.xlsx"

    opex_df = pd.read_excel(opex_data_path)
    opex_df = opex_df[opex_df['dhk_name'] == 'OPEX']

    train_all_kpis(opex_df)


if __name__ == "__main__":
    main()
