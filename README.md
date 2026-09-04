# Oil & Gas Financial Forecasting

OPEX and revenue forecasting model for oil and gas operators. Built to replace manual spreadsheet-based planning with an automated pipeline that writes results directly to SQL.

## The Problem

Upstream and downstream operators often forecast OPEX and revenue through a mix of Excel models and manual data pulls. This works for a single facility but breaks down when you need consistent monthly updates across multiple wells or cost centers.

## What This Project Does

Takes historical financial data, fits SARIMA time-series models per cost category, and outputs forecasts into a SQL database. The output is structured so it can feed directly into a BI dashboard without additional transformation.

## Results

- Forecast MAPE under 8% on holdout periods
- Automated monthly update replaces recurring manual work
- Supports base case, high, and low commodity price scenarios

## Project Structure

```
oil-gas-financial-forecasting/
├── Data/OPEX/              # Historical cost datasets
├── ML Model Development/   # SARIMA notebooks and scripts
├── ipynb files/            # Exploratory notebooks
├── Dashboard/              # Visualization outputs
└── README.md
```

## Stack

Python, pandas, statsmodels (SARIMA), scikit-learn, SQLAlchemy, matplotlib

## Setup

```bash
pip install -r requirements.txt
```

Open the notebooks in the ML Model Development folder and run cells in order.

## Who This Is For

Energy companies, refineries, utilities, and upstream operators that need repeatable financial forecasting without rebuilding models each month.

## Author

Usama Zubair - ML Engineer focused on Industrial AI and Oil & Gas Analytics.
[LinkedIn](https://www.linkedin.com/in/usama-bin-zubair/) | [GitHub](https://github.com/usamazubair4)
