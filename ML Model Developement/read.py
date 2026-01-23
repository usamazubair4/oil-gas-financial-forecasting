import pandas as pd


def load_opex_data(file_path: str) -> pd.DataFrame:
    """
    Load OPEX data from Excel and filter only OPEX records.
    """
    df = pd.read_excel(file_path)
    df = df[df['dhk_name'] == 'OPEX']
    return df


def build_main_category_dfs(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of dataframes for each main OPEX category (dmk_name).
    """
    category_df = {}

    main_categories = df['dmk_name'].unique()

    for cat in main_categories:
        category_df[cat] = df[df['dmk_name'] == cat]
        print(f"{cat} dataframe created.")

    return category_df


def build_sub_category_map(category_df: dict) -> dict:
    """
    Create a dictionary mapping main categories to their sub-categories (dsk_name).
    """
    category_name = {}

    for main_cat, sub_df in category_df.items():
        sub_categories = sub_df['dsk_name'].unique()
        category_name[main_cat] = sub_categories

    return category_name


def main():
    opex_data_path = r"C:\Users\Usama Zubair\Projects\Financial Envelope\Data\OPEX\OPEX_data.xlsx"

    # Load and filter data
    opex_data = load_opex_data(opex_data_path)

    # Build main category dataframes
    category_df = build_main_category_dfs(opex_data)

    # Build main → sub category mapping
    category_name = build_sub_category_map(category_df)

    # Example access
    # print(category_df['Field and other services'].head())
    # print(category_name['Field and other services'])


if __name__ == "__main__":
    main()
