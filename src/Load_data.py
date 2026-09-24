from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_sales_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the synthetic Ontario sales dataset."""
    sales_path = Path(path) if path else DATA_DIR / "retail_sales_ontario_synthetic.csv"
    df = pd.read_csv(sales_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["total_sales"] = df["units_sold"] * df["unit_price"]
    return df


def load_city_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load city metadata for geographic enrichment."""
    metadata_path = Path(path) if path else DATA_DIR / "city_metadata.csv"
    df = pd.read_csv(metadata_path)
    return df


def merge_sales_and_city_metadata(
    sales_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge sales data with city metadata on the city name."""
    if sales_df is None:
        sales_df = load_sales_data()
    if metadata_df is None:
        metadata_df = load_city_metadata()

    merged = sales_df.merge(metadata_df, on="city", how="left")
    return merged


def preprocess_sales_data(
    sales_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean the combined dataset for analysis."""
    df = merge_sales_and_city_metadata(sales_df=sales_df, metadata_df=metadata_df)

    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["store_type"] = df["store_type"].fillna("Unknown")
    df["category"] = df["category"].fillna("Uncategorized")
    df["population_2023"] = pd.to_numeric(df["population_2023"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    return df


def main() -> None:
    sales_df = load_sales_data()
    city_df = load_city_metadata()
    processed_df = preprocess_sales_data(sales_df=sales_df, metadata_df=city_df)

    print("Sales rows loaded:", len(sales_df))
    print("City metadata rows loaded:", len(city_df))
    print("Merged dataset rows:", len(processed_df))
    print("\nPreview:")
    print(processed_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
