from pathlib import Path
import pandas as pd


# Find the root folder of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default location of the primary dataset
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "retail_sales_ontario_synthetic.csv"


class LoadSales:
    """
    Load and clean the e-commerce sales dataset.
    """

    def __init__(self, file_path=DEFAULT_DATA_PATH):
        self.file_path = Path(file_path)

        # Load the raw CSV file
        self.data = pd.read_csv(
            self.file_path,
            low_memory=False
        )

        self.cleaned = False

    def getSales(self):
        """
        Return a copy of the loaded sales data.
        """
        return self.data.copy()

    def clean(self):
        """
        Clean and standardize the sales data.
        """

        df = self.data.copy()

        # -------------------------------------------------
        # 1. Clean text columns
        # -------------------------------------------------
        text_columns = [
            "customer_id",
            "product",
            "product_category",
            "coupon_code",
            "payment_method",
            "shipping_city",
            "shipping_province"
        ]

        for column in text_columns:
            if column in df.columns:
                df[column] = (
                    df[column]
                    .astype("string")
                    .str.strip()
                )

        # Standardize coupon codes
        if "coupon_code" in df.columns:
            df["coupon_code"] = (
                df["coupon_code"]
                .str.upper()
                .fillna("NO_COUPON")
            )

        # -------------------------------------------------
        # 2. Convert date column
        # -------------------------------------------------
        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

        # -------------------------------------------------
        # 3. Convert numeric columns
        # -------------------------------------------------
        numeric_columns = [
            "price",
            "quantity",
            "discount_pct",
            "sales_amount"
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # -------------------------------------------------
        # 4. Remove rows missing required information
        # -------------------------------------------------
        required_columns = [
            "date",
            "customer_id",
            "product",
            "price",
            "quantity",
            "shipping_city"
        ]

        required_columns = [
            column
            for column in required_columns
            if column in df.columns
        ]

        df = df.dropna(subset=required_columns)

        # -------------------------------------------------
        # 5. Remove invalid prices
        # -------------------------------------------------
        if "price" in df.columns:
            df = df[df["price"] > 0]

        # -------------------------------------------------
        # 6. Remove invalid quantities
        # -------------------------------------------------
        if "quantity" in df.columns:
            df = df[df["quantity"] > 0]

        # -------------------------------------------------
        # 7. Fix discount percentages
        # -------------------------------------------------
        if "discount_pct" in df.columns:
            df["discount_pct"] = (
                df["discount_pct"]
                .fillna(0)
                .clip(lower=0, upper=100)
            )

        # -------------------------------------------------
        # 8. Remove exact duplicate rows
        # -------------------------------------------------
        df = df.drop_duplicates()

        # -------------------------------------------------
        # 9. Fill missing sales amount when possible
        # -------------------------------------------------
        if "sales_amount" in df.columns:

            discount = (
                df["discount_pct"]
                if "discount_pct" in df.columns
                else pd.Series(0, index=df.index)
            )

            calculated_sales = (
                df["price"]
                * df["quantity"]
                * (1 - discount / 100)
            )

            df["sales_amount"] = (
                df["sales_amount"]
                .fillna(calculated_sales)
            )

        self.data = df.reset_index(drop=True)
        self.cleaned = True

        return self.data.copy()

    def total(self):
        """
        Return total sales amount.
        """

        if "sales_amount" in self.data.columns:
            return float(self.data["sales_amount"].sum())

        return float(
            (
                self.data["price"]
                * self.data["quantity"]
            ).sum()
        )