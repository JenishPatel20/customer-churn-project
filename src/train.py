import os
import pandas as pd

def main():
    path = os.path.join("data", "raw", "telco.csv")
    if not os.path.exists(path):
        print("Missing dataset:", path)
        print("Put your churn CSV at data/raw/telco.csv")
        return

    df = pd.read_csv(path)

    print("Loaded:", df.shape)
    print(df.head())

    # quick sanity checks
    print("\nColumns:", list(df.columns))
    print("\nChurn counts:\n", df["Churn"].value_counts(dropna=False))
    print("\nMissing values per column:\n", df.isna().sum().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()
