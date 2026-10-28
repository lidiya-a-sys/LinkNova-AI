import pandas as pd

# Load the dataset
file_path = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("DATASET LABEL CHECK")
print("-" * 40)

# Count each label
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

# Display label proportions
print("\nLabel proportions:")
print(df["label"].value_counts(normalize=True).sort_index() * 100)

# Check missing values in important columns
print("\nMissing URL values:", df["URL"].isna().sum())
print("Missing label values:", df["label"].isna().sum())

# Check duplicate URLs
print("\nDuplicate URLs:", df["URL"].duplicated().sum())

# Display example URLs for each label
for label_value in sorted(df["label"].dropna().unique()):
    print(f"\nExamples for label {label_value}:")
    print(
        df.loc[df["label"] == label_value, "URL"]
        .head(5)
        .to_string(index=False)
    )