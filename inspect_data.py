import pandas as pd

# Location of our dataset
file_path = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

print("Loading dataset...")

# Read the CSV file
df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

# Show number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Show first five records
print("\nFirst five records:")
print(df.head())

# Show information about the columns
print("\nDataset information:")
df.info()