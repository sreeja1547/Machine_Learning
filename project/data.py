import pandas as pd

INPUT_FILE = "dataset_ (1).csv"
OUTPUT_FILE = "credit_risk_dataset.csv"

print("Reading ARFF dataset...")

# Find @DATA section
with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

data_start = None

for i, line in enumerate(lines):
    if line.strip().upper() == "@DATA":
        data_start = i + 1
        break

if data_start is None:
    raise ValueError("@DATA section not found!")

# Attribute names
columns = []

for line in lines:
    line = line.strip()

    if line.upper().startswith("@ATTRIBUTE"):
        parts = line.split()

        # @ATTRIBUTE column_name type
        columns.append(parts[1])

# Read only the actual data after @DATA
data_lines = []

for line in lines[data_start:]:
    line = line.strip()

    if line and not line.startswith("%"):
        data_lines.append(line)

# Convert to DataFrame
from io import StringIO

df = pd.read_csv(
    StringIO("\n".join(data_lines)),
    header=None,
    names=columns
)

# Save proper CSV
df.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("CONVERSION SUCCESSFUL")
print("========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved as:")
print(OUTPUT_FILE)

print(df["loan_status"].value_counts())
print(df["loan_status"].value_counts(normalize=True) * 100)