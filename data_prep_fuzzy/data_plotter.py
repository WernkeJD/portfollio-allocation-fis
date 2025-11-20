import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv(r"C:\Users\Connor\OneDrive\fuzzyport\portfollio-allocation-fis\data_prep_fuzzy\fred\zscored_clipped.csv")

# Try to detect a time column
time_col = None
for c in df.columns:
    if "date" in c.lower() or "time" in c.lower():
        time_col = c
        break

# If no obvious time column, just use index
x = df[time_col] if time_col else df.index

# Plot all numeric columns
plt.figure(figsize=(12,6))
for col in df.select_dtypes(include='number').columns:
    plt.plot(x, df[col], label=col)

plt.xlabel(time_col if time_col else "Index")
plt.ylabel("Value")
plt.title("All Columns Over Time")
plt.legend()
plt.tight_layout()
plt.show()
