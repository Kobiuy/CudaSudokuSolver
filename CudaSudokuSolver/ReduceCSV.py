import pandas as pd

# Load the CSV file
file_path = "quatersudoku_cluewise.csv"  # Replace with your file's path
df = pd.read_csv(file_path)

# Drop every second row (keep only rows with even indices)
df_filtered = df.iloc[::2]

# Save the updated DataFrame to a new CSV
df_filtered.to_csv("smalloutput_file.csv", index=False)
print("Processed file saved as 'output_file.csv'")
