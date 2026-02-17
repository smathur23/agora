import pandas as pd


# Read the CSV
df = pd.read_csv("evaluation/questions_only.csv")

# Keep the first row unchanged
first_row = df.iloc[:1]

# Shuffle the remaining rows
shuffled = df.iloc[1:].sample(frac=1, random_state=42).reset_index(drop=True)

# Combine back together
result = pd.concat([first_row, shuffled], ignore_index=True)

# Save to CSV
result.to_csv("evaluation/questions_only.csv", index=False)