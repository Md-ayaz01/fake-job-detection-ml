import pandas as pd
import os

# Load your combined dataset (make sure this file exists)
df = pd.read_csv('data/processed/combined_data.csv')

# Separate fake and legitimate jobs
fake_jobs = df[df['fraudulent'] == 1]
legit_jobs = df[df['fraudulent'] == 0]

# Balance the datasets by alternating
min_len = min(len(fake_jobs), len(legit_jobs))

alternated_rows = []

for i in range(min_len):
    alternated_rows.append(fake_jobs.iloc[i])
    alternated_rows.append(legit_jobs.iloc[i])

alternated_df = pd.DataFrame(alternated_rows)

# Save the alternated dataset
os.makedirs('data/processed', exist_ok=True)
alternated_df.to_csv('data/processed/alternated_combined_data.csv', index=False)

print("✅ Alternated dataset created at data/processed/alternated_combined_data.csv")
