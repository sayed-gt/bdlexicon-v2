import pandas as pd

df = pd.read_csv('bdlexicon_analysis.csv')

# Convert to datetime, coercing invalid formats to NaT (missing value)
if 'appearance' in df.columns:
    df['appearance'] = pd.to_datetime(df['appearance'], errors='coerce')

# Filter only the problematic/invalid dates (where conversion failed)
invalid_df = df[df['appearance'].isna()]


# now put the invalid dates in the filtered df to replace none
# You can match by word

# Save those invalid rows
invalid_df.to_csv('problematic_dates.csv', index=False)
