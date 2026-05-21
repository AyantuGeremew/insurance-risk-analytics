import pandas as pd

def load_data(filepath):

    # Load CSV
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column safely
    if 'date' in df.columns:

        df['date'] = pd.to_datetime(
            df['date'],
            format='mixed',
            errors='coerce',
            utc=True
        )

    return df