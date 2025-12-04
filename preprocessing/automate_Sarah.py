import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean())

    scaler = StandardScaler()
    numeric = df.select_dtypes(include=['int64', 'float64'])
    df[numeric.columns] = scaler.fit_transform(numeric)

    return df

def save_preprocessed(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    raw = load_raw("namadataset_raw/Heart_disease_statlog.csv")
    clean = preprocess(raw)
    save_preprocessed(clean, "preprocessing/Heart_preprocessed.csv")
    print("Preprocessing selesai dan tersimpan!")
