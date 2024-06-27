import pandas as pd

def load_file(file):
    if file is not None:
        file_extension = file.name.split('.')[-1].lower()
        try:
            if file_extension == 'csv':
                df = pd.read_csv(file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(file)
            elif file_extension == 'json':
                df = pd.read_json(file)
            else:
                return None
            return df
        except Exception as e:
            return None
    return None