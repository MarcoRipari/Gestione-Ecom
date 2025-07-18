import pandas as pd
import chardet

def read_csv_auto_encoding(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding=encoding)

def not_in_array(array, required):
    return not all(col in array for col in required)
