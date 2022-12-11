import pandas as pd
import requests

if __name__ == "__main__":
    data = pd.read_csv("short_dataset.csv")
    request_features = list(data.columns[:-1])

    for i in range(100):
        response = requests.get(
            "http://127.0.0.1:8100/predict",
            json={"data": [data.iloc[i].tolist()[:-1]], "features": request_features}
        )
        print(f"ground true: {data.iloc[i].tolist()[-1]}")
        print(f"model label: {response.json()}")
