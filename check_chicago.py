import requests
import pandas as pd

datasets = {
    "historical_traffic": "4g9f-3jbs",
    "road_segments": "n4j6-wkkf"
}

for name, dataset_id in datasets.items():
    url = (
        "https://data.cityofchicago.org/"
        f"resource/{dataset_id}.json"
    )

    response = requests.get(
        url,
        params={"$limit": 10},
        timeout=60
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json())

    print(f"\nDataset: {name}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))

    df.to_csv(f"{name}_sample.csv", index=False)

print("\nChicago API check completed.")
