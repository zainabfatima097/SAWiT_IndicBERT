import pandas as pd
import json

from pathlib import Path
file_path = Path("C:/Users/zaina/Downloads/Dataset_SAWIT.AI.csv")
df = pd.read_csv(file_path, encoding="utf-8")

# Convert to JSON and save (force_ascii=False is crucial for Hindi)
df.to_json("dataset.json", orient="records", indent=4, force_ascii=False)
