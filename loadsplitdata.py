import pandas as pd
from datasets import Dataset
import os

jsonl_filepath = r"C:\Users\zaina\Downloads\your_dataset.jsonl"  # **REPLACE THIS WITH YOUR ACTUAL PATH**
print(f"Loading data from: {jsonl_filepath}")

df = pd.read_json(jsonl_filepath, lines=True)  # Load as Pandas DataFrame

# Convert labels to integers (if they are strings)
if df['Responses'].dtype == 'object': # Check if labels are strings
    df['Responses'] = df['Responses'].astype('category').cat.codes  # Convert to numerical labels

dataset = Dataset.from_pandas(df)  # Convert to Dataset

dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset['test'].train_test_split(test_size=0.5, seed=42)
dataset['validation'] = test_dataset['test']
dataset['test'] = test_dataset['train']

save_directory = "split_datasets"
print(f"Saving data to: {save_directory}")

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

dataset.save_to_disk(save_directory)

print(f"Datasets saved to: {save_directory}")