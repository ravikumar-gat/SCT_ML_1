import os
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")

csv_path = os.path.join(path, "Mall_Customers.csv")
df = pd.read_csv(csv_path)

print(df.head())
