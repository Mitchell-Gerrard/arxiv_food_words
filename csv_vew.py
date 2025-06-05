import pandas as pd

file_path="FoodData_Central_csv_2025-04-24/food.csv"

data=pd.read_csv(file_path)
print(data.head())