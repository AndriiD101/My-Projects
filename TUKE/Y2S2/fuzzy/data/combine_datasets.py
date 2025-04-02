import pandas as pd

df = pd.read_csv(r"D:\tuke\summer_semester_24_25\fuzzy_systemy\zapocet\data\complete_data.csv")
print(df.isnull().values.any())
print(df.describe())
df = df.dropna()
print(df.describe())
df.to_csv(r"D:\tuke\summer_semester_24_25\fuzzy_systemy\zapocet\data\complete_data.csv", index=False)