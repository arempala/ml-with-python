import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("FuelConsumptionCo2.csv")

print(type(df))

print(df.head())

print(df['ENGINESIZE'])

plt.scatter( df['ENGINESIZE'], df['CO2EMISSIONS'] )
plt.show()