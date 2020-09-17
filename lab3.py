import pandas
import numpy

# load the data
path = "automobile.csv"
df = pandas.read_csv(path)
print(df.info)

