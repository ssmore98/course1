import pandas

# import the data set
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pandas.read_csv(path, header=None)
print("The first 5 rows of the dataframe") 
print(df.head())
print("The last 10 rows of the dataframe") 
print(df.tail(10))

# create and assign headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
        "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
        "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
        "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
print(df.head())

# find the names of the columns
print(df.columns)

# drone row with no price
df.dropna(subset=["price"], axis=0)

# save the data set
df.to_csv("automobile.csv", index=False)

