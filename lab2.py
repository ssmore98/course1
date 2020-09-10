import pandas
import matplotlib.pylab
import numpy
import matplotlib
from matplotlib import pyplot

def find_missing_data(df):
    # missing data
    print("MISSING DATA")
    missing_data = df.isnull()
    for column in missing_data.columns.values.tolist():
        if True in missing_data[column].value_counts().keys():
            print(column)
            print(missing_data[column].value_counts())
            print("")    

# load the data
path = "automobile.csv"
df = pandas.read_csv(path)
print(df.info)

# replace unknown data
print(df.head())
df.replace("?", numpy.nan, inplace = True)
print(df.head())


find_missing_data(df)

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(numpy.nan, avg_norm_loss, inplace=True)


print(df[["num-of-doors"]].describe())
df["num-of-doors"].replace(numpy.nan, 4, inplace=True)


print(df[["bore"]].describe())
avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(numpy.nan, avg_bore, inplace=True)


print(df[["horsepower"]].describe())
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)
df["horsepower"].replace(numpy.nan, avg_horsepower, inplace=True)


print(df[["stroke"]].describe())
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(numpy.nan, avg_stroke, inplace=True)


print(df[["peak-rpm"]].describe())
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of stroke:", avg_peak_rpm)
df["peak-rpm"].replace(numpy.nan, avg_peak_rpm, inplace=True)


df.dropna(subset=["price"], axis=0, inplace=True)
# reset index, because we droped rows
df.reset_index(drop=True, inplace=True)


find_missing_data(df)

print(df.dtypes)
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["horsepower"]] = df[["horsepower"]].astype("int")
df[["bore"]] = df[["bore"]].astype("float")
df[["stroke"]] = df[["stroke"]].astype("float")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print(df.dtypes)


#normalization
df['city-L/100km'] = 235/df["city-mpg"]
df['highway-L/100km'] = 235/df["highway-mpg"]

df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

print(df)

# binning
matplotlib.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
matplotlib.pyplot.xlabel("horsepower")
matplotlib.pyplot.ylabel("count")
matplotlib.pyplot.title("horsepower bins")
# matplotlib.pyplot.show()

bins = numpy.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pandas.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))
print(df["horsepower-binned"].value_counts())

matplotlib.pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
matplotlib.pyplot.xlabel("horsepower")
matplotlib.pyplot.ylabel("count")
matplotlib.pyplot.title("horsepower bins")
matplotlib.pyplot.show()
