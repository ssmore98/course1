import pandas
import numpy
import matplotlib.pyplot
import seaborn

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

df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["horsepower"]] = df[["horsepower"]].astype("int")
df[["bore"]] = df[["bore"]].astype("float")
df[["stroke"]] = df[["stroke"]].astype("float")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# list the data types for each column
print(df.dtypes)

print(df.corr())

print(df[['bore','stroke' ,'compression-ratio','horsepower']].corr())

# Engine size as potential predictor variable of price
seaborn.regplot(x="engine-size", y="price", data=df)
matplotlib.pyplot.ylim(0,)
# matplotlib.pyplot.show()

# highway mpg size as potential predictor variable of price
seaborn.regplot(x="highway-mpg", y="price", data=df)
matplotlib.pyplot.ylim(0,)
# matplotlib.pyplot.show()

print(df[["engine-size", "price"]].corr())
print(df[["highway-mpg", "price"]].corr())

seaborn.regplot(x="peak-rpm", y="price", data=df)
matplotlib.pyplot.ylim(0,)
# matplotlib.pyplot.show()
print(df[["peak-rpm", "price"]].corr())

print(df[["stroke","price"]].corr())
matplotlib.pyplot.clf()
seaborn.regplot(x="stroke", y="price", data=df)
matplotlib.pyplot.ylim(0,)
# matplotlib.pyplot.show()

matplotlib.pyplot.clf()
seaborn.boxplot(x="body-style", y="price", data=df)
# matplotlib.pyplot.show()

matplotlib.pyplot.clf()
seaborn.boxplot(x="engine-location", y="price", data=df)
# matplotlib.pyplot.show()

matplotlib.pyplot.clf()
seaborn.boxplot(x="drive-wheels", y="price", data=df)
# matplotlib.pyplot.show()

print(df.describe())

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

engine_location_counts = df['engine-location'].value_counts().to_frame()
engine_location_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_location_counts.index.name = 'engine-location'
print(engine_location_counts)

print(df['drive-wheels'].unique())

df_group_one = df[['drive-wheels','body-style','price']]
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)

df_gptest2 = df[['body-style','price']]
grouped_test2 = df_gptest2.groupby(['body-style'],as_index=False).mean()
print(grouped_test2)

#use the grouped results
matplotlib.pyplot.clf()
matplotlib.pyplot.pcolor(grouped_pivot, cmap='RdBu')
matplotlib.pyplot.colorbar()
fig, ax = matplotlib.pyplot.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(numpy.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(numpy.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
matplotlib.pyplot.xticks(rotation=90)

fig.colorbar(im)
matplotlib.pyplot.show()
