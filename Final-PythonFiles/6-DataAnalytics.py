import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('6 mamaearth.csv')

#drop nan values and index col
df.dropna(inplace= True)
df = df.reset_index()
df.drop(['index'], axis = 1, inplace= True) 

#add brand names to the dataframe and show unique brands
def get_brand(x):
    return x.split('-')[0].lower()

df['brand']= df['name'].apply(get_brand)
df['brand'].unique()

#create count of items for each brand
df1= pd.DataFrame(df.groupby('brand')['asin'].count().reset_index())
df1.columns= ['brand', 'count'] #rename cols
sort_df= df1.sort_values(['count'], ascending= True)

# brands with least no of reviews plot
X = sort_df['brand'].iloc[:5]
Y = sort_df['count'].iloc[:5]
plt.figure(figsize = (16,8))
plt.bar(X,Y)
plt.xlabel("brands")
plt.ylabel("count")
plt.title('Top 5 brands with least number of reviews')
plt.show()

#brands with most number of reviews plot
X = sort_df['brand'].iloc[-10:]
Y = sort_df['count'].iloc[-10:]
plt.figure(figsize = (16,8))
plt.bar(X,Y)
plt.xlabel("brands")
plt.ylabel("count")
plt.title('Top 10 brands with MOST number of reviews')
plt.show()