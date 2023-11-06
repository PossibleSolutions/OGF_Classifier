#Kmeans clustering and correlation testing

#method imports
import numpy as np
from sklearn.cluster import KMeans
import geopandas as gpd
import pandas as pd
import dask_geopandas as dask_gpd
from sklearn.impute import SimpleImputer
from scipy import stats

#filereads
filename = r"YourFile"
ddf = dask_gpd.read_file(filename, npartitions=4)
df = ddf.compute()
print(f"Length of input is {len(ddf)}")

#Drop columns that include over 99% of Nulls that have little or no impact in classification (optional)
filter = (df.isnull().mean() * 100)>99
drop_me = filter[filter].index.to_list()
df = df.drop(columns=drop_me)

#Columns to use in the equation/class separation. Do the variables have pos or neg impact to the desired class label
Pos_columns = ['Ika_MaxYhdistetty_Mean', 'KKorkeus8m_Mean', 'KokTilavuusUusi_Mean', 'MZonationValt2_Mean']
Neg_columns = ['MKI_Perc', 'GFW_Perc', 'Ojitus_Perc','Ihmispaine16m_Mean']
All_columns = Pos_columns + Neg_columns

#Classifiers dont work with empty values.
#Either fill nans with zeroes, imputer or other more complex method. comment out the unwanted method

#Fill NaNs with zeroes OR...
df[All_columns] = df[All_columns]. fillna(0)

# ...OR Fill NaNs with imputer. Only use Simpleimputer if marginal amount of NaNs, is a very simple imputer!
#https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# Create an imputer object
#imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', etc.
# Fit the imputer on the data and transform X
#X_imputed = imputer.fit_transform(df[All_columns])

#additionally check that there is correlation between dependent and independent variables
print("Correlation between dependent variable and the following independent variables:")
for column in All_columns:
    print(f"{column} {stats.pearsonr(df[column],df['OGF'])}")

# Number of clusters. Test the optimal amount of cluster with e.g. Elbow plot. In this case binary classification problem
#is or is not OGF is checked with n_clusters = 2
n_clusters = 2
print(f"Assigned number of clusters is {n_clusters}")
# Create and fit the K-means model
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df[All_columns])
#kmeans.fit(X_imputed) # if using the imputer instead of fillna

# Get the cluster assignments for each data point and assign to variable
df['cluster'] = kmeans.labels_

#print clusters and their sizes
print(df.cluster.value_counts())

#pivot to see if certain location or other factor dominates the clustering
pivot_df = df.pivot_table(index='ELY', columns='cluster', values='Score', aggfunc='size', fill_value=0)
print(pivot_df)

#uneven or unrealistic cluster sizes (e.g 99% data points being in one and the rest in the other) indicate posible outliers
#homogenous mass or other factor that leads into unsuccesfull clustering
#if the clustering appears to result into the desired output, save it
#df.to_file(r"YourOutput.gpkg", driver="GPKG")