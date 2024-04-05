import pandas as pd 
import matplotlib.pyplot as plt

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering 

from clusteval import clusteval
import numpy as np

from sklearn import metrics


# **Import the data**

from sqlalchemy import create_engine
from urllib.parse import quote

df = pd.read_csv(r"D:/360DigiTMG/DS/Dataset/DS_(1)_Hierarchial Clustering/AirTraffic_Passenger_Statistics.csv")

# Credentials to connect to Database
user = 'root' # user name
pw = quote('1234') # password
db = 'air_routes_db' # database
# creating engine to connect MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('airline_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from airline_tbl;'
df = pd.read_sql_query(sql, engine)

# Data types
df.info()
df.isnull().sum()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

df.describe()

df.duplicated().sum()

# my_report = sweetviz.analyze([df, "df"])
# my_report.show_html('Report.html')

# As we can see there are multiple columns in our dataset, 
# but for cluster analysis we will use 
# Operating Airline, Geo Region, Passenger Count and Flights held by each airline.
df1 = df[["Operating Airline", "GEO Region", "Passenger Count"]]

airline_count = df1["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)

passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)

'''So as this algorithms is working with distances it is very sensitive to outliers, 
that’s why before doing cluster analysis we have to identify outliers and remove them from the dataset. 
In order to find outliers more accurately, we will build the scatter plot.'''

df2 = pd.concat([airline_count, passenger_count], axis=1)
# x = airline_count.values
# y = passenger_count.values
plt.figure(figsize = (10,10))
plt.scatter(df2['count'], df2['Passenger Count'])
plt.xlabel("Flights held")
plt.ylabel("Passengers")
for i, txt in enumerate(airline_count.index.values):
    a = plt.gca()
    plt.annotate(txt, (df2['count'][i], df2['Passenger Count'][i]))
plt.show()

df2.index
# We can see that most of the airlines are grouped together in the bottom left part of the plot, 
# some are above them, and it has 2 outliers United Airlines and Unites Airlines — Pre 07/01/2013.
# So let’s get rid of them.

index_labels_to_drop = ['United Airlines', 'United Airlines - Pre 07/01/2013']
df3 = df2.drop(index_labels_to_drop)


######### Model Building #########
# ### Hierarchical Clustering - Agglomerative Clustering

# from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.cluster import AgglomerativeClustering 
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline') --- if running in jupyter notebook

plt.figure(1, figsize = (16, 8))
tree_plot = dendrogram(linkage(df3, method  = "ward"))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Euclidean distance')
plt.show()


# Applying AgglomerativeClustering and grouping data into 2 clusters 
# based on the above dendrogram as a reference

hc1 = AgglomerativeClustering(n_clusters = 2, metric = 'euclidean', linkage = 'ward')

y_hc1 = hc1.fit_predict(df3)
y_hc1

# Analyzing the Results obtained
hc1.labels_   # Referring to the cluster labels assigned

cluster_labels = pd.Series(hc1.labels_ ) 

# Combine the labels obtained with the data
df3.reset_index(inplace=True)
df_clust = pd.concat([cluster_labels, df3], axis = 1) 
df3.set_index('index', inplace=True)
df_clust.set_index('index', inplace=True)
df_clust.head()

df_clust.columns
df_clust = df_clust.rename(columns = {0: 'cluster'})
df_clust.head()

df_clust['cluster'].value_counts()

# # Clusters Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of the clustering technique, and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(df3, cluster_labels)

'''Alternatively, we can use:'''
# **Calinski Harabasz:**
# Higher value of the CH index means clusters are well separated.
# There is no thumb rule which is an acceptable cut-off value.
metrics.calinski_harabasz_score(df3, cluster_labels)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
metrics.davies_bouldin_score(df3, cluster_labels)


'''Hyperparameter Optimization for Hierarchical Clustering'''
# Experiment to obtain the best clusters by altering the parameters

# ## Cluster Evaluation Library

# pip install clusteval
# Refer to link: https://pypi.org/project/clusteval

# from clusteval import clusteval
# import numpy as np

# Silhouette cluster evaluation. 
ce = clusteval(evaluate = 'silhouette')

df_array = np.array(df3)

# Fit
ce.fit(df_array)

# Plot
ce.plot()

##clusteval also given 2 

df_clust.to_csv('Air.csv', encoding = 'utf-8')

import os
os.getcwd()

