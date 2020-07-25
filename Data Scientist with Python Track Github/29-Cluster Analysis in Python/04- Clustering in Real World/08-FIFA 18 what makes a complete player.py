'''

FIFA 18: what makes a complete player?

The overall level of a player in FIFA 18 is defined by six characteristics: pace (pac), shooting (sho), passing (pas),
dribbling (dri), defending (def), physical (phy).

Here is a sample card:

Eden Hazard Player Card

In this exercise, you will use all six characteristics to create clusters. The data for this exercise is stored
in a Pandas dataframe, fifa. features is the list of these column names and scaled_features is the list of columns
which contains their scaled values. The following have been pre-loaded: kmeans, vq from scipy.cluster.vq,
matplotlib.pyplot as plt, seaborn as sns.

Before you start the exercise, you may wish to explore scaled_features in the console to check out the list of
six scaled columns names.

Instructions

    1   Use the kmeans() algorithm to create 2 clusters using the list of columns, scaled_features. - 35 XP

    2   Assign cluster labels to each row using vq() and print cluster centers of scaled_features using
        the .mean() method of Pandas.       - 35 XP

    3   Plot a bar chart of scaled attributes of each cluster center using the .plot() method of Pandas.    - 0 XP

    4   Print the names of first 5 players in each cluster, using the name column.  - 30 XP

'''

# Create centroids with kmeans for 2 clusters
cluster_centers,_ = kmeans(fifa[scaled_features], 2)

# Assign cluster labels and print cluster centers
fifa['cluster_labels'], _ = vq(fifa[scaled_features], cluster_centers)
print(fifa.groupby('cluster_labels')[scaled_features].mean())


# Plot cluster centers to visualize clusters
fifa.groupby('cluster_labels')[scaled_features].mean().plot(legend=True, kind='bar')
plt.show()


# Get the name column of first 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])