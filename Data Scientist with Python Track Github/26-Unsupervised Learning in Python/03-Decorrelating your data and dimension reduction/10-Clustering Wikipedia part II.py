'''
Clustering Wikipedia part II

It is now time to put your pipeline from the previous exercise to work! You are given an array articles
of tf-idf word-frequencies of some popular Wikipedia articles, and a list titles of their titles.
Use your pipeline to cluster the Wikipedia articles.

A solution to the previous exercise has been pre-loaded for you, so a Pipeline pipeline chaining TruncatedSVD
with KMeans is available.

INSTRUCTIONS
100XP



    1   Import pandas as pd.

    2   Fit the pipeline to the word-frequency array articles.

    3   Predict the cluster labels.

    4   Align the cluster labels with the list titles of article titles by creating a DataFrame df with
        labels and titles as columns. This has been done for you.

    5   Use the .sort_values() method of df to sort the DataFrame by the 'label' column, and print the result.

    6   Hit 'Submit Answer' and take a moment to investigate your amazing clustering of Wikipedia pages!

'''


# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
