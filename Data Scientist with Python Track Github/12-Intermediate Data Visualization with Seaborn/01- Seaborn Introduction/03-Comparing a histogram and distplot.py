'''

Comparing a histogram and distplot

The pandas library supports simple plotting of data, which is very convenient when data is already likely
to be in a pandas DataFrame.

Seaborn generally does more statistical analysis on data and can provide more sophisticated insight into the data.
In this exercise, we will compare a pandas histogram vs the seaborn distplot.


Instructions

    1 Use the pandas' plot.hist() function to plot a histogram of the Award_Amount column. - 50 XP

    2 Use Seaborn's distplot() function to plot a distribution plot of the same column. - 50 XP
'''

# Display pandas histogram
df['Award_Amount'].plot.hist()
plt.show()

# Clear out the pandas histogram
plt.clf()


# Display a Seaborn distplot
sns.distplot(df['Award_Amount'])
plt.show()

# Clear the distplot
plt.clf()