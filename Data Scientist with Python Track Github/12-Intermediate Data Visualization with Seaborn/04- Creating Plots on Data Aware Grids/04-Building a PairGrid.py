'''

Building a PairGrid

When exploring a dataset, one of the earliest tasks is exploring the relationship between pairs of variables.
This step is normally a precursor to additional investigation.

Seaborn supports this pair-wise analysis using the PairGrid. In this exercise, we will look at the Car
Insurance Premium data we analyzed in Chapter 1. All data is available in the df variable.


Instructions

    1 Compare "fatal_collisions" to "premiums" by using a scatter plot mapped to a PairGrid(). - 50 XP

    2 Create another PairGrid but plot a histogram on the diagonal and scatter plot on the off diagonal. - 50 XP

'''


# Create a PairGrid with a scatter plot for fatal_collisions and premiums
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map(plt.scatter)

plt.show()
plt.clf()



# Create the same PairGrid but map a histogram on the diag
g = sns.PairGrid(df, vars=["fatal_collisions", "premiums"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)

plt.show()
plt.clf()