'''
Regression plot parameters

Seaborn's regression plot supports several parameters that can be used to configure the plots
and drive more insight into the data.

For the next exercise, we can look at the relationship between tuition and the percent of students
that receive Pell grants. A Pell grant is based on student financial need and subsidized by the US Government.
In this data set, each University has some percentage of students that receive these grants. Since this data is
continuous, using x_bins can be useful to break the percentages into categories in order to summarize and understand the data.


Instructions

    1 Plot a regression plot of Tuition and PCTPELL. - 30 XP

    2 Create another plot that breaks the PCTPELL column into 5 different bins. - 30 XP

    3 Create a final regression plot that includes a 2nd order polynomial regression line. - 40 XP

'''

# Plot a regression plot of Tuition and the Percentage of Pell Grants
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL")

plt.show()
plt.clf()


# Create another plot that estimates the tuition by PCTPELL
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL",
            x_bins=5)

plt.show()
plt.clf()



# The final plot should include a line using a 2nd order polynomial
sns.regplot(data=df,
            y='Tuition',
            x="PCTPELL",
            x_bins=5,
            order=2)

plt.show()
plt.clf()