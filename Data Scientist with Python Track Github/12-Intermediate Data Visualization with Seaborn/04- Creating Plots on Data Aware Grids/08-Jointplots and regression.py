'''

Jointplots and regression

Since the previous plot does not show a relationship between humidity and rental amounts,
we can look at another variable that we reviewed earlier. Specifically, the relationship between temp and total_rentals.

Instructions

    1 Create a jointplot with a 2nd order polynomial regression plot comparing temp and total_rentals. - 50 XP

    2 Use a residual plot to check the appropriateness of the model. - 50 XP
'''


# Plot temp vs. total_rentals as a regression plot
sns.jointplot(x="temp",
              y="total_rentals",
              kind='reg',
              data=df,
              order=2,
              xlim=(0, 1))

plt.show()
plt.clf()




# Plot a jointplot showing the residuals
sns.jointplot(x="temp",
        y="total_rentals",
        kind='resid',
        data=df,
        order=2)

plt.show()
plt.clf()