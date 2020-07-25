'''
Pairs bootstrap of literacy/fertility data

Using the function you just wrote, perform pairs bootstrap to plot a histogram describing the estimate of the slope
from the illiteracy/fertility data. Also report the 95% confidence interval of the slope. The data is available to
you in the NumPy arrays illiteracy and fertility.

As a reminder, draw_bs_pairs_linreg() has a function signature of draw_bs_pairs_linreg(x, y, size=1), and it returns
two values: bs_slope_reps and bs_intercept_reps.

Instructions
100 XP

    1   Use your draw_bs_pairs_linreg() function to take 1000 bootstrap replicates of the slope and intercept.
        The x-axis data is illiteracy and y-axis data is fertility.

    2   Compute and print the 95% bootstrap confidence interval for the slope.

    3   Plot and show a histogram of the slope replicates. Be sure to label your axes.
        This has been done for you, so click 'Submit Answer' to see your histogram!

'''

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()
