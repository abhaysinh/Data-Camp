'''
Bootstrap replicates of other statistics

We saw in a previous exercise that the mean is Normally distributed. This does not necessarily hold for
other statistics, but no worry: as hackers, we can always take bootstrap replicates! In this exercise,
you'll generate bootstrap replicates for the variance of the annual rainfall at the Sheffield Weather
Station and plot the histogram of the replicates.

Here, you will make use of the draw_bs_reps() function you defined a few exercises ago. It is provided
below for your reference:

def draw_bs_reps(data, func, size=1):
    return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])

INSTRUCTIONS
100XP

    1   Draw 10000 bootstrap replicates of the variance in annual rainfall using your draw_bs_reps() function.
        Hint: Pass in np.var for computing the variance.

    2   Divide your variance replicates - bs_replicates - by 100 to put the variance in units of square
        centimeters for convenience.

    3   Make a histogram of bs_replicates using the normed=True keyword argument and 50 bins.

'''


# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, normed=True, bins=50)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()
