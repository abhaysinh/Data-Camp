'''
Parameter estimates of beak depths

Estimate the difference of the mean beak depth of the G. scandens samples from 1975 and 2012 and report a 95% confidence interval.

Since in this exercise you will use the draw_bs_reps() function you wrote in chapter 2, it may be helpful to refer back to it.

Instructions
100 XP

    1   Compute the difference of the sample means.

    2   Take 10,000 bootstrap replicates of the mean for the 1975 beak depths using your draw_bs_reps() function.
        Also get 10,000 bootstrap replicates of the mean for the 2012 beak depths.

    3   Subtract the 1975 replicates from the 2012 replicates to get bootstrap replicates of the difference of means.

    4   Use the replicates to compute the 95% confidence interval.

    5   Hit 'Submit Answer' to view the results!

'''

# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, 10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')