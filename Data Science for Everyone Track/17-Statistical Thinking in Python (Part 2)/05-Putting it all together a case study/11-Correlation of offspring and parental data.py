"""
Correlation of offspring and parental data

In an effort to quantify the correlation between offspring and parent beak depths,
we would like to compute statistics, such as the Pearson correlation coefficient, between parents and offspring.
To get confidence intervals on this, we need to do a pairs bootstrap.

You have already written a function to do pairs bootstrap to get estimates for parameters derived from linear regression.
Your task in this exercise is to make a new function with call signature draw_bs_pairs(x, y, func, size=1)
that performs pairs bootstrap and computes a single statistic on pairs samples defined. The statistic of interest
is computed by calling func(bs_x, bs_y). In the next exercise, you will use pearson_r for func.


Instructions
100 XP

    1   Set up an array of indices to sample from. (Remember, when doing pairs bootstrap,
        we randomly choose indices and use those to get the pairs.)

    2   Initialize the array of bootstrap replicates. This should be a one-dimensional array of length size.

    3   Write a for loop to draw the samples.

    4   Randomly choose indices from the array of indices you previously set up.

    5   Extract x values and y values from the input array using the indices you just chose to generate a bootstrap sample.

    6   Use func to compute the statistic of interest from the bootstrap samples of x and y
        and store it in your array of bootstrap replicates.

    7   Return the array of bootstrap replicates.

"""

def draw_bs_pairs(x, y, func, size=1):
    """Perform pairs bootstrap for single statistic."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates