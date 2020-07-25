'''
Visualizing bootstrap samples

In this exercise, you will generate bootstrap samples from the set of annual rainfall data measured at
the Sheffield Weather Station in the UK from 1883 to 2015. The data are stored in the NumPy array
rainfall in units of millimeters (mm). By graphically displaying the bootstrap samples with an ECDF,
you can get a feel for how bootstrap sampling allows probabilistic descriptions of data.

INSTRUCTIONS
100XP

    1   Write a for loop to acquire 50 bootstrap samples of the rainfall data and plot their ECDF.
        -Use np.random.choice() to generate a bootstrap sample from the NumPy array rainfall. Be sure that
            the size of the resampled array is len(rainfall).
        -Use the function ecdf() that you wrote in the prequel to this course to generate the x and y values
            for the ECDF of the bootstrap sample bs_sample.
        -Plot the ECDF values. Specify color='gray' (to make gray dots) and alpha=0.1 (to make them
            semi-transparent, since we are overlaying so many) in addition to the marker='.' and linestyle='none'
            keyword arguments.

    2   Use ecdf() to generate x and y values for the ECDF of the original rainfall data available in the
        array rainfall.

    3   Plot the ECDF values of the original data.
    4   Hit 'Submit Answer' to visualize the samples!

'''


for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()
