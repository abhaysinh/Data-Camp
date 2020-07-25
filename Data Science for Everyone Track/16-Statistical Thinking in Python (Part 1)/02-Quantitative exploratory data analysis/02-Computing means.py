'''
Computing means

The mean of all measurements gives an indication of the typical magnitude of a measurement.
It is computed using np.mean().

Instructions
100xp

    1   Compute the mean petal length of Iris versicolor from Anderson's classic data set.
        The variable versicolor_petal_length is provided in your namespace. Assign the mean to mean_length_vers.

    2   Hit submit to print the result.

'''

# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)

# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')
