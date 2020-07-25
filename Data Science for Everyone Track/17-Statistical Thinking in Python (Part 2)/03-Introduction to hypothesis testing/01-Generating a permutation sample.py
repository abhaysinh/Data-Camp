'''
Generating a permutation sample

In the video, you learned that permutation sampling is a great way to simulate the hypothesis that two
variables have identical probability distributions. This is often a hypothesis you want to test, so in this
exercise, you will write a function to generate a permutation sample from two data sets.

Remember, a permutation sample of two arrays having respectively n1 and n2 entries is constructed by
concatenating the arrays together, scrambling the contents of the concatenated array, and then taking the
first n1 entries as the permutation sample of the first array and the last n2 entries as the permutation
sample of the second array.

Instructions
100 XP

    1   Concatenate the two input arrays into one using np.concatenate(). Be sure to pass in data1 and data2
        as one argument (data1, data2).

    2   Use np.random.permutation() to permute the concatenated array.
    3   Store the first len(data1) entries of permuted_data as perm_sample_1 and the last len(data2) entries
        of permuted_data as perm_sample_2. In practice, this can be achieved by using :len(data1) and len(data1):
        to slice permuted_data.

    4   Return perm_sample_1 and perm_sample_2.

'''

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2
