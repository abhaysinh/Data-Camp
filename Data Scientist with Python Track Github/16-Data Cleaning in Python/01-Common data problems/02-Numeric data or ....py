'''
Numeric data or ... ?

In this exercise, and throughout this chapter, you'll be working with bicycle ride sharing data in
San Francisco called ride_sharing. It contains information on the start and end stations, the trip duration,
and some user information for a bike sharing service.

The user_type column contains information on whether a user is taking a free ride and takes on the following values:

    1 for free riders.
    2 for pay per ride.
    3 for monthly subscribers.

In this instance, you will print the information of ride_sharing using .info() and see a firsthand example
of how an incorrect data type can flaw your analysis of the dataset. The pandas package is imported as pd.

Instructions

    1. Print the information of ride_sharing.
    Use .describe() to print the summary statistics of the user_type column from ride_sharing. - 35 XP

    2 Question   - 35 XP

    By looking at the summary statistics - they don't really seem to offer much description on how users
    are distributed along their purchase type, why do you think that is?

    Possible Answers

    The user_type column is not of the correct type, it should be converted to str.
    The user_type column has an infinite set of possible values, it should be converted to category.
    The user_type column has an finite set of possible values that represent groupings of data, it should be
    converted to category.

Answer : The user_type column has an finite set of possible values that represent groupings of data, it should
be converted to category.


    3 Convert user_type into categorical by assigning it the 'category' data type and store it in the user_type_cat column.
     Make sure you converted user_type_cat correctly by using an assert statement.   - 30 XP
'''

# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())



# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

# Convert user_type from integer to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')

# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'

# Print new summary statistics
print(ride_sharing['user_type_cat'].describe())