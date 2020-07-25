'''
Summing strings and concatenating numbers

In the previous exercise, you were able to identify that category is the correct data type for
user_type and convert it in order to extract relevant statistical summaries that shed light on the distribution of user_type.

Another common data type problem is importing what should be numerical values as strings, as mathematical
operations such as summing and multiplication lead to string concatenation, not numerical outputs.

In this exercise, you'll be converting the string column duration to the type int. Before that however,
you will need to make sure to strip "minutes" from the column in order to make sure pandas reads it as numerical.


The pandas package has been imported as pd.
Instructions
100 XP

    Use the .strip() method to strip duration of "minutes" and store it in the duration_trim column.
    Convert duration_trim to int and store it in the duration_time column.
    Write an assert statement that checks if duration_time's data type is now an int.
    Print the average ride duration.

'''


# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')

# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')

# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'

# Print formed columns and calculate average ride duration
print(ride_sharing[['duration','duration_trim','duration_time']])
print(ride_sharing['duration_time'].mean())