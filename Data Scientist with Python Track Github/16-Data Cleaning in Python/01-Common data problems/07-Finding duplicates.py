'''
Finding duplicates

A new update to the data pipeline feeding into ride_sharing has added the ride_id column,
which represents a unique identifier for each ride.

The update however coincided with radically shorter average ride duration times and irregular user birth dates
set in the future. Most importantly, the number of rides taken has increased by 20% overnight, leading you to think
there might be both complete and incomplete duplicates in the ride_sharing DataFrame.

In this exercise, you will confirm this suspicion by finding those duplicates.
A sample of ride_sharing is in your environment, as well as all the packages you've been working with thus far.


Instructions
100 XP

    Find duplicated rows of ride_id in the ride_sharing DataFrame while setting keep to False.
    Subset ride_sharing on duplicates and sort by ride_id and assign the results to duplicated_rides.
    Print the ride_id, duration and user_birth_year columns of duplicated_rides in that order.

'''




# Find duplicates
duplicates = ride_sharing.duplicated(subset = 'ride_id', keep = False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns
print(duplicated_rides[['ride_id','duration','user_birth_year']])