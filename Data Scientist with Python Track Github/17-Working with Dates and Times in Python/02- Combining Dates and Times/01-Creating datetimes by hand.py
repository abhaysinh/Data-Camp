'''

Creating datetimes by hand

Often you create datetime objects based on outside data. Sometimes though, you want to create a datetime object from scratch.

You're going to create a few different datetime objects from scratch to get the hang of that process.
These come from the bikeshare data set that you'll use throughout the rest of the chapter.


Instructions

    1   Import the datetime class.   - 50 XP
        Create a datetime for October 1, 2017 at 15:26:26.
        Print the results in ISO format.

    2   Import the datetime class.    - 50 XP
        Create a datetime for December 31, 2017 at 15:19:13.
        Print the results in ISO format.

    3   Create a new datetime by replacing the year in dt with 1917 (instead of 2017)
'''


# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 10, 1, 15, 26, 26)

# Print the results in ISO 8601 format
print(dt.isoformat())


# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Print the results in ISO 8601 format
print(dt.isoformat())



# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format
print(dt_old)