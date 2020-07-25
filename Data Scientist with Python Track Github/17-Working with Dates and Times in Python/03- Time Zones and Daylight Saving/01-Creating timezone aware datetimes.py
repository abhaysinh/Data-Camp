'''

Creating timezone aware datetimes

In this exercise, you will practice setting timezones manually.
Instructions 1/3


    1   Import timezone.   -35 XP
        Set the tzinfo to UTC, without using timedelta.

    2   Set pst to be a timezone set for UTC-8.   -35 XP
        Set dt's timezone to be pst.

    3   Set tz to be a timezone set for UTC+11.   -30 XP
        Set dt's timezone to be tz.

'''

# Import datetime, timezone
from datetime import datetime, timezone

# October 1, 2017 at 15:26:26, UTC
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=timezone.utc)

# Print results
print(dt.isoformat())



# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Pacific Standard Time, or UTC-8
pst = timezone(timedelta(hours=-8))

# October 1, 2017 at 15:26:26, UTC-8
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=pst)

# Print results
print(dt.isoformat())


# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Australian Eastern Daylight Time, or UTC+11
aedt = timezone(timedelta(hours=11))

# October 1, 2017 at 15:26:26, UTC+11
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=aedt)

# Print results
print(dt.isoformat())