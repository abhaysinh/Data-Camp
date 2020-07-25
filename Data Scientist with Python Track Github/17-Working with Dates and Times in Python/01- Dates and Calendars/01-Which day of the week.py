'''

Which day of the week?

Hurricane Andrew, which hit Florida on August 24, 1992, was one of the costliest and deadliest
 hurricanes in US history. Which day of the week did it make landfall?

Let's walk through all of the steps to figure this out.
Instructions

    1 Import date from datetime. - 35 XP

    2 Create a date object for August 24, 1992. - 35 XP

    3 Now ask Python what day of the week Hurricane Andrew hit    - 30 XP
    (remember that Python counts days of the week starting from Monday as 0, Tuesday as 1, and so on).


'''


# Import date from datetime
from datetime import date

# Import date from datetime
from datetime import date

# Create a date object
hurricane_andrew = date(1992, 8, 24)


# Import date from datetime
from datetime import date

# Create a date object
hurricane_andrew = date(1992, 8, 24)

# Which day of the week is the date?
print(hurricane_andrew.weekday())