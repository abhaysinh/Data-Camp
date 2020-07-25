'''

The cutoff point

In this exercise, and throughout this chapter, you'll be working with the restaurants DataFrame which has data
on various restaurants. Your ultimate goal is to create a restaurant recommendation engine, but you need to
first clean your data.

This version of restaurants has been collected from many sources, where the cuisine_type column is riddled with typos,
and should contain only italian, american and asian cuisine types. There are so many unique categories that remapping
them manually isn't scalable, and it's best to use string similarity instead.

Before doing so, you want to establish the cutoff point for the similarity score using the fuzzywuzzy's
process.extract() function by finding the similarity score of the most distant typo of each category.

Instructions

    1 Import process from fuzzywuzzy.                 - 50 XP
     Store the unique cuisine_types into unique_types.
     Calculate the similarity of 'asian', 'american', and 'italian' to all possible cuisine_types using
     process.extract(), while returning all possible matches.

    2 Question                 - 50 XP

    Take a look at the output, what do you think should be the similarity cutoff point when remapping categories?

    Possible Answers

    80
    70
    60

Answer : 80
'''


# Import process from fuzzywuzzy
from fuzzywuzzy import process

# Store the unique values of cuisine_type in unique_types
unique_types = restaurants['cuisine_type'].unique()

# Calculate similarity of 'asian' to all values of unique_types
print(process.extract('asian', unique_types, limit = len(unique_types)))

# Calculate similarity of 'american' to all values of unique_types
print(process.extract('american', unique_types, limit = len(unique_types)))

# Calculate similarity of 'italian' to all values of unique_types
print(process.extract('italian', unique_types, limit = len(unique_types)))