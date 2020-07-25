'''

Finding consistency

In this exercise and throughout this chapter, you'll be working with the airlines DataFrame which contains
survey responses on the San Francisco Airport from airline customers.

The DataFrame contains flight metadata such as the airline, the destination, waiting times as well as answers
to key questions regarding cleanliness, safety, and satisfaction. Another DataFrame named categories was created,
containing all correct possible values for the survey columns.

In this exercise, you will use both of these DataFrames to find survey answers with inconsistent values,
 and drop them, effectively performing an outer and inner join on both these DataFrames as seen in the video exercise.
 The pandas package has been imported as pd, and the airlines and categories DataFrames are in your environment.


Instructions

    1 Print the categories DataFrame and take a close look at all possible correct categories of the survey columns.
      Print the unique values of the survey columns in airlines using the .unique() method.    - 35 XP

    2 Question   - 15 XP

    Take a look at the output. Out of the cleanliness, safety and satisfaction columns, which one has an
    inconsistent category and what is it?


    Possible Answers

    cleanliness because it has an Unacceptable category.
    cleanliness because it has a Terribly dirty category.
    satisfaction because it has a Very satisfied category.
    safety because it has a Neutral category.

Answer : cleanliness because it has an Unacceptable category.

    3 Create a set out of the cleanliness column in airlines using set() and find the inconsistent category by finding the difference in the cleanliness column of categories.
      Find rows of airlines with a cleanliness value not in categories and print the output.    - 25 XP

    4 Print the rows with the consistent categories of cleanliness only.    - 25 XP

'''



# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(),"\n")


# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])


# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Print rows with consistent categories only
print(airlines[~cat_clean_rows])