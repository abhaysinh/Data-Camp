'''
Determine the average age by population

As Jason discussed in the video, to calculate a weighted average, we first find the total sum of weights multiplied by the values we're averaging, then divide by the sum of all the weights.

For example, if we wanted to find a weighted average of data = [10, 30, 50] weighted by weights = [2,4,6], we would compute 2⋅10+4⋅30+6⋅502+4+6

, or sum(weights * data) / sum(weights).

In this exercise, however, you will make use of func.sum() together with select to select the weighted average of a column from a table. You will still work with the census data, and you will compute the average of age weighted by state population in the year 2000, and then group this weighted average by sex.
Instructions
100 XP

    1   Import select and func from sqlalchemy.
        Write a statement to select the average of age (age) weighted by population in 2000 (pop2000) from census.

    2   Modify the select statement to alias the new column with weighted average as 'average_age' using .label().

    3   Modify the select statement to select the sex column of census in addition to the weighted average,
            with the sex column coming first.
        Group by the sex column of census.


    4   Execute the statement on the connection (which has been created for you) and fetch all the results.
        Loop over the results and print the values in the sex and average_age columns for each record in the results.

'''


# Import select and func
from sqlalchemy import select, func

# Select the average of age weighted by pop2000
stmt = select([func.sum(census.columns.pop2000 * census.columns.age)
  					/ func.sum(census.columns.pop2000)
			  ])




# Import select and func
from sqlalchemy import select, func

# Relabel the new column as average_age
stmt = select([(func.sum(census.columns.pop2000 * census.columns.age)
  					/ func.sum(census.columns.pop2000)).label('average_age')
			  ])



# Import select and func
from sqlalchemy import select, func

# Add the sex column to the select statement
stmt = select([census.columns.sex,
  			   (func.sum(census.columns.pop2000 * census.columns.age)
  					/ func.sum(census.columns.pop2000)).label('average_age')
			  ])

# Group by sex
stmt = stmt.group_by(census.columns.sex)




# Import select and func
from sqlalchemy import select, func

# Select sex and average age weighted by 2000 population
stmt = select([(func.sum(census.columns.pop2000 * census.columns.age)
  					/ func.sum(census.columns.pop2000)).label('average_age'),
               census.columns.sex
			  ])

# Group by sex
stmt = stmt.group_by(census.columns.sex)

# Execute the query and fetch all the results
results = connection.execute(stmt).fetchall()

# Print the sex and average age column for each result
for result in results:
    print(result.sex, result.average_age)