'''

Filter data selected from a Table - Expressions

In addition to standard Python comparators, we can also use methods such as in_() to create more powerful where() clauses. You can see a full list of expressions in the SQLAlchemy Documentation.

Method in_(), when used on a column, allows us to include records where the value of a column is among a list of possible values. For example, where(census.columns.age.in_([20, 30, 40])) will return only records for people who are exactly 20, 30, or 40 years old.

In this exercise, you will continue working with the census table, and select the records for people from the three most densely populated states. The list of those states has already been created for you.

Instructions
100 XP

    Select all records from the census table.
    Modify the argument of the where clause to use in_() to return all the records where the value in the census.columns.state column is in the states list.
    Loop over the ResultProxy connection.execute(stmt) and print the state and pop2000 columns from each record.

'''

# Define a list of states for which we want results
states = ['New York', 'California', 'Texas']

# Create a query for the census table: stmt
stmt = select([census])

# Append a where clause to match all the states in_ the list states
stmt = stmt.where(census.columns.state.in_(states))

# Loop over the ResultProxy and print the state and its population in 2000
for result in connection.execute(stmt):
    print(result.state, result.pop2000)
