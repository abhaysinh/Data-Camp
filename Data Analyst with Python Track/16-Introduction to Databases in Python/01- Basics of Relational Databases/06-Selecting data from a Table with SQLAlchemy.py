'''

Selecting data from a Table with SQLAlchemy

Excellent work so far! It's now time to build your first select statement using SQLAlchemy. SQLAlchemy provides a nice "Pythonic" way of interacting with databases. When you used raw SQL in the last exercise, you queried the database directly. When using SQLAlchemy, you will go through a Table object instead, and SQLAlchemy will take case of translating your query to an appropriate SQL statement for you. So rather than dealing with the differences between specific dialects of traditional SQL such as MySQL or PostgreSQL, you can leverage the Pythonic framework of SQLAlchemy to streamline your workflow and more efficiently query your data. For this reason, it is worth learning even if you may already be familiar with traditional SQL.

In this exercise, you'll once again build a statement to query all records from the census table. This time, however, you'll make use of the select() function of the sqlalchemy module. This function requires a list of tables or columns as the only required argument: for example, select([my_table]).

You will also fetch only a few records of the ResultProxy by using .fetchmany() with a size argument specifying the number of records to fetch.

Table and MetaData have already been imported. The metadata is available as metadata and the connection to the database as connection.

Instructions
100 XP

    Import select from the sqlalchemy module.
    Reflect the census table. This code is already written for you.
    Create a query using the select() function to retrieve all the records in the census table. To do so, pass a list to select() containing a single element: census.
    Print stmt to see the actual SQL query being created. This code has been written for you.
    Fetch 10 records from the census table and store then in results. To do this:
        Use the .execute() method on connection with stmt as the argument to retrieve the ResultProxy.
        Use .fetchmany() with the appropriate size argument on connection.execute(stmt) to retrieve the ResultSet.

'''


# Import select
from sqlalchemy import select

# Reflect census table via engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Build select statement for census table: stmt
stmt = select([census])

# Print the emitted statement to see the SQL string
print(stmt)

# Execute the statement on connection and fetch 10 records: results
results = connection.execute(stmt).fetchmany(size=10)

# Execute the statement and print the results
print(results)