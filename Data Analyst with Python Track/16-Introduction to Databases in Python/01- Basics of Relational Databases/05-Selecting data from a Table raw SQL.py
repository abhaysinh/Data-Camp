'''

Selecting data from a Table: raw SQL

As you have seen in the video, to access and manipulate the data in the database, we will first need to establish a connection to it by using the .connect() method on the engine. This is because the create_engine() function that you have used before returns an instance of an engine, but it does not actually open a connection until an action is called that would require a connection, such as a query.

Using what we just learned about SQL and applying the .execute() method on our connection, we can leverage a raw SQL query to query all the records in our census table. The object returned by the .execute() method is a ResultProxy. On this ResultProxy, we can then use the .fetchall() method to get our results - that is, the ResultSet.

In this exercise, you'll use a traditional SQL query. Notice that when you execute a query using raw SQL, you will query the table in the database directly. In particular, no reflection step is needed.

In the next exercise, you'll move to SQLAlchemy and begin to understand its advantages. Go for it!

Instructions
100 XP

    Use the .connect() method of engine to create a connection.
    Build a SQL statement to query all the columns from census and store it in stmt. Note that your SQL statement must be a string.
    Use the .execute() and .fetchall() methods on connection and store the result in results. Remember that .execute() comes before .fetchall() and that stmt needs to be passed to .execute().
    Print results.

'''


from sqlalchemy import create_engine
engine = create_engine('sqlite:///census.sqlite')

# Create a connection on engine
connection = engine.connect()

# Build select statement for census table: stmt
stmt = 'SELECT * FROM census'

# Execute the statement and fetch the results: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)