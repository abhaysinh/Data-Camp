'''
Inserting a single row

There are several ways to perform an insert with SQLAlchemy; however, we are going to focus on the one that follows the same pattern as the select statement.

It uses an insert statement where you specify the table as an argument, and supply the data you wish to insert into the value via the .values() method as keyword arguments. For example, if my_table contains columns my_col_1 and my_col_2, then insert(my_table).values(my_col_1=5, my_col_2="Example") will create a row in my_table with the value in my_col_1 equal to 5 and value in my_col_2 equal to "Example".

Notice the difference in syntax: when appending a where statement to an existing statement, we include the name of the table as well as the name of the column, for example new_stmt = old_stmt.where(my_tbl.columns.my_col == 15). This is necessary because the existing statement might involve several tables.

On the other hand, you can only insert a record into a single table, so you do not need to include the name of the table when using values() to insert, e.g. stmt = insert(my_table).values(my_col = 10).

Here, the name of the table is data. You can run repr(data) in the console to examine the structure of the table.

Instructions
100 XP

    Import insert and select from the sqlalchemy module.
    Build an insert statement insert_stmt for the data table to set name to 'Anna', count to 1, amount to 1000.00, and valid to True.
    Execute insert_stmt with the connection and store the results.
    Print the .rowcount attribute of results to see how many records were inserted.
    Build a select statement to query data for the record with the name of 'Anna'.
    Hit 'Run Solution' to print the results of executing the select statement.

'''

# Import insert and select from sqlalchemy
from sqlalchemy import insert, select

# Build an insert statement to insert a record into the data table: insert_stmt
insert_stmt = insert(data).values(name='Anna', count=1, amount=1000.00, valid=True)

# Execute the insert statement via the connection: results
results = connection.execute(insert_stmt)

# Print result rowcount
print(results.rowcount)

# Build a select statement to validate the insert: select_stmt
select_stmt = select([data]).where(data.columns.name == 'Anna')

# Print the result of executing the query.
print(connection.execute(select_stmt).first())
