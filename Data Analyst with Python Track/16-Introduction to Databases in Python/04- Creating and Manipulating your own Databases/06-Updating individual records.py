'''

Updating individual records

The update statement is very similar to an insert statement. For example, you can update all wages in the employees table as follows:

stmt = update(employees).values(wage=100.00)

The update statement also typically uses a where clause to help us determine what data to update. For example, to only update the record for the employee with ID 15, you would append the previous statement as follows:

stmt = stmt.where(employees.id == 15)

You'll be using the FIPS state code here, which is appropriated by the U.S. government to identify U.S. states and certain other associated areas.

For your convenience, the names of the tables and columns of interest in this exercise are: state_fact (Table), name (Column), and fips_state (Column).

Instructions
100 XP

    1   Build a statement to select all columns from the state_fact table where the value in the name column is 'New York'.
            Call it select_stmt.
        Fetch all the results and assign them to results.
        Print the results and the fips_state column of the first row of the results.

    2   Notice that there is only one record in state_fact for the state of New York. It currently has the FIPS code of 0.

            Build an update statement to change the fips_state column code to 36, save it as update_stmt.
            Use a where clause to filter for states with the name of 'New York' in the state_fact table.
            Execute update_stmt via the connection and save the output as update_results.

    3   Now you will confirm that the record for New York was updated by selecting all the records for New York from
        state_fact and repeating what you did in Step 1.

            Execute select_stmt again, fetch all the results, and assign them to new_results.
                Print the new_results and the fips_state column of the first row of the new_results.

'''



# Build a select statement: select_stmt
select_stmt = select([state_fact]).where(state_fact.columns.name == 'New York')

# Execute select_stmt and fetch the results
results = connection.execute(select_stmt).fetchall()

# Print the results of executing the select_stmt
print(results)

# Print the FIPS code for the first row of the result
print(results[0]['fips_state'])



select_stmt = select([state_fact]).where(state_fact.columns.name == 'New York')
results = connection.execute(select_stmt).fetchall()
print(results)
print(results[0]['fips_state'])

# Build a statement to update the fips_state to 36: update_stmt
update_stmt = update(state_fact).values(fips_state = 36)

# Append a where clause to limit it to records for New York state
update_stmt = update_stmt.where(state_fact.columns.name == 'New York')

# Execute the update statement: update_results
update_results = connection.execute(update_stmt)



select_stmt = select([state_fact]).where(state_fact.columns.name == 'New York')
results = connection.execute(select_stmt).fetchall()
print(results)
print(results[0]['fips_state'])

update_stmt = update(state_fact).values(fips_state = 36)
update_stmt = update_stmt.where(state_fact.columns.name == 'New York')
update_results = connection.execute(update_stmt)

# Execute select_stmt again and fetch the new results
new_results = connection.execute(select_stmt).fetchall()

# Print the new_results
print(new_results)

# Print the FIPS code for the first row of the new_results
print(new_results[0]['fips_state'])