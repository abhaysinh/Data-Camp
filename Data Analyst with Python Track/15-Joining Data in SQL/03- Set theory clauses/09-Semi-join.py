'''

Semi-join

You are now going to use the concept of a semi-join to identify languages spoken in the Middle East.

Instructions
100 XP

    1   Flash back to our Intro to SQL for Data Science course and begin by selecting all country codes in the
        Middle East as a single field result using SELECT, FROM, and WHERE.

    2   Comment out the answer to the previous tab by surrounding it in /* and */. You'll come back to it!
        Below the commented code, select only unique languages by name appearing in the languages table.
        Order the resulting single field table by name in ascending order.

    3   Now combine the previous two queries into one query:

        Add a WHERE IN statement to the SELECT DISTINCT query, and use the commented out query from the
        first instruction in there. That way, you can determine the unique languages spoken in the Middle East.

    Carefully review this result and its code after completing it. It serves as a great example of subqueries,
    which are the focus of Chapter 4.


'''


-- Select code
SELECT code
  -- From countries
  FROM countries
-- Where region is Middle East
WHERE region = 'Middle East';



/*
SELECT code
  FROM countries
WHERE region = 'Middle East';
*/

-- Select field
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Order by name
ORDER BY name;



-- Select distinct fields
SELECT DISTINCT name
  -- From languages
  FROM languages
-- Where in statement
WHERE code IN
  -- Subquery
  (SELECT code
   FROM countries
   WHERE region = 'Middle East')
-- Order by name
ORDER BY name;