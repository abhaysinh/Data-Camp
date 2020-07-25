'''

A table of two cities

This exercise looks to explore languages potentially and most frequently spoken in the cities of Hyderabad, India and Hyderabad, Pakistan.

You will begin with a cross join with cities AS c on the left and languages AS l on the right. Then you will modify the query using an inner join in the next tab.

Instructions
100 XP

    1   Create the cross join as described above. (Recall that cross joins do not use ON or USING.)
        Make use of LIKE and Hyder% to choose Hyderabad in both countries.
        Select only the city name AS city and language name AS language


    2   Use an inner join instead of a cross join. Think about what the difference will be in the results for this
        inner join result and the one for the cross join.


'''


-- 4. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  CROSS JOIN languages AS l
-- 3. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';



-- 5. Select fields
SELECT c.name AS city, l.name AS language
-- 1. From cities (alias as c)
FROM cities AS c
  -- 2. Join to languages (alias as l)
  INNER JOIN languages AS l
    -- 3. Match on country code
    ON c.country_code = l.code
-- 4. Where c.name like Hyderabad
WHERE c.name LIKE 'Hyder%';