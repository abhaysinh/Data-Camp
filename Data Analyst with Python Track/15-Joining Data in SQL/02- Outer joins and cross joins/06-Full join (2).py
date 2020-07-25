'''

Full join (2)

You'll now investigate a similar exercise to the last one, but this time focused on using a table with more records on the left than the right. You'll work with the languages and countries tables.

Begin with a full join with languages on the left and countries on the right. Appropriate fields have been selected for you again here.

Instructions
100 XP

    1   Choose records in which countries.name starts with the capital letter 'V' or is NULL.
        Arrange by countries.name in ascending order to more clearly see the results.

    2   Repeat the same query as above but use a left join instead of a full join. Note what has changed
        compared to the full join result!

    3   Repeat once more, but use an inner join instead of a left join. Note what has changed compared to the
        full join and left join results.

'''


SELECT countries.name, code, languages.name AS language
-- 3. From languages
FROM languages
  -- 4. Join to countries
  FULL JOIN countries
    -- 5. Match on code
    USING (code)
-- 1. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
-- 2. Order by ascending countries.name
ORDER BY countries.name;



SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  LEFT JOIN countries
    -- 2. Match using code
    USING (code)
-- 3. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;



SELECT countries.name, code, languages.name AS language
FROM languages
  -- 1. Join to countries
  INNER JOIN countries
    USING (code)
-- 2. Where countries.name starts with V or is null
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;