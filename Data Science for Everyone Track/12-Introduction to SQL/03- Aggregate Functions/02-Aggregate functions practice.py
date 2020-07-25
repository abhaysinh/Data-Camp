'''
Aggregate functions practice

Good work. Aggregate functions are important to understand, so let's get some more practice!

Instructions
100 XP

    Use the SUM function to get the total amount grossed by all films.

    Get the average amount grossed by all films.

    Get the amount grossed by the worst performing film.

    Get the amount grossed by the best performing film.

'''

SELECT SUM(gross)
FROM films;


SELECT AVG(gross)
FROM films;


SELECT MIN(gross)
FROM films;


SELECT MAX(gross)
FROM films;