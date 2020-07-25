'''
Compatibility with different versions

A common case for using environments is in developing scripts or Jupyter notebooks that rely on particular software versions for their functionality. Over time, the underlying tools might change, making updating the scripts worthwhile. Being able to switch between environments with different versions of the underlying packages installed makes this development process much easier.

In this chapter, you will walk through steps of doing this with a very simple script as an example.

Instructions
100 XP

    1   The file weekly_humidity.py is stored in the current session. First just take a look at it using the Unix tool cat.
        You will see that the purpose of this script is rather trivial: it shows the last few days of the rolling mean of
        humidity taken from a data file. It would be easy to generalize this with switches to show different periods,
        different rolling intervals, different data sources, etc.

Solution :

cat weekly_humidity.py

    2   Run the script (in the current base environment).

Solution :

python weekly_humidity.py

    3   The script ran and produced a little report of the rolling mean of humidity. However, it also produced some
        rather noisy complaints about deprecated syntax in the Pandas library (called a FutureWarning).
        You now remember that you created this script a few years ago when you were using the pd-2015 environment.
        Switch to that environment.

Solution :

conda activate pd-2015

    4   Run the script in the current pd-2015 environment. You will notice that the report itself is the same,
        but the FutureWarning is not present. For a first step, this is how to utilize this script.

Solution : 

python weekly_humidity.py

'''