'''
Updating a script

You certainly have the easy option to continue using the pd-2015 environment whenever you need to run the weekly_humidity.py script. Environments can be kept around as long as you like and will assure that all your old scripts (and notebooks, libraries, etc) continue to run the same way they always have.

But quite likely you would like to update your script for your colleagues who use more recent versions of Python. Ideally, you would like them not to have to worry about this message:

FutureWarning: pd.rolling_mean is deprecated for Series and will be removed in a future version, replace with
     Series.rolling(window=7,center=False).mean()
  print(pd.rolling_mean(humidity, 7).tail(5))

Happily, the warning itself pretty much tells you exactly how to update your script.

Instructions
100 XP

    1   Use the nano text editor to modify weekly_humidity.py so the last line is changed to:

            print(humidity.rolling(7).mean().tail(5))

        If you are more familiar with them, the editors vim and emacs are also installed in this session.

Solution :

# You almost surely want to use a text editor like 'nano' rather than the StreamEDitor yourself:
sed -i '$ d' weekly_humidity.py && echo 'print(humidity.rolling(7).mean().tail(5))' >> weekly_humidity.py


    2   Run the modified script in the active base environment that contains Panda 0.22. The FutureWarning should be gone now.

Solution :

python weekly_humidity.py

    3   APIs do change over time. You should check whether your script, as modified, works in the older Pandas 0.17
        installed in pd-2015. So, switch to the pd-2015 environment.

Solution :

conda activate pd-2015

    4   Now, run the script in the current pd-2015 environment. You will notice that a new failure mode occurs now
        because Pandas 0.17 does not support the newer API you have used in your modified script.

Solution : 

python weekly_humidity.py

'''