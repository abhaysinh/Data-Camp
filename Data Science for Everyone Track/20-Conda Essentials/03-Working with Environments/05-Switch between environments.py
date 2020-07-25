'''

Switch between environments

Simply having different environments is not of much use; you need to be able to switch between environments. Most typically this is done at the command line, using the conda command. With some other interfaces (like Anaconda Navigator or Jupyter with nb_conda installed), other techniques for selecting environment are available. But for this course, you will learn about command-line use.

To activate an environment, you simply use conda activate ENVNAME. To deactivate an environment, you use conda deactivate, which returns you to the root/base environment.

If you used conda outside this course, and prior to version 4.4, you may have seen a more platform specific style. On older versions, Windows users would type activate ENVNAME and deactivate, while Linux and OSX users would type source activate ENVNAME and source deactivate. The unified style across platforms is more friendly. Related to the change to conda activate, version 4.4 and above use a special environment called base that is equivalent to what was called root in older versions. However, in old versions of conda you would not typically see an environment listed on the terminal prompt when you were in the root environment.

Instructions
100 XP

    1   Activate the environment called course-env in the current session.

Solution :

conda activate course-env

    2   Suppose you did some work within the course-env environment. Now you wish to utilize another environment. Activate the environment called pd-2015 in the current session.

Solution :

conda activate pd-2015

    3   Deactivate the current environment you switched in the last step. This will bring you back to the base environment.

Solution :

conda deactivate

'''