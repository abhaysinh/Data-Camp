'''
Create a new environment

This course is configured with several environments, but in your use you will need to create environments meeting your own purposes. The basic command for creating environments is conda create. You will always need to specify a name for your environment, using --name (or short form -n), and you may optionally specify packages (with optional versions) that you want in that environment initially. You do not need to specify any packages when creating; either way you can add or remove whatever packages you wish from an environment later.

The general syntax is similar to:

conda create --name recent-pd python=3.6 pandas=0.22 scipy statsmodels

This command will perform consistency resolution on those packages and versions indicated, in the same manner as a conda install will. Notice that even though this command works with environments it is conda create rather than a conda env ... command.

Instructions
100 XP

    1   Create a new environment called conda-essentials that contains attrs version 19.1.0 and the best available version of cytoolz (we pick these examples for illustration largely because they are small and have few dependencies).

Solution :

conda create --name conda-essentials attrs=19.1.0 cytoolz -y

    2   Switch into the environment you just created named conda-essentials.

Solution :

conda activate conda-essentials

    3   Examine all the software packages installed in the current conda-essentials environment.

Solution :  

conda list

'''