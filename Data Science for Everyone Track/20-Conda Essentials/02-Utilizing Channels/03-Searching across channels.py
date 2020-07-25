'''
Searching across channels

Although the conda command and its subcommands are used for nearly everything in this course, the package anaconda-client provides the command anaconda that searches in a different manner that is often more useful. For instance, you may know the name of the textadapter package, but you may not know in which channel (or channels) it may be published (or by which users). You can search across all channels and all platforms using:

$ anaconda search textadapter
Using Anaconda API: https://api.anaconda.org
Packages:
     Name                      |  Version | Package Types   | Platforms       | Builds
     ------------------------- |   ------ | --------------- | --------------- | ----------
     DavidMertz/textadapter    |    2.0.0 | conda           | linux-64, osx-64 | py36_0, py35_0, py27_0
     conda-forge/textadapter   |    2.0.0 | conda           | linux-64, win-32, osx-64, win-64 | py35_0, py27_0
     gbrener/textadapter       |    2.0.0 | conda           | linux-64, osx-64 | py35_0, py27_0
                                          : python interface Amazon S3, and large data files
     sseefeld/textadapter      |    2.0.0 | conda           | win-64          | py36_0, py34_0, py35_0, py27_0
                                          : python interface Amazon S3, and large data files
Found 4 packages


Following this example, use anaconda search to determine the latest available version of the package boltons.
Instructions
50 XP
Possible Answers

    0.2
    15.0.1
    16.4.1
    17.1.0
    19.2.0
    20.0.0

Answer : 20.0.0
'''