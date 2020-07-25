'''
Create an environment from a shared specification

You may recreate an environment from one of the YAML (Yet Another Markup Language) format files produced by conda env export. However, it is also easy to hand write an environment specification with less detail. For example, you might describe an environment you need and want to share with colleagues as follows:

$ cat shared-project.yml
name: shared-project
channels:
  - defaults
dependencies:
  - python=3.6
  - pandas>=0.21
  - scikit-learn
  - statsmodels

Clearly this version is much less specific than what conda env export produces. But it indicates the general tools, with some version specification, that will be required to work on a shared project. Actually creating an environment from this sketched out specification will fill in all the dependencies of those large projects whose packages are named, and this will install dozens of packages not explicitly listed. Often you are happy to have other dependencies in the manner conda decides is best.

Of course, a fully fleshed out specification like we saw in the previous exercise are equally usable. Non-relevant details like the path to the environment on the local system are ignored when installing to a different machine or to a different platform altogether, which will work equally well.

To create an environment from file-name.yml, you can use the following command:

conda env create --file file-name.yml

As a special convention, if you use the plain command conda env create without specifying a YAML file, it will assume you mean the file environment.yml that lives in the local directory.

Instructions
100 XP

    1   A file environment.yml exists in the local directory within the current session. Use this file to create an
        environment called shared-project.

Solution :

conda env create

    2   The current session directory also has a file named shared-config.yml. Create an environment based on this
        specification. The name of this environment will be functional-data.

Solution :

conda env create -f shared-config.yml

'''