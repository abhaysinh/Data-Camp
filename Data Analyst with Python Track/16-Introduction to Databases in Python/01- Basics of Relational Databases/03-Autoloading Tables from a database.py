'''
Autoloading Tables from a database

SQLAlchemy can be used to automatically load tables from a database using something called reflection. Reflection is the process of reading the database and building the metadata based on that information. It's the opposite of creating a Table by hand and is very useful for working with existing databases.

To perform reflection, you will first need to import and initialize a MetaData object. MetaData objects contain information about tables stored in a database. During reflection, the MetaData object will be populated with information about the reflected table automatically, so we only need to initialize it before reflecting by calling MetaData().

You will also need to import the Table object from the SQLAlchemy package. Then, you use this Table object to read your table from the engine, autoload the columns, and populate the metadata. This can be done with a single call to Table(): using the Table object in this manner is a lot like passing arguments to a function. For example, to autoload the columns with the engine, you have to specify the keyword arguments autoload=True and autoload_with=engine to Table().

Finally, to view information about the object you just created, you will use the repr() function. For any Python object, repr() returns a text representation of that object. For SQLAlchemy Table objects, it will return the information about that table contained in the metadata.

In this exercise, your job is to reflect the "census" table available on your engine into a variable called census. We already pre-filled the code to create the engine that you wrote in the previous exercise.

Instructions
100 XP

    Import the Table and MetaData from sqlalchemy.
    Create a MetaData object: metadata
    Reflect the census table by using the Table object with the arguments:
        The name of the table as a string ('census').
        The metadata you just initialized.
        autoload=True
        The engine to autoload with - in this case, engine.
    Print the details of census using the repr() function.

'''

# Import create_engine, MetaData, and Table
from sqlalchemy import create_engine, MetaData, Table

# Create engine: engine
engine = create_engine('sqlite:///census.sqlite')

# Create a metadata object: metadata
metadata = MetaData()

# Reflect census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)

# Print census table metadata
print(repr(census))

