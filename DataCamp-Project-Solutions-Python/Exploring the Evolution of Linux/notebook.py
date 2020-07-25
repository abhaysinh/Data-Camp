
# coding: utf-8

# ## 1. Introduction
# <p><a href="https://commons.wikimedia.org/wiki/File:Tux.svg">
# <img style="float: right;margin:5px 20px 5px 1px" width="150px" src="https://assets.datacamp.com/production/project_111/img/tux.png" alt="Tux - the Linux mascot">
# </a></p>
# <p>Version control repositories like CVS, Subversion or Git can be a real gold mine for software developers. They contain every change to the source code including the date (the "when"), the responsible developer (the "who"), as well as a little message that describes the intention (the "what") of a change.</p>
# <p>In this notebook, we will analyze the evolution of a very famous open-source project &ndash; the Linux kernel. The Linux kernel is the heart of some Linux distributions like Debian, Ubuntu or CentOS. Our dataset at hand contains the history of kernel development of almost 13 years (early 2005 - late 2017). We get some insights into the work of the development efforts by </p>
# <ul>
# <li>identifying the TOP 10 contributors and</li>
# <li>visualizing the commits over the years.</li>
# </ul>

# In[2]:


# Printing the content of git_log_excerpt.csv
print(open('datasets/git_log_excerpt.csv'))


# In[3]:


get_ipython().run_cell_magic('nose', '', '\ndef test_listing_of_file_contents():\n    \n    # FIXME1: if student executes cell more than once, variable _i2 is then not defined. Solution?\n    \n    #PATH = "datasets/git_log_excerpt.csv"\n    # hard coded cell number: maybe a little bit fragile\n    #cell_input_from_sample_code = _i2\n    #assert PATH in cell_input_from_sample_code, \\\n    #"The file %s should be read in." % PATH\n    \n    # FIXME2: can\'t access the sample code cell\'s output here because of the use of \'print\'\n    \n    # test currently deactivated: too hard to create a table test case\n    assert True')


# ## 2. Reading in the dataset
# <p>The dataset was created by using the command <code>git log --encoding=latin-1 --pretty="%at#%aN"</code> in late 2017. The <code>latin-1</code> encoded text output was saved in a header-less CSV file. In this file, each row is a commit entry with the following information:</p>
# <ul>
# <li><code>timestamp</code>: the time of the commit as a UNIX timestamp in seconds since 1970-01-01 00:00:00 (Git log placeholder "<code>%at</code>")</li>
# <li><code>author</code>: the name of the author that performed the commit (Git log placeholder "<code>%aN</code>")</li>
# </ul>
# <p>The columns are separated by the number sign <code>#</code>. The complete dataset is in the <code>datasets/</code> directory. It is a <code>gz</code>-compressed csv file named <code>git_log.gz</code>.</p>

# In[4]:


# Loading in the pandas module as 'pd'
import pandas as pd

# Reading in the log file
git_log = pd.read_csv(
    'datasets/git_log.gz',
    sep='#',
    encoding='latin-1',
    header=None,
    names=['timestamp', 'author']
)

# Printing out the first 5 rows
git_log.head(5)


# In[5]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_is_pandas_loaded_as_pd():\n    \n    try:\n        pd # throws NameError\n        pd.DataFrame # throws AttributeError\n    except NameError:\n        assert False, "Module pandas not loaded as pd."\n    except AttributeError:\n        assert False, "Variable pd is used as short name for another module."\n    \n    \ndef test_is_git_log_data_frame_existing():\n    \n    try:\n        # checks implicitly if git_log by catching the NameError exception\n        assert isinstance(git_log, pd.DataFrame), "git_log isn\'t a DataFrame."\n              \n    except NameError as e:\n        assert False, "Variable git_log doesn\'t exist."\n\n\ndef test_has_git_log_correct_columns():\n    \n    expected = [\'timestamp\', \'author\']\n    assert all(git_log.columns.get_values() == expected), \\\n        "Expected columns are %s" % expected\n        \n\ndef test_is_logfile_content_read_in_correctly():\n    \n    correct_git_log = pd.read_csv(\n        \'datasets/git_log.gz\',\n        sep=\'#\',\n        encoding=\'latin-1\',\n        header=None,\n        names=[\'timestamp\', \'author\'])\n    \n    assert correct_git_log.equals(git_log), \\\n        "The content of datasets/git_log.gz wasn\'t correctly read into git_log. Check the parameters of read_csv."')


# ## 3. Getting an overview
# <p>The dataset contains the information about every single code contribution (a "commit") to the Linux kernel over the last 13 years. We'll first take a look at the number of authors and their commits to the repository.</p>

# In[6]:


# calculating number of commits
number_of_commits = git_log['timestamp'].count()

# calculating number of authors
number_of_authors = git_log['author'].value_counts(dropna=True).count()

# printing out the results
print("%s authors committed %s code changes." % (number_of_authors, number_of_commits))


# In[7]:


get_ipython().run_cell_magic('nose', '', '\ndef test_basic_statistics():\n    assert number_of_commits == len(git_log), \\\n    "The number of commits should be right."\n    assert number_of_authors == len(git_log[\'author\'].dropna().unique()), \\\n    "The number of authors should be right."')


# ## 4. Finding the TOP 10 contributors
# <p>There are some very important people that changed the Linux kernel very often. To see if there are any bottlenecks, we take a look at the TOP 10 authors with the most commits.</p>

# In[8]:


# Identifying the top 10 authors
top_10_authors = git_log['author'].value_counts().head(10)

# Listing contents of 'top_10_authors'
top_10_authors


# In[9]:


get_ipython().run_cell_magic('nose', '', '\n\ndef test_is_series_or_data_frame():\n    \n    assert isinstance(top_10_authors, pd.Series) or isinstance(top_10_authors, pd.DataFrame), \\\n    "top_10_authors isn\'t a Series or DataFrame, but of type %s." % type(top_10_authors)\n\n    \ndef test_is_result_structurally_alright():\n    \n    top10 = top_10_authors.squeeze()\n    # after a squeeze(), the DataFrame with one Series should be converted to a Series\n    assert isinstance(top10, pd.Series), \\\n    "top_10_authors should only contain the data for authors and the number of commits."\n    \n\ndef test_is_right_number_of_entries():\n    \n    expected_number_of_entries = 10\n    assert len(top_10_authors.squeeze()) is expected_number_of_entries, \\\n    "The number of TOP 10 entries should be %r. Be sure to store the result into the \'top_10_authors\' variable." % expected_number_of_entries \n    \n    \ndef test_is_expected_top_author():\n    \n    expected_top_author = "Linus Torvalds"\n    assert top_10_authors.squeeze().index[0] == expected_top_author, \\\n    "The number one contributor should be %s." % expected_top_author\n    \n    \ndef test_is_expected_top_commits():    \n    expected_top_commits = 23361\n    assert top_10_authors.squeeze()[0] == expected_top_commits, \\\n    "The number of the most commits should be %r." % expected_top_commits')


# ## 5. Wrangling the data
# <p>For our analysis, we want to visualize the contributions over time. For this, we use the information in the <code>timestamp</code> column to create a time series-based column.</p>

# In[10]:


# converting the timestamp column
git_log['timestamp'] = pd.to_datetime(git_log['timestamp'], unit='s')

# summarizing the converted timestamp column
git_log['timestamp'].describe()


# In[11]:


get_ipython().run_cell_magic('nose', '', "\ndef test_timestamps():\n    \n    START_DATE = '1970-01-01 00:00:01'\n    assert START_DATE in str(git_log['timestamp'].min()), \\\n    'The first timestamp should be %s.' % START_DATE\n    \n    END_DATE = '2037-04-25 08:08:26'\n    assert END_DATE in str(git_log['timestamp'].max()), \\\n    'The last timestamp should be %s.' % END_DATE")


# ## 6. Treating wrong timestamps
# <p>As we can see from the results above, some contributors had their operating system's time incorrectly set when they committed to the repository. We'll clean up the <code>timestamp</code> column by dropping the rows with the incorrect timestamps.</p>

# In[12]:


# determining the first real commit timestamp
first_commit_timestamp = git_log['timestamp'].iloc[-1]

# determining the last sensible commit timestamp
last_commit_timestamp = pd.to_datetime('today')

# filtering out wrong timestamps
corrected_log = git_log[(git_log['timestamp']>=first_commit_timestamp)&(git_log['timestamp']<=last_commit_timestamp)]

# summarizing the corrected timestamp column
corrected_log['timestamp'].describe()


# In[13]:


get_ipython().run_cell_magic('nose', '', "\ndef test_corrected_timestamps():\n    \n    FIRST_REAL_COMMIT = '2005-04-16 22:20:36'\n    assert FIRST_REAL_COMMIT in str(corrected_log['timestamp'].min()), \\\n    'The first real commit timestamp should be %s.' % FIRST_REAL_COMMIT\n    \n    LAST_REAL_COMMIT = '2017-10-03 12:57:00'\n    assert LAST_REAL_COMMIT in str(corrected_log['timestamp'].max()), \\\n    'The last real commit timestamp should be %s.' % LAST_REAL_COMMIT")


# ## 7. Grouping commits per year
# <p>To find out how the development activity has increased over time, we'll group the commits by year and count them up.</p>

# In[14]:


# Counting the no. commits per year
commits_per_year = corrected_log.groupby(
    pd.Grouper(
        key='timestamp', 
        freq='AS'
        )
    ).count()

# Listing the first rows
commits_per_year.head()


# In[15]:


get_ipython().run_cell_magic('nose', '', "\ndef test_number_of_commits_per_year():\n    \n    YEARS = 13\n    assert len(commits_per_year) == YEARS, \\\n    'Number of years should be %s.' % YEARS\n    \n    \ndef test_new_beginning_of_git_log():\n    \n    START = '2005-01-01 00:00:00'\n    assert START in str(commits_per_year.index[0]), \\\n    'DataFrame should start at %s' % START")


# ## 8. Visualizing the history of Linux
# <p>Finally, we'll make a plot out of these counts to better see how the development effort on Linux has increased over the the last few years. </p>

# In[16]:


# Setting up plotting in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the data
commits_per_year.plot(kind='line', title='Development effort on Linux', legend=False)


# In[17]:


get_ipython().run_cell_magic('nose', '', '\ndef test_call_to_plot():\n    \n    # FIXME: Different results local and on build server.\n    # - local (expected): AssertionError: Plot type should be a bar chart.\n    # - build server: NameError: name \'_i20\' is not defined\n    # deactivating tests\n    \n    #assert "kind=\'bar\'" in _i20, "Plot type should be a bar chart."\n    \n    # test currently deactivated: too hard to create a table test case\n    assert True')


# ## 9.  Conclusion
# <p>Thanks to the solid foundation and caretaking of Linux Torvalds, many other developers are now able to contribute to the Linux kernel as well. There is no decrease of development activity at sight!</p>

# In[18]:


# calculating or setting the year with the most commits to Linux
year_with_most_commits = 2016


# In[19]:


get_ipython().run_cell_magic('nose', '', '\ndef test_year_with_most_commits():\n    assert str(year_with_most_commits).endswith("16") , \\\n        "Write the year with the most commits as 20??, but with ?? replaced."')

