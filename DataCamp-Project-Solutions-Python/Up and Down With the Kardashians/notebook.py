#!/usr/bin/env python
# coding: utf-8

# ## 1. The sisters and Google Trends
# <p>While I'm not a fan nor a hater of the Kardashians and Jenners, the polarizing family intrigues me. Why? Their marketing prowess. Say what you will about them and what they stand for, they are great at the hype game. Everything they touch turns to content.</p>
# <p>The sisters in particular over the past decade have been especially productive in this regard. Let's get some facts straight. I consider the "sisters" to be the following daughters of Kris Jenner. Three from her first marriage to lawyer <a href="https://en.wikipedia.org/wiki/Robert_Kardashian">Robert Kardashian</a>:</p>
# <ul>
# <li><a href="https://en.wikipedia.org/wiki/Kourtney_Kardashian">Kourtney Kardashian</a> (daughter of Robert Kardashian, born in 1979)</li>
# <li><a href="https://en.wikipedia.org/wiki/Kim_Kardashian">Kim Kardashian</a> (daughter of Robert Kardashian, born in 1980)</li>
# <li><a href="https://en.wikipedia.org/wiki/Khlo%C3%A9_Kardashian">Khloé Kardashian</a> (daughter of Robert Kardashian, born in 1984)</li>
# </ul>
# <p>And two from her second marriage to Olympic gold medal-winning decathlete, <a href="https://en.wikipedia.org/wiki/Caitlyn_Jenner">Caitlyn Jenner</a> (formerly Bruce):</p>
# <ul>
# <li><a href="https://en.wikipedia.org/wiki/Kendall_Jenner">Kendall Jenner</a> (daughter of Caitlyn Jenner, born in 1995)</li>
# <li><a href="https://en.wikipedia.org/wiki/Kylie_Jenner">Kylie Jenner</a> (daughter of Caitlyn Jenner, born in 1997)</li>
# </ul>
# <p><img src="https://assets.datacamp.com/production/project_538/img/kardashian_jenner_family_tree.png" alt="Kardashian Jenner sisters family tree"></p>
# <p>This family tree can be confusing, but we aren't here to explain it. We're here to explore the data underneath the hype, and we'll do it using search interest data from Google Trends. We'll recreate the Google Trends plot to visualize their ups and downs over time, then make a few custom plots of our own. And we'll answer the big question: <strong>is Kim even the most famous sister anymore?</strong></p>
# <p>First, let's load and inspect our Google Trends data, which was downloaded in CSV form. The <a href="https://trends.google.com/trends/explore?date=2007-01-01%202019-03-21&q=%2Fm%2F0261x8t,%2Fm%2F043p2f2,%2Fm%2F043ttm7,%2Fm%2F05_5_yx,%2Fm%2F05_5_yh">query</a> parameters: each of the sisters, worldwide search data, 2007 to present day. (2007 was the year Kim became "active" according to Wikipedia.)</p>

# In[2]:


# Load pandas
import pandas as pd

# Read in dataset
trends = pd.read_csv("datasets/trends_kj_sisters.csv")

# Inspect data
trends.head()


# In[3]:


get_ipython().run_cell_magic('nose', '', '\n# One or more tests of the students code.\n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_output = _\n\ndef test_pandas_loaded():\n    assert \'pd\' in globals(), \\\n    \'Did you import the pandas library aliased as pd?\'\n    \ndef test_trends_correctly_loaded():\n    correct_trends = pd.read_csv(\'datasets/trends_kj_sisters.csv\')\n    assert correct_trends.equals(trends), "The variable trends should contain the data in trends_kj_sisters.csv."\n\ndef test_head_output():\n    try:\n        assert "2007-05" in last_output.to_string()\n    except AttributeError:\n        assert False, "Please use trends.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, "The output of the cell is not what we expected. You should see \'2007-05\' in the fifth row of the trends DataFrame."')


# ## 2. Better "kolumn" names
# <p>So we have a column for each month since January 2007 and a column for the worldwide search interest for each of the sisters each month. By the way, Google defines the values of search interest as:</p>
# <blockquote>
#   <p>Numbers represent search interest relative to the highest point on the chart for the given region and time. A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular. A score of 0 means there was not enough data for this term.</p>
# </blockquote>
# <p>Okay, that's great Google, but you are not making this data easily analyzable for us. I see a few things. Let's do the column names first. A column named "Kim Kardashian: (Worldwide)" is not the most usable for coding purposes. Let's shorten those so we can access their values better. Might as well standardize all column formats, too. I like lowercase, short column names.</p>

# In[4]:


# Make column names easier to work with
trends.columns = ["month", "kim", "khloe", "kourtney", "kendall", "kylie"]

# Inspect data
trends.head()


# In[5]:


get_ipython().run_cell_magic('nose', '', '\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_output = _\n\ndef test_column_names():\n    assert list(trends.columns) == [\'month\', \'kim\', \'khloe\', \'kourtney\', \'kendall\', \'kylie\'], \\\n    \'At least one column name is incorrect or out of order.\'\n    \ndef test_head_output():\n    try:\n        assert "2007-05" in last_output.to_string()\n    except AttributeError:\n        assert False, "Please use trends.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, "The output of the cell is not what we expected. You should see \'2007-05\' in the fifth row of the trends DataFrame."')


# ## 3. Pesky data types
# <p>That's better. We don't need to scroll our eyes across the table to read the values anymore since it is much less wide. And seeing five columns that all start with the letter "k" ... the aesthetics ... we should call them "kolumns" now! (Bad joke.)</p>
# <p>The next thing I see that is going to be an issue is that "&lt;" sign. If <em>"a score of 0 means there was not enough data for this term,"</em> "&lt;1" must mean it is between 0 and 1 and Google does not want to give us the fraction from google.trends.com for whatever reason. That's fine, but this "&lt;" sign means we won't be able to analyze or visualize our data right away because those column values aren't going to be represented as numbers in our data structure. Let's confirm that by inspecting our data types.</p>

# In[6]:


# Inspect data types
trends.info()


# In[7]:


get_ipython().run_cell_magic('nose', '', '\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_info_command():\n    assert \'trends\' in last_input, \\\n        "We expected the trend variable in your input."\n    assert \'info\' in last_input, \\\n        "We expected the info method in your input."')


# ## 4. From object to integer
# <p>Yes, okay, the <code>khloe</code>, <code>kourtney</code>, and <code>kendall</code> columns aren't integers like the <code>kim</code> and <code>kylie</code> columns are. Again, because of the "&lt;" sign that indicates a search interest value between zero and one. Is this an early hint at the hierarchy of sister popularity? We'll see shortly. Before that, we'll need to remove that pesky "&lt;" sign. Then we can change the type of those columns to integer.</p>

# In[8]:


# Loop through columns
for column in trends.columns:
    # Only modify columns that have the "<" sign
    if "<" in trends[column].to_string():
        # Remove "<" and convert dtype to integer
        trends[column] = trends[column].str.replace("<", " ")
        trends[column] = pd.to_numeric(trends[column])

# Inspect data types and data
trends.info()
trends.head()


# In[9]:


get_ipython().run_cell_magic('nose', '', '\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_output = _\n\ndef test_removed_sign():\n    assert "<" not in trends.to_string(), \\\n    \'The "<" character is still present. Did you remember to overwrite each column with the output of str.replace().\'\n    \ndef test_kim_dtype():\n    assert trends.kim.dtype == \'int64\', \\\n    \'The kim column is not of dtype int64.\'\n    \ndef test_khloe_dtype():\n    assert trends.khloe.dtype == \'int64\', \\\n    \'The khloe column is not of dtype int64.\'\n\ndef test_kourtney_dtype():\n    assert trends.kourtney.dtype == \'int64\', \\\n    \'The kourtney column is not of dtype int64.\'\n    \ndef test_kendall_dtype():\n    assert trends.kendall.dtype == \'int64\', \\\n    \'The kendall column is not of dtype int64.\'\n\ndef test_kylie_dtype():\n    assert trends.kylie.dtype == \'int64\', \\\n    \'The kylie column is not of dtype int64.\'\n    \ndef test_head_output():\n    try:\n        assert "2007-05" in last_output.to_string()\n    except AttributeError:\n        assert False, "Please use trends.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, "The output of the cell is not what we expected. You should see \'2007-05\' in the fifth row of the trends DataFrame."')


# ## 5. From object to datetime
# <p>Okay, great, no more "&lt;" signs. All the sister columns are of integer type.</p>
# <p>Now let's convert our <code>month</code> column from type object to datetime to make our date data more accessible.</p>

# In[10]:


# Convert month to type datetime
trends.month = pd.to_datetime(trends.month)

# Inspect data types and data
trends.info()
trends.head()


# In[11]:


get_ipython().run_cell_magic('nose', '', '\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_output = _\n\ndef test_month_dtype():\n    assert trends.month.dtype == \'datetime64[ns]\', \\\n    \'The month column is not of dtype datetime64[ns].\'\n    \ndef test_head_output():\n    try:\n        assert "2007-05-01" in last_output.to_string()\n    except AttributeError:\n        assert False, "Please use trends.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, "The output of the cell is not what we expected. You should see \'2007-05-01\' in the fifth row of the trends DataFrame."')


# ## 6. Set month as index
# <p>And finally, let's set the <code>month</code> column as our index to wrap our data cleaning. Having <code>month</code> as index rather than the zero-based row numbers will allow us to write shorter lines of code to create plots, where <code>month</code> will represent our x-axis.</p>

# In[12]:


# Set month as DataFrame index
trends = trends.set_index("month")

# Inspect the data
trends.head()


# In[13]:


get_ipython().run_cell_magic('nose', '', '\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_output = _\n\ndef test_index_name():\n    assert trends.index.name == \'month\', \\\n    \'The index of the DataFrame is not named month (case-sensitive).\'\n    \ndef test_trends_shape():\n    assert trends.shape[1] == 5, \\\n    \'There should be five columns in the DataFrame, one for the first name of each sister.\'\n    \ndef test_head_output():\n    try:\n        assert "2007-05-01" in last_output.to_string()\n    except AttributeError:\n        assert False, "Please use trends.head() as the last line of code in the cell to inspect the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, "The output of the cell is not what we expected. You should see \'2007-05-01\' in the fifth row of the trends DataFrame."')


# ## 7. The early Kim hype
# <p>Okay! So our data is ready to plot. Because we cleaned our data, we only need one line of code (and just <em>thirteen</em> characters!) to remake the Google Trends chart, plus another line to make the plot show up in our notebook.</p>

# In[14]:


# Plot search interest vs. month
get_ipython().run_line_magic('matplotlib', 'inline')

trends.plot()


# In[15]:


get_ipython().run_cell_magic('nose', '', "\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_value = _\n\ndef test_plot_exists_7():\n    assert type(last_value) == type(trends.plot()), \\\n    'A plot was not the last output of the code cell.'\n    \ndef test_x_data_7():\n    assert len(last_value.get_lines()[0].get_xdata()) == len(list(trends.index)), \\\n    'The x-axis data looks incorrect.'\n    \ndef test_y_data_7():\n    assert len(last_value.get_lines()[0].get_ydata()) == len(list(trends.kim)), \\\n    'The y-axis data looks incorrect.'\n\n# def test_x_label_7():\n#     assert last_value.get_xlabel() == 'month', \\\n#     'The x-axis is not labeled `month`.'")


# ## 8. Kylie's rise
# <p>Oh my! There is so much to make sense of here. Kim's <a href="https://en.wikipedia.org/wiki/Kim_Kardashian#2007%E2%80%932009:_Breakthrough_with_reality_television">sharp rise in 2007</a>, with the beginning of <a href="https://en.wikipedia.org/wiki/Keeping_Up_with_the_Kardashians"><em>Keeping Up with the Kardashians</em></a>, among other things. There was no significant search interest for the other four sisters until mid-2009 when Kourtney and Khloé launched the reality television series, <a href="https://en.wikipedia.org/wiki/Kourtney_and_Kim_Take_Miami"><em>Kourtney and Khloé Take Miami</em></a>. Then there was Kim's rise from famous to <a href="https://trends.google.com/trends/explore?date=all&geo=US&q=%2Fm%2F0261x8t,%2Fm%2F0d05l6">literally more famous than God</a> in 2011. This Cosmopolitan <a href="https://www.cosmopolitan.com/uk/entertainment/a12464842/who-is-kim-kardashian/">article</a> covers the timeline that includes the launch of music videos, fragrances,  iPhone and Android games, another television series, joining Instagram, and more. Then there was Kim's ridiculous spike in December 2014: posing naked on the cover of Paper Magazine in a bid to break the internet will do that for you.</p>
# <p>A curious thing starts to happen after that bid as well. Let's zoom in...</p>

# In[16]:


# Zoom in from January 2014
trends.loc["2014-01-01" :].plot()


# In[17]:


get_ipython().run_cell_magic('nose', '', "\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_value = _\n\ndef test_plot_exists_8():\n    assert type(last_value) == type(trends.loc['2014-01-01':].plot()), \\\n    'A plot was not the last output of the code cell.'\n    \ndef test_x_data_8():\n    assert len(last_value.get_lines()[0].get_xdata()) == len(list(trends.loc['2014-01-01':].index)), \\\n    'The x-axis data looks incorrect.'\n    \ndef test_y_data_8():\n    assert len(last_value.get_lines()[0].get_ydata()) == len(list(trends.loc['2014-01-01':].kim)), \\\n    'The y-axis data looks incorrect.'")


# ## 9. Smooth out the fluctuations with rolling means
# <p>It looks like my suspicion may be true: Kim is not always the most searched Kardashian or Jenner sister. Since late-2016, at various months, Kylie overtakes Kim. Two big spikes where she smashed Kim's search interest: in September 2017 when it was reported that Kylie was expecting her first child with rapper <a href="https://en.wikipedia.org/wiki/Travis_Scott">Travis Scott</a> and in February 2018 when she gave birth to her daughter, Stormi Webster. The continued success of Kylie Cosmetics has kept her in the news, not to mention making her the "The Youngest Self-Made Billionaire Ever" <a href="https://www.forbes.com/sites/natalierobehmed/2019/03/05/at-21-kylie-jenner-becomes-the-youngest-self-made-billionaire-ever/#57e612c02794">according to Forbes</a>.</p>
# <p>These fluctuations are descriptive but do not really help us answer our question: is Kim even the most famous sister anymore? We can use rolling means to smooth out short-term fluctuations in time series data and highlight long-term trends. Let's make the window twelve months a.k.a. one year.</p>

# In[18]:


# Smooth the data with rolling means
trends.rolling(window=12).mean().plot()


# In[19]:


get_ipython().run_cell_magic('nose', '', "\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nlast_value = _\n# print(last_value.get_lines()[0].get_ydata())\n# print(trends.rolling(window=12).mean().kim.values)\n\ndef test_plot_exists_9():\n    assert type(last_value) == type(trends.rolling(window=12).mean().plot()), \\\n    'A plot was not the last output of the code cell.'\n    \ndef test_x_data_9():\n    assert len(last_value.get_lines()[0].get_xdata()) == len(list(trends.index)), \\\n    'The x-axis data looks incorrect.'\n\ndef test_y_data_9():\n    assert all(last_value.get_lines()[0].get_ydata()[11:] == trends.rolling(window=12).mean().kim.values[11:]), \\\n    'The y-axis data looks incorrect.'")


# ## 10. Who's more famous? The Kardashians or the Jenners?
# <p>Whoa, okay! So by this metric, Kim is still the most famous sister despite Kylie being close and nearly taking her crown. Honestly, the biggest takeaway from this whole exercise might be Kendall not showing up that much. It makes sense, though, despite her <a href="http://time.com/money/5033357/kendall-jenner-makes-more-than-gisele-bundchen/">wildly successful modeling career</a>. Some have called her "<a href="https://www.nickiswift.com/5681/kendall-jenner-normal-one-family/">the only normal one in her family</a>" as she tends to shy away from the more dramatic and controversial parts of the media limelight that generate oh so many clicks.</p>
# <p>Let's end this analysis with one last plot. In it, we will plot (pun!) the Kardashian sisters against the Jenner sisters to see which family line is more popular now. We will use average search interest to make things fair, i.e., total search interest divided by the number of sisters in the family line.</p>
# <p><strong>The answer?</strong> Since 2015, it has been a toss-up. And in the future? With this family and their penchant for big events, who knows?</p>

# In[20]:


# Average search interest for each family line
trends['kardashian'] = (trends.kim + trends.khloe + trends.kourtney)/3
trends['jenner'] = (trends.kendall + trends.kylie)/2

# Plot average family line search interest vs. month
trends[['kardashian','jenner']].plot()


# In[21]:


get_ipython().run_cell_magic('nose', '', "\nlast_value = _\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef test_kardashian_created():\n    assert 'kardashian' in list(trends.columns), \\\n    'The kardashian column (case-sensitive) does not look like it was added to the DataFrame correctly.'\n\ndef test_jenner_created():\n    assert 'jenner' in list(trends.columns), \\\n    'The jenner column (case-sensitive) does not look like it was added to the DataFrame correctly.'\n\ndef test_kim_remains():\n    assert 'kim' in list(trends.columns), \\\n    'The kim column looks like it was removed from the trends DataFrame. It should remain.'\n\ndef test_khloe_remains():\n    assert 'khloe' in list(trends.columns), \\\n    'The khloe column looks like it was removed from the trends DataFrame. It should remain.'\n    \ndef test_kourtney_remains():\n    assert 'kourtney' in list(trends.columns), \\\n    'The kourtney column looks like it was removed from the trends DataFrame. It should remain.'\n    \ndef test_kendall_remains():\n    assert 'kendall' in list(trends.columns), \\\n    'The kendall column looks like it was removed from the trends DataFrame. It should remain.'\n    \ndef test_kylie_remains():\n    assert 'kylie' in list(trends.columns), \\\n    'The kylie column looks like it was removed from the trends DataFrame. It should remain.'\n\ndef test_kardashian_correct():\n    assert all(trends['kardashian'].round() == ((trends.kim + trends.khloe + trends.kourtney) / 3).round()), \\\n    'The data in the kardashian column looks incorrect.'\n    \ndef test_jenner_correct():\n    assert all(trends['jenner'].round() == ((trends.kendall + trends.kylie) / 2).round()), \\\n    'The data in the kardashian column looks incorrect.'\n\ndef test_plot_exists_10():\n    assert type(last_value) == type(trends[['kardashian', 'jenner']].plot()), \\\n    'A plot was not the last output of the code cell.'\n    \ndef test_x_data_10():\n    assert len(last_value.get_lines()[0].get_xdata()) == len(list(trends.index)), \\\n    'The x-axis data looks incorrect. It should contain every month from 2007-01-01 through 2019-03-01.'\n    \ndef test_y_legend_10():\n    assert set(last_value.get_legend_handles_labels()[1]) == set(['kardashian', 'jenner']), \\\n    'The y-axis data looks incorrect. It should contain the average search interest for each family line, i.e., the newly created kardashian and jenner columns.'")

