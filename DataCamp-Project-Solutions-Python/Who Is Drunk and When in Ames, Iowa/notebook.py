
# coding: utf-8

# ## 1. Breath alcohol tests in Ames, Iowa, USA
# <p>Ames, Iowa, USA is the home of Iowa State University, a land grant university with over 36,000 students. By comparison, the city of Ames, Iowa, itself only has about 65,000 residents. As with any other college town, Ames has had its fair share of alcohol-related incidents. (For example, Google 'VEISHEA riots 2014'.) We will take a look at some breath alcohol test data from Ames that is published by the State of Iowa.</p>
# <p><img style="width:500px" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_208/img/usa.jpg"> </p>
# <p>The data file 'breath_alcohol_ames.csv' contains 1,556 readings from breath alcohol tests administered by the Ames and Iowa State University Police Departments from January 2013 to December 2017. The columns in this data set are year, month, day, hour, location, gender, Res1, Res2.</p>

# In[2]:


# import pandas
import pandas as pd

# read the data into your workspace
ba_data = pd.read_csv('datasets/breath_alcohol_ames.csv')

# quickly inspect the data
print(ba_data.head())

# obtain counts for each year 
ba_year = ba_data['year'].value_counts()


# In[3]:


get_ipython().run_cell_magic('nose', '', '\nimport pandas as pd\n\n# # check read file\n# def test_breath_alcohol_ames():\n#     correct_pulls = pd.read_csv("datasets/breath_alcohol_ames.csv")\n#     assert correct_pulls.equals(breath_alcohol_ames), \\\n#     \'Read in "datasets/breath_alcohol_ames.csv" using read_csv().\'\n\n# # check head\n\n# # check value counts    \n# def test_value_counts():\n#     assert len(ba_year) == 5, \\\n#     \'The rows are not arraged by year. Did you select the correct value to count?\'\n\ndef test_task_1a():\n    correct_ba_data = pd.read_csv("datasets/breath_alcohol_ames.csv")\n    assert correct_ba_data.equals(ba_data), "The variable `ba_data` should contain the data in `breath_alcohol_ames.csv`"\n    \ndef test_task_1b():\n    correct_ba_year = ba_data[\'year\'].value_counts()\n    assert correct_ba_year.equals(ba_year), "The variable `ba_year` should contain the counts of years in `ba_data`. Did you use the `value_counts` method?"')


# ## 2. What is the busiest police department in Ames?
# <p>There are two police departments in the data set: the Iowa State University Police Department and the Ames Police Department. Which one administers more breathalyzer tests? </p>

# In[4]:


# use value_counts to tally up the totals for each department
pds = ba_data['location'].value_counts()
pds


# In[5]:


get_ipython().run_cell_magic('nose', '', '\n# # check value counts    \n# def test_value_counts():\n#     assert len(ba_year) == 2, \\\n#     \'The rows are not arraged by location. Did you select the correct value to count?\'\n\ndef test_task_2():\n    correct_pds = ba_data[\'location\'].value_counts()\n    assert correct_pds.equals(pds), "The variable `pds` should contain the counts of locations in `ba_data`. Did you use the `value_counts` method?"')


# ## 3. Nothing Good Happens after 2am
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_208/img/himym02.jpg" style="float: left;margin:5px 20px 5px 1px;width:300px"></p>
# <p>We all know that "nothing good happens after 2am." Thus, there are inevitably some times of the day when breath alcohol tests, especially in a college town like Ames, are most and least common. Which hours of the day have the most and least breathalyzer tests?  </p>

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')

# count by hour 
hourly = ba_data.groupby(['hour']).size()

# create a vertical bar graph of the arrest count by hour
hourly.plot.bar(x=hourly)


# In[7]:


get_ipython().run_cell_magic('nose', '', '\n# groupby, count, and sort values\n# check counts    \ndef test_sort_values():\n    assert len(hourly) == 24, \\\n    \'The rows are not arranged by hour. Did you select "hour" to group by?\'\n    \n# no test for plots')


# ## 4. Breathalyzer tests by month
# <p>Now that we have discovered which time of day is most common for breath alcohol tests, we will determine which time of the year has the most breathalyzer tests. Which month will have the most recorded tests?</p>

# In[8]:


# count by month and arrange by descending frequency
monthly = ba_data.groupby(['month']).size()

# use plot.bar to make the appropriate bar chart
monthly.plot.bar(x='monthly')


# In[9]:


get_ipython().run_cell_magic('nose', '', '\n# groupby, count, and sort values\n# check counts    \ndef test_sort_values():\n    assert len(monthly) == 12, \\\n    \'The rows are not arranged by month. Did you select "month" to group by?\'\n    ')


# ## 5. COLLEGE
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_208/img/PF2081John-Belushi-College-Posters.jpg" style="float: left;margin:5px 20px 5px 1px"> </p>
# <p>When we think of (binge) drinking in college towns in America, we usually think of something like this image at the left. And so, one might suspect that breath alcohol tests are given to men more often than women and that men drink more than women. </p>

# In[10]:


# count by gender
counts_gender = ba_data['gender'].value_counts()

# create a dataset with no NAs in gender 
gen = ba_data.dropna(subset=['gender'])

# create a mean test result variable
mean_bas = gen.assign(meanRes=(gen.Res1+gen.Res2)/2)

# # create side-by-side boxplots to compare the mean blood alcohol levels of men and women
mean_bas.boxplot(['meanRes'], by='gender')


# In[11]:


get_ipython().run_cell_magic('nose', '', '\n# gender value_counts\ndef test_task_5b():\n    correct_counts_gender = ba_data[\'gender\'].value_counts()\n    assert correct_counts_gender.equals(counts_gender), \\\n    "The variable `counts_gender` should contain the counts of gender in `ba_data`. Did you use the `value_counts` method?"\n\n# dropna\ndef test_dropna():\n    assert len(gen) == 1527, \\\n    \'Did you use dropna() on `gender`?\'\n    \n# create meanRes in mean_bas\ndef test_month_year_column():\n    assert \'meanRes\' in mean_bas, \\\n    "Did you create the `meanRes` column."\n    \n# no test for plots')


# ## 6. Above the legal limit
# <p>In the USA, it is illegal to drive with a blood alcohol concentration (BAC) above 0.08%. This is the case for <a href="https://www.dmv.org/automotive-law/dui.php">all 50 states</a>. Assuming everyone tested in our data was driving (though we have no way of knowing this from the data), if either of the results (<code>Res1</code>, <code>Res2</code>) are above 0.08, the person would be charged with DUI (driving under the influence). </p>

# In[12]:


# Filter the data
duis = ba_data[(ba_data.Res1 > 0.08) | (ba_data.Res2 > 0.08)]

# proportion of tests that would have resulted in a DUI
p_dui = duis.shape[0] / ba_data.shape[0]
p_dui


# In[13]:


get_ipython().run_cell_magic('nose', '', "\n# check filter    \ndef test_duis():\n    assert len(duis) == 1159, \\\n    'Did you use the logical OR operator (|) to filter ba_data?'\n    \n# check p_dui\ndef test_p_dui():\n    correct_p_dui = duis.shape[0] / ba_data.shape[0]\n    assert correct_p_dui == p_dui")


# ## 7. Breathalyzer tests: is there a pattern over time?
# <p>We previously saw that 2am is the most common time of day for breathalyzer tests to be administered, and August is the most common month of the year for breathalyzer tests. Now, we look at the weeks in the year over time. </p>

# In[14]:


# Create date variable
ba_data['date'] = pd.to_datetime(ba_data[['year', 'month', 'day']])

# Create a week variable
ba_data['week'] = ba_data['date'].dt.week

# Check your work
ba_data.head()


# In[15]:


get_ipython().run_cell_magic('nose', '', '\n# create date var    \ndef test_date():\n    assert \'date\' in ba_data, \\\n    "You did not create the variable for date."\n    \n# create week var\ndef test_week():\n    assert \'week\' in ba_data, \\\n    "You did not create the variable for week."    ')


# ## 8. Looking at timelines
# <p>How do the weeks differ over time? One of the most common data visualizations is the time series, a line tracking the changes in a variable over time. We will use the new <code>week</code> variable to look at test frequency over time. We end with a time series plot showing the frequency of breathalyzer tests by week in year, with one line for each year. </p>

# In[16]:


# choose and count the variables of interest  
timeline = ba_data.groupby(['week','year']).count()['Res1']

# unstack and plot
timeline.unstack().plot(title='VEISHEA DUIs', legend=True)


# In[17]:


get_ipython().run_cell_magic('nose', '', "\n# check timeline  \ndef test_timeline():\n    assert len(timeline) == 259, \\\n    'Did you group by week and year?'\n    \n# plot")


# ## 9. The end of VEISHEA
# <p>From <a href="https://en.wikipedia.org/wiki/VEISHEA">Wikipedia</a>: 
# "VEISHEA was an annual week-long celebration held each spring on the campus of Iowa State University in Ames, Iowa. The celebration featured an annual parade and many open-house demonstrations of the university facilities and departments. Campus organizations exhibited products, technologies, and held fundraisers for various charity groups. In addition, VEISHEA brought speakers, lecturers, and entertainers to Iowa State. [...] VEISHEA was the largest student-run festival in the nation, bringing in tens of thousands of visitors to the campus each year."</p>
# <p>This over 90-year tradition in Ames was <a href="https://www.news.iastate.edu/news/2014/08/07/veisheaend">terminated permanently</a> after <a href="https://www.desmoinesregister.com/story/news/crime-and-courts/2014/04/09/veishea-ames-car-tipping/7495935/">riots in 2014</a>, where drunk celebrators flipped over multiple vehicles and tore light poles down. This was not the first incidence of violence and severe property damage in VEISHEA's history. Did former President Leath make the right decision by canceling VEISHEA?</p>

# In[18]:


## Was it right to permanently cancel VEISHEA? TRUE or FALSE?  
canceling_VEISHEA_was_right = True


# In[19]:


get_ipython().run_cell_magic('nose', '', '\ndef test_bool():\n    assert isinstance(canceling_VEISHEA_was_right, bool), "Did you assign `True` or `False` to `canceling_VEISHEA_was_right`?"\n\n## The original R version did not have a test regarding opinion.\n# def test_opinion():\n#     assert canceling_VEISHEA_was_right == \'False\', \\\n#     \'The previous year, 2013, had peak DUIs around week 30, and August has the highest amounts of DUIs per month. VEISHEA was held in April.\'')

