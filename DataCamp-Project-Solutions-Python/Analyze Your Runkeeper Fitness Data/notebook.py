#!/usr/bin/env python
# coding: utf-8

# ## 1. Obtain and review raw data
# <p>One day, my old running friend and I were chatting about our running styles, training habits, and achievements, when I suddenly realized that I could take an in-depth analytical look at my training. I have been using a popular GPS fitness tracker called <a href="https://runkeeper.com/">Runkeeper</a> for years and decided it was time to analyze my running data to see how I was doing.</p>
# <p>Since 2012, I've been using the Runkeeper app, and it's great. One key feature: its excellent data export. Anyone who has a smartphone can download the app and analyze their data like we will in this notebook.</p>
# <p><img src="https://assets.datacamp.com/production/project_727/img/runner_in_blue.jpg" alt="Runner in blue" title="Explore world, explore your data!"></p>
# <p>After logging your run, the first step is to export the data from Runkeeper (which I've done already). Then import the data and start exploring to find potential problems. After that, create data cleaning strategies to fix the issues. Finally, analyze and visualize the clean time-series data.</p>
# <p>I exported seven years worth of my training data, from 2012 through 2018. The data is a CSV file where each row is a single training activity. Let's load and inspect it.</p>

# In[2]:


# Import pandas
import pandas as pd

# Define file containing dataset
runkeeper_file = 'datasets/cardioActivities.csv'

# Create DataFrame with parse_dates and index_col parameters 
df_activities = pd.read_csv(runkeeper_file, parse_dates=True, index_col='Date')

# First look at exported data: select sample of 3 random rows 
display(df_activities.sample(3))

# Print DataFrame summary
df_activities.info()


# In[3]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_pandas_loaded():\n    assert \'pd\' in globals(), \\\n    \'Did you import the pandas library aliased as pd?\'\n    \ndef test_activities_correctly_loaded():\n    correct_activities = pd.read_csv(runkeeper_file,  parse_dates=True, index_col=\'Date\')\n    assert correct_activities.equals(df_activities), \\\n    "The variable df_activities should contain data read from runkeeper_file with dates parsed and the index column set to date."\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_sample_command():\n    assert (\'df_activities.sample(n=3)\' in last_input) or (\'df_activities.sample(3)\' in last_input), \\\n        "We expected the sample() method with the n parameter set to 3 in your input."\n\ndef test_info_command():\n    assert \'df_activities.info()\' in last_input, \\\n        "We expected the info() method to be called on df_activities in your input."')


# ## 2. Data preprocessing
# <p>Lucky for us, the column names Runkeeper provides are informative, and we don't need to rename any columns.</p>
# <p>But, we do notice missing values using the <code>info()</code> method. What are the reasons for these missing values? It depends. Some heart rate information is missing because I didn't always use a cardio sensor. In the case of the <code>Notes</code> column, it is an optional field that I sometimes left blank. Also, I only used the <code>Route Name</code> column once, and never used the <code>Friend's Tagged</code> column.</p>
# <p>We'll fill in missing values in the heart rate column to avoid misleading results later, but right now, our first data preprocessing steps will be to:</p>
# <ul>
# <li>Remove columns not useful for our analysis.</li>
# <li>Replace the "Other" activity type to "Unicycling" because that was always the "Other" activity.</li>
# <li>Count missing values.</li>
# </ul>

# In[4]:


# Define list of columns to be deleted
cols_to_drop = ['Friend\'s Tagged','Route Name','GPX File','Activity Id','Calories Burned', 'Notes']


# Delete unnecessary columns
df_activities.drop(cols_to_drop, axis=1, inplace=True)

# Count types of training activities
display(df_activities['Type'].value_counts())

# Rename 'Other' type to 'Unicycling'
df_activities['Type'] = df_activities['Type'].str.replace('Other', 'Unicycling')

# Count missing values for each column
df_activities.isnull().sum()


# In[5]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\nlast_output = _\n\ncols_to_drop_correct = [\'Friend\\\'s Tagged\',\'Route Name\',\'GPX File\',\'Activity Id\',\'Calories Burned\', \'Notes\']\n\ndef test_columns_deleted():\n    for cld in cols_to_drop_correct:\n        assert cld not in list(df_activities.columns), \\\n        "Did you drop unnecessary columns as described in the instructions?"\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_values_count():\n    assert "df_activities" in last_input, \\\n        "We expected the df_activities variable in your input."\n    assert \'value_counts\' in last_input, \\\n        "We expected the value_counts() method in your input."\n    \ndef test_replace():\n    assert df_activities[df_activities[\'Type\'] == \'Unicycling\'][\'Type\'].count() == 2, \\\n        "The count of activities of type \'Unicycling\' is incorrect."\n    \ndef test_missing_output():\n    try:\n        assert "214" in last_output.to_string()\n    except AttributeError:\n        assert False, \\\n        "Please use isnull().sum() as the last line of code in the cell to count missing values."\n    except AssertionError:\n        assert False, \\\n        "The output of the cell is not what we expected. You should see 214 missing values."')


# ## 3. Dealing with missing values
# <p>As we can see from the last output, there are 214 missing entries for my average heart rate.</p>
# <p>We can't go back in time to get those data, but we can fill in the missing values with an average value. This process is called <em>mean imputation</em>. When imputing the mean to fill in missing data, we need to consider that the average heart rate varies for different activities (e.g., walking vs. running). We'll filter the DataFrames by activity type (<code>Type</code>) and calculate each activity's mean heart rate, then fill in the missing values with those means.</p>

# In[6]:


# Calculate sample means for heart rate for each training activity type 
avg_hr_run = df_activities[df_activities['Type'] == 'Running']['Average Heart Rate (bpm)'].mean()
avg_hr_cycle = df_activities[df_activities['Type'] == 'Cycling']['Average Heart Rate (bpm)'].mean()

# Split whole DataFrame into several, specific for different activities
df_run = df_activities[df_activities['Type'] == 'Running'].copy()
df_walk = df_activities[df_activities['Type'] == 'Walking'].copy()
df_cycle = df_activities[df_activities['Type'] == 'Cycling'].copy()

# Filling missing values with counted means  
df_walk['Average Heart Rate (bpm)'].fillna(110, inplace=True)
df_run['Average Heart Rate (bpm)'].fillna(int(avg_hr_run), inplace=True)
df_cycle['Average Heart Rate (bpm)'].fillna(int(avg_hr_cycle), inplace=True)

# Count missing values for each column in running data
df_run.isnull().sum()


# In[7]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\n\nlast_output = _\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_avg_hr_cycle():\n    assert avg_hr_cycle == 124.4, \\\n    "The value of avg_hr_rate is not what we expected. Use code similar to avg_hr_run with \'Type\' == \'Cycling\'."\n\ndef test_df_cycle_correct():\n    assert len(df_cycle) == 29, \\\n    "The variable df_cycle should contain data filtered from df_activities with \'Type\' == \'Cycling\'."\n\ndef test_df_cycle_fillna():\n    assert df_cycle[\'Average Heart Rate (bpm)\'].isnull().sum() == 0, \\\n    "There are missing values in df_cycle \'Average Heart Rate (bpm)\' column."\n\ndef test_isnull_output():\n    try:\n        assert "Average Heart Rate (bpm)    0" in last_output.to_string()\n    except AttributeError:\n        assert False, "\'Average Heart Rate (bpm)    0\' should be the output of the cell. Please do not use the print() or display() functions; ensure the isnull().sum() code is the last line of code in the cell."\n    except AssertionError:\n        assert False, \\\n        "The output of the cell is not what we expected. You should see \'Average Heart Rate (bpm)    0\'."')


# ## 4. Plot running data
# <p>Now we can create our first plot! As we found earlier, most of the activities in my data were running (459 of them to be exact). There are only 29, 18, and two instances for cycling, walking, and unicycling, respectively. So for now, let's focus on plotting the different running metrics.</p>
# <p>An excellent first visualization is a figure with four subplots, one for each running metric (each numerical column). Each subplot will have a different y-axis, which is explained in each legend. The x-axis, <code>Date</code>, is shared among all subplots.</p>

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Import matplotlib, set style and ignore warning
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
plt.style.use('ggplot')
warnings.filterwarnings(
    action='ignore', module='matplotlib.figure', category=UserWarning,
    message=('This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.')
)

# Prepare data subsetting period from 2013 till 2018
runs_subset_2013_2018 = df_run.loc['20190101':'20130101']

# Create, plot and customize in one step
runs_subset_2013_2018.plot(subplots=True,
                           sharex=False,
                           figsize=(12,16),
                           linestyle='none',
                           marker='o',
                           markersize=3,
                          )

# Show plot
plt.show()


# In[9]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\n# def test_set_style():\n#     assert "plt.style.use(\'ggplot\')" in last_input, \\\n#         "We expected the plt.style.use(\'ggplot\') in your input."\n    \ndef test_subset_correct():\n    correct_subset = df_run[\'2018\':\'2013\']\n    assert correct_subset.equals(runs_subset_2013_2018), \\\n    "The data in runs_subset_2013_2018 is not what we expected."\n\ndef test_subplots_param():\n    assert "subplots=True" in last_input, \\\n        "We expected the subplots=True in your input WITHOUT whitespaces around \'=\' sign."\n    \ndef test_plt_show():\n    assert "plt.show()" in last_input, \\\n        "We expected plt.show() in your input."')


# ## 5. Running statistics
# <p>No doubt, running helps people stay mentally and physically healthy and productive at any age. And it is great fun! When runners talk to each other about their hobby, we not only discuss our results, but we also discuss different training strategies. </p>
# <p>You'll know you're with a group of runners if you commonly hear questions like:</p>
# <ul>
# <li>What is your average distance?</li>
# <li>How fast do you run?</li>
# <li>Do you measure your heart rate?</li>
# <li>How often do you train?</li>
# </ul>
# <p>Let's find the answers to these questions in my data. If you look back at plots in Task 4, you can see the answer to, <em>Do you measure your heart rate?</em> Before 2015: no. To look at the averages, let's only use the data from 2015 through 2018.</p>
# <p>In pandas, the <code>resample()</code> method is similar to the <code>groupby()</code> method - with <code>resample()</code> you group by a specific time span. We'll use <code>resample()</code> to group the time series data by a sampling period and apply several methods to each sampling period. In our case, we'll resample annually and weekly.</p>

# In[10]:


# Prepare running data for the last 4 years
runs_subset_2015_2018 = df_run.loc['20190101':'20150101']

# Calculate annual statistics
print('How my average run looks in last 4 years:')
display(runs_subset_2015_2018.resample('A').mean())

# Calculate weekly statistics
print('Weekly averages of last 4 years:')
display(runs_subset_2015_2018.resample('W').mean().mean())

# Mean weekly counts
weekly_counts_average = runs_subset_2015_2018['Distance (km)'].resample('W').count().mean()
print('How many trainings per week I had on average:', weekly_counts_average)


# In[11]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\nlast_output = _\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_subset_correct5():\n    correct_subset = df_run[\'2018\':\'2015\']\n    assert correct_subset.equals(runs_subset_2015_2018), \\\n    "The data in runs_subset_2015_2018 is not what we expected."\n\ndef test_annual_stats():\n    assert "resample(\'A\').mean()" in last_input, \\\n    "Did you use resample(\'A\').mean() to count annual averages for each year?"\n\ndef test_weekly_average():\n    assert "resample(\'W\').mean().mean()" in last_input, \\\n    "Did you use resample(\'W\').mean().mean() to count average weekly statistics?"\n\ndef test_weekly_count_syntax():\n    assert "runs_subset_2015_2018[\'Distance (km)\']" in last_input, \\\n    "Did you filter column \'Distance (km)\' from data subset using single quotes?"\n    \ndef test_weekly_count():\n    assert weekly_counts_average == 1.5, \\\n    "We expected 1.5 trainings per week on average."')


# ## 6. Visualization with averages
# <p>Let's plot the long term averages of my distance run and my heart rate with their raw data to visually compare the averages to each training session. Again, we'll use the data from 2015 through 2018.</p>
# <p>In this task, we will use <code>matplotlib</code> functionality for plot creation and customization.</p>

# In[12]:


# Prepare data
runs_distance = runs_subset_2015_2018['Distance (km)']
runs_hr = runs_subset_2015_2018['Average Heart Rate (bpm)']

# Create plot
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))

# Plot and customize first subplot
runs_distance.plot(ax=ax1)
ax1.set(ylabel='Distance (km)', title='Historical data with averages')
ax1.axhline(runs_distance.mean(), color='blue', linewidth=1, linestyle='-.')

# Plot and customize second subplot
runs_hr.plot(ax=ax2, color='gray')
ax2.set(xlabel='Date', ylabel='Average Heart Rate (bpm)')
ax2.axhline(runs_hr.mean(), color='blue', linewidth=1, linestyle='-.')

# Show plot
plt.show()


# In[13]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_data_correct6():\n    cor_runs_distance = runs_subset_2015_2018[\'Distance (km)\']\n    cor_runs_hr = runs_subset_2015_2018[\'Average Heart Rate (bpm)\']\n    assert (cor_runs_distance.equals(runs_distance) & cor_runs_hr.equals(runs_hr)), \\\n    "Did you use the correct column names to select distance and heart rate?"\n\ndef test_plot_exist6():\n    assert (ax2.get_geometry()[0] == 2 and fig.get_figwidth() == 12), \\\n    "The size of plot and the number of subplots are not as expected."\n    \ndef test_first_subplot6():\n    assert all(ax1.lines[0].get_ydata() == runs_distance.sort_index()), \\\n    "Did you use runs_distance.plot(ax=ax1) to plot the first plot?"\n    \ndef test_horline6():\n    assert ax2.lines[1].get_ydata()[0] == runs_hr.mean(), \\\n    "The value of average heart rate is incorrect on the plot."')


# ## 7. Did I reach my goals?
# <p>To motivate myself to run regularly, I set a target goal of running 1000 km per year. Let's visualize my annual running distance (km) from 2013 through 2018 to see if I reached my goal each year. Only stars in the green region indicate success.</p>

# In[14]:


# Prepare data
df_run_dist_annual = df_run.sort_index()['20130101':'20181231']['Distance (km)']                     .resample('A').sum()

# Create plot
fig = plt.figure(figsize=(8, 5))

# Plot and customize
ax = df_run_dist_annual.plot(marker='*', markersize=14, linewidth=0, 
                             color='blue')
ax.set(ylim=[0, 1210], 
       xlim=['2012','2019'],
       ylabel='Distance (km)',
       xlabel='Years',
       title='Annual totals for distance')

ax.axhspan(1000, 1210, color='green', alpha=0.4)
ax.axhspan(800, 1000, color='yellow', alpha=0.3)
ax.axhspan(0, 800, color='red', alpha=0.2)

# Show plot
plt.show()


# In[15]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_data_correct7():\n    cor_run_dist_annual = df_run[\'2018\':\'2013\'][\'Distance (km)\'].resample(\'A\').sum()\n    assert cor_run_dist_annual.equals(df_run_dist_annual), \\\n    "The data in df_run_dist_annual is not what we expected."\n\ndef test_figsize7():\n    assert (fig.get_figwidth() == 8.0 and fig.get_figheight() == 5.0), \\\n    "The figure size is incorrect. Did you set figsize=(8, 5) in plt.figure()?"\n\ndef test_axspan7():\n    assert len(ax.patches) == 3, \\\n    "Did you use ax.axhspan(0, 800, ...)?"\n\ndef test_show7():\n    assert \'plt.show()\' in last_input , \\\n    "We expected plt.show() in your input."')


# ## 8. Am I progressing?
# <p>Let's dive a little deeper into the data to answer a tricky question: am I progressing in terms of my running skills? </p>
# <p>To answer this question, we'll decompose my weekly distance run and visually compare it to the raw data. A red trend line will represent the weekly distance run.</p>
# <p>We are going to use <code>statsmodels</code> library to decompose the weekly trend.</p>

# In[16]:


# Import required library
import statsmodels.api as sm

# Prepare data
df_run_dist_wkly = df_run.loc['20190101':'20130101']['Distance (km)']                     .resample('W').bfill()
decomposed = sm.tsa.seasonal_decompose(df_run_dist_wkly, 
                                       extrapolate_trend=1, freq=52)

# Create plot
fig = plt.figure(figsize=(12, 5))

# Plot and customize
ax = decomposed.trend.plot(label='Trend', linewidth=2)
ax = decomposed.observed.plot(label='Observed', linewidth=0.5)

ax.legend()
ax.set_title('Running distance trend')

# Show plot
plt.show()


# In[17]:


get_ipython().run_cell_magic('nose', '', '\'\'\'\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\'\'\'\n\ndef test_sm_imported():\n    assert \'sm\' in globals(), \\\n    "Did you import statsmodels.api aliased as sm?"\n\ndef test_data_correct8():\n    cor_run_dist_wkly = df_run[\'2018\':\'2013\'][\'Distance (km)\'].resample(\'W\').bfill()\n    assert cor_run_dist_wkly.equals(df_run_dist_wkly), \\\n    "The data in df_run_dist_wkly is not what we expected."\n\ndef test_figsize8():\n    assert (fig.get_figwidth() == 12.0 and fig.get_figheight() == 5.0), \\\n    "The figure size is incorrect. Did you set figsize=(12, 5) in plt.figure()?"')


# ## 9. Training intensity
# <p>Heart rate is a popular metric used to measure training intensity. Depending on age and fitness level, heart rates are grouped into different zones that people can target depending on training goals. A target heart rate during moderate-intensity activities is about 50-70% of maximum heart rate, while during vigorous physical activity it’s about 70-85% of maximum.</p>
# <p>We'll create a distribution plot of my heart rate data by training intensity. It will be a visual presentation for the number of activities from predefined training zones. </p>

# In[18]:


# Prepare data
hr_zones = [100, 125, 133, 142, 151, 173]
zone_names = ['Easy', 'Moderate', 'Hard', 'Very hard', 'Maximal']
zone_colors = ['green', 'yellow', 'orange', 'tomato', 'red']
df_run_hr_all = df_run.loc['20190101':'20150301']['Average Heart Rate (bpm)']

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot and customize
n, bins, patches = ax.hist(df_run_hr_all, bins=hr_zones, alpha=0.5)
for i in range(0, len(patches)):
    patches[i].set_facecolor(zone_colors[i])

ax.set(title='Distribution of HR', ylabel='Number of runs')
ax.xaxis.set(ticks=hr_zones)
ax.set_xticklabels(labels=zone_names, rotation=-30, ha='left')

# Show plot
plt.show()


# In[19]:


get_ipython().run_cell_magic('nose', '', '\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\n\ndef test_data_correct9():\n    corr_data = df_run[\'2018\':\'2015-03\'][\'Average Heart Rate (bpm)\']\n    assert corr_data.equals(df_run_hr_all), \\\n    "The data in df_run_hr_all is not what we expected."\n\ndef test_subplots_used9():\n    assert \'plt.subplots\' in last_input, \\\n    "We expected plt.subplots in your input."\n    \ndef test_figsize9():\n    assert (fig.get_figwidth() == 8.0 and fig.get_figheight() == 5.0), \\\n    "The figure size is incorrect. Did you set figsize=(8, 5) in plt.subplots()?"\n    \ndef test_xticklabels9():\n    lbls = [lb.get_text() for lb in ax.get_xticklabels()]\n    del lbls[-1]\n    assert lbls == zone_names, \\\n    "The x-axis labels are incorrect. Did you use ax.set_xticklabels(labels=zone_names, ...) ?"\n    \ndef test_show9():\n    assert \'plt.show()\' in last_input , \\\n    "We expected plt.show() in your input."')


# ## 10. Detailed summary report
# <p>With all this data cleaning, analysis, and visualization, let's create detailed summary tables of my training. </p>
# <p>To do this, we'll create two tables. The first table will be a summary of the distance (km) and climb (m) variables for each training activity. The second table will list the summary statistics for the average speed (km/hr), climb (m), and distance (km) variables for each training activity.</p>

# In[20]:


# Concatenating three DataFrames
df_run_walk_cycle = df_run.append([df_walk, df_cycle]).sort_index(ascending=False)

dist_climb_cols, speed_col = ['Distance (km)', 'Climb (m)'], ['Average Speed (km/h)']

# Calculating total distance and climb in each type of activities
df_totals = df_run_walk_cycle.groupby('Type')[dist_climb_cols].sum()

print('Totals for different training types:')
display(df_totals)

# Calculating summary statistics for each type of activities 
df_summary = df_run_walk_cycle.groupby('Type')[dist_climb_cols + speed_col].describe()

# Combine totals with summary
for i in dist_climb_cols:
    df_summary[i, 'total'] = df_totals[i]

print('Summary statistics for different training types:')
df_summary.stack()


# In[21]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef strip_comment_lines(cell_input):\n    """Returns cell input string with comment lines removed."""\n    return \'\\n\'.join(line for line in cell_input.splitlines() if not line.startswith(\'#\'))\n\nlast_input = strip_comment_lines(In[-2])\nlast_output = _\n\ndef test_data_correct10():\n    corr_data = df_run.append(df_walk).append(df_cycle).sort_index(ascending=False)\n    assert corr_data.equals(df_run_walk_cycle), \\\n    "The data in df_run_walk_cycle is not what we expected. Did you apply sort_index(ascending=False) ?"\n\ndef test_display_totals10():\n    assert \'display(df_totals)\' in last_input, \\\n    "Did you use display(df_totals) to show the calculations?"\n    \ndef test_totals10():\n    corr_tot = df_totals[\'Distance (km)\'][\'Running\'].round(1)\n    assert corr_tot == 5224.5, \\\n    "We expected different values for totals. You should see 5224.5 km as total distance."\n\ndef test_stack_used10():\n    assert \'stack()\' in last_input, \\\n    "Did you use stack() to create a compact view of the results?"')


# ## 11. Fun facts
# <p>To wrap up, let’s pick some fun facts out of the summary tables and solve the last exercise.</p>
# <p>These data (my running history) represent 6 years, 2 months and 21 days. And I remember how many running shoes I went through–7.</p>
# <pre><code>FUN FACTS
# - Average distance: 11.38 km
# - Longest distance: 38.32 km
# - Highest climb: 982 m
# - Total climb: 57,278 m
# - Total number of km run: 5,224 km
# - Total runs: 459
# - Number of running shoes gone through: 7 pairs
# </code></pre>
# <p>The story of Forrest Gump is well known–the man, who for no particular reason decided to go for a "little run." His epic run duration was 3 years, 2 months and 14 days (1169 days). In the picture you can see Forrest’s route of 24,700 km.  </p>
# <pre><code>FORREST RUN FACTS
# - Average distance: 21.13 km
# - Total number of km run: 24,700 km
# - Total runs: 1169
# - Number of running shoes gone through: ...
# </code></pre>
# <p>Assuming Forest and I go through running shoes at the same rate, figure out how many pairs of shoes Forrest needed for his run.</p>
# <p><img src="https://assets.datacamp.com/production/project_727/img/Forrest_Gump_running_route.png" alt="Forrest's route" title="Little run of Forrest Gump"></p>

# In[22]:


# Count average shoes per lifetime (as km per pair) using our fun facts
average_shoes_lifetime = 5224 / 7

# Count number of shoes for Forrest's run distance
shoes_for_forrest_run = 24700 // average_shoes_lifetime

print('Forrest Gump would need {} pairs of shoes!'.format(shoes_for_forrest_run))


# In[23]:


get_ipython().run_cell_magic('nose', '', '\ndef test_forrest_shoes():\n    assert shoes_for_forrest_run == 33, \\\n    "We expected shoes_for_forrest_run to be a whole integer. Forrest would need 33 pairs."')

