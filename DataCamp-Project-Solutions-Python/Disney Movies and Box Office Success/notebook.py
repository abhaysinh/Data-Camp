#!/usr/bin/env python
# coding: utf-8

# ## 1. The dataset
# <p>Walt Disney Studios is the foundation on which The Walt Disney Company was built. The Studios has produced more than 600 films since their debut film,  Snow White and the Seven Dwarfs in 1937. While many of its films were big hits, some of them were not. In this notebook, we will explore a dataset of Disney movies and analyze what contributes to the success of Disney movies.</p>
# <p><img src="https://assets.datacamp.com/production/project_740/img/jorge-martinez-instagram-jmartinezz9-431078-unsplash_edited.jpg" alt=""></p>
# <p>First, we will take a look at the Disney data compiled by <a href="https://data.world/kgarrett/disney-character-success-00-16">Kelly Garrett</a>. The data contains 579 Disney movies with six features: movie title, release date, genre, MPAA rating, total gross, and inflation-adjusted gross. </p>
# <p>Let's load the file and see what the data looks like.</p>

# In[2]:


# Import pandas library
import pandas as pd

# Read the file into gross
gross = pd.read_csv("datasets/disney_movies_total_gross.csv", 
                    parse_dates=["release_date"])

# Print out gross
gross.head()


# In[3]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_pandas_loaded():\n    assert \'pd\' in globals(), \\\n    \'Did you import the pandas library aliased as pd?\'\n\ndef test_parse_date():\n    assert gross[\'release_date\'].dtype == \'<M8[ns]\', \\\n    "The release_date column is not of dtype <M8[ns]."\n      \ndef test_gross_correctly_loaded():\n    correct_gross = pd.read_csv(\'./datasets/disney_movies_total_gross.csv\', parse_dates=[\'release_date\'])\n    assert correct_gross.equals(gross), "The DataFrame gross should contain the data in disney_movies_total_gross.csv."')


# ## 2. Top ten movies at the box office
# <p>Let's started by exploring the data. We will check which are the 10 Disney movies that have earned the most at the box office. We can do this by sorting movies by their inflation-adjusted gross (we will call it adjusted gross from this point onward). </p>

# In[4]:


# Sort data by the adjusted gross in descending order 
inflation_adjusted_gross_desc = gross.sort_values(by='inflation_adjusted_gross', 
                                                  ascending=False)

# Display the top 10 movies 
inflation_adjusted_gross_desc.head(10)


# In[5]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\nlast_output = _\n\ndef test_sort_output():\n    try:\n        assert (last_output.iloc[0][0] == \'Snow White and the Seven Dwarfs\'  and \n        last_output.iloc[9][0] == \'The Lion King\')\n    except AttributeError:\n        assert False, \\\n            "Please use head() as the last line of code in the cell to display the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "The data was not sorted correctly. You should see Snow White and the Seven Dwarf in the first row and The Lion King in the tenth row of the output."\n    except IndexError:\n        assert False, \\\n            "Did you return the first 10 rows of the output?"\n        \ndef test_head_output():\n    try:\n        assert last_output.shape == (10,6)\n    except AttributeError:\n        assert False, \\\n            "Please use head() as the last line of code in the cell to display the data, not the display() or print() functions."\n    except AssertionError:\n        assert False, \\\n            "Did you return the first 10 rows of the output?"')


# ## 3. Movie genre trend
# <p>From the top 10 movies above, it seems that some genres are more popular than others. So, we will check which genres are growing stronger in popularity. To do this, we will group movies by genre and then by year to see the adjusted gross of each genre in each year.</p>

# In[6]:


# Extract year from release_date and store it in a new column
gross['release_year'] = pd.DatetimeIndex(gross["release_date"]).year 

# Compute mean of adjusted gross per genre and per year
group = gross.groupby(['genre','release_year']).mean()

# Convert the GroupBy object to a DataFrame
genre_yearly = group.reset_index()

# Inspect genre_yearly 
genre_yearly.head(10)


# In[7]:


get_ipython().run_cell_magic('nose', '', 'import numpy \n# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ncorrect_group_result = gross.groupby([\'genre\',\'release_year\']).mean()\n\ndef test_release_year_exists():\n    assert \'release_year\' in gross, \\\n        "The column release_year was not correctly created."\n\ndef test_release_year_correctly_created():\n    correct_release_year = pd.Series(pd.DatetimeIndex(gross[\'release_date\']).year )\n    assert numpy.array_equal(gross[\'release_year\'].values,correct_release_year.values), \\\n    "The values in the column release_year looks incorrect." \n\ndef test_group_correctly_created():\n    assert isinstance(group, pd.DataFrame),\\\n    "The variable group was not created correctly."\n\ndef test_group_value():\n    assert group.iloc[2][\'total_gross\'] == 17577696 ,\\\n    "The variable group has incorrect values."\n    \ndef test_group_shape():\n    assert correct_group_result.shape == group.shape, \\\n    "The variable group should have 218 rows and 2 columns."\n \ndef test_group_column_names():\n     assert correct_group_result.index.names == group.index.names, \\\n    "The variable group should have two index names in this order: \'genre\', \'release_year\'."\n        \ndef test_genre_yearly_correctly_created():\n    correct_genre_yearly_result = group.reset_index()\n    assert correct_genre_yearly_result.equals(genre_yearly), \\\n    "The variable genre_yearly looks incorrect."\n    #assert isinstance(genre_yearly, pd.DataFrame),\\\n    #"The variable genre_yearly is not a DataFrame." \n\ndef test_genre_yearly_shape():\n    assert genre_yearly.shape[1] == 4, \\\n    \'There should be four columns in the genre_yearly DataFrame.\'')


# ## 4. Visualize the genre popularity trend
# <p>We will make a plot out of these means of groups to better see how box office revenues have changed over time.</p>

# In[8]:


# Import seaborn library
import seaborn as sns

# Plot the data  
sns.relplot(x='release_year', y='inflation_adjusted_gross', kind='line', 
            hue='genre',
            data=genre_yearly)


# In[9]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\nlast_value = _\ndef test_sns_exists():\n    assert \'sns\' in globals(), \\\n        "Did you import the seaborn library aliased as sns?\'"\n\ndef test_plot_exists():\n    assert isinstance(last_value, sns.axisgrid.FacetGrid),\\\n    "Did you created a line plot yet?"\n\ndef test_line_plot():\n    assert len(last_value.ax.get_lines()) == 25,\\\n    "The line plot looks incorrect."\n\ndef test_x_data(): # Test the first x value in the first line \n    #assert list(last_value.ax.get_lines()[0].get_xdata()) == list(genre_yearly[genre_yearly[\'genre\'] ==\'Action\'][\'release_year\']), \\\n    assert last_value.ax.get_lines()[0].get_xdata()[0] in list(gross[\'release_year\']), \\\n    \'The x-axis data looks incorrect.\'\n    \ndef test_y_data(): # Test the first y value in the first line \n        assert last_value.ax.get_lines()[0].get_ydata()[0] in list(gross[\'inflation_adjusted_gross\']), \\\n    \'The y-axis data looks incorrect.\'\n    \ndef test_hue_data():\n    assert list(last_value.ax.get_legend_handles_labels()[1][1::])  == list(genre_yearly[\'genre\'].unique()), \\\n    \'The hue data looks incorrect.\'')


# ## 5. Data transformation
# <p>The line plot supports our belief that some genres are growing faster in popularity than others. For Disney movies, Action and Adventure genres are growing the fastest. Next, we will build a linear regression model to understand the relationship between genre and box office gross. </p>
# <p>Since linear regression requires numerical variables and the genre variable is a categorical variable, we'll use a technique called one-hot encoding to convert the categorical variables to numerical. This technique transforms each category value into a new column and assigns a 1 or 0 to the column. </p>
# <p>For this dataset, there will be 11 dummy variables, one for each genre except the action genre which we will use as a baseline. For example, if a movie is an adventure movie, like The Lion King, the adventure variable will be 1 and other dummy variables will be 0. Since the action genre is our baseline, if a movie is an action movie, such as The Avengers, all dummy variables will be 0.</p>

# In[10]:


# Convert genre variable to dummy variables 
genre_dummies =  pd.get_dummies(data=gross['genre'], drop_first=True)

# Inspect genre_dummies
genre_dummies.head()


# In[11]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_genre_dummies_exists():\n    assert \'genre_dummies\' in globals(), \\\n     "The variable genre_dummies was not correctly created."\n\ndef test_correct_dummies_numbers():\n    assert isinstance(genre_dummies, pd.DataFrame) and genre_dummies.shape == (579, 11), \\\n        "The genre_dummies should be a DataFrame with 11 columns and 579 rows."  \n\ndef test_correct_dummies_values():\n    assert list(genre_dummies.iloc[0]) ==  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], \\\n        "The genre_dummies should contain one 1 and ten 0s for all genres except the action genre."')


# ## 6. The genre effect
# <p>Now that we have dummy variables, we can build a linear regression model to predict the adjusted gross using these dummy variables.</p>
# <p>From the regression model, we can check the effect of each genre by looking at its coefficient given in units of box office gross dollars. We will focus on the impact of action and adventure genres here. (Note that the intercept and the first coefficient values represent the effect of action and adventure genres respectively). We expect that movies like the Lion King or Star Wars would perform better for box office.</p>

# In[12]:


# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Build a linear regression model
regr = LinearRegression()

# Fit regr to the dataset
regr.fit(genre_dummies, gross["inflation_adjusted_gross"])

# Get estimated intercept and coefficient values 
action =  regr.intercept_
adventure = regr.coef_[[0]][0]

# Inspect the estimated intercept and coefficient values 
print((action, adventure))


# In[13]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\nfrom sklearn.utils import validation \nfrom sklearn.exceptions import NotFittedError\n\n\ndef test_LinearRegression_loaded():\n    assert \'LinearRegression\' in globals(), \\\n    \'Did you import LinearRegression from sklearn.linear_model?\'\n\ndef test_regr_exists():\n    assert "regr" in globals(), \\\n        "The variable regr was not correctly created."\n    \ndef test_regr_is_model():\n    assert isinstance(regr, LinearRegression),\\\n    "regr should be a LinearRegression model." \n\ndef test_regr_is_fitted():\n    try:\n        validation.check_is_fitted(regr, "coef_")\n \n    except NotFittedError:\n        assert False, "regr is not fitted yet."\n\ndef test_correct_features():\n    assert   regr.coef_.size == 11, \\\n    "regr doesn\'t have a correct number of coefficients. Did you use the correct regressors X?"\n\n        \ndef test_correct_intercept():\n    assert round(action) == 102921757,\\\n    "The value of action looks incorrect."\n    ')


# ## 7. Confidence intervals for regression parameters  (i)
# <p>Next, we will compute 95% confidence intervals for the intercept and coefficients. The 95% confidence intervals for the intercept  <b><i>a</i></b> and coefficient <b><i>b<sub>i</sub></i></b> means that the intervals have a probability of 95% to contain the true value <b><i>a</i></b> and coefficient <b><i>b<sub>i</sub></i></b> respectively. If there is a significant relationship between a given genre and the adjusted gross, the confidence interval of its coefficient should exclude 0.      </p>
# <p>We will calculate the confidence intervals using the pairs bootstrap method. </p>

# In[14]:


# Import a module
import numpy as np

# Create an array of indices to sample from 
inds = np.arange(len(gross['genre']))

# Initialize 500 replicate arrays
size = 500
bs_action_reps =  np.empty(size)
bs_adventure_reps =  np.empty(size)


# In[15]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_inds_exists():\n    assert "inds" in globals(), \\\n        "The variable inds was not correctly created."\n    \ndef test_inds_type(): \n    assert  isinstance(inds, np.ndarray) ,\\\n    "Type of inds should be ndarray."\n\n#def test_inds_shape(): \n#    assert  inds.size==579  ,\\\n#    "Length of inds should be 579."    \n\ndef test_inds_shape(): \n    try:\n        assert inds.size==579\n    except AttributeError:\n        assert False, \\\n            "Make sure that inds is an ndarray of length 579."\n    except AssertionError:\n        assert False, \\\n            "Length of inds should be 579."  \n        \ndef test_bs_action_reps_type(): \n    assert  isinstance(bs_action_reps, np.ndarray) ,\\\n    "Type of bs_action_reps should be ndarray."\n    \n#def test_bs_action_reps_shape(): \n#    assert bs_action_reps.size == 1000  ,\\\n#    "Length of bs_action_reps should be 1000."\n\ndef test_bs_action_reps_shape(): \n    try:\n        assert bs_action_reps.size == 500\n    except AttributeError:\n        assert False, \\\n            "Make sure that bs_action_reps is an ndarray of length 500."\n    except AssertionError:\n        assert False, \\\n            "Length of bs_action_reps should be 500."\n        \ndef test_bs_adventure_reps_type(): \n    assert  isinstance(bs_adventure_reps, np.ndarray) ,\\\n    "Type of bs_adventure_reps should be ndarray."\n \n\n#def test_bs_adventure_reps_shape(): \n#    assert bs_adventure_reps.size == 500  ,\\\n#    "Length of bs_adventure_reps should be 500."\n    \ndef test_bs_adventure_reps_shape(): \n    try:\n        assert bs_adventure_reps.size == 500\n    except AttributeError:\n        assert False, \\\n            "Make sure that bs_adventure_reps is an ndarray of length 500."\n    except AssertionError:\n        assert False, \\\n            "Length of bs_adventure_reps should be 500."')


# ## 8. Confidence intervals for regression parameters  (ii)
# <p>After the initialization, we will perform pair bootstrap estimates for the regression parameters. Note that we will draw a sample from a set of (genre, adjusted gross) data where the genre is the original genre variable. We will perform one-hot encoding after that. </p>

# In[16]:


# Generate replicates  
for i in range(size):
    
    # Resample the indices 
    bs_inds = np.random.choice(inds, size=len(inds))
    
    # Get the sampled genre and sampled adjusted gross
    bs_genre = gross['genre'][bs_inds] 
    bs_gross = gross['inflation_adjusted_gross'][bs_inds]
    
    # Convert sampled genre to dummy variables
    bs_dummies = pd.get_dummies(data=gross['genre'] , drop_first=True)
   
    # Build and fit a regression model 
    regr = LinearRegression().fit(bs_dummies, bs_gross)
    
    # Compute replicates of estimated intercept and coefficient
    bs_action_reps[i] = regr.intercept_
    bs_adventure_reps[i] = regr.coef_[[0]][0]


# In[17]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_bs_inds_exists():\n     assert \'bs_inds\' in globals(), \\\n    \'Did you create the variable bs_inds to store the resampled indices?\'\n\ndef test_bs_inds_type():\n     assert isinstance(bs_inds, np.ndarray), \\\n    \'Type of bs_inds should be ndarray.\' \n  \ndef test_bs_inds_size():\n     assert bs_inds.size == 579, \\\n    \'Size of bs_inds should be 579.\' \n   \ndef test_bs_gross_type(): \n    assert  isinstance(bs_gross, pd.core.series.Series) ,\\\n    "Type of bs_gross should be Pandas Series."\n    \ndef test_bs_gross_dtype(): \n    assert  bs_gross.dtype == \'int64\' ,\\\n    "Type of elements in bs_gross should be int64."\n    \ndef test_bs_gross_shape(): \n    assert  bs_gross.size == 579  ,\\\n    "Size of bs_gross should be 579."\n\ndef test_bs_gross_is_randomized(): \n    assert not bs_gross.equals(gross[\'inflation_adjusted_gross\'])  ,\\\n    "bs_gross should not be the same as the inflation_adjusted_gross."\n    \ndef test_bs_dummies_exists():\n     assert \'bs_dummies\' in globals(), \\\n    \'The variable bs_dummies was not correctly created.\'\n\ndef test_bs_dummies_DataFrame():\n    assert isinstance(bs_dummies, pd.DataFrame), \\\n        "bs_dummies should be a DataFrame."  \n    \ndef test_correct_bs_dummies_numbers():\n    assert  bs_dummies.shape == (579, 11), \\\n        "bs_dummies should be a DataFrame with 11 columns and 579 rows."  \n\ndef test_correct_bs_dummies_values(): \n    assert set(bs_dummies.iloc[0]) == {0, 1} or set(bs_dummies.iloc[0]) == {0}, \\\n        "The values of bs_dummies looks incorrect."  \n\ndef test_bs_action_reps_type(): \n    assert  isinstance(bs_action_reps, np.ndarray) ,\\\n    "Type of bs_action_reps should be a NumPy array."\n\ndef test_bs_action_reps_shape(): \n    assert  bs_action_reps.size == 500  ,\\\n    "Size of bs_gross should be 500."\n    \ndef test_bs_action_reps_value():\n    assert bs_action_reps[499] == regr.intercept_,\\\n        "The value of bs_action_reps looks incorrect."')


# ## 9. Confidence intervals for regression parameters (iii)
# <p>Finally, we compute 95% confidence intervals for the intercept and coefficient and examine if they exclude 0. If one of them (or both) does, then it is unlikely that the value is 0 and we can conclude that there is a significant relationship between that genre and the adjusted gross. </p>

# In[18]:


# Compute 95% confidence intervals for intercept and coefficient values
confidence_interval_action = np.percentile(bs_action_reps, [2.5, 97.5]) 
confidence_interval_adventure = np.percentile(bs_adventure_reps, [2.5, 97.5])
    
# Inspect the confidence intervals
print(confidence_interval_action)
print(confidence_interval_adventure)


# In[19]:


get_ipython().run_cell_magic('nose', '', '# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student\'s code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ncorrect_confidence_interval_action = np.percentile(bs_action_reps, [2.5, 97.5])\ncorrect_confidence_interval_adventure = np.percentile(bs_adventure_reps, [2.5, 97.5])\n\ndef test_confidence_interval_action_exists():\n    assert \'confidence_interval_action\' in globals(),\\\n    "confidence_interval_action was not correctly created."\n    \ndef test_confidence_interval_adventure_exists():\n    assert \'confidence_interval_adventure\' in globals(),\\\n    "confidence_interval_adventure was not correctly created."\n   \ndef test_confidence_interval_action_type(): \n    assert  isinstance(confidence_interval_action, np.ndarray) ,\\\n    "Type of confidence_interval_action should be a NumPy array."\n    \ndef test_confidence_interval_action_values(): \n    assert (confidence_interval_action == correct_confidence_interval_action).all(),\\\n    "Values in confidence_interval_action look incorrect."\n    \ndef test_confidence_interval_action_shape(): \n    assert  confidence_interval_action.size == 2  ,\\\n    "Length of confidence_interval_action should be 2."\n\ndef test_confidence_interval_adventure_type(): \n    assert  isinstance(confidence_interval_adventure, np.ndarray) ,\\\n    "Type of confidence_interval_adventure should be a Numpy array."\n    \ndef test_confidence_interval_adventure_values(): \n    assert  (confidence_interval_adventure == correct_confidence_interval_adventure).all(),\\\n    "Values in confidence_interval_adventure look incorrect."\n      \ndef test_confidence_interval_adventure_shape(): \n    assert  confidence_interval_adventure.size == 2  ,\\\n    "Length of confidence_interval_adventure should be 2."')


# ## 10. Should Disney make more action and adventure movies?
# <p>The confidence intervals from the bootstrap method for the intercept and coefficient do not contain the value zero, as we have already seen that lower and upper bounds of both confidence intervals are positive. These tell us that it is likely that the adjusted gross is significantly correlated with the action and adventure genres. </p>
# <p>From the results of the bootstrap analysis and the trend plot we have done earlier, we could say that Disney movies with plots that fit into the action and adventure genre, according to our data, tend to do better in terms of adjusted gross than other genres. So we could expect more Marvel, Star Wars, and live-action movies in the upcoming years!</p>

# In[20]:


# should Disney studios make more action and adventure movies? 
more_action_adventure_movies = True


# In[21]:


get_ipython().run_cell_magic('nose', '', "# %%nose needs to be included at the beginning of every @tests cell\n\n# One or more tests of the student's code\n# The @solution should pass the tests\n# The purpose of the tests is to try to catch common errors and\n# to give the student a hint on how to resolve these errors\n\ndef test_more_action_adventure_movies_type():\n    assert isinstance(more_action_adventure_movies, bool), \\\n        'more_action_adventure_movies should be of Boolean type.'\n    \ndef test_conclusion():\n    assert more_action_adventure_movies, \\\n        'That is not a reasonable conclusion given the values of confidence intervals.'")

