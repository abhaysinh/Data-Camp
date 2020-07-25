
# coding: utf-8

# ## 1. Introduction
# <p>Everyone loves Lego (unless you ever stepped on one). Did you know by the way that "Lego" was derived from the Danish phrase leg godt, which means "play well"? Unless you speak Danish, probably not. </p>
# <p>In this project, we will analyze a fascinating dataset on every single lego block that has ever been built!</p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_10/datasets/lego-bricks.jpeg" alt="lego"></p>

# In[2]:


# Nothing to do here


# In[3]:


get_ipython().run_cell_magic('nose', '', 'def test_default():\n  assert True')


# ## 2. Reading Data
# <p>A comprehensive database of lego blocks is provided by <a href="https://rebrickable.com/downloads/">Rebrickable</a>. The data is available as csv files and the schema is shown below.</p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_10/datasets/downloads_schema.png" alt="schema"></p>
# <p>Let us start by reading in the colors data to get a sense of the diversity of lego sets!</p>

# In[4]:


# Import modules
import pandas as pd

# Read colors data
colors = pd.read_csv('datasets/colors.csv')

# Print the first few rows
colors.head()


# In[5]:


get_ipython().run_cell_magic('nose', '', 'def test_colors_exists():\n    assert \'colors\' in globals(), "You should read the data into a variable named `colors`"')


# ## 3. Exploring Colors
# <p>Now that we have read the <code>colors</code> data, we can start exploring it! Let us start by understanding the number of colors available.</p>

# In[6]:


# How many distinct colors are available?
num_colors = colors.shape[0]

num_colors


# In[7]:


get_ipython().run_cell_magic('nose', '', 'def test_num_colors():\n    assert num_colors == 135, "The variable num_colors should equal 135"')


# ## 4. Transparent Colors in Lego Sets
# <p>The <code>colors</code> data has a column named <code>is_trans</code> that indicates whether a color is transparent or not. It would be interesting to explore the distribution of transparent vs. non-transparent colors.</p>

# In[8]:


# colors_summary: Distribution of colors based on transparency
colors_summary = colors.groupby(colors['is_trans']).count()
colors_summary


# In[9]:


get_ipython().run_cell_magic('nose', '', 'def test_colors_summary_exists():\n    assert \'colors_summary\' in globals(), "You should have defined a variable named `colors_summary`"\ndef test_colors_summary():\n    assert colors_summary.shape == (2, 3), "The DataFrame colors_summary should contain 2 rows and 3 columns"')


# ## 5. Explore Lego Sets
# <p>Another interesting dataset available in this database is the <code>sets</code> data. It contains a comprehensive list of sets over the years and the number of parts that each of these sets contained. </p>
# <p><img src="https://imgur.com/1k4PoXs.png" alt="sets_data"></p>
# <p>Let us use this data to explore how the average number of parts in Lego sets has varied over the years.</p>

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Read sets data as `sets`
sets = pd.read_csv('datasets/sets.csv')

print(sets.head())

# Create a summary of average number of parts by year: `parts_by_year`
parts_by_year = sets[['year', 'num_parts']].groupby('year', as_index=False).count()
# Plot trends in average number of parts by year
parts_by_year.plot(x = 'year', y = 'num_parts')
parts_by_year.head()


# In[11]:


get_ipython().run_cell_magic('nose', '', 'def test_sets_exists():\n    assert \'sets\' in globals(), "You should read the data into a variable named `sets`"\ndef test_parts_by_year_exists():\n    assert \'parts_by_year\' in globals(), "You should have defined a variable named `parts_by_year`"')


# ## 6. Lego Themes Over Years
# <p>Lego blocks ship under multiple <a href="https://shop.lego.com/en-US/Themes">themes</a>. Let us try to get a sense of how the number of themes shipped has varied over the years.</p>

# In[12]:


# themes_by_year: Number of themes shipped by year
themes_by_year = sets[['year', 'theme_id']].groupby('year', as_index=False).count()
themes_by_year.head()


# In[13]:


get_ipython().run_cell_magic('nose', '', 'def test_themes_by_year_exists():\n    assert \'themes_by_year\' in globals(), "You should have defined a variable named `themes_by_year`"\ndef test_themes_by_year():\n    assert themes_by_year.shape == (66, 2), "The DataFrame themes_by_year should contain 66 rows and 2 columns"\ndef test_themes_by_year_names():\n    colnames = [\'year\', \'theme_id\']\n    assert all(name in themes_by_year for name in colnames), "Your DataFrame, bnames, should have columns named: year, theme_id"')


# ## 7. Wrapping It All Up!
# <p>Lego blocks offer an unlimited amount of fun across ages. We explored some interesting trends around colors, parts, and themes. </p>

# In[14]:


# Nothing to do here


# In[15]:


get_ipython().run_cell_magic('nose', '', 'def test_default():\n  assert True')

