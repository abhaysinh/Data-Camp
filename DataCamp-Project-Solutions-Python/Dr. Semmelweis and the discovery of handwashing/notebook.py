
# coding: utf-8

# ## 1. Meet Dr. Ignaz Semmelweis
# <p><img style="float: left;margin:5px 20px 5px 1px" src="https://assets.datacamp.com/production/project_20/img/ignaz_semmelweis_1860.jpeg"></p>
# <!--
# <img style="float: left;margin:5px 20px 5px 1px" src="https://assets.datacamp.com/production/project_20/datasets/ignaz_semmelweis_1860.jpeg">
# -->
# <p>This is Dr. Ignaz Semmelweis, a Hungarian physician born in 1818 and active at the Vienna General Hospital. If Dr. Semmelweis looks troubled it's probably because he's thinking about <em>childbed fever</em>: A deadly disease affecting women that just have given birth. He is thinking about it because in the early 1840s at the Vienna General Hospital as many as 10% of the women giving birth die from it. He is thinking about it because he knows the cause of childbed fever: It's the contaminated hands of the doctors delivering the babies. And they won't listen to him and <em>wash their hands</em>!</p>
# <p>In this notebook, we're going to reanalyze the data that made Semmelweis discover the importance of <em>handwashing</em>. Let's start by looking at the data that made Semmelweis realize that something was wrong with the procedures at Vienna General Hospital.</p>

# In[30]:


# importing modules
import pandas as pd 

# Read datasets/yearly_deaths_by_clinic.csv into yearly
yearly = pd.read_csv('datasets/yearly_deaths_by_clinic.csv')

# Print out yearly
print(yearly)


# In[31]:


get_ipython().run_cell_magic('nose', '', '\nimport pandas as pd\n\ndef test_yearly_exists():\n    assert "yearly" in globals(), \\\n        "The variable yearly should be defined."\n        \ndef test_yearly_correctly_loaded():\n    correct_yearly = pd.read_csv("datasets/yearly_deaths_by_clinic.csv")\n    try:\n        pd.testing.assert_frame_equal(yearly, correct_yearly)\n    except AssertionError:\n        assert False, "The variable yearly should contain the data in yearly_deaths_by_clinic.csv"\n        ')


# ## 2. The alarming number of deaths
# <p>The table above shows the number of women giving birth at the two clinics at the Vienna General Hospital for the years 1841 to 1846. You'll notice that giving birth was very dangerous; an <em>alarming</em> number of women died as the result of childbirth, most of them from childbed fever.</p>
# <p>We see this more clearly if we look at the <em>proportion of deaths</em> out of the number of women giving birth. Let's zoom in on the proportion of deaths at Clinic 1.</p>

# In[32]:


# Calculate proportion of deaths per no. births
yearly["proportion_deaths"] = yearly.deaths/yearly.births

# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2
yearly1 = yearly[yearly['clinic'] == 'clinic 1']
yearly2 = yearly[yearly['clinic'] == 'clinic 2']

# Print out yearly1
print(yearly)


# In[33]:


get_ipython().run_cell_magic('nose', '', '\ndef test_proportion_deaths_exists():\n    assert \'proportion_deaths\' in yearly, \\\n        "The DataFrame yearly should have the column proportion_deaths"\n\ndef test_proportion_deaths_is_correctly_calculated():\n    assert all(yearly["proportion_deaths"] == yearly["deaths"] / yearly["births"]), \\\n        "The column proportion_deaths should be the number of deaths divided by the number of births."\n   \ndef test_yearly1_correct_shape():\n    assert yearly1.shape == yearly[yearly["clinic"] == "clinic 1"].shape, \\\n        "yearly1 should contain the rows in yearly from clinic 1"\n\ndef test_yearly2_correct_shape():\n    assert yearly2.shape == yearly[yearly["clinic"] == "clinic 2"].shape, \\\n        "yearly2 should contain the rows in yearly from clinic 2"')


# ## 3. Death at the clinics
# <p>If we now plot the proportion of deaths at both clinic 1 and clinic 2  we'll see a curious pattern...</p>

# In[34]:


# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot yearly proportion of deaths at the two clinics
ax = yearly1.plot(x='year', y='proportion_deaths', label='clinic1')
yearly2.plot(x='year', y='proportion_deaths', label='clinic2', ax=ax)
ax.set_ylabel('Proportion deaths')


# In[35]:


get_ipython().run_cell_magic('nose', '', '\ndef test_ax_exists():\n    assert \'ax\' in globals(), \\\n        "The result of the plot method should be assigned to a variable called ax"\n        \ndef test_plot_plots_correct_data():\n    y0 = ax.get_lines()[0].get_ydata()\n    y1 = ax.get_lines()[1].get_ydata()\n    assert (\n        (all(yearly1["proportion_deaths"] == y0) and\n         all(yearly2["proportion_deaths"] == y1))\n        or\n        (all(yearly1["proportion_deaths"] == y1) and\n         all(yearly2["proportion_deaths"] == y0))), \\\n        "The data from clinic 1 and clinic 2 should be plotted as two separate lines."')


# ## 4. The handwashing begins
# <p>Why is the proportion of deaths constantly so much higher in Clinic 1? Semmelweis saw the same pattern and was puzzled and distressed. The only difference between the clinics was that many medical students served at Clinic 1, while mostly midwife students served at Clinic 2. While the midwives only tended to the women giving birth, the medical students also spent time in the autopsy rooms examining corpses. </p>
# <p>Semmelweis started to suspect that something on the corpses, spread from the hands of the medical students, caused childbed fever. So in a desperate attempt to stop the high mortality rates, he decreed: <em>Wash your hands!</em> This was an unorthodox and controversial request, nobody in Vienna knew about bacteria at this point in time. </p>
# <p>Let's load in monthly data from Clinic 1 to see if the handwashing had any effect.</p>

# In[36]:


# Read datasets/monthly_deaths.csv into monthly
monthly = pd.read_csv('datasets/monthly_deaths.csv', parse_dates=['date'])

# Calculate proportion of deaths per no. births
monthly['proportion_deaths'] = monthly.deaths/monthly.births

# Print out the first rows in monthly
monthly.head()


# In[37]:


get_ipython().run_cell_magic('nose', '', '\ndef test_monthly_exists():\n    assert "monthly" in globals(), \\\n        "The variable monthly should be defined."\n        \ndef test_monthly_correctly_loaded():\n    correct_monthly = pd.read_csv("datasets/monthly_deaths.csv")\n    try:\n        pd.testing.assert_series_equal(monthly["births"], correct_monthly["births"])\n    except AssertionError:\n        assert False, "The variable monthly should contain the data in monthly_deaths.csv"\n\ndef test_date_correctly_converted():\n    assert monthly.date.dtype == pd.to_datetime(pd.Series("1847-06-01")).dtype, \\\n        "The column date should be converted using the pd.to_datetime() function"        \n        \ndef test_proportion_deaths_is_correctly_calculated():\n    assert all(monthly["proportion_deaths"] == monthly["deaths"] / monthly["births"]), \\\n        "The column proportion_deaths should be the number of deaths divided by the number of births."')


# ## 5. The effect of handwashing
# <p>With the data loaded we can now look at the proportion of deaths over time. In the plot below we haven't marked where obligatory handwashing started, but it reduced the proportion of deaths to such a degree that you should be able to spot it!</p>

# In[38]:


# Plot monthly proportion of deaths
# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plot yearly proportion of deaths at the two clinics
ax = monthly['proportion_deaths'].plot(x='date', y='proportion_deaths', label='clinic1') 
monthly['proportion_deaths'].plot(x='date', y='proportion_deaths', label='clinic2', ax=ax)
ax.set_ylabel("Proportion deaths")


# In[39]:


get_ipython().run_cell_magic('nose', '', '        \ndef test_ax_exists():\n    assert \'ax\' in globals(), \\\n        "The result of the plot method should be assigned to a variable called ax"\n\ndef test_plot_plots_correct_data():\n    y0 = ax.get_lines()[0].get_ydata()\n    assert all(monthly["proportion_deaths"] == y0), \\\n        "The plot should show the column \'proportion_deaths\' in monthly."')


# ## 6. The effect of handwashing highlighted
# <p>Starting from the summer of 1847 the proportion of deaths is drastically reduced and, yes, this was when Semmelweis made handwashing obligatory. </p>
# <p>The effect of handwashing is made even more clear if we highlight this in the graph.</p>

# In[40]:


# Date when handwashing was made mandatory
get_ipython().run_line_magic('matplotlib', 'inline')

# Date when handwashing was made mandatory
import pandas as pd
handwashing_start = pd.to_datetime('1847-06-01')

# Split monthly into before and after handwashing_start
before_washing = monthly[monthly['date'] < handwashing_start]
after_washing = monthly[monthly['date'] >= handwashing_start]

# Plot monthly proportion of deaths before and after handwashing
ax = before_washing.plot(x='date', y='proportion_deaths', label='clinic1') 
after_washing.plot(x='date', y='proportion_deaths', label='clinic2', ax=ax)
ax.set_ylabel("Proportion deaths")


# In[41]:


get_ipython().run_cell_magic('nose', '', '\ndef test_before_washing_correct():\n    correct_before_washing = monthly[monthly["date"] < handwashing_start]\n    try:\n        pd.testing.assert_frame_equal(before_washing, correct_before_washing)\n    except AssertionError:\n        assert False, "before_washing should contain the rows of monthly < handwashing_start" \n\ndef test_after_washing_correct():\n    correct_after_washing = monthly[monthly["date"] >= handwashing_start]\n    try:\n        pd.testing.assert_frame_equal(after_washing, correct_after_washing)\n    except AssertionError:\n        assert False, "after_washing should contain the rows of monthly >= handwashing_start" \n\ndef test_ax_exists():\n    assert \'ax\' in globals(), \\\n        "The result of the plot method should be assigned to a variable called ax"\n\n        \ndef test_plot_plots_correct_data():\n    y0_len = ax.get_lines()[0].get_ydata().shape[0]\n    y1_len = ax.get_lines()[1].get_ydata().shape[0]\n    assert (\n        (before_washing["proportion_deaths"].shape[0] == y0_len and\n         after_washing["proportion_deaths"].shape[0] == y1_len)\n        or\n        (before_washing["proportion_deaths"].shape[0] == y0_len and\n         after_washing["proportion_deaths"].shape[0] == y1_len)), \\\n        "The data in before_washing and after_washing should be plotted as two separate lines."')


# ## 7. More handwashing, fewer deaths?
# <p>Again, the graph shows that handwashing had a huge effect. How much did it reduce the monthly proportion of deaths on average?</p>

# In[42]:


# Difference in mean monthly proportion of deaths due to handwashing
before_proportion = before_washing['proportion_deaths']
after_proportion = after_washing['proportion_deaths']
mean_diff = after_proportion.mean() - before_proportion.mean()
mean_diff


# In[43]:


get_ipython().run_cell_magic('nose', '', '        \ndef test_before_proportion_exists():\n    assert \'before_proportion\' in globals(), \\\n        "before_proportion should be defined"\n        \ndef test_after_proportion_exists():\n    assert \'after_proportion\' in globals(), \\\n        "after_proportion should be defined"\n        \ndef test_mean_diff_exists():\n    assert \'mean_diff\' in globals(), \\\n        "mean_diff should be defined"\n        \ndef test_before_proportion_is_a_series():\n     assert hasattr(before_proportion, \'__len__\') and len(before_proportion) == 76, \\\n        "before_proportion should be 76 elements long, and not a single number."\n\ndef test_correct_mean_diff():\n    correct_before_proportion = before_washing["proportion_deaths"]\n    correct_after_proportion = after_washing["proportion_deaths"]\n    correct_mean_diff = correct_after_proportion.mean() - correct_before_proportion.mean()\n    assert mean_diff == correct_mean_diff, \\\n        "mean_diff should be calculated as the mean of after_proportion minus the mean of before_proportion."')


# ## 8. A Bootstrap analysis of Semmelweis handwashing data
# <p>It reduced the proportion of deaths by around 8 percentage points! From 10% on average to just 2% (which is still a high number by modern standards). </p>
# <p>To get a feeling for the uncertainty around how much handwashing reduces mortalities we could look at a confidence interval (here calculated using the bootstrap method).</p>

# In[44]:


# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = before_proportion.sample(frac=1, replace=True)
    boot_after = after_proportion.sample(frac=1, replace=True)
    boot_mean_diff.append(boot_after.mean() - boot_before.mean())

# Calculating a 95% confidence interval from boot_mean_diff 
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])
confidence_interval


# In[45]:


get_ipython().run_cell_magic('nose', '', '\ndef test_confidence_interval_exists():\n    assert \'confidence_interval\' in globals(), \\\n        "confidence_interval should be defined"\n\ndef test_boot_before_correct_length():\n    assert len(boot_before) == len(before_proportion), \\\n        ("boot_before have {} elements and before_proportion have {}." + \n         "They should have the same number of elements."\n        ).format(len(boot_before), len(before_proportion))\n        \ndef test_confidence_interval_correct():\n    assert ((0.09 < abs(confidence_interval).max() < 0.11) and\n            (0.055 < abs(confidence_interval).min() < 0.075)) , \\\n        "confidence_interval should be calculated as the [0.025, 0.975] quantiles of boot_mean_diff."')


# ## 9. The fate of Dr. Semmelweis
# <p>So handwashing reduced the proportion of deaths by between 6.7 and 10 percentage points, according to a 95% confidence interval. All in all, it would seem that Semmelweis had solid evidence that handwashing was a simple but highly effective procedure that could save many lives.</p>
# <p>The tragedy is that, despite the evidence, Semmelweis' theory — that childbed fever was caused by some "substance" (what we today know as <em>bacteria</em>) from autopsy room corpses — was ridiculed by contemporary scientists. The medical community largely rejected his discovery and in 1849 he was forced to leave the Vienna General Hospital for good.</p>
# <p>One reason for this was that statistics and statistical arguments were uncommon in medical science in the 1800s. Semmelweis only published his data as long tables of raw data, but he didn't show any graphs nor confidence intervals. If he would have had access to the analysis we've just put together he might have been more successful in getting the Viennese doctors to wash their hands.</p>

# In[46]:


# The data Semmelweis collected points to that:
doctors_should_wash_their_hands = True


# In[47]:


get_ipython().run_cell_magic('nose', '', '\ndef test_doctors_should_was_their_hands():\n    assert doctors_should_wash_their_hands, \\\n        "Semmelweis would argue that doctors_should_wash_their_hands should be True ."')

