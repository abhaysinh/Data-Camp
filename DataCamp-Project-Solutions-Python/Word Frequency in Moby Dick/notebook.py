
# coding: utf-8

# ## 1. Tools for text processing
# <p><img style="float: right ; margin: 5px 20px 5px 10px; width: 45%" src="https://s3.amazonaws.com/assets.datacamp.com/production/project_38/img/Moby_Dick_p510_illustration.jpg"> </p>
# <p>What are the most frequent words in Herman Melville's novel, Moby Dick, and how often do they occur?</p>
# <p>In this notebook, we'll scrape the novel <em>Moby Dick</em> from the website <a href="https://www.gutenberg.org/">Project Gutenberg</a> (which contains a large corpus of books) using the Python package <code>requests</code>. Then we'll extract words from this web data using <code>BeautifulSoup</code>. Finally, we'll dive into analyzing the distribution of words using the Natural Language ToolKit (<code>nltk</code>). </p>
# <p>The <em>Data Science pipeline</em> we'll build in this notebook can be used to visualize the word frequency distributions of any novel that you can find on Project Gutenberg. The natural language processing tools used here apply to much of the data that data scientists encounter as a vast proportion of the world's data is unstructured data and includes a great deal of text.</p>
# <p>Let's start by loading in the three main Python packages we are going to use.</p>

# In[22]:


# Importing requests, BeautifulSoup and nltk
import requests
from bs4 import BeautifulSoup
import nltk


#http://docs.python-requests.org/en/master/user/quickstart/#response-content
#https://www.crummy.com/software/BeautifulSoup/bs4/doc/#quick-start
#http://www.nltk.org/api/nltk.tokenize.html?highlight=regexp#module-nltk.tokenize.regexp
#http://www.nltk.org/book/ch02.html#wordlist-corpora


# In[23]:


get_ipython().run_cell_magic('nose', '', "\nimport sys\n\ndef test_example():\n    assert ('requests' in sys.modules and \n            'bs4' in sys.modules and\n            'nltk' in sys.modules ), \\\n    'The modules requests, BeautifulSoup, and nltk should be imported.'")


# ## 2. Request Moby Dick
# <p>To analyze Moby Dick, we need to get the contents of Moby Dick from <em>somewhere</em>. Luckily, the text is freely available online at Project Gutenberg as an HTML file: https://www.gutenberg.org/files/2701/2701-h/2701-h.htm .</p>
# <p><strong>Note</strong> that HTML stands for Hypertext Markup Language and is the standard markup language for the web.</p>
# <p>To fetch the HTML file with Moby Dick we're going to use the <code>request</code> package to make a <code>GET</code> request for the website, which means we're <em>getting</em> data from it. This is what you're doing through a browser when visiting a webpage, but now we're getting the requested page directly into Python instead. </p>

# In[24]:


# Getting the Moby Dick HTML 
r = requests.get('https://s3.amazonaws.com/assets.datacamp.com/production/project_147/datasets/2701-h.htm')


# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Printing the first 2000 characters in html
print(html[0:2000])


# In[25]:


get_ipython().run_cell_magic('nose', '', '\ndef test_r_correct():\n    assert r.request.path_url == \'/assets.datacamp.com/production/project_147/datasets/2701-h.htm\', \\\n    "r should be a get request for \'https://s3.amazonaws.com/assets.datacamp.com/production/project_147/datasets/2701-h.htm\'"\n\ndef test_text_read_in_correctly():\n    assert len(html) == 1500996, \\\n    \'html should contain the text of the request r.\'')


# ## 3. Get the text from the HTML
# <p>This HTML is not quite what we want. However, it does <em>contain</em> what we want: the text of <em>Moby Dick</em>. What we need to do now is <em>wrangle</em> this HTML to extract the text of the novel. For this we'll use the package <code>BeautifulSoup</code>.</p>
# <p>Firstly, a word on the name of the package: Beautiful Soup? In web development, the term "tag soup" refers to structurally or syntactically incorrect HTML code written for a web page. What Beautiful Soup does best is to make tag soup beautiful again and to extract information from it with ease! In fact, the main object created and queried when using this package is called <code>BeautifulSoup</code>. After creating the soup, we can use its <code>.get_text()</code> method to extract the text.</p>

# In[26]:


# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, 'html.parser')

# Getting the text out of the soup
text = soup.get_text()

# Printing out text between characters 32000 and 34000
print(text[32000:34000])


# In[27]:


get_ipython().run_cell_magic('nose', '', "\nimport bs4\n\ndef test_text_correct_type():\n    assert isinstance(text, str), \\\n    'text should be a string.'\n    \ndef test_soup_correct_type():\n    assert isinstance(soup, bs4.BeautifulSoup), \\\n    'soup should be a BeautifulSoup object.'\n    ")


# ## 4. Extract the words
# <p>We now have the text of the novel! There is some unwanted stuff at the start and some unwanted stuff at the end. We could remove it, but this content is so much smaller in amount than the text of Moby Dick that, to a first approximation, it is okay to leave it in.</p>
# <p>Now that we have the text of interest, it's time to count how many times each word appears, and for this we'll use <code>nltk</code> – the Natural Language Toolkit. We'll start by tokenizing the text, that is, remove everything that isn't a word (whitespace, punctuation, etc.) and then split the text into a list of words.</p>

# In[28]:


# Creating a tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')

# Tokenizing the text
tokens = tokenizer.tokenize(text)

# Printing out the first 8 words / tokens 
print(tokens[0:8])


# In[29]:


get_ipython().run_cell_magic('nose', '', "\nimport nltk\n\ndef test_correct_tokenizer():\n    correct_tokenizer = nltk.tokenize.RegexpTokenizer('\\w+')\n    assert isinstance(tokenizer, nltk.tokenize.regexp.RegexpTokenizer), \\\n    'tokenizer should be created using the function nltk.tokenize.RegexpTokenizer.'\n    \ndef test_correct_tokens():\n    correct_tokenizer = nltk.tokenize.RegexpTokenizer('\\w+')\n    correct_tokens = correct_tokenizer.tokenize(text)\n    assert isinstance(tokens, list) and len(tokens) > 150000 , \\\n    'tokens should be a list with the words in text.'")


# ## 5. Make the words lowercase
# <p>OK! We're nearly there. Note that in the above 'Or' has a capital 'O' and that in other places it may not, but both 'Or' and 'or' should be counted as the same word. For this reason, we should build a list of all words in <em>Moby Dick</em> in which all capital letters have been made lower case.</p>

# In[30]:


# A new list to hold the lowercased words
words = []

# Looping through the tokens and make them lower case
for word in tokens:
    words.append(word.lower())

# Printing out the first 8 words / tokens 
print(words[0:8])


# In[31]:


get_ipython().run_cell_magic('nose', '', "\ncorrect_words = []\nfor word in tokens:\n    correct_words.append(word.lower())\n\ndef test_correct_words():\n    assert correct_words == words, \\\n    'words should contain every element in tokens, but lowercased.'")


# ## 6. Load in stop words
# <p>It is common practice to remove words that appear a lot in the English language such as 'the', 'of' and 'a' because they're not so interesting. Such words are known as <em>stop words</em>. The package <code>nltk</code> includes a good list of stop words in English that we can use.</p>

# In[32]:


# Getting the English stop words from nltk
from nltk.corpus import stopwords
sw = stopwords.words('english')

# Printing out the first eight stop words
print(sw[0:5])


# In[33]:


get_ipython().run_cell_magic('nose', '', "\ndef test_correct_sw():\n    correct_sw = nltk.corpus.stopwords.words('english')\n    assert correct_sw == sw, \\\n    'sw should contain the stop words from nltk.'")


# ## 7. Remove stop words in Moby Dick
# <p>We now want to create a new list with all <code>words</code> in Moby Dick, except those that are stop words (that is, those words listed in <code>sw</code>). One way to get this list is to loop over all elements of <code>words</code> and add each word to a new list if they are <em>not</em> in <code>sw</code>.</p>

# In[34]:


# A new list to hold Moby Dick with No Stop words
words_ns = []

# Appending to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)

# Printing the first 5 words_ns to check that stop words are gone
print(words_ns[0:5])


# In[35]:


get_ipython().run_cell_magic('nose', '', "\ndef test_correct_words_ns():\n    correct_words_ns = []\n    for word in words:\n        if word not in sw:\n            correct_words_ns.append(word)\n    assert correct_words_ns == words_ns, \\\n    'words_ns should contain all words of Moby Dick but with the stop words removed.'")


# ## 8. We have the answer
# <p>Our original question was:</p>
# <blockquote>
#   <p>What are the most frequent words in Herman Melville's novel Moby Dick and how often do they occur?</p>
# </blockquote>
# <p>We are now ready to answer that! Let's create a word frequency distribution plot using <code>nltk</code>. </p>

# In[36]:


# This command display figures inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Creating the word frequency distribution
freqdist = nltk.FreqDist(words_ns)


# Plotting the word frequency distribution
freqdist.plot(25)


# In[37]:


get_ipython().run_cell_magic('nose', '', "\ndef test_correct_freqdist():\n    correct_freqdist = nltk.FreqDist(words_ns)\n    assert correct_freqdist == freqdist, \\\n    'freqdist should contain the word frequencies of words_ns.'")


# ## 9. The most common word
# <p>Nice! The frequency distribution plot above is the answer to our question. </p>
# <p>The natural language processing skills we used in this notebook are also applicable to much of the data that Data Scientists encounter as the vast proportion of the world's data is unstructured data and includes a great deal of text. </p>
# <p>So, what word turned out to (<em>not surprisingly</em>) be the most common word in Moby Dick?</p>

# In[38]:


# What's the most common word in Moby Dick?
most_common_word = 'whale'


# In[39]:


get_ipython().run_cell_magic('nose', '', '\ndef test_most_common_word():\n    assert most_common_word.lower() == \'whale\', \\\n    "That\'s not the most common word in Moby Dick."')

