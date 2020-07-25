
# coding: utf-8

# ## 1. Darwin's bibliography
# <p><img src="https://assets.datacamp.com/production/project_607/img/CharlesDarwin.jpg" alt="Charles Darwin" width="300px"></p>
# <p>Charles Darwin is one of the few universal figures of science. His most renowned work is without a doubt his "<em>On the Origin of Species</em>" published in 1859 which introduced the concept of natural selection. But Darwin wrote many other books on a wide range of topics, including geology, plants or his personal life. In this notebook, we will automatically detect how closely related his books are to each other.</p>
# <p>To this purpose, we will develop the bases of <strong>a content-based book recommendation system</strong>, which will determine which books are close to each other based on how similar the discussed topics are. The methods we will use are commonly used in text- or documents-heavy industries such as legal, tech or customer support to perform some common task such as text classification or handling search engine queries.</p>
# <p>Let's take a look at the books we'll use in our recommendation system.</p>

# In[2]:


# Import library
import glob

# The books files are contained in this folder
folder = "datasets/"

# List all the .txt files and sort them alphabetically
files = glob.glob(folder + "*.txt")
files.sort()
files


# In[3]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code.\n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef test_files_type():\n    assert isinstance(files, list), \\\n    'The files variable should be a list.'\n    \ndef test_glob_len():\n    assert len(files) == 20, \\\n    'The files variable should contain 20 elements. Make sure you only selected files ending by .txt.'\n    \ndef test_is_files_list_sorted():\n    assert all(files[i] <= files[i+1] for i in range(len(files)-1)), \\\n    'The files list should be sorted by using the .sort() method.'")


# ## 2. Load the contents of each book into Python
# <p>As a first step, we need to load the content of these books into Python and do some basic pre-processing to facilitate the downstream analyses. We call such a collection of texts <strong>a corpus</strong>. We will also store the titles for these books for future reference and print their respective length to get a gauge for their contents.</p>

# In[4]:


# Import libraries
import re, os

# Initialize the object that will contain the texts and titles
txts = []
titles = []

for n in files:
    # Open each file
    f = open(n, encoding='utf-8-sig')
    # Remove all non-alpha-numeric characters
    data = re.sub('[\W_]+', ' ', f.read())
    # Store the texts and titles of the books in two separate lists
    txts.append(data)
    titles.append(os.path.basename(n).replace('.txt', ''))
    
# Print the length, in characters, of each book
[len(t) for t in txts]


# In[5]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\nimport _io\n\n# For some reasons, the index isn\'t the same between my local notebook and the DataCamp version.\n# Therefore, I can not hardcode the value and have to generate it here.\n\ndef test_txts_titles_len():\n    assert len(txts) == 20 & len(titles) == 20, \\\n    \'The txts and titles variable should contain 20 elements.\'\n    \n\ndef test_f_type():\n    assert isinstance(f, _io.TextIOWrapper), \\\n    \'The f variable should be of type _io.TextIOWrapper and be generated with the open() function.\'\n    \ndef test_file_encoding():\n    assert f.encoding == \'utf-8-sig\', \\\n    \'Make sure you open the text file encoded as utf-8-sig.\'   \n    \ndef test_txt_in_title():\n    assert all([".txt" not in title for title in titles]), \\\n    \'The titles contained in the titles variable should not contain the .txt string. Use the .replace() method to remove them.\'   \n\ndef test_folder_in_title():\n    assert all([folder not in title for title in titles]), \\\n    \'The titles contained in the titles variable should not contain the name of the folder (\' + str(folder) + \'). Use the os.path.basename() function to remove them.\'   \n    \n    \ndef test_alphanumeric_in_txt():\n    assert not sum([len(i) for i in [re.findall(\'^[\\w-]+$\', t) for t in txts]]), \\\n    \'The elements of the txts variable should not contain non-alphanumeric characters.\' ')


# ## 3. Find "On the Origin of Species"
# <p>For the next parts of this analysis, we will often check the results returned by our method for a given book. For consistency, we will refer to Darwin's most famous book: "<em>On the Origin of Species</em>." Let's find to which index this book is associated.</p>

# In[6]:


# Browse the list containing all the titles
for i in range(len(titles)):
    # Store the index if the title is "OriginofSpecies"
    if titles[i] == 'OriginofSpecies':
        ori = i
        break

# Print the stored index
ori


# In[7]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\n\n# For some reasons, the index isn\'t the same between my local notebook and the DataCamp version.\n# Therefore, I can not hardcode the value and have to generate it here.\n\nfor i in range(len(titles)):\n    # Store the index if the title is "OriginofSpecies"\n    if(titles[i]=="OriginofSpecies"):\n        ori_test = i\n\n\ndef test_ori_existence():\n    assert \'ori\' in globals(), \\\n    \'Make sure you created a variable called ori.\'\n\ndef test_ori_type():\n    assert type(ori) == int, \\\n    \'The ori variable should be of type integer.\'\n    \ndef test_ori_value():\n    assert ori == ori_test, \\\n    \'The ori variable does not contain the correct index number.\'')


# ## 4. Tokenize the corpus
# <p>As a next step, we need to transform the corpus into a format that is easier to deal with for the downstream analyses. We will tokenize our corpus, i.e., transform each text into a list of the individual words (called tokens) it is made of. To check the output of our process, we will print the first 20 tokens of "<em>On the Origin of Species</em>".</p>

# In[8]:


# Define a list of stop words
stoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())

# Convert the text to lower case 
txts_lower_case = [txt.lower() for txt in txts]

# Transform the text into tokens 
txts_split = [txt.split() for txt in txts_lower_case]

# Remove tokens which are part of the list of stop words
texts = [[word for word in txt if word not in stoplist] for txt in txts_split]

# Print the first 20 tokens for the "On the Origin of Species" book
texts[ori][: 20]


# In[9]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\nstoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())\n\ndef intersection(lst1, lst2): \n    lst3 = [value for value in lst1 if value in lst2] \n    return lst3 \n\n## Variables existence\n\ndef test_var_texts_existence():\n    assert 'texts' in globals(), \\\n    'The results should be stored in a variable called texts.'\n\ndef test_var_lowercase_existence():\n    assert 'txts_lower_case' in globals(), \\\n    'The variable txts_lower_case should exist.'\n    \ndef test_var_split_existence():\n    assert 'txts_split' in globals(), \\\n    'The variable txts_split should exist.'\n\n## Variables type and length    \ndef test_var_texts_type():\n    assert isinstance(txts, list), \\\n    'The texts variable should be a list.'\n    \ndef test_var_texts_lowercase_type():\n    assert isinstance(txts_lower_case, list), \\\n    'The txts_lower_case variable should be a list.'\n\ndef test_var_texts_split_type():\n    assert isinstance(txts_split, list), \\\n    'The txts_split variable should be a list.'\n\n## Variables length\ndef test_var_texts_len():\n    assert len(texts) == 20, \\\n    'The texts list should contain 20 elements.'\n\ndef test_var_texts_lowercase_len():\n    assert len(txts_lower_case) == 20, \\\n    'The txts_lower_case list should contain 20 elements.'\n\ndef test_var_texts_split_len():\n    assert len(txts_split) == 20, \\\n    'The txts_split list should contain 20 elements.'\n    \n## Variable content\n\ndef test_lower_case():\n    assert all([t.islower() for t in txts_lower_case]), \\\n    'The texts in the txts_lower_case list should all be in lower case.'\n\ndef test_split_list():\n    assert all([isinstance(t, list) for t in txts_split]), \\\n    'Each element of the txts_split list should be a list'\n    \n    \ndef test_stopwords():\n    assert not sum([len(i) for i in [intersection(t, stoplist) for t in texts]]), \\\n    'You should remove stop words from the final token sets contained in the texts variable.'")


# ## 5. Stemming of the tokenized corpus
# <p>If you have read <em>On the Origin of Species</em>, you will have noticed that Charles Darwin can use different words to refer to a similar concept. For example, the concept of selection can be described by words such as <em>selection</em>, <em>selective</em>, <em>select</em> or <em>selects</em>. This will dilute the weight given to this concept in the book and potentially bias the results of the analysis.</p>
# <p>To solve this issue, it is a common practice to use a <strong>stemming process</strong>, which will group together the inflected forms of a word so they can be analysed as a single item: <strong>the stem</strong>. In our <em>On the Origin of Species</em> example, the words related to the concept of selection would be gathered under the <em>select</em> stem.</p>
# <p>As we are analysing 20 full books, the stemming algorithm can take several minutes to run and, in order to make the process faster, we will directly load the final results from a pickle file and review the method used to generate it.</p>

# In[10]:


import pickle

# Load the stemmed tokens list from the pregenerated pickle file
texts_stem = pickle.load(open('datasets/texts_stem.p', 'rb'))

# Print the 20 first stemmed tokens from the "On the Origin of Species" book
texts_stem[ori][: 20]


# In[11]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\ndef test_var_existence():\n    assert 'texts_stem'in globals(), \\\n    'The content of the pickle file should be loaded into a variable named texts_stem.'\n\ndef test_var_type():\n    assert isinstance(texts_stem, list), \\\n    'The texts_stem variable should be a list.'\n    \ndef test_var_len():\n    assert len(texts_stem) == 20, \\\n    'The texts_stem list should contain 20 elements.'")


# ## 6. Building a bag-of-words model
# <p>Now that we have transformed the texts into stemmed tokens, we need to build models that will be useable by downstream algorithms.</p>
# <p>First, we need to will create a universe of all words contained in our corpus of Charles Darwin's books, which we call <em>a dictionary</em>. Then, using the stemmed tokens and the dictionary, we will create <strong>bag-of-words models</strong> (BoW) of each of our texts. The BoW models will represent our books as a list of all uniques tokens they contain associated with their respective number of occurrences. </p>
# <p>To better understand the structure of such a model, we will print the five first elements of one of the "<em>On the Origin of Species</em>" BoW model.</p>

# In[12]:


# Load the functions allowing to create and use dictionaries
from gensim import corpora

# Create a dictionary from the stemmed tokens
dictionary = corpora.Dictionary(texts_stem)

# Create a bag-of-words model for each book, using the previously generated dictionary
bows = [dictionary.doc2bow(txt) for txt in texts_stem]

# Print the first five elements of the On the Origin of species' BoW model
bows[ori][: 5]


# In[13]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\nimport gensim.corpora.dictionary\nfrom gensim import corpora\n\ndictionary_test = corpora.Dictionary(texts_stem)\nbows_test = [dictionary.doc2bow(text) for text in texts_stem]\nbows_len =[len(b) for b in bows]\nbows_test_len =[len(b) for b in bows_test]\n\n## Dictionary variable\ndef test_dictionary_type():\n    assert isinstance(dictionary, gensim.corpora.dictionary.Dictionary), \\\n    \'The dictionary should be created using the corpora.Dictionary() function and the resulting object should be of type "gensim.corpora.dictionary.Dictionary".\'\n    \ndef test_dictionary_len():\n    assert len(dictionary) == len(dictionary_test), \\\n    \'The dictionary should contain \' +  str(len(dictionary_test)) + \' tokens. Make sure you generated the dictionary from the texts_stem object.\'\n    \n## bows variable\n\ndef test_bows_type():\n    assert isinstance(bows, list), \\\n    \'The bows variable should be a list.\'\n    \ndef test_bows_list_len():\n    assert bows_len == bows_test_len, \\\n    \'The lengths of the bows are not those expected. Make sure you generated them using the texts_stem object.\'\n    \ndef test_bows_len():\n    assert len(bows) == 20, \\\n    \'The bows object should have 20 elements, one model per text.\'\n    \ndef test_bows_content_type():\n    assert all([isinstance(b, list) for b in bows]), \\\n    \'Each elements in the bows list should be a list.\'\n    ')


# ## 7. The most common words of a given book
# <p>The results returned by the bag-of-words model is certainly easy to use for a computer but hard to interpret for a human. It is not straightforward to understand which stemmed tokens are present in a given book from Charles Darwin, and how many occurrences we can find.</p>
# <p>In order to better understand how the model has been generated and visualize its content, we will transform it into a DataFrame and display the 10 most common stems for the book "<em>On the Origin of Species</em>".</p>

# In[14]:


# Import pandas to create and manipulate DataFrames
import pandas as pd

# Convert the BoW model for "On the Origin of Species" into a DataFrame
df_bow_origin = pd.DataFrame(bows[ori])

# Add the column names to the DataFrame
df_bow_origin.columns = ['index', 'occurrences']

# Add a column containing the token corresponding to the dictionary index
df_bow_origin['token'] = df_bow_origin['index'].apply(lambda x: dictionary[x])

# Sort the DataFrame by descending number of occurrences and print the first 10 values
df_bow_origin = df_bow_origin.sort_values('occurrences', ascending=False)
df_bow_origin.head(10)


# In[15]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nimport pandas.core.frame\n\n# Hardcoded numbers might not hold once on Datacamp\'s servers.\n# Hence the values are recomputed here.\n\ndf_bow_origin_test = pd.DataFrame(bows[ori])\nn_rows_test = df_bow_origin_test.shape[0]\n\n#### Tests\n\ndef test_df_type():\n    assert isinstance(df_bow_origin, pandas.core.frame.DataFrame), \\\n    \'The df_bow_origin variable should be a pandas DataFrame of type pandas.core.frame.DataFrame.\'\n\ndef test_df_dim1():\n    assert df_bow_origin.shape[1] == 3, \\\n    \'The df_bow_origin DataFrame should have 3 columns.\'\n    \ndef test_df_dim2():\n    assert df_bow_origin.shape[0] == n_rows_test, \\\n    \'The df_bow_origin DataFrame should have \' + str(n_rows_test) + \' rows.\'    \n    \ndef test_df_columns():\n    assert all(df_bow_origin.columns == list(["index", "occurrences", "token"])), \\\n    \'The columns of the df_bow_origin DataFrame should be named "index", "occurrences" and "token".\'    \n        \n    ')


# ## 8. Build a tf-idf model
# <p>If it wasn't for the presence of the stem "<em>speci</em>", we would have a hard time to guess this BoW model comes from the <em>On the Origin of Species</em> book. The most recurring words are, apart from few exceptions, very common and unlikely to carry any information peculiar to the given book. We need to use an additional step in order to determine which tokens are the most specific to a book.</p>
# <p>To do so, we will use a <strong>tf-idf model</strong> (term frequency–inverse document frequency). This model defines the importance of each word depending on how frequent it is in this text and how infrequent it is in all the other documents. As a result, a high tf-idf score for a word will indicate that this word is specific to this text.</p>
# <p>After computing those scores, we will print the 10 words most specific to the "<em>On the Origin of Species</em>" book (i.e., the 10 words with the highest tf-idf score).</p>

# In[16]:


# Load the gensim functions that will allow us to generate tf-idf models
from gensim.models import TfidfModel

# Generate the tf-idf model
model = TfidfModel(bows)

# Print the model for "On the Origin of Species"
model[bows[ori]]


# In[17]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\n\nimport gensim.models.tfidfmodel\n\n\ndef test_model_type():\n    assert isinstance(model, gensim.models.tfidfmodel.TfidfModel), \\\n    \'The tf-idf model should be created using the TfidfModel() function and the model variable should be of type "gensim.models.tfidfmodel.TfidfModel".\'')


# ## 9. The results of the tf-idf model
# <p>Once again, the format of those results is hard to interpret for a human. Therefore, we will transform it into a more readable version and display the 10 most specific words for the "<em>On the Origin of Species</em>" book.</p>

# In[18]:


# Convert the tf-idf model for "On the Origin of Species" into a DataFrame
df_tfidf = pd.DataFrame(model[bows[ori]])

# Name the columns of the DataFrame id and score
df_tfidf.columns = ['id', 'score']

# Add the tokens corresponding to the numerical indices for better readability
df_tfidf['token'] = df_tfidf['id'].apply(lambda x: dictionary[x])

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
df_tfidf = df_tfidf.sort_values('score', ascending=False)
df_tfidf.head(10)


# In[19]:


get_ipython().run_cell_magic('nose', '', '# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nimport pandas.core.frame\n\n### Recomputing object to avoid hard-coded values\ndf_tfidf_test = pd.DataFrame(model[bows[ori]])\nn_rows_test_tfidf = df_tfidf_test.shape[0]\n\ndef test_df_type():\n    assert isinstance(df_tfidf, pandas.core.frame.DataFrame), \\\n    \'The df_tfidf variable should contain a pandas DataFrame of type pandas.core.frame.DataFrame.\'\n    \ndef test_df_dim1():\n    assert df_tfidf.shape[1] == 3, \\\n    \'The df_bow_origin DataFrame should have 3 columns.\'\n    \ndef test_df_dim2():\n    assert df_tfidf.shape[0] == n_rows_test_tfidf, \\\n    \'The df_bow_origin DataFrame should have \' + str(n_rows_test_tfidf) + \' rows.\'    \n    \ndef test_df_columns():\n    assert all(df_tfidf.columns == list(["id", "score", "token"])), \\\n    \'The columns of the df_bow_origin DataFrame should be named "id", "score" and "token".\'    \n        \n    ')


# ## 10. Compute distance between texts
# <p>The results of the tf-idf algorithm now return stemmed tokens which are specific to each book. We can, for example, see that topics such as selection, breeding or domestication are defining "<em>On the Origin of Species</em>" (and yes, in this book, Charles Darwin talks quite a lot about pigeons too). Now that we have a model associating tokens to how specific they are to each book, we can measure how related to books are between each other.</p>
# <p>To this purpose, we will use a measure of similarity called <strong>cosine similarity</strong> and we will visualize the results as a distance matrix, i.e., a matrix showing all pairwise distances between Darwin's books.</p>

# In[20]:


# Load the library allowing similarity computations
from gensim import similarities

# Compute the similarity matrix (pairwise distance between all texts)
sims = similarities.MatrixSimilarity(model[bows])

# Transform the resulting list into a dataframe
sim_df = pd.DataFrame(list(sims))

# Add the titles of the books as columns and index of the dataframe
sim_df.columns = titles
sim_df.index = titles

# Print the resulting matrix
sim_df


# In[21]:


get_ipython().run_cell_magic('nose', '', "\nimport gensim.similarities.docsim\nimport pandas.core.frame\nimport pandas.core.indexes.base\n\ndef test_sims_type():\n    assert isinstance(sims, gensim.similarities.docsim.MatrixSimilarity), \\\n    'The sims variable should be created using the similarities.MatrixSimilarity() function and be of type gensim.similarities.docsim.MatrixSimilarity.'\n    \ndef test_sim_df_type():\n    assert isinstance(sim_df, pandas.core.frame.DataFrame), \\\n    'The sim_df variable should be a pandas DataFrame of type pandas.core.frame.DataFrame.'\n    \ndef test_df_dims():\n    assert sim_df.shape[0] == 20 and sim_df.shape[1] == 20 , \\\n    'The sim_df DataFrame should have 20 rows and 20 columns.'    \n    \ndef test_df_columns():\n    assert all(sim_df.columns == titles), \\\n    'The columns of the sim_df DataFrame should be the titles of the books (contained in the titles variable).'    \n        \n# This test isn't working properly and is redundant with the following one. Hence, it is disabled.\n#def test_df_index_type():\n#    assert isinstance(sim_df.index, pandas.core.indexes.base.Index), \\\n#    'The index of the sim_df DataFrame should be of type pandas.core.indexes.base.Index. Make sure the index contains the titles of the books, and not numbers.'    \n        \ndef test_df_index_content():\n    assert  list(sim_df.index) == titles , \\\n    'The index of the sim_df DataFrame should contain the titles of the books, contained in the titles variable. For example, use: sim_df.index = titles.'    \n          ")


# ## 11. The book most similar to "On the Origin of Species"
# <p>We now have a matrix containing all the similarity measures between any pair of books from Charles Darwin! We can now use this matrix to quickly extract the information we need, i.e., the distance between one book and one or several others. </p>
# <p>As a first step, we will display which books are the most similar to "<em>On the Origin of Species</em>," more specifically we will produce a bar chart showing all books ranked by how similar they are to Darwin's landmark work.</p>

# In[22]:


# This is needed to display plots in a notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Import libraries
import matplotlib.pyplot as plt

# Select the column corresponding to "On the Origin of Species" and 
v = sim_df['OriginofSpecies']

# Sort by ascending scores
v_sorted = v.sort_values()

# Plot this data has a horizontal bar plot
v_sorted.plot.barh(x='lab', y='val', rot=0).plot()

# Modify the axes labels and plot title for a better readability
plt.xlabel("Score")
plt.ylabel("Book")
plt.title("Similarity")


# In[23]:


get_ipython().run_cell_magic('nose', '', "# This needs to be included at the beginning of every @tests cell.\n\n# One or more tests of the students code. \n# The @solution should pass the tests.\n# The purpose of the tests is to try to catch common errors and to \n# give the student a hint on how to resolve these errors.\n\nimport pandas.core.series\n\nv_sorted_list = list(v_sorted) # We collapse this into a list to avoid issues if indices are wrong\n\n## Variable types\ndef test_v_type():\n    assert isinstance(v, pandas.core.series.Series), \\\n    'The v variable should be a series of type pandas.core.series.Series.'\n\ndef test_v_sorted_type():\n    assert isinstance(v_sorted, pandas.core.series.Series), \\\n    'The v_sorted variable should be a series of type pandas.core.series.Series.'\n\n## Variable lengths    \ndef test_v_len():\n    assert len(v) == 20, \\\n    'The v series should be of length 20.'\n\ndef test_v_sorted_len():\n    assert len(v_sorted) == 20, \\\n    'The v_sorted series should be of length 20.'\n    \n## Is the series sorted?\ndef test_is_series_sorted():\n    assert all(v_sorted_list[i] <= v_sorted_list[i+1] for i in range(len(v_sorted_list)-1)), \\\n    'The v_sorted series should be sorted by ascending scores.'")


# ## 12. Which books have similar content?
# <p>This turns out to be extremely useful if we want to determine a given book's most similar work. For example, we have just seen that if you enjoyed "<em>On the Origin of Species</em>," you can read books discussing similar concepts such as "<em>The Variation of Animals and Plants under Domestication</em>" or "<em>The Descent of Man, and Selection in Relation to Sex</em>." If you are familiar with Darwin's work, these suggestions will likely seem natural to you. Indeed, <em>On the Origin of Species</em> has a whole chapter about domestication and <em>The Descent of Man, and Selection in Relation to Sex</em> applies the theory of natural selection to human evolution. Hence, the results make sense.</p>
# <p>However, we now want to have a better understanding of the big picture and see how Darwin's books are generally related to each other (in terms of topics discussed). To this purpose, we will represent the whole similarity matrix as a dendrogram, which is a standard tool to display such data. <strong>This last approach will display all the information about book similarities at once.</strong> For example, we can find a book's closest relative but, also, we can visualize which groups of books have similar topics (e.g., the cluster about Charles Darwin personal life with his autobiography and letters). If you are familiar with Darwin's bibliography, the results should not surprise you too much, which indicates the method gives good results. Otherwise, next time you read one of the author's book, you will know which other books to read next in order to learn more about the topics it addressed.</p>

# In[24]:


# Import libraries
from scipy.cluster import hierarchy

# Compute the clusters from the similarity matrix,
# using the Ward variance minimization algorithm
Z = hierarchy.linkage(sims, 'ward')

# Display this result as a horizontal dendrogram
hierarchy.dendrogram(Z, leaf_font_size=8, labels=sim_df.index,
                     orientation='left')


# In[25]:


get_ipython().run_cell_magic('nose', '', "\nimport numpy\n\ndef test_Z_type():\n    assert isinstance(Z, numpy.ndarray), \\\n    'The Z variable should be generated by the hierarchy.linkage() function and of type numpy.ndarray.'")

