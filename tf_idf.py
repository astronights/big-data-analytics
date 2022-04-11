import os
import re
import sys
import math
import itertools
from pyspark import SparkConf, SparkContext

###################################################
# Spark Initialization and Argument Parsing
###################################################

# Obtain variables from arguments
data_folder = sys.argv[1]
query_file = sys.argv[2]
stopwords_file = sys.argv[3]
out_file = sys.argv[4]

# Initialization of Spark Context
conf = SparkConf()
sc = SparkContext(conf=conf)

###################################################
# Helper Functions
#
# List of function to help in preprocessing, TF-IDF
# calculation and cosine similarity
###################################################


# Function to remove punctuations, bullet points
def remove_nonalphanum(doc_string):
	return([re.sub('\W+','',word) for word in doc_string.split()])

# Function to check if string is only number
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

# Function to remove numbers
def remove_numbers(words):
	return([word for word in words if (len(word) > 0) and not is_number(word)])

# Function to remove stopwords
def remove_stopwords(words):
	return([word.lower() for word in words if word.lower() not in stopwords]) 

# Function to read document in lines
def doc_lines(doc):
	return sc.textFile(doc).map(lambda line: (doc, line)).collect()

# Function to calculate Term Frequencies
# TF = Count of tokens in the document (or) sentence
def calc_tf(token_list):
	tf_map_rdd = token_list.flatMap(lambda words_pair: [((words_pair[0], word), 1) for word in words_pair[1]])
	tf_red_rdd = tf_map_rdd.reduceByKey(lambda x, y: x + y)
	tf_rdd = tf_red_rdd.map(lambda token_pair: (token_pair[0][1], (token_pair[0][0], token_pair[1])))
	return(tf_rdd)

# Function to calculate Inverse Document Frequency for Documents
# IDF = log10(Total Number of documents/ # Documents with token)
def calc_doc_idf(token_tfs):
	idf_map_rdd = token_tfs.map(lambda tf_v: (tf_v[0], 1))
	idf_red_rdd = idf_map_rdd.reduceByKey(lambda x, y: x + y)
	idf_rdd = idf_red_rdd.map(lambda token_pair: (token_pair[0], math.log10(n/token_pair[1])))
	return(idf_rdd)

# Function to calculate Inverse Document Frequency for Sentences per Document
# IDF = log10(Total Number of sentences in document/ # Sentences with token)
def calc_sent_idf(token_tfs):
	idf_sent_count_rdd = token_tfs.map(lambda tf_v: tf_v[1][0]).map(lambda sent_v: (sent_v[0], 1)).reduceByKey(lambda x, y: x + y) 
	idf_map_rdd = token_tfs.map(lambda tf_v: ((tf_v[0], tf_v[1][0][0]), 1))
	idf_red_rdd = idf_map_rdd.reduceByKey(lambda x, y: x + y).map(lambda idf_v: (idf_v[0][1], (idf_v[0][0], idf_v[1])))
	idf_rdd = idf_red_rdd.join(idf_sent_count_rdd).map(lambda token_pair: ((token_pair[1][0][0], token_pair[0]), math.log10(token_pair[1][1]/token_pair[1][0][1])))
	return(idf_rdd)

# Function to calculate TF-IDF Score for Document
# TF-IDF = (1+log10(TF))*IDF
def calc_doc_tf_idf(tf_rdd, idf_rdd):
	tf_idf_rdd = tf_rdd.join(idf_rdd).map(lambda token_pair: (token_pair[1][0][0], (token_pair[0], token_pair[1][0][1], token_pair[1][1], (1+math.log10(token_pair[1][0][1]))*token_pair[1][1])))
	return(tf_idf_rdd)

# Function to calculate TF-IDF Score for Sentence
# TF-IDF = (1+log10(TF))*IDF
def calc_sent_tf_idf(tf_rdd, idf_rdd):
	tf_idf_rdd = tf_rdd.map(lambda tf_pair: ((tf_pair[0], tf_pair[1][0][0]), (tf_pair[1][0][1], tf_pair[1][1]))).join(idf_rdd).map(lambda token_pair: ((token_pair[0][1], token_pair[1][0][0]), (token_pair[0][0], token_pair[1][0][1], token_pair[1][1], (1+math.log10(token_pair[1][0][1]))*token_pair[1][1])))
	return(tf_idf_rdd)


# Function to calculate normalized TF-IDF Scores
def calc_normal_tf_idf(tf_idf_rdd):
	ss_map_rdd = tf_idf_rdd.map(lambda tf_idf_pair: (tf_idf_pair[0], tf_idf_pair[1][3]**2))
	ss_red_rdd = ss_map_rdd.reduceByKey(lambda x,y: x + y)
	#ss_red_rdd -> (document, sum_of_square_TF-IDF)

	normal_rdd = tf_idf_rdd.join(ss_red_rdd).map(lambda joined_pair: (joined_pair[0], (joined_pair[1][0][0], joined_pair[1][0][3]/math.sqrt(joined_pair[1][1])))).sortByKey()
	return(normal_rdd)

# Function to calculate cosine similarity given a set of tokens and a query
def calc_similarity(doc, q):
	if(sum([x[1] for x in doc]) == 0):
		return(0.0)
	return(sum([word_pair[1] for word_pair in doc if word_pair[0] in q])/(math.sqrt(sum([word_pair[1]**2 for word_pair in doc]))*math.sqrt(len(q))))

# Function to calculate document similarities
def calc_doc_similarity(normal_tf_idf_rdd):
	vec_rdd = normal_tf_idf_rdd.groupByKey().mapValues(list)
	similar_rdd = vec_rdd.map(lambda doc_vec: (doc_vec[0], calc_similarity(doc_vec[1], query)))
	return(similar_rdd)

###################################################
# Data Reading and Preprocessing
# 
# Data is read for one file at a time. Preprocessing
# involves the following steps:
# 1. Split content of document by whitespace
# 2. Remove non alphanumeric characters
# 3. Remove any individual numbers (without text)
# 4. Remove stopwords
###################################################

# Get List of Stopwords
stopwords = sc.textFile(stopwords_file).collect()

# Read Query to Dict
query = sc.textFile(query_file).map(lambda text: remove_nonalphanum(text)).flatMap(lambda tokens: remove_stopwords(remove_numbers(tokens))).map(lambda word: (word, 1)).reduceByKey(lambda x, y: x).collectAsMap()

# Read files 
data_files = sorted(os.listdir(data_folder))
n = float(len(data_files))
raw_data = sc.wholeTextFiles(','.join([os.path.join(data_folder,data_file) for data_file in data_files]))
#raw_data -> (document, content)

# Clean and preprocess text
clean_data = raw_data.map(lambda doc: (doc[0], remove_nonalphanum(doc[1]))).map(lambda doc: (doc[0], remove_numbers(doc[1]))).map(lambda doc: (doc[0], remove_stopwords(doc[1])))
#clean_data -> (document, [token])

###################################################
# Step 1: Compute Term Frequency
#
# TF is calculated using the helper function
###################################################

# Calculate Term Frequency
doc_tf_rdd = calc_tf(clean_data)
#doc_tf_rdd -> (token, (document, TF))

###################################################
# Step 2: Compute TF-IDF for tokens in document
#
# First IDF values are calculated for each token
# Then TF and IDF values are combined with formula
###################################################

# Compute Inverse Document Frequency for every token
doc_idf_rdd = calc_doc_idf(doc_tf_rdd)
#doc_idf_rdd -> (token, IDF)

# Join TF and IDF scores to get TF-IDF Score
doc_tf_idf_rdd = calc_doc_tf_idf(doc_tf_rdd, doc_idf_rdd)
#doc_tf_idf_rdd -> (document, (token, TF, IDF, TF-IDF))

###################################################
# Step 3: Compute Normalized TF-IDF scores
#
# TF-IDF values are normalized using sum of squares
###################################################

# Normalize TF-IDF scores
doc_normal_rdd = calc_normal_tf_idf(doc_tf_idf_rdd)
#doc_normal_rdd -> (document, (token, normal_TF-IDF))

###################################################
# Step 4: Compute Relevance of Document wrt query
#
# Cosine similarity is calculated for document
###################################################

# Generate vectors for document and compute similarity
doc_similar_rdd = calc_doc_similarity(doc_normal_rdd)
#doc_similar_rdd -> (document, similarity)

###################################################
# Step 5: Sort and get Top 10 Document
#
# Documents are sorted by similarity and top 10
###################################################

# Sort and take top 10 documents
top_10_doc = doc_similar_rdd.sortBy(lambda doc: -doc[1]).take(10)
#top_10_doc -> [(document, similarity)]

###################################################
# Step 6: Relevance of sentences in document
#
# Approach similar to document similarity calculation
# 1. Each of the 10 documents are read line by line
# 2. Preprocessed similar to documents to clean text
# 3. TF score calculated for token in sentence
# 4. IDF calculated on a per document basis
# 5. TF-IDF joined and normalized as previously
# 6. Cosine similarity calculated with query
###################################################

# Generate tokens for each sentence
top_10_doc_lines = [doc_lines(doc[0]) for doc in top_10_doc]
top_10_doc_lines_rdd = sc.parallelize(itertools.chain.from_iterable(top_10_doc_lines))
top_10_doc_words = top_10_doc_lines_rdd.map(lambda doc: ((doc[0], doc[1]), remove_nonalphanum(doc[1]))).map(lambda doc: (doc[0],  remove_numbers(doc[1]))).map(lambda doc: (doc[0], remove_stopwords(doc[1]))).sortByKey()
#top_10_doc_words -> ((document, sentence), [token])

# Calculate Term Frequency
sent_tf_rdd = calc_tf(top_10_doc_words)
#sent_tf_rdd -> (token, ((document, sentence), TF))

# Compute Inverse Document Frequency for every token
sent_idf_rdd = calc_sent_idf(sent_tf_rdd)
#sent_idf_rdd -> ((token, document), IDF)

# Join TF and IDF scores to get TF-IDF Score
sent_tf_idf_rdd = calc_sent_tf_idf(sent_tf_rdd, sent_idf_rdd)
#sent_tf_idf_rdd -> ((document, sentence), (token, TF, IDF, TF-IDF))

# Normalize TF-IDF scores
sent_normal_rdd = calc_normal_tf_idf(sent_tf_idf_rdd)
#sent_normal_rdd -> ((document, sentence), (token, normal_TF-IDF))

# Calculate sentence similarity
sent_similar_rdd = calc_doc_similarity(sent_normal_rdd)
#sent_similar_rdd -> ((document, sentence), similarity)

###################################################
# Step 7: Most relevant sentence in documents
#
# Get sentence with highest similarity per document
# Join sentence similarity with document similarity
###################################################

# Get most relevant sentence for each document
temp_sent_rdd = sent_similar_rdd.map(lambda pair: (pair[0][0], (pair[0][1], pair[1])))
relevant_sentence_rdd = temp_sent_rdd.reduceByKey(lambda x, y: x if x[1] >= y[1] else y)
#relevant_sentence_rdd -> (document, (sentence, similarity))

# Join with document relevance
relevance_rdd = relevant_sentence_rdd.join(doc_similar_rdd).map(lambda vals: (vals[0].split('/')[-1], vals[1][1], vals[1][0][0].encode('utf-8'), vals[1][0][1])).sortBy(lambda x: -x[1])
#relevant_rdd -> (document, (document_similarity, sentence, sentence_similarity))

relevant_docs = relevance_rdd.collect()

###################################################
# Output Results
#
# Print and save to file, final results
###################################################

# Output Product details:
for x in relevant_docs:
	print('{} {} {} {}'.format(x[0], x[1], x[2], x[3]))

# Write to File using Python
with open(out_file, 'w') as out:
	for x in relevant_docs:
		out.write('{} {} {} {}\n'.format(x[0], x[1], x[2], x[3]))

###################################################
# Stop Spark
###################################################

sc.stop()
