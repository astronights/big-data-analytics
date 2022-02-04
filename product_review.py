import sys
import json
from pyspark import SparkConf, SparkContext, SQLContext

# Initialization of Spark and SQL Context for JSON file input and processing
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlc = SQLContext(sc)

# Creating RDD of average review of product from the reviews file
review_lines = sqlc.read.option('allowBackslashEscapingAnyCharacter', True).json(sys.argv[1]).rdd
review_scores = review_lines.map(lambda l: (l['asin'], l['overall']))
review_pairs = review_scores.map(lambda s: (s[0], (s[1], 1)))
review_counts = review_pairs.reduceByKey(lambda p1, p2: (p1[0]+p2[0], p1[1]+p2[1])) 
review_averages = review_counts.map(lambda c: (c[0], (c[1][1], c[1][0]/c[1][1])))

# Creating RDD of price of product from metadata file
meta_lines = sqlc.read.option('allowBackslashEscapingAnyCharacter', True).json(sys.argv[2]).rdd
meta_prices = meta_lines.map(lambda l: (l['asin'], l['price']))

# Join RDDs
joined_rdd = meta_prices.join(review_averages)
sorted_rdd = joined_rdd.sortBy(lambda r: -r[1][1][0])
output_rdd = sorted_rdd.map(lambda s: (s[0], s[1][1][1], s[1][0]))
out_list = output_rdd.take(15)

# Output Product details:
for x in out_list:
	print('{} {} {}'.format(x[0], x[1], x[2]))

# Write to File using Python
with open(sys.argv[3], 'w') as out:
	for x in out_list:
		out.write('{} {} {}\n'.format(x[0], x[1], x[2]))

# Stoppage of Spark Context
sc.stop()
