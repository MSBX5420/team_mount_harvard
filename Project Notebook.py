
# coding: utf-8

# In[25]:


import findspark
findspark.init()
import pandas as pd
import pyspark as ps
import warnings
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, StructType, StructField
import csv
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler


# In[26]:


sc = SparkContext()
sqlContext = SQLContext(sc)


# In[27]:


# sc.stop()


# In[28]:


schema = StructType([
    StructField("col_01", StringType()),
    StructField("col_02", StringType()),
    StructField("col_03", StringType()),
    StructField("col_04", StringType()),
    StructField("col_05", IntegerType()),
    StructField("col_06", IntegerType()),
    StructField("col_07", IntegerType()),
    StructField("col_08", IntegerType()),
    
])


# In[29]:


data = []
with open('small_tweets.csv', 'r' ) as doc:
    reader = csv.DictReader(doc)
    for line in reader:
        data.append(line)

df = sc.parallelize(data).toDF()


# In[30]:


#data = sqlContext.read.format('csv') \
#.options(header='true', schema=schema) \
#.load('small_tweets.csv')


# In[31]:


df.show()


# In[32]:


df.printSchema()


# In[33]:


flag1 = df.withColumn("favourites_count", df["favourites_count"].cast(IntegerType()))


# In[34]:


flag2 = flag1.withColumn("retweet_count", flag1["retweet_count"].cast(IntegerType()))


# In[35]:


flag3 = flag2.withColumn("followers_count", flag2["followers_count"].cast(IntegerType()))


# In[36]:


dfnew = flag3.withColumn("verified", flag3["verified"].cast(IntegerType()))


# In[37]:


dfnew.printSchema()


# In[38]:


dfnew.show(5)


# In[46]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="text_features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "favourites_count", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(dfnew)
model_df = pipelineFit.transform(dfnew)
model_df.show(5)


# In[47]:


vectorAssembler = VectorAssembler(inputCols = ['followers_count', 'retweet_count', 'verified', 'tf', 'text_features'], outputCol = 'features')
model_df = vectorAssembler.transform(model_df)
model_df = model_df.select(['features', 'label'])
model_df.show(3)


# In[48]:


(train_set, val_set, test_set) = model_df.randomSplit([0.8, 0.1, 0.1], seed = 2000)


# In[49]:


lr = LinearRegression(featuresCol = 'features', labelCol='label')
lr_model = lr.fit(train_set)
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[ ]:


# predictions = lr_model.transform(test_set)
# predictions.select("prediction","favourites_count","features").show()

