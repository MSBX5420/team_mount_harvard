#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyspark as ps
import warnings
from pyspark.sql import SparkSession
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


# In[2]:


spark = SparkSession.builder.getOrCreate()


# In[3]:


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


# In[4]:


spark = SparkSession.builder.getOrCreate()
df = spark.read.format("csv").option("header", "true").option("multiLine","true").load("s3://msbx5420-2020/small_tweets.csv")
df.show(5)


# In[5]:


df.printSchema()


# In[6]:


flag1 = df.withColumn("favourites_count", df["favourites_count"].cast(IntegerType()))


# In[7]:


flag2 = flag1.withColumn("retweet_count", flag1["retweet_count"].cast(IntegerType()))


# In[8]:


flag3 = flag2.withColumn("followers_count", flag2["followers_count"].cast(IntegerType()))


# In[9]:


dfnew = flag3.withColumn("verified", flag3["verified"].cast(IntegerType()))


# In[10]:


dfnew = dfnew.na.drop()


# In[11]:


dfnew.printSchema()


# In[122]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="text_features", minDocFreq=5) #minDocFreq: remove sparse terms
# label_stringIdx = StringIndexer(inputCol = 'favourites_count', outputCol = "fav_idx")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf])


# In[123]:


pipelineFit = pipeline.fit(dfnew)
model_df = pipelineFit.transform(dfnew)


# In[124]:


data2 = model_df.select(model_df.followers_count, 
                        model_df.retweet_count, 
                        model_df.verified, 
                        model_df.tf,
                        model_df.text_features,
#                         model_df.fav_idx,
                        model_df.favourites_count.alias('label'))


# In[125]:


data2.printSchema()


# In[126]:


# train, test = data2.randomSplit([0.7,0.3])


# In[127]:


assembler = VectorAssembler().setInputCols(['followers_count', 'retweet_count', 'verified', 'tf']).setOutputCol('features')


# In[128]:


train01 = assembler.transform(data2)
train02 = train01.select("features","label")


# In[129]:


lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train02)
trainingSummary = lr_model.summary
print("r2: %f" % trainingSummary.r2)


# In[ ]:




