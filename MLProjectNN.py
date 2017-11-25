
# coding: utf-8

# In[1]:

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import SparseVector, DenseVector 
spark = SparkSession     .builder     .appName("MLProjectNN")     .config("spark.some.config.option", "some-value")     .getOrCreate()


# In[2]:

def indexData(df_sample):   
    df_sche = df_sample.schema.fields
    for s in df_sche:
        n = s.name
        if (n !="target")&(n!="id"):
            print(n)
            indexer = StringIndexer(inputCol=n, outputCol=n+"_index").fit(df_sample)
            df_sample = indexer.transform(df_sample).drop(n)
        elif n=="id":
            indexer = StringIndexer(inputCol=n, outputCol=n+"_index").fit(df_sample)
            df_sample = indexer.transform(df_sample)
    return df_sample


# In[3]:

def assembleData(df_sample,case):
    vecAssembler = VectorAssembler(inputCols = ["msno_index","song_id_index","source_system_tab_index",
            "source_screen_name_index","source_type_index"], outputCol = "features")
    df_0 = vecAssembler.transform(df_sample)
    #turn sparse vectors to dense vectors
    if case=="train":
        df_1 = df_0.withColumn("target",df_0["target"].cast("double")).select("target","features")
        return df_1
    elif case=="test":
        df_1 = df_0.select("features","id")
        return df_1
    else:
        print("No such type")
        return df_sample


# In[4]:

df_train = spark.read.csv("/Users/Bobby/Documents/class/Machine Learning/finalProject/train.csv",
                header="true")
df_test = spark.read.csv("/Users/Bobby/Documents/class/Machine Learning/finalProject/test.csv",
                header="true")
# df.printSchema()
# print(df_sample.count())
df_train_sample = df_train.sample(False, 0.001, None).na.fill('unknown')
df_test_sample = df_test.na.fill('unknown')
indexed_train_data = indexData(df_train_sample)
indexed_test_data = indexData(df_test_sample)


# In[5]:

# df_obj = df1.join(df2,df1.msno == df2.msno,how="right").show(10)
print(indexed_train_data.count())
print(indexed_train_data.dropna().count())
indexed_train_data.show(10)


# In[6]:

df_train = assembleData(indexed_train_data,"train")
df_test = assembleData(indexed_test_data,"test")


# In[13]:

# df_train.show(10)
df_test.show(10)
# print(df_test.count())
# print(df_test.dropna().count())


# In[14]:

#set parameters for a KNN model

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

layers = [[5,3,2],[5,4,2],[5,5,2]]
maxAccuracy = 0
bestLayer = []

for layer in layers:
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layer, blockSize=128)
    param = trainer.setParams(featuresCol = "features",labelCol="target")
    #use K-Fold validation to tune the model
    #pyspark library
    grid = ParamGridBuilder().build()
    # .addGrid(trainer.maxIter, [0, 1]) random forest
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",labelCol="target")
    cv = CrossValidator(estimator=trainer, estimatorParamMaps=grid, evaluator=evaluator,numFolds=5)
    cv.extractParamMap()
    cvModel = cv.fit(df_train)
    print(layer)
    cvModel.avgMetrics[0]
    print(evaluator.evaluate(cvModel.transform(df_train)))
    if(evaluator.evaluate(cvModel.transform(df_train))>maxAccuracy):
        bestLayer = layer
        maxAccuracy = evaluator.evaluate(cvModel.transform(df_train))
print("best layer:")
print(bestLayer)
print("max accuracy:")
print(maxAccuracy)



# In[15]:

trainer = MultilayerPerceptronClassifier(maxIter=100, layers=bestLayer, blockSize=128,labelCol="target")
model = trainer.fit(df_train)
result = model.transform(df_test)
df_final = result.selectExpr("id as id","prediction as target")


# In[19]:

df_final.coalesce(1).write.csv("/Users/Bobby/Documents/class/Machine Learning/finalProject/testResult.csv",header=True)


# In[ ]:

df_final.show(10)


# In[ ]:



