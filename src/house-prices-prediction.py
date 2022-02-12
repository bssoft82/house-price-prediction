# Databricks notebook source
# MAGIC %md
# MAGIC # HOUSE SALE PRICE ESTIMATOR - GROUP 23

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Install pyspark and required modules

# COMMAND ----------

!pip install pyspark
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install seaborn

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Initialize Spark and other required modules

# COMMAND ----------

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from pyspark.sql import SparkSession, functions as F, DataFrame
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, VectorIndexer, Bucketizer, OneHotEncoder, MinMaxScaler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GeneralizedLinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType,BooleanType,DateType
from pyspark.sql.functions import col,isnan,when,count
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3. Creating Spark Session

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Load Data

# COMMAND ----------

# Train data
df_train = spark.read.csv('/FileStore/tables/house_sales_train.csv', inferSchema=True, header=True)

# Test data
df_test = spark.read.csv('/FileStore/tables/house_sales_test.csv', inferSchema=True, header=True)

# Sample Submission
df_sample_submission = spark.read.csv('/FileStore/tables/sample_submission.csv', inferSchema=True, header=True)

# Columns to be part of Submission
col_sample_submission = ['Id','SalePrice']
# df_train.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * There are 1460 instances of training data and 1460 of test data. Total number of attributes equals 81, of which 36 is quantitative, 43 categorical in addition to Id and SalePrice.
# MAGIC 
# MAGIC * Quantitative
# MAGIC 1stFlrSF, 2ndFlrSF, 3SsnPorch, BedroomAbvGr, BsmtFinSF1, BsmtFinSF2, BsmtFullBath, BsmtHalfBath, BsmtUnfSF, EnclosedPorch, Fireplaces, FullBath, GarageArea, GarageCars, GarageYrBlt, GrLivArea, HalfBath, KitchenAbvGr, LotArea, LotFrontage, LowQualFinSF, MSSubClass, MasVnrArea, MiscVal, MoSold, OpenPorchSF, OverallCond, OverallQual, PoolArea, ScreenPorch, TotRmsAbvGrd, TotalBsmtSF, WoodDeckSF, YearBuilt, YearRemodAdd, YrSold
# MAGIC 
# MAGIC * Qualitative
# MAGIC Alley, BldgType, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtQual, CentralAir, Condition1, Condition2, Electrical, ExterCond, ExterQual, Exterior1st, Exterior2nd, Fence, FireplaceQu, Foundation, Functional, GarageCond, GarageFinish, GarageQual, GarageType, Heating, HeatingQC, HouseStyle, KitchenQual, LandContour, LandSlope, LotConfig, LotShape, MSZoning, MasVnrType, MiscFeature, Neighborhood, PavedDrive, PoolQC, RoofMatl, RoofStyle, SaleCondition, SaleType, Street, Utilities,

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Exploratory Data Analysis

# COMMAND ----------

# Schema of Train data
df_train.printSchema()

# COMMAND ----------

# Schema of Test data
df_test.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC * We observe there is discrepancy in the datatypes of the dataframe for train and test dataset
# MAGIC * Few integer columns in test dataset have NA, None values due to which the type is inferred as string by pyspark
# MAGIC * The datatype of few columns of test dataset need to be modified from string to int type

# COMMAND ----------

# Train Data Description
df_train.summary().toPandas().head()

# COMMAND ----------

# Test Data Description
df_test.summary().toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * We observe compared to remaining column values , the values for columns GarageYrBlt, GrLivArea, 1stFlrSF, TotalBsmtSF ,YearBuilt ,YearRemodAdd and LotArea are higher. These columns need to be scaled appropriately

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 6. Data Cleaning and Transformation

# COMMAND ----------

# Modifying datatype for few columns of Test dataset
def cast_to_int(_sdf: DataFrame,col_list: list) -> DataFrame:
    for col in col_list:
        _sdf = _sdf.withColumn(col, _sdf[col].cast('int'))
    return _sdf

features_to_be_modified = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea','BsmtFullBath', 'BsmtHalfBath']
df_test_typecast = cast_to_int(df_test, features_to_be_modified)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.1 Find count for empty, None, Null, Nan with string literals.

# COMMAND ----------

df_train.select([count(when(col(c).contains('None'), c )).alias(c) for c in df_train.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

df_train.select([count(when(col(c).contains('NULL'), c )).alias(c) for c in df_train.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

df_train.select([count(when(col(c) == '', c )).alias(c) for c in df_train.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

df_train.select([count(when(col(c).isNull(), c )).alias(c) for c in df_train.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

df_train.select([count(when(isnan(c), c )).alias(c) for c in df_train.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

df_train.select([count(when(isnan(c), c )).alias(c) for c in df_train.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

df_test.select([count(when(isnan(c), c )).alias(c) for c in df_test.columns]).show(truncate=False, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC As can be seen about except for *MasVnrType* there are no columns where are empty or missing records. There are no nan records as well.

# COMMAND ----------

df_train.select( "LotFrontage" ).distinct().collect()

# COMMAND ----------

# Get Integer and String features of Train Dataset
def get_features (df_train):
    str_features = [] 
    int_features = []
    for col in  df_train.dtypes:
        if col[0] not in ('Id'):
            if col[1] == 'string':
                str_features += [col[0]]
            else:
                int_features += [col[0]]
    return str_features, int_features


# COMMAND ----------

print("Train Dataset :\n")

str_features, int_features = get_features (df_train)
print(f'str_features : {str_features}', "\n")
print(f'int_features: {int_features}')

# COMMAND ----------

print("Typecasted Test Dataset :\n")

str_features, int_features = get_features (df_test_typecast)
print(f'str_features : {str_features}', "\n")
print(f'int_features: {int_features}')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * We observe the dataypes of all the columns of train dataset and typecasted test dataset match perfectly

# COMMAND ----------

df_test_typecast.select(int_features).limit(5).toPandas().T

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2 Covariance Analysis

# COMMAND ----------

saleprice_cov = {} 
for col in  df_train.dtypes:
    if col[0] != 'SalePrice' and col[1] != 'string':
        saleprice_cov[col[0]] = df_train.cov('SalePrice', col[0])

dict(sorted(saleprice_cov.items(), key=lambda item: item[1], reverse= True))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.3 Correlation Analysis

# COMMAND ----------

corr = df_train.toPandas().corr()
corr[['SalePrice']].sort_values(by='SalePrice',ascending=False).style.background_gradient(cmap='viridis', axis=None)

# COMMAND ----------

# MAGIC %md
# MAGIC * We can see that OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF ,1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, Fireplaces, BsmtFinSF1, WoodDeckSF, 2ndFlrSF, OpenPorchSF, HalfBath, LotArea, BsmtFinSF1 are the top influencers of the Sale Price
# MAGIC 
# MAGIC * We can obseve the columns BedroomAbvGr, ScreenPorch, PoolArea, MoSold, 3SsnPorch BsmtFinSF2, BsmtHalfBath , MiscVal , LowQualFinSF, YrSold, OverallCond, MSSubClass, EnclosedPorch do not impact the Sale Price

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Dropping Low Influencing Columns (Feature Selection)

# COMMAND ----------

LowInfluenceColumns = [ 'BedroomAbvGr', 'ScreenPorch', 'PoolArea', 'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 'EnclosedPorch']
df_train=df_train.drop(*LowInfluenceColumns)
df_test_typecast=df_test_typecast.drop(*LowInfluenceColumns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Cleaning Train Dataset

# COMMAND ----------

# Removing Duplicates
df_train = df_train.distinct()

# Replacing Null, Na values
df_train.na.fill('NA', subset=["Alley"])
df_train.na.fill('NP', subset=["PoolQC"])
df_train.na.fill('NF', subset=["Fence"])
df_train.na.fill('None', subset=["MiscFeature"])
df_train.na.fill('No FP', subset=["FireplaceQu"])

# Dropping records with No value for Important Features
df_train = df_train.na.drop(subset=("OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF" ,"1stFlrSF", "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "Fireplaces", "BsmtFinSF1", "WoodDeckSF", "2ndFlrSF", "OpenPorchSF", "HalfBath", "LotArea", "BsmtFinSF1"))

# Shape of the train dataset
print("Shape of train dataset : " ,df_train.count(),"," ,len(df_train.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 9. Preparing Test Dataset

# COMMAND ----------

# Removing Duplicates
df_test_typecast = df_test_typecast.distinct()

# Replacing Null, Na values
df_test_typecast.na.fill('NA', subset=["Alley"])
df_test_typecast.na.fill('NP', subset=["PoolQC"])
df_test_typecast.na.fill('NF', subset=["Fence"])
df_test_typecast.na.fill('None', subset=["MiscFeature"])
df_test_typecast.na.fill('No FP', subset=["FireplaceQu"])

# Appending a SalePrice column with 0 literal value
df_test_typecast = df_test_typecast.withColumn("SalePrice", lit(0))

# Shape of the test dataset
print("Shape of test dataset : " ,df_test_typecast.count(),"," ,len(df_test_typecast.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10. Transform categorical data

# COMMAND ----------

# MAGIC %md
# MAGIC #### 10.1 Encode a string column of labels to a column of label indices

# COMMAND ----------

for col in  df_train.dtypes:
    if col[1] != 'string':
        output_col = "" + col[0] + "_int"
        indexer = StringIndexer(inputCol=col[0], outputCol=output_col)
        indexed = indexer.fit(df_train).transform(df_train)
        
# # Shape of the dataset
# print("Shape of indexed dataset : " ,indexed.count(),"," ,len(indexed.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 10.2 Assembler combines all integer and create a vector which is used as input to predict.

# COMMAND ----------

str_features, int_features = get_features (df_train)
assembler= VectorAssembler(inputCols=int_features,outputCol="features")

output= assembler.transform(indexed)
output.select("features","SalePrice")

#We can see column features is dense vector
final=output.select("features","SalePrice")
final.head(1)

#We will split data into train and validate
train_df,valid_df= final.randomSplit([0.7,0.3])
train_df.describe().show()

#initializing and fitting model
lr= LinearRegression(labelCol="SalePrice")
model= lr.fit(train_df)

#fitting model of validation set
validate=model.evaluate(valid_df)

#let's check how model performed
print(validate.rootMeanSquaredError)
print(validate.r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11. Build Pipeline for Train

# COMMAND ----------

_stages = []

##Imputer
null_impute = Imputer(inputCols= int_features, outputCols=int_features) 
_stages += [null_impute]

##Encoder
str_indexer = [StringIndexer(inputCol=column,
                           outputCol=f'{column}_StringIndexer',
                            handleInvalid='keep') 
               for column in str_features]
_stages += str_indexer

#Assembler
assembler_input = [f for f in int_features] 
assembler_input += [f'{column}_StringIndexer' 
                    for column  in str_features] 
feature_vector = VectorAssembler(inputCols=assembler_input, 
                                 outputCol='features', 
                                 handleInvalid = 'keep' )
_stages += [feature_vector]

#Vector Encoder
vect_indexer = VectorIndexer(inputCol='features', 
                             outputCol= 'features_indexed', 
                             handleInvalid = 'keep' )
_stages += [vect_indexer]

# COMMAND ----------

splits = df_train.randomSplit([0.7, 0.3])
train = splits[0]
val = splits[1]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 12. Linear Regression

# COMMAND ----------

#LR Model
LR = LinearRegression(featuresCol='features_indexed', 
                      labelCol= 'SalePrice',
                     maxIter=10,
                     regParam=0.3,
                     elasticNetParam=0.8)

ml_pipeline = Pipeline(stages=_stages + [LR])
lr_model = ml_pipeline.fit(train)

# COMMAND ----------

lr_predictions = lr_model.transform(val)
lr_predictions.select("Id","prediction","SalePrice","features").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 13. Predicting the House Prices of Test Dataset By Linear Regression

# COMMAND ----------

test_predictions = lr_model.transform(df_test_typecast)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 14. Display Test Dataset Predictionsc By Linear Regression

# COMMAND ----------

test_predictions.select("Id","prediction","features").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 15. Collate and Format the Predictions

# COMMAND ----------

pred = test_predictions.select("Id","prediction")
pred = pred.withColumnRenamed("prediction","SalePrice")

from pyspark.sql.types import FloatType, IntegerType

#pred.printSchema()
pred = pred.withColumn("Id", pred["Id"].cast(IntegerType()))
pred = pred.withColumn("SalePrice", pred["SalePrice"].cast(FloatType()))
pred = pred.sort("Id")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 16. Save the Predictions

# COMMAND ----------

pred.toPandas().head()

# COMMAND ----------

# save in databricks dbfs file system
ct = datetime.datetime.now()
pred.write.option("header",True).csv("dbfs:/FileStore/tables/submission-lr-"+str(ct)+".csv")

# save in local csv file
pred.toPandas().to_csv("submission-lr-"+str(ct)+".csv")

# COMMAND ----------

# display few records from submission csv
df_submission = spark.read.csv('/FileStore/tables/submission-lr-'+str(ct)+'.csv', inferSchema=True, header=True)
df_submission.toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 17. Random Forest Regression

# COMMAND ----------

rf = RandomForestRegressor(featuresCol = 'features', labelCol='SalePrice', 
                           maxDepth=20, 
                           minInstancesPerNode=2,
                           bootstrap=True,
                           maxBins=350
                          )
ml_pipeline = Pipeline(stages=_stages + [rf])
rf_model = ml_pipeline.fit(df_train)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 18. Predictions Using Random Forest Regression

# COMMAND ----------

rf_predictions = rf_model.transform(df_test_typecast)
rf_predictions.select("Id","prediction","features").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 19. Collate the Predictions Using Random Forest Regression

# COMMAND ----------

pred = rf_predictions.select("Id","prediction")
pred = pred.withColumnRenamed("prediction","SalePrice")

from pyspark.sql.types import FloatType, IntegerType

#pred.printSchema()
pred = pred.withColumn("Id", pred["Id"].cast(IntegerType()))
pred = pred.withColumn("SalePrice", pred["SalePrice"].cast(FloatType()))
pred = pred.sort("Id")

pred.toPandas().head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 20. Save the Predictions of Random Forest Regression

# COMMAND ----------

# save in databricks dbfs file system
ct = datetime.datetime.now()
pred.write.option("header",True).csv("dbfs:/FileStore/tables/submission-rf-"+str(ct)+".csv")

# save in local csv file
pred.toPandas().to_csv("submission-rf-"+str(ct)+".csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 21. Explainability

# COMMAND ----------

rf_predictions.select(['Id','GrLivArea','prediction']).sort(['prediction'],ascending=False).toPandas().plot(x="GrLivArea", y="prediction")

# COMMAND ----------

rf_predictions.select(['Id','OverallQual','prediction']).sort(['prediction'],ascending=False).toPandas().plot(x="OverallQual", y="prediction")

# COMMAND ----------

rf_predictions.select(['Id','GarageCars','prediction']).sort(['prediction'],ascending=False).toPandas().plot(x="GarageCars", y="prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * From the histogram plots we could observe the predictions are more relatively for houses with more OverallQual, GrLivArea and GarageCars
# MAGIC 
# MAGIC * We can see that OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF ,1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, Fireplaces, BsmtFinSF1, WoodDeckSF, 2ndFlrSF, OpenPorchSF, HalfBath, LotArea, BsmtFinSF1 are the top influencers of the Sale Price
# MAGIC 
# MAGIC * The predictions from Random Forest Regression are better than those from Linear Regression. This is because there are 18 important features after feature extraction, which is a large number and more complex. Even though there is a risk of overfitting the data using Random Forest. It is still better than Linear Regression where important features would be missed and underfitting would occur.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 22. Model Versioning
# MAGIC 
# MAGIC * The Models are versioned based on type of model and timestamps when they were created. These are stored of databrick file system hosted on AWS S3 buckets of the databricks stack on AWS.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 23. Future Scope
# MAGIC 
# MAGIC * We want to further explain the model using SHAP library and analyse bias and variance factors in the underlying data
# MAGIC 
# MAGIC * We want to also create a data versioning to explain the model and concept drifts in the future.
