import logging

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, functions as F, DataFrame
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, VectorIndexer, Bucketizer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GeneralizedLinearRegression, \
    RandomForestRegressor, GBTRegressor
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType, BooleanType, DateType
from pyspark.sql.functions import col, isnan, when, count
import pyspark.sql.functions as F

# ---------------------------------------------------------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

logging.basicConfig(filename="house_sale_price.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------

def get_features (df_train):
    str_features = []
    int_features = []
    for col in  df_train.dtypes:
        if col[1] == 'string':
            str_features += [col[0]]
        else:
            int_features += [col[0]]
        # str_features, int_features = get_features (df_train)
    logger.info(f'Qualitative (String) Features ({len(str_features)}): {str_features}')
    logger.info(f'Quantitative (Int) Features ({len(int_features)}): {int_features}')
    return str_features, int_features

def count_missings(spark_df,sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(df) == 0:
        logger.info("There are no any missing values!")
        return None

    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count",ascending=False)
    logger.info(df)
    return df

def check_feature_cov(df_train):
    saleprice_cov = {}
    for col in df_train.dtypes:
        if col[0] != 'SalePrice' and col[1] != 'string':
            saleprice_cov[col[0]] = df_train.cov('SalePrice', col[0])

    dict(sorted(saleprice_cov.items(), key=lambda item: item[1]))


def analyze_features(train_data):
    str_features, int_features = get_features(train_data)
    count_missings(train_data)
    check_feature_cov(train_data)

def get_imputer(df_train):
    # One Hot Encoding and nan transformation
    data = pd.get_dummies(df_train)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(df_train.toPandas())

    # Log transformation
    data = np.log(df_train.toPandas())
    labels = np.log(data.SalePrice)

    # Change -inf to 0 again
    data[data == -np.inf] = 0


def get_categorical_encoder():
    pass


def get_assembler():
    pass


def get_vector_encoder():
    pass


def get_lr_model():
    pass


def get_ml_pipeline(train_data):
    analyze_features(train_data)
    _stages = []
    _stages += [get_imputer()]
    _stages += [get_categorical_encoder()]
    _stages += [get_assembler()]
    _stages += [get_vector_encoder()]
    _stages += [get_lr_model()]

    return Pipeline(stages=_stages)


def predict_house_price(train_data, test_data):
    pipeline = get_ml_pipeline(train_data)
    #model = pipeline.fit()
    return '';#model.transform(test_data)
