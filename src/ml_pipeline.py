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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

logging.basicConfig(filename="../logs/house_sale_price.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------

def get_features(df_train):
    str_features = []
    int_features = []
    for col in df_train.dtypes:
        if col[1] == 'string':
            str_features += [col[0]]
        else:
            int_features += [col[0]]
        # str_features, int_features = get_features (df_train)
    logger.info(f'Qualitative (String) Features ({len(str_features)}): {str_features}')
    logger.info(f'Quantitative (Int) Features ({len(int_features)}): {int_features}')
    return str_features, int_features


def count_missings(spark_df, sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c, c_type) in spark_df.dtypes if
                          c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(df) == 0:
        logger.info("There are no any missing values!")
        return None

    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count", ascending=False)
    logger.info(df)
    return df


def check_feature_cov(df_train):
    saleprice_cov = {}
    for col in df_train.dtypes:
        if col[0] != 'SalePrice' and col[1] != 'string':
            saleprice_cov[col[0]] = df_train.cov('SalePrice', col[0])

    saleprice_cov = dict(sorted(saleprice_cov.items(), key=lambda item: item[1]))
    logger.info(f'saleprice_cov: {saleprice_cov}')
    return saleprice_cov


def do_pca(df_train):
    # One Hot Encoding and nan transformation
    data = pd.get_dummies(df_train.toPandas())

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(data)

    # Supress/hide the warning
    np.seterr(invalid='ignore')

    # Log transformation
    data = np.log(data)
    labels = df_train.toPandas()["SalePrice"]
    labels = np.log(labels)

    # Change -inf to 0 again
    data[data == -np.inf] = 0

    pca = PCA(whiten=True)
    pca.fit(data)
    variance = pd.DataFrame(pca.explained_variance_ratio_)
    logger.info(f'PCA - variance: {variance}')
    pca_cumsum = np.cumsum(pca.explained_variance_ratio_)
    logger.info(f'PCA - cumulative sum: {pca_cumsum}')

    pca = PCA(n_components=36, whiten=True)
    pca = pca.fit(data)
    dataPCA = pca.transform(data)
    logger.info(f'PCA Data: {dataPCA}')
    return dataPCA


def analyze_features(train_data):
    count_missings(train_data)
    saleprice_cov = check_feature_cov(train_data)
    dataPCA = do_pca(train_data)


def get_imputer(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(data)
    return data


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

    str_features, int_features = get_features(train_data)

    # One Hot Encoding and nan transformation
    data = pd.get_dummies(train_data.toPandas())

    _stages = []

    # Impute
    null_impute = Imputer(inputCols=int_features, outputCols=int_features)
    # SimpleImputer(missing_values=np.nan, strategy='mean')
    _stages += [null_impute]

    # Category Encoder
    str_indexer = [StringIndexer(inputCol=column,
                                 outputCol=f'{column}_StringIndexer',
                                 handleInvalid='keep')
                   for column in str_features]
    _stages += str_indexer

    # Assembler
    assembler_input = [f for f in int_features]
    assembler_input += [f'{column}_StringIndexer'
                        for column in str_features]
    feature_vector = VectorAssembler(inputCols=assembler_input,
                                     outputCol='features',
                                     handleInvalid='keep')
    _stages += [feature_vector]

    # Vector Encoder
    vect_indexer = VectorIndexer(inputCol='features',
                                 outputCol='features_indexed',
                                 handleInvalid='keep')
    _stages += [vect_indexer]

    # Model
    LR = LinearRegression(featuresCol='features_indexed',
                          labelCol='SalePrice',
                          maxIter=10,
                          regParam=0.3,
                          elasticNetParam=0.8)
    _stages += [LR]

    return Pipeline(stages=_stages)


def cast_to_int_1(_sdf, col_list: list):
    for col in col_list:
        # _sdf[col] = _sdf[col].astype('int')
        _sdf = _sdf.withColumn(col, _sdf[col].cast(IntegerType))
    return _sdf

def cast_to_int_2(_sdf):
    for col in _sdf.dtypes:
        # _sdf[col] = _sdf[col].astype('int')
        print(f'col: {col[0]}, col_type: {col[1]}')
        if col[1] != 'int':
            _sdf = _sdf.withColumn(col[0], _sdf[col[0]].cast(IntegerType))
    return _sdf


def cast_to_int(_sdf):
    for col in _sdf.dtypes:
        print(f'col: {col[0]}, col_type: {col[1]}')
        indexer = StringIndexer(inputCol=col[0], outputCol=col[0]+"1")
        indexed = indexer.fit(_sdf).transform(_sdf)
    return indexed


def predict_house_price(train_data, test_data):
    # train_data = pd.get_dummies(train_data)
    # imp = SimpleImputer(missing_values='NaN', strategy='most_frequent')
    # train_data = imp.fit_transform(train_data)
    #
    # test_data = pd.get_dummies(test_data)
    # imp = SimpleImputer(missing_values='NaN', strategy='most_frequent')
    # test_data = imp.fit_transform(test_data)

    # str_features_train, int_features_train = get_features(train_data)
    # sdf_train_filter = train_data.select (str_features_train + int_features_train)
    # train_data = cast_to_int(sdf_train_filter, int_features_train)

    # train_data = cast_to_int(train_data)
    # test_data = cast_to_int(test_data)

    pipeline = get_ml_pipeline(train_data)
    model = pipeline.fit(train_data)
    try:
        # str_features_test, int_features_test = get_features(test_data)
        # sdf_test_filter = test_data.select(str_features_test + int_features_test)
        # test_data = cast_to_int(sdf_test_filter, int_features_test)

        df_predict = model.transform(test_data)
        logger.info(f'df_predict: {df_predict.show(10)}')
    except Exception as e:
        logger.error("Exception occurred during transform %s", e.args[1])
    return df_predict
