# ---------------------------------------------------------------------------------------------------------------
# Import Section
# ---------------------------------------------------------------------------------------------------------------
import os
import logging

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from ml_pipeline import predict_house_price

# ---------------------------------------------------------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------------------------------------------------------

logging.basicConfig(filename="house_sale_price.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

train_data_file = "data/house-sales-train.csv"
test_data_file = "data/house-sales-test.csv"


# ---------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------
def init_spark():
    conf = SparkConf();
    #conf.setMaster("spark:////172.31.21.121:7077")
    conf.setAppName("House Price Predictor")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    logger.info("Spark Engine Initialized: %s", spark.sparkContext._conf.getAll()  )
    return spark


def load_train_data(spark):
    logger.info("loading training data")
    df_train = spark.read.csv(train_data_file, inferSchema=True, header=True)
    #logger.info("loaded training data from ", train_data_file)
    df_train.cache()
    logger.info("Train Data Record Count: %s", df_train.count())
    return df_train


def load_test_data(spark):
    logger.info("loading test data")
    df_test = spark.read.csv(test_data_file, inferSchema=True, header=True)
    #logger.info("loaded test data from ", train_data_file)
    logger.info("Test Data Record Count: %s", df_test.count())
    df_test.cache()
    return df_test


# ---------------------------------------------------------------------------------------------------------------
# Entry point to the app
# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    spark = init_spark()
    train_data = load_train_data(spark);
    test_data = load_test_data(spark);
    df_predict = predict_house_price(train_data, test_data)
    # df_predict.withColumnRenamed('prediction', 'SalePrice') \
    #      .select('Id', 'SalePrice') \
    #      .coalesce(1) \
    #      .write.csv('submission', mode='overwrite', header=True)
    # print(os.listdir('submission'))
