#!/usr/bin/python

import pandas as pd
import json
import time, sys, os, shutil, glob, io, requests
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.sql import SQLContext
import dsx_core_utils
sys.path.insert(0, '/user-home/.scripts/common-helpers')
import published_model_util

# define variables
args={"threshold": {"metric": "accuracyScore", "min_value": 0.3, "mid_value": 0.7}, "evaluator_type": "binary", "published": "false", "dataset": "/datasets/customer_churn.csv"}
model_path = os.getenv("DSX_PROJECT_DIR")+"/models/Telco Churn Prediction Model 02/1/model"
if(False):
    input_data = args.get("dataset")
else:
    input_data = os.getenv("DSX_PROJECT_DIR")+args.get("dataset")
user_id = os.environ['DSX_USER_ID']
project_name = os.environ['DSX_PROJECT_NAME']

published_path = ''
# if published use model path
if(args.get('published').lower() == 'true'):
    copy_result = json.loads(published_model_util.copy_model(project_name, "Telco Churn Prediction Model 02"))
    if(copy_result['code'] == 200):
        model_path = copy_result['path'] + "/model"
        published_path = copy_result['path']
    else:
        raise Exception('Unable to evaluate published model: ' + copy_result['description'])

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# load the input data
if(False):
    dataSet = dsx_core_utils.get_remote_data_set_info(input_data.get('dataset'))
    dataSource = dsx_core_utils.get_data_source_info(dataSet['datasource'])
    dbTableOrQuery = (dataSet['schema'] + '.' if(len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
    dataframe = spark.read.format("jdbc").option("url", dataSource['URL']).option("dbtable",dbTableOrQuery).option("user",dataSource['user']).option("password",dataSource['password']).load()
else:
    dataframe = SQLContext(sc).read.csv(input_data , header='true', inferSchema = 'true')

# load the model from disk 
model_rf = PipelineModel.load(model_path)

# generate predictions
predictions = model_rf.transform(dataframe)

# Create Evalutation JSON
evaluation = dict()
evaluation["metrics"] = dict()

threshold={u'metric': 'accuracyScore', u'min_value': 0.3, u'mid_value': 0.7}

# replace "label" below with the numeric representation of the label column that you defined while training the model
labelCol = "label"

# create evaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol=labelCol)

# compute evaluations
evaluation["metrics"]["accuracyScore"] = predictions.rdd.filter(lambda x: x[labelCol] == x["prediction"]).count() * 1.0 / predictions.count()
evaluation["metrics"]["areaUnderPR"] = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
evaluation["metrics"]["areaUnderROC"] = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
evaluation["metrics"]["threshold"] = threshold

if(evaluation["metrics"][threshold.get('metric','INVALID_METRIC')] >= threshold.get('mid_value', 0.70)):
    evaluation["performance"] = "good"
elif(evaluation["metrics"][threshold.get('metric','INVALID_METRIC')] <= threshold.get('min_value', 0.25)):
    evaluation["performance"] = "poor"
else:
    evaluation["performance"] = "fair"


evaluation["modelName"] = "Telco Churn Prediction Model 02"
evaluation["startTime"] = int(time.time())

if(args.get('published').lower() == 'true'):
    evaluations_file_path = published_path +'/evaluations.json'
    evaluation["deployment"] = "default"
else:
    evaluations_file_path = os.getenv("DSX_PROJECT_DIR") + '/models/' + str("Telco Churn Prediction Model 02") + '/' + str("1") + '/evaluations.json'
    evaluation["modelVersion"] = "1"

if(os.path.isfile(evaluations_file_path)):
    current_evaluations = json.load(open(evaluations_file_path))
else:
    current_evaluations = []
current_evaluations.append(evaluation)

with open(evaluations_file_path, 'w') as outfile:
    json.dump(current_evaluations, outfile, indent=4, sort_keys=True)

#copy to dir with helper function
if (len(published_path) > 0):
    published_model_util.update_evaluation_metrics(project_name, "Telco Churn Prediction Model 02")
    published_model_util.delete_temp_model()