#!/usr/bin/python

import sys, os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
from pyspark.sql import SQLContext
import pandas
import json
sys.path.insert(0, '/user-home/.scripts/common-helpers')
import published_model_util, dsx_core_utils

# define variables
args={u'target': u'/datasets/custtomer_subset_scored', u'output_datasource_type': u'', u'execution_type': u'DSX', u'source': u'/datasets/customer_churn_subset.csv', u'output_type': u'Localfile', u'sysparm': u''}
input_data = os.getenv("DSX_PROJECT_DIR")+args.get("source")
output_data = os.getenv("DSX_PROJECT_DIR")+args.get("target")
model_name = 'Telco Churn Prediction Model 02'
model_path = os.getenv("DSX_PROJECT_DIR")+"/models/Telco Churn Prediction Model 02/1/model"
project_name = 'DSXL-Demo'
is_published = 'false'

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# read test dataframe (inputJson = "input.json") 
testDF = SQLContext(sc).read.csv(input_data , header='true', inferSchema = 'true')

published_path = ''
if is_published == 'true':
    copy_result = json.loads(published_model_util.copy_model(project_name, model_name))
    if(copy_result['code'] == 200):
        model_path = copy_result['path'] + '/model'
        published_path = copy_result['path']
    else:
        raise Exception('Unable to score published model: ' + copy_result['description'])

#load model
model_rf = PipelineModel.load(model_path)

#prediction
outputDF = model_rf.transform(testDF) 

# save scoring result to given target
scoring_df = outputDF.toPandas()

# save output to csv
scoring_df.to_csv(output_data)

if (len(published_path) > 0):
    published_model_util.delete_temp_model()