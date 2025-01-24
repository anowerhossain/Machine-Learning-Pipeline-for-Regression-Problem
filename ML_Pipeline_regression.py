import warnings
warnings.filterwarnings('ignore')

def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Install necessary packages
!pip install pyspark==3.1.2 -q
!pip install findspark -q

# Initialize FindSpark to simplify using Spark with Python
import findspark
findspark.init()

# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Step 1: Create SparkSession
spark = SparkSession.builder.appName("Practice Project").getOrCreate()

# Step 2: Download dataset
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/datasets/mpg-raw.csv

# Step 3: Load dataset into a Spark DataFrame
data = spark.read.csv("mpg-raw.csv", header=True, inferSchema=True)

# Step 4: Preview the dataset
data.show(5)

# Step 5: Group data by 'Origin' column and display count
origin_grouped = data.groupBy('Origin').count().orderBy('count')
origin_grouped.show()

# Step 6: Data Cleaning - Count initial rows
initial_row_count = data.count()
print("Total initial rows: ", initial_row_count)

# Remove duplicate rows
data_no_duplicates = data.dropDuplicates()
post_deduplication_count = data_no_duplicates.count()
print("Rows after removing duplicates: ", post_deduplication_count)

# Remove rows with null values
data_cleaned = data_no_duplicates.dropna()
final_row_count = data_cleaned.count()
print("Rows after removing null values: ", final_row_count)

# Rename column for better readability
data_cleaned = data_cleaned.withColumnRenamed("Engine Disp", "Engine_Disp")

# Save cleaned data as Parquet file
data_cleaned.write.mode("overwrite").parquet("mpg-cleaned.parquet")

# Validate the Parquet file creation
print("Parquet file created: ", os.path.isdir("mpg-cleaned.parquet"))

# Step 7: Feature Engineering and Pipeline Creation
# Encode the 'Origin' column
origin_indexer = StringIndexer(inputCol="Origin", outputCol="OriginIndex")

# Assemble input features into a single vector
feature_assembler = VectorAssembler(
    inputCols=['Cylinders', 'Engine_Disp', 'Horsepower', 'Weight', 'Accelerate', 'Year'],
    outputCol="features"
)

# Standardize the feature vector
feature_scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Define the linear regression model
linear_regression = LinearRegression(featuresCol="scaledFeatures", labelCol="MPG")

# Create a pipeline with all stages
pipeline = Pipeline(stages=[origin_indexer, feature_assembler, feature_scaler, linear_regression])

# Split data into training and testing sets
training_data, testing_data = data_cleaned.randomSplit([0.7, 0.3], seed=42)

# Train the pipeline model
pipeline_model = pipeline.fit(training_data)

# Step 8: Evaluate the Pipeline
# Get pipeline stage information
pipeline_stages = [str(stage).split("_")[0] for stage in pipeline.getStages()]
print("Pipeline Stages: ", pipeline_stages)

# Confirm the label column for regression
print("Label Column: ", linear_regression.getLabelCol())

# Step 9: Make Predictions and Evaluate the Model
predictions = pipeline_model.transform(testing_data)

# Define evaluators for different metrics
evaluator_mse = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="mse")
evaluator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="mae")
evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="r2")

# Calculate and display evaluation metrics
mse = evaluator_mse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("Mean Squared Error (MSE): ", round(mse, 2))
print("Mean Absolute Error (MAE): ", round(mae, 2))
print("R-Squared (R2): ", round(r2, 2))

# Retrieve and display model intercept
lr_model = pipeline_model.stages[-1]  # Get the linear regression model
print("Model Intercept: ", round(lr_model.intercept, 2))

# Step 10: Save and Reload the Pipeline Model
pipeline_model.write().save("Practice_Project")

# Reload the saved pipeline model
reloaded_model = PipelineModel.load("Practice_Project")

# Validate predictions with the reloaded model
reloaded_predictions = reloaded_model.transform(testing_data)
reloaded_predictions.select("MPG", "prediction").show()

# Step 11: Analyze Model Coefficients
print("Number of Stages in Pipeline: ", len(reloaded_model.stages))

# Display coefficients of each input feature
input_features = reloaded_model.stages[1].getInputCols()
for feature, coef in zip(input_features, lr_model.coefficients):
    print(f"Coefficient for {feature}: {round(coef, 4)}")

# Stop the Spark session
spark.stop()
