# Machine-Learning-Pipeline-for-Regression-Problem
## Step-by-step guide for the project, each with an explanation and corresponding code. 🚀
# Step 1: 📂 Import Necessary Libraries

```python
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
```

- Import essential libraries for Spark and machine learning.
- findspark initializes Spark for Python.

# Step 2: 🚀 Initialize Spark Session

```python
spark = SparkSession.builder.appName("Vehicle MPG Prediction Pipeline").getOrCreate()
```

- Create a Spark session to process data and run machine learning tasks.

# Step 3: 📥 Download and Load the Dataset

```bash
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/datasets/mpg-raw.csv
```
```python
df = spark.read.csv("mpg-raw.csv", header=True, inferSchema=True)
```
- Download the dataset and load it into a Spark DataFrame.

# Step 4: 🔍 Explore Dataset

```python
df.show(5)
df.printSchema()
df.groupBy('Origin').count().orderBy('count').show()
```
- Inspect the first few rows, schema, and unique counts for the Origin column.


# Step 5: ✂️ Clean Data
```python
rowcount1 = df.count()
df = df.dropDuplicates()
rowcount2 = df.count()
df = df.dropna()
rowcount3 = df.count()
df = df.withColumnRenamed("Engine Disp", "Engine_Disp")
```

- Remove duplicates.
- Handle missing values by dropping rows with null values.
- Rename columns for compatibility.



# Step 6: 💾 Save Cleaned Data
```python
df.write.mode("overwrite").parquet("mpg-cleaned.parquet")
```

- Save the cleaned data as a Parquet file for optimized storage.
# Step 7: 🏗️ Build ML Pipeline
```python
indexer = StringIndexer(inputCol="Origin", outputCol="OriginIndex")
assembler = VectorAssembler(inputCols=['Cylinders', 'Engine_Disp', 'Horsepower', 
                                        'Weight', 'Accelerate', 'Year'], 
                            outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="MPG")

pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])
```
- Transform categorical data using StringIndexer.
- Combine multiple features into a single vector using VectorAssembler.
- Standardize features with StandardScaler.
- Use LinearRegression for prediction.


# Step 8: 📊 Split Data

```python
(trainingData, testingData) = df.randomSplit([0.7, 0.3], seed=42)
```
- Split the data into training (70%) and testing (30%) sets.

# Step 9: 🏋️ Train Model

```python
pipelineModel = pipeline.fit(trainingData)
```

- Train the pipeline on the training dataset.

# Step 10: 🧪 Evaluate Model

```python
predictions = pipelineModel.transform(testingData)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="mse")
mse = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="mae")
mae = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="MPG", metricName="r2")
r2 = evaluator.evaluate(predictions)

print("Mean Squared Error = ", round(mse, 2))
print("Mean Absolute Error = ", round(mae, 2))
print("R Squared = ", round(r2, 2))
```

- Use RegressionEvaluator to compute MSE, MAE, and R² scores.

# Step 11: 💾 Save and Reload Model
```python
pipelineModel.write().save("Vehicle_MPG_Pipeline")
loadedPipelineModel = PipelineModel.load("Vehicle_MPG_Pipeline")
```
- Save the trained pipeline and reload it for reuse.

# Step 12: 🔍 Inspect Model Coefficients

```python
loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[1].getInputCols()

print("Number of stages in the pipeline = ", totalstages)
for feature, coef in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {feature} = {round(coef, 4)}")
```
- Inspect the pipeline structure and regression coefficients.
