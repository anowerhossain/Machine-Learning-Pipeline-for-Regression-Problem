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


