# Отправка dataset'ов на удаленные сервера

1. Отправка на 1 ноду:

```bash
scp -P 22 -i /home/arthur/Project/GeekBrains/Spark_streaming/_ADDS/id_rsa_student910_14 -r /home/arthur/Project/GeekBrains/Spark_streaming/HW8/source/ student910_14@37.139.32.56:/home/student910_14/8/
```

2. Отправка на worker ноду:

```bash
scp -r source 10.0.0.19:/home/student910_14/hw8/
```

3. Конвертация из csv в json:

```python
import csv
import json

for dataset in ['train','test']:
    csvfile = open(f'{dataset}.csv', 'r')
    jsonfile = open(f'{dataset}.json', 'w')
    fieldnames = ('Id','DistrictId','Rooms','Square','LifeSquare','KitchenSquare','Floor','HouseFloor','HouseYear','Ecology_1','Ecology_2','Ecology_3','Social_1','Social_2','Social_3','Healthcare_1','Helthcare_2','Shops_1','Shops_2','Price')
    reader = csv.DictReader(csvfile, fieldnames)
    for row in reader:
        json.dump(row, jsonfile)
        jsonfile.write('\n')
```

4. Отправка в HDFS:

```bash
hdfs dfs -put ./source /user/student910_14/hw8/
```

# Работа с Kafka

1. Создание topic `910_14_testDataset` с входящими данными:

```bash
/usr/hdp/3.1.4.0-315/kafka/bin/kafka-topics.sh --create --topic 910_14_testDataset --zookeeper bigdataanalytics2-worker-shdpt-v31-1-4:2181 --partitions 3 --replication-factor 2 --config retention.ms=17280000000
```

2. Создание topic `910_14_resDataset` для выгрузки предсказаний:

```bash
/usr/hdp/3.1.4.0-315/kafka/bin/kafka-topics.sh --create --topic 910_14_resDataset --zookeeper bigdataanalytics2-worker-shdpt-v31-1-4:2181 --partitions 3 --replication-factor 2 --config retention.ms=17280000000
```

3. Загрузка данных в соответствующий topic - `910_14_testDataset`:

```python
### orders_data_uploader.py ###

from kafka import KafkaProducer
import sys
import json

def upload_orders_to_kafka(host, topic, orders):
    producer = KafkaProducer(bootstrap_servers=host)
    orders_topic_name = topic
    for order in orders:
        producer.send(orders_topic_name, bytes(order, 'utf-8'))


def read_orders(orders_file_path):
    with open(orders_file_path, 'r') as f:
        orders_json = f.readlines()

    return orders_json


if __name__ == '__main__':
    args = sys.argv[1:]

    # arg 1 - orders file path
    orders_file_path = args[0]
    print("Orders file path: {}".format(orders_file_path))

    # arg 2 - kafka host
    kafka_host = args[1]
    print("Kafka host: {}".format(kafka_host))

    # arg 3 - kafka target orders topic
    kafka_target_topic = args[2]
    print("Kafka topic: {}".format(kafka_target_topic))

    # read orders json file
    orders = read_orders(orders_file_path)
    # upload orders to kafka
    upload_orders_to_kafka(kafka_host, kafka_target_topic, orders)
```

```bash
python3.7 orders_data_uploader.py test.json bigdataanalytics2-worker-shdpt-v31-1-5:6667 910_14_testDataset
```

# Обучение и сохранение модели

```bash
/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]
```

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer


## CONSTANTS
data_path = '/user/student910_14/hw8/source/train.csv'
model_dir = '/user/student910_14/hw8/models'
feature_names = ['Rooms', 'Square', 'LifeSquare', 'KitchenSquare', 
                 'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Social_1']
target_name = 'Price'

# необходимые функции
def prepare(data_to_prepare):
    # работа с выбросами
    data_to_prepare = data_to_prepare.where('Square < 300')
    data_to_prepare = data_to_prepare.where('KitchenSquare < 100')
    data_to_prepare = data_to_prepare.where('HouseFloor < 50')
    data_to_prepare = data_to_prepare.where('HouseFloor > 2')
    data_to_prepare = data_to_prepare.where('HouseYear <= 2020')
    
    # работа с NaN
    data_to_prepare = data_to_prepare.na.fill(0)
    
    # векторизация фичей (spark работает с одной колонкой фичей)
    assembler = VectorAssembler(inputCols=feature_names, 
                                outputCol='features')
    data_to_prepare = assembler.transform(data_to_prepare)
    data_to_prepare = data_to_prepare.select('features', target_name)
    return data_to_prepare

if __name__ == '__main__':
    spark = SparkSession.builder.appName("ML910_14")\
                                .master("local[*]")\
                                .getOrCreate()
    # подготовка данных
    data = spark\
        .read\
        .format("csv")\
        .options(inferSchema=True, header=True) \
        .load(data_path)
    #data = data.repartition(20)
    model_data = prepare(data)
    train, test = model_data.randomSplit([0.8, 0.2], seed=42)
    
    # обучение модели. maxIter должно быть 100 (но модель получается тяжелой и кластер ее не пускает)
    model = GBTRegressor(featuresCol="features", 
                         labelCol=target_name, 
                         maxIter=16)
    model = model.fit(train)

    # оценка модели
    evaluator = RegressionEvaluator() \
            .setMetricName("r2") \
            .setLabelCol(target_name) \
            .setPredictionCol("prediction")
    prediction = model.transform(test)
    evaluation_result = evaluator.evaluate(prediction)
    print("Evaluation result: {}".format(evaluation_result)) # 0.716358

    # сохранение модели
    model.write().overwrite().save(model_dir + "/model")
```

# Использование модели на данных из потока

```python
### submit_batch-predictions.py ###
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.feature import VectorAssembler, StringIndexer

# .config('spark.jars', 'jar/snowflake-jdbc-3.13.4.jar,jar/spark-snowflake_2.12-2.9.0-spark_3.1.jar') \
spark = SparkSession \
        .builder \
        .appName("ML910_14_predict") \
        .getOrCreate()

## CONSTANTS
checkpoint_location = "tmp/ml_checkpoint"
data_path = '/user/student910_14/hw8/source/*.csv'
model_dir = '/user/student910_14/hw8/models'
feature_names = ['Rooms', 'Square', 'LifeSquare', 'KitchenSquare', 
                 'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Social_1']
target_name = 'Price'
schema = StructType() \
		 .add("Id", StringType()) \
		 .add("DistrictId", StringType()) \
		 .add("Rooms", StringType()) \
		 .add("Square", StringType()) \
		 .add("LifeSquare", StringType()) \
		 .add("KitchenSquare", StringType()) \
		 .add("Floor", StringType()) \
		 .add("HouseFloor", StringType()) \
		 .add("HouseYear", StringType()) \
		 .add("Ecology_1", StringType()) \
		 .add("Ecology_2", StringType()) \
		 .add("Ecology_3", StringType()) \
		 .add("Social_1", StringType()) \
		 .add("Social_2", StringType()) \
		 .add("Social_3", StringType()) \
		 .add("Healthcare_1", StringType()) \
		 .add("Helthcare_2", StringType()) \
		 .add("Shops_1", StringType()) \
		 .add("Shops_2", StringType())

# необходимые функции
def prepare(data_to_prepare):
    # работа с выбросами
    data_to_prepare = data_to_prepare.where('Square < 300')
    data_to_prepare = data_to_prepare.where('KitchenSquare < 100')
    data_to_prepare = data_to_prepare.where('HouseFloor < 50')
    data_to_prepare = data_to_prepare.where('HouseFloor > 2')
    data_to_prepare = data_to_prepare.where('HouseYear <= 2020')
    
    # работа с NaN
    data_to_prepare = data_to_prepare.na.fill(0)
    
    # векторизация фичей (spark работает с одной колонкой фичей)
    assembler = VectorAssembler(inputCols=feature_names, 
                                outputCol='features')
    data_to_prepare = assembler.transform(data_to_prepare)
    data_to_prepare = data_to_prepare.select('features', target_name)
    return data_to_prepare

def process_batch(df, epoch):
    model_data = prepare(df)
    prediction = model.transform(model_data)
    kafka_output(prediction, 5, kafka_topic_res)
    prediction.show()

def foreach_batch_output(df):
    from datetime import datetime as dt
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df\
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", checkpoint_location + "/" + date)\
        .start()

def console_output(df, freq):
	from datetime import datetime as dt
	date = dt.now().strftime("%Y%m%d%H%M%S")
	return df.writeStream \
	    .format("console") \
	    .trigger(processingTime='%s seconds' % freq) \
	    .option("checkpointLocation", checkpoint_location + "/" + date) \
	    .options(truncate=False) \
	    .start()

def kafka_output(df, freq, topic):
    return df \
        .writeStream \
        .format("kafka") \
        .trigger(processingTime='%s seconds' % freq) \
        .option("topic", topic) \
        .option("kafka.bootstrap.servers", kafka_brokers) \
        .option("checkpointLocation", checkpoint_location) \
        .start()


kafka_brokers = "bigdataanalytics2-worker-shdpt-v31-1-0:6667"
kafka_topic = "910_14_testDataset"
kafka_topic_res = "910_14_resDataset"

data = spark.readStream. \
       format("kafka"). \
       option("kafka.bootstrap.servers", kafka_brokers). \
       option("subscribe", kafka_topic). \
       option("maxOffsetsPerTrigger", "20"). \
       option("startingOffsets", "earliest"). \
       load()
data = data \
       .select(F.from_json(F.col("value").cast("String"), schema).alias("value"), "offset") \
       .select("value.*")
# stream = console_output(data, 10)

model = GBTRegressionModel.load(model_dir + "/model")
stream = foreach_batch_output(data)
#stream.stop()

stream.awaitTermination()
```

Запуск файла

```bash
/spark2.4/bin/spark-submit ~/hw8/source/submit_batch-predictions.py
```
