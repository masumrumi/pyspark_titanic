![plot](./images/titanic-submersible-oceangate-illustration-andrea-gatti.jpg)

This kernel will give a tutorial for starting out with PySpark using Titanic dataset. Let's get started.

### Kernel Goals

<a id="aboutthiskernel"></a>

---

There are three primary goals of this kernel.

- <b>Provide a tutorial for someone who is starting out with pyspark.
- <b>Do an exploratory data analysis(EDA)</b> of titanic with visualizations and storytelling.
- <b>Predict</b>: Use machine learning classification models to predict the chances of passengers survival.

### What is Spark, anyway?

Spark is a platform for cluster computing. Spark lets us spread data and computations over clusters with multiple nodes (think of each node as a separate computer). Splitting up data makes it easier to work with very large datasets because each node only works with a small amount of data.
As each node works on its own subset of the total data, it also carries out a part of the total calculations required, so that both data processing and computation are performed in parallel over the nodes in the cluster. It is a fact that parallel computation can make certain types of programming tasks much faster.

Deciding whether or not Spark is the best solution for your problem takes some experience, but you can consider questions like:

- Is my data too big to work with on a single machine?
- Can my calculations be easily parallelized?

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    ../input/titanic/test.csv
    ../input/titanic/train.csv

```python
## installing pyspark
!pip install pyspark
```

    Requirement already satisfied: pyspark in /Users/masumrumi/.local/share/virtualenvs/pyspark_titanic-AKXOLEKJ/lib/python3.11/site-packages (3.5.1)
    Requirement already satisfied: py4j==0.10.9.7 in /Users/masumrumi/.local/share/virtualenvs/pyspark_titanic-AKXOLEKJ/lib/python3.11/site-packages (from pyspark) (0.10.9.7)

The first step in using Spark is connecting to a cluster. In practice, the cluster will be hosted on a remote machine that's connected to all other nodes. There will be one computer, called the master that manages splitting up the data and the computations. The master is connected to the rest of the computers in the cluster, which are called worker. The master sends the workers data and calculations to run, and they send their results back to the master.

We definitely don't need may clusters for Titanic dataset. In addition to that, the syntax for running locally or using many clusters are pretty similar. To start working with Spark DataFrames, we first have to create a SparkSession object from SparkContext. We can think of the SparkContext as the connection to the cluster and SparkSession as the interface with that connection. Let's create a SparkSession.

# Beginner Tutorial

This part is solely for beginners. I recommend starting from here to get a good understanding of the flow.

```python
## creating a spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tutorial').getOrCreate()
```

Let's read the dataset.

```python
df_train = spark.read.csv('../input/titanic/train.csv', header = True, inferSchema=True)
df_test = spark.read.csv('../input/titanic/test.csv', header = True, inferSchema=True)
```

```python
titanic_train = df_train.alias("titanic_train")
```

```python
## So, what is df_train?
type(df_train)
```

    pyspark.sql.dataframe.DataFrame

```python
## As you can see it's a Spark dataframe. Let's take a look at the preview of the dataset.
df_train.show(truncate=False)
```

    +-----------+--------+------+-------------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|Name                                                   |Sex   |Age |SibSp|Parch|Ticket          |Fare   |Cabin|Embarked|
    +-----------+--------+------+-------------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |1          |0       |3     |Braund, Mr. Owen Harris                                |male  |22.0|1    |0    |A/5 21171       |7.25   |NULL |S       |
    |2          |1       |1     |Cumings, Mrs. John Bradley (Florence Briggs Thayer)    |female|38.0|1    |0    |PC 17599        |71.2833|C85  |C       |
    |3          |1       |3     |Heikkinen, Miss. Laina                                 |female|26.0|0    |0    |STON/O2. 3101282|7.925  |NULL |S       |
    |4          |1       |1     |Futrelle, Mrs. Jacques Heath (Lily May Peel)           |female|35.0|1    |0    |113803          |53.1   |C123 |S       |
    |5          |0       |3     |Allen, Mr. William Henry                               |male  |35.0|0    |0    |373450          |8.05   |NULL |S       |
    |6          |0       |3     |Moran, Mr. James                                       |male  |NULL|0    |0    |330877          |8.4583 |NULL |Q       |
    |7          |0       |1     |McCarthy, Mr. Timothy J                                |male  |54.0|0    |0    |17463           |51.8625|E46  |S       |
    |8          |0       |3     |Palsson, Master. Gosta Leonard                         |male  |2.0 |3    |1    |349909          |21.075 |NULL |S       |
    |9          |1       |3     |Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)      |female|27.0|0    |2    |347742          |11.1333|NULL |S       |
    |10         |1       |2     |Nasser, Mrs. Nicholas (Adele Achem)                    |female|14.0|1    |0    |237736          |30.0708|NULL |C       |
    |11         |1       |3     |Sandstrom, Miss. Marguerite Rut                        |female|4.0 |1    |1    |PP 9549         |16.7   |G6   |S       |
    |12         |1       |1     |Bonnell, Miss. Elizabeth                               |female|58.0|0    |0    |113783          |26.55  |C103 |S       |
    |13         |0       |3     |Saundercock, Mr. William Henry                         |male  |20.0|0    |0    |A/5. 2151       |8.05   |NULL |S       |
    |14         |0       |3     |Andersson, Mr. Anders Johan                            |male  |39.0|1    |5    |347082          |31.275 |NULL |S       |
    |15         |0       |3     |Vestrom, Miss. Hulda Amanda Adolfina                   |female|14.0|0    |0    |350406          |7.8542 |NULL |S       |
    |16         |1       |2     |Hewlett, Mrs. (Mary D Kingcome)                        |female|55.0|0    |0    |248706          |16.0   |NULL |S       |
    |17         |0       |3     |Rice, Master. Eugene                                   |male  |2.0 |4    |1    |382652          |29.125 |NULL |Q       |
    |18         |1       |2     |Williams, Mr. Charles Eugene                           |male  |NULL|0    |0    |244373          |13.0   |NULL |S       |
    |19         |0       |3     |Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)|female|31.0|1    |0    |345763          |18.0   |NULL |S       |
    |20         |1       |3     |Masselmani, Mrs. Fatima                                |female|NULL|0    |0    |2649            |7.225  |NULL |C       |
    +-----------+--------+------+-------------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 20 rows

```python
## It looks a bit messi. See what I did there? ;). Anyway, how about using .toPandas() for change.
df_train.toPandas()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>"Johnston, Miss. Catherine Helen ""Carrie"""</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>None</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>

```python
## how about a summary.
result = df_train.describe().toPandas()
```

    24/05/30 18:55:58 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.


```python
result
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>714</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>446.0</td>
      <td>0.3838383838383838</td>
      <td>2.308641975308642</td>
      <td>None</td>
      <td>None</td>
      <td>29.69911764705882</td>
      <td>0.5230078563411896</td>
      <td>0.38159371492704824</td>
      <td>260318.54916792738</td>
      <td>32.2042079685746</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>257.3538420152301</td>
      <td>0.48659245426485753</td>
      <td>0.8360712409770491</td>
      <td>None</td>
      <td>None</td>
      <td>14.526497332334035</td>
      <td>1.1027434322934315</td>
      <td>0.8060572211299488</td>
      <td>471609.26868834975</td>
      <td>49.69342859718089</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>"Andersson, Mr. August Edvard (""Wennerstrom"")"</td>
      <td>female</td>
      <td>0.42</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>0.0</td>
      <td>A10</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>891</td>
      <td>1</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>80.0</td>
      <td>8</td>
      <td>6</td>
      <td>WE/P 5735</td>
      <td>512.3292</td>
      <td>T</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

```python
# getting the total row count
df_train.count()
```

    891

```python
# We can also convert a pandas dataframe to spark dataframe. Here is how we do it.
print(f"Before: {type(result)}")
spark_temp = spark.createDataFrame(result)
print(f"After: {type(spark_temp)}")
```

    Before: <class 'pandas.core.frame.DataFrame'>
    After: <class 'pyspark.sql.dataframe.DataFrame'>

```python
# pyspark version
spark_temp.show()
```

    +-------+-----------------+-------------------+------------------+--------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
    |summary|      PassengerId|           Survived|            Pclass|                Name|   Sex|               Age|             SibSp|              Parch|            Ticket|             Fare|Cabin|Embarked|
    +-------+-----------------+-------------------+------------------+--------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+
    |  count|              891|                891|               891|                 891|   891|               714|               891|                891|               891|              891|  204|     889|
    |   mean|            446.0| 0.3838383838383838| 2.308641975308642|                NULL|  NULL| 29.69911764705882|0.5230078563411896|0.38159371492704824|260318.54916792738| 32.2042079685746| NULL|    NULL|
    | stddev|257.3538420152301|0.48659245426485753|0.8360712409770491|                NULL|  NULL|14.526497332334035|1.1027434322934315| 0.8060572211299488|471609.26868834975|49.69342859718089| NULL|    NULL|
    |    min|                1|                  0|                 1|"Andersson, Mr. A...|female|              0.42|                 0|                  0|            110152|              0.0|  A10|       C|
    |    max|              891|                  1|                 3|van Melkebeke, Mr...|  male|              80.0|                 8|                  6|         WE/P 5735|         512.3292|    T|       S|
    +-------+-----------------+-------------------+------------------+--------------------+------+------------------+------------------+-------------------+------------------+-----------------+-----+--------+

```python
# pandas version
result
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>714</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>446.0</td>
      <td>0.3838383838383838</td>
      <td>2.308641975308642</td>
      <td>None</td>
      <td>None</td>
      <td>29.69911764705882</td>
      <td>0.5230078563411896</td>
      <td>0.38159371492704824</td>
      <td>260318.54916792738</td>
      <td>32.2042079685746</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>257.3538420152301</td>
      <td>0.48659245426485753</td>
      <td>0.8360712409770491</td>
      <td>None</td>
      <td>None</td>
      <td>14.526497332334035</td>
      <td>1.1027434322934315</td>
      <td>0.8060572211299488</td>
      <td>471609.26868834975</td>
      <td>49.69342859718089</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>"Andersson, Mr. August Edvard (""Wennerstrom"")"</td>
      <td>female</td>
      <td>0.42</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>0.0</td>
      <td>A10</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>891</td>
      <td>1</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>80.0</td>
      <td>8</td>
      <td>6</td>
      <td>WE/P 5735</td>
      <td>512.3292</td>
      <td>T</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Cool, Let's print the schema of the df using .printSchema()
df_train.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)

```python
# similar approach
df_train.dtypes
```

    [('PassengerId', 'int'),
     ('Survived', 'int'),
     ('Pclass', 'int'),
     ('Name', 'string'),
     ('Sex', 'string'),
     ('Age', 'double'),
     ('SibSp', 'int'),
     ('Parch', 'int'),
     ('Ticket', 'string'),
     ('Fare', 'double'),
     ('Cabin', 'string'),
     ('Embarked', 'string')]

The data in the real world is not this clean. We often have to create our own schema and implement it. We will describe more about it in the future. Since we are talking about schema, are you wondering if you would be able to implement sql with Spark?. Yes, you can.

One of the best advantage of Spark is that you can run sql commands to do analysis. If you are like that nifty co-worker of mine, you would probably want to use sql with spark. Let's do an example.

```python
## First, we need to register a sql temporary view.
df_train.createOrReplaceTempView("mytable");

## Then, we use spark.sql and write sql inside it, which returns a spark Dataframe.
result = spark.sql("SELECT * FROM mytable ORDER BY Fare DESC LIMIT 10")
result.show(truncate=False)
```

    +-----------+--------+------+-----------------------------------------+------+----+-----+-----+--------+--------+---------------+--------+
    |PassengerId|Survived|Pclass|Name                                     |Sex   |Age |SibSp|Parch|Ticket  |Fare    |Cabin          |Embarked|
    +-----------+--------+------+-----------------------------------------+------+----+-----+-----+--------+--------+---------------+--------+
    |680        |1       |1     |Cardeza, Mr. Thomas Drake Martinez       |male  |36.0|0    |1    |PC 17755|512.3292|B51 B53 B55    |C       |
    |259        |1       |1     |Ward, Miss. Anna                         |female|35.0|0    |0    |PC 17755|512.3292|NULL           |C       |
    |738        |1       |1     |Lesurer, Mr. Gustave J                   |male  |35.0|0    |0    |PC 17755|512.3292|B101           |C       |
    |89         |1       |1     |Fortune, Miss. Mabel Helen               |female|23.0|3    |2    |19950   |263.0   |C23 C25 C27    |S       |
    |28         |0       |1     |Fortune, Mr. Charles Alexander           |male  |19.0|3    |2    |19950   |263.0   |C23 C25 C27    |S       |
    |342        |1       |1     |Fortune, Miss. Alice Elizabeth           |female|24.0|3    |2    |19950   |263.0   |C23 C25 C27    |S       |
    |439        |0       |1     |Fortune, Mr. Mark                        |male  |64.0|1    |4    |19950   |263.0   |C23 C25 C27    |S       |
    |312        |1       |1     |Ryerson, Miss. Emily Borie               |female|18.0|2    |2    |PC 17608|262.375 |B57 B59 B63 B66|C       |
    |743        |1       |1     |"Ryerson, Miss. Susan Parker ""Suzette"""|female|21.0|2    |2    |PC 17608|262.375 |B57 B59 B63 B66|C       |
    |119        |0       |1     |Baxter, Mr. Quigg Edmond                 |male  |24.0|0    |1    |PC 17558|247.5208|B58 B60        |C       |
    +-----------+--------+------+-----------------------------------------+------+----+-----+-----+--------+--------+---------------+--------+

Similarly we can also register another sql temp view.

```python
df_test.createOrReplaceTempView("df_test")
```

Now that we have registered two tables within this spark session, wondering how we can see which once are registered?

```python
spark.catalog.listTables()
```

    [Table(name='df_test', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True),
     Table(name='mytable', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True)]

```python
# similarly
spark.sql("SHOW views").show()
```

    +---------+--------+-----------+
    |namespace|viewName|isTemporary|
    +---------+--------+-----------+
    |         | df_test|       true|
    |         | mytable|       true|
    +---------+--------+-----------+

```python
# or
spark.sql("SHOW tables").show()
```

    +---------+---------+-----------+
    |namespace|tableName|isTemporary|
    +---------+---------+-----------+
    |         |  df_test|       true|
    |         |  mytable|       true|
    +---------+---------+-----------+

```python
# We can also create spark dataframe out of these tables using spark.table
temp_table = spark.table("df_test")
print(type(temp_table))
temp_table.show(5)
```

    <class 'pyspark.sql.dataframe.DataFrame'>
    +-----------+------+--------------------+------+----+-----+-----+-------+-------+-----+--------+
    |PassengerId|Pclass|                Name|   Sex| Age|SibSp|Parch| Ticket|   Fare|Cabin|Embarked|
    +-----------+------+--------------------+------+----+-----+-----+-------+-------+-----+--------+
    |        892|     3|    Kelly, Mr. James|  male|34.5|    0|    0| 330911| 7.8292| NULL|       Q|
    |        893|     3|Wilkes, Mrs. Jame...|female|47.0|    1|    0| 363272|    7.0| NULL|       S|
    |        894|     2|Myles, Mr. Thomas...|  male|62.0|    0|    0| 240276| 9.6875| NULL|       Q|
    |        895|     3|    Wirz, Mr. Albert|  male|27.0|    0|    0| 315154| 8.6625| NULL|       S|
    |        896|     3|Hirvonen, Mrs. Al...|female|22.0|    1|    1|3101298|12.2875| NULL|       S|
    +-----------+------+--------------------+------+----+-----+-----+-------+-------+-----+--------+
    only showing top 5 rows

```python
# pretty cool, We will dive deep in sql later.
# Let's go back to dataFrame and do some nitty-gritty stuff.
# What if want the column names only.
df_train.columns
```

    ['PassengerId',
     'Survived',
     'Pclass',
     'Name',
     'Sex',
     'Age',
     'SibSp',
     'Parch',
     'Ticket',
     'Fare',
     'Cabin',
     'Embarked']

```python
# What about just a column?
df_train['Age']
```

    Column<'Age'>

```python
# similarly
df_train.Age
```

    Column<'Age'>

```python
type(df_train['Age'])
```

    pyspark.sql.column.Column

```python
# Well, that's not what we pandas users have expected.
# Yes, in order to get a column we need to use select().
# df.select(df['Age']).show()
df_train.select('Age').show()
```

    +----+
    | Age|
    +----+
    |22.0|
    |38.0|
    |26.0|
    |35.0|
    |35.0|
    |NULL|
    |54.0|
    | 2.0|
    |27.0|
    |14.0|
    | 4.0|
    |58.0|
    |20.0|
    |39.0|
    |14.0|
    |55.0|
    | 2.0|
    |NULL|
    |31.0|
    |NULL|
    +----+
    only showing top 20 rows

```python
# similarly...
df_train[['Age']].show()
```

    +----+
    | Age|
    +----+
    |22.0|
    |38.0|
    |26.0|
    |35.0|
    |35.0|
    |NULL|
    |54.0|
    | 2.0|
    |27.0|
    |14.0|
    | 4.0|
    |58.0|
    |20.0|
    |39.0|
    |14.0|
    |55.0|
    | 2.0|
    |NULL|
    |31.0|
    |NULL|
    +----+
    only showing top 20 rows

```python
## What if we want multiple columns?
df_train.select(['Age', 'Fare']).show()
```

    +----+-------+
    | Age|   Fare|
    +----+-------+
    |22.0|   7.25|
    |38.0|71.2833|
    |26.0|  7.925|
    |35.0|   53.1|
    |35.0|   8.05|
    |NULL| 8.4583|
    |54.0|51.8625|
    | 2.0| 21.075|
    |27.0|11.1333|
    |14.0|30.0708|
    | 4.0|   16.7|
    |58.0|  26.55|
    |20.0|   8.05|
    |39.0| 31.275|
    |14.0| 7.8542|
    |55.0|   16.0|
    | 2.0| 29.125|
    |NULL|   13.0|
    |31.0|   18.0|
    |NULL|  7.225|
    +----+-------+
    only showing top 20 rows

```python
# similarly
df_train[['Age', 'Fare']].show()
```

    +----+-------+
    | Age|   Fare|
    +----+-------+
    |22.0|   7.25|
    |38.0|71.2833|
    |26.0|  7.925|
    |35.0|   53.1|
    |35.0|   8.05|
    |NULL| 8.4583|
    |54.0|51.8625|
    | 2.0| 21.075|
    |27.0|11.1333|
    |14.0|30.0708|
    | 4.0|   16.7|
    |58.0|  26.55|
    |20.0|   8.05|
    |39.0| 31.275|
    |14.0| 7.8542|
    |55.0|   16.0|
    | 2.0| 29.125|
    |NULL|   13.0|
    |31.0|   18.0|
    |NULL|  7.225|
    +----+-------+
    only showing top 20 rows

```python
# or
df_train[df_train.Age,
         df_train.Fare].show()
```

    +----+-------+
    | Age|   Fare|
    +----+-------+
    |22.0|   7.25|
    |38.0|71.2833|
    |26.0|  7.925|
    |35.0|   53.1|
    |35.0|   8.05|
    |NULL| 8.4583|
    |54.0|51.8625|
    | 2.0| 21.075|
    |27.0|11.1333|
    |14.0|30.0708|
    | 4.0|   16.7|
    |58.0|  26.55|
    |20.0|   8.05|
    |39.0| 31.275|
    |14.0| 7.8542|
    |55.0|   16.0|
    | 2.0| 29.125|
    |NULL|   13.0|
    |31.0|   18.0|
    |NULL|  7.225|
    +----+-------+
    only showing top 20 rows

As you can see pyspark dataframe syntax is pretty simple with a lot of ways to implement. Which syntex is best implemented depends on what we are trying to accomplish. I will discuss more on this as we go on. Now let's see how we can access a row.

```python
df_train.head(1)
```

    [Row(PassengerId=1, Survived=0, Pclass=3, Name='Braund, Mr. Owen Harris', Sex='male', Age=22.0, SibSp=1, Parch=0, Ticket='A/5 21171', Fare=7.25, Cabin=None, Embarked='S')]

```python
type(df_train.head(1))
```

    list

```python
## returns a list. let's get the item in the list
row = df_train.head(1)[0]
row
```

    Row(PassengerId=1, Survived=0, Pclass=3, Name='Braund, Mr. Owen Harris', Sex='male', Age=22.0, SibSp=1, Parch=0, Ticket='A/5 21171', Fare=7.25, Cabin=None, Embarked='S')

```python
type(row)
```

    pyspark.sql.types.Row

```python
## row can be converted into dict using .asDict()
row.asDict()
```

    {'PassengerId': 1,
     'Survived': 0,
     'Pclass': 3,
     'Name': 'Braund, Mr. Owen Harris',
     'Sex': 'male',
     'Age': 22.0,
     'SibSp': 1,
     'Parch': 0,
     'Ticket': 'A/5 21171',
     'Fare': 7.25,
     'Cabin': None,
     'Embarked': 'S'}

```python
## Then the value can be accessed from the row dictionaly.
row.asDict()['PassengerId']
```

    1

```python
## similarly
row.asDict()['Name']
```

    'Braund, Mr. Owen Harris'

```python
## let's say we want to change the name of a column. we can use withColumnRenamed
# df.withColumnRenamed('exsisting name', 'anticipated name');
df_train.withColumnRenamed("Age", "newA").limit(5).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex|newA|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| NULL|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| NULL|       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| NULL|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+

```python
# Let's say we want to modify a column, for example, add in this case; adding $20 with every fare.
## df.withColumn('existing column', 'calculation with the column(we have to put df not just column)')
## so not df.withColumn('Fare', 'Fare' +20).show()
df_train.withColumn('Fare', df_train['Fare']+20).limit(5).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|  27.25| NULL|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|91.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282| 27.925| NULL|       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   73.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|  28.05| NULL|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+

Now this change isn't permanent since we are not assigning it to any variables.

```python
## let's say we want to get the average fare.
# we will use the "mean" function from pyspark.sql.functions(this is where all the functions are stored) and
# collect the data using ".collect()" instead of using .show()
# collect returns a list so we need to get the value from the list using index
```

```python
from pyspark.sql.functions import mean
fare_mean = df_train.select(mean("Fare")).collect()
fare_mean[0][0]
```

    32.2042079685746

```python
fare_mean = fare_mean[0][0]
fare_mean
```

    32.2042079685746

#### Filter

```python
# What if we want to filter data and see all fare above average.
# there are two approaches of this, we can use sql syntex/passing a string
# or just dataframe approach.
df_train.filter("Fare > 32.20" ).limit(3).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|  Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|PC 17599|71.2833|  C85|       C|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|  113803|   53.1| C123|       S|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|   17463|51.8625|  E46|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+

```python
# similarly
df_train[df_train.Fare > 32.20].limit(3).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|  Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|PC 17599|71.2833|  C85|       C|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|  113803|   53.1| C123|       S|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|   17463|51.8625|  E46|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+

```python
# or we can use the dataframe approach
df_train.filter(df_train['Fare'] > fare_mean).limit(3).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|  Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|PC 17599|71.2833|  C85|       C|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|  113803|   53.1| C123|       S|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|   17463|51.8625|  E46|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+

```python
## What if we want to filter by multiple columns.
# passenger with below average fare with a sex equals male
temp_df = df_train.filter((df_train['Fare'] < fare_mean) &
          (df_train['Sex'] ==  'male')
         )
temp_df.show(5)
```

    +-----------+--------+------+--------------------+----+----+-----+-----+---------+------+-----+--------+
    |PassengerId|Survived|Pclass|                Name| Sex| Age|SibSp|Parch|   Ticket|  Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+----+----+-----+-----+---------+------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|male|22.0|    1|    0|A/5 21171|  7.25| NULL|       S|
    |          5|       0|     3|Allen, Mr. Willia...|male|35.0|    0|    0|   373450|  8.05| NULL|       S|
    |          6|       0|     3|    Moran, Mr. James|male|NULL|    0|    0|   330877|8.4583| NULL|       Q|
    |          8|       0|     3|Palsson, Master. ...|male| 2.0|    3|    1|   349909|21.075| NULL|       S|
    |         13|       0|     3|Saundercock, Mr. ...|male|20.0|    0|    0|A/5. 2151|  8.05| NULL|       S|
    +-----------+--------+------+--------------------+----+----+-----+-----+---------+------+-----+--------+
    only showing top 5 rows

```python
# similarly
df_train[(df_train.Fare < fare_mean) &
         (df_train.Sex == "male")].show(5)
```

    +-----------+--------+------+--------------------+----+----+-----+-----+---------+------+-----+--------+
    |PassengerId|Survived|Pclass|                Name| Sex| Age|SibSp|Parch|   Ticket|  Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+----+----+-----+-----+---------+------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|male|22.0|    1|    0|A/5 21171|  7.25| NULL|       S|
    |          5|       0|     3|Allen, Mr. Willia...|male|35.0|    0|    0|   373450|  8.05| NULL|       S|
    |          6|       0|     3|    Moran, Mr. James|male|NULL|    0|    0|   330877|8.4583| NULL|       Q|
    |          8|       0|     3|Palsson, Master. ...|male| 2.0|    3|    1|   349909|21.075| NULL|       S|
    |         13|       0|     3|Saundercock, Mr. ...|male|20.0|    0|    0|A/5. 2151|  8.05| NULL|       S|
    +-----------+--------+------+--------------------+----+----+-----+-----+---------+------+-----+--------+
    only showing top 5 rows

```python
# passenger with below average fare and are not male
filter1_less_than_mean_fare = df_train['Fare'] < fare_mean
filter2_sex_not_male = df_train['Sex'] != "male"
df_train.filter((filter1_less_than_mean_fare) &
                (filter2_sex_not_male)).show(10)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| NULL|       S|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333| NULL|       S|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708| NULL|       C|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|         PP 9549|   16.7|   G6|       S|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|          113783|  26.55| C103|       S|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0|          350406| 7.8542| NULL|       S|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|          248706|   16.0| NULL|       S|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|          345763|   18.0| NULL|       S|
    |         20|       1|     3|Masselmani, Mrs. ...|female|NULL|    0|    0|            2649|  7.225| NULL|       C|
    |         23|       1|     3|"McGowan, Miss. A...|female|15.0|    0|    0|          330923| 8.0292| NULL|       Q|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 10 rows

```python
# We can also apply it this way
# passenger with below fare and are not male
# creating filters
filter1_less_than_mean_fare = df_train['Fare'] < fare_mean
filter2_sex_not_male = df_train['Sex'] != "male"
# applying filters
df_train.filter(filter1_less_than_mean_fare).filter(filter2_sex_not_male).show(10)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| NULL|       S|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333| NULL|       S|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708| NULL|       C|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|         PP 9549|   16.7|   G6|       S|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|          113783|  26.55| C103|       S|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0|          350406| 7.8542| NULL|       S|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|          248706|   16.0| NULL|       S|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|          345763|   18.0| NULL|       S|
    |         20|       1|     3|Masselmani, Mrs. ...|female|NULL|    0|    0|            2649|  7.225| NULL|       C|
    |         23|       1|     3|"McGowan, Miss. A...|female|15.0|    0|    0|          330923| 8.0292| NULL|       Q|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 10 rows

```python
# we can also filter by using builtin functions.
# between
df_train.select("PassengerId", "Fare").filter(df_train.Fare.between(10,40)).show()
```

    +-----------+-------+
    |PassengerId|   Fare|
    +-----------+-------+
    |          8| 21.075|
    |          9|11.1333|
    |         10|30.0708|
    |         11|   16.7|
    |         12|  26.55|
    |         14| 31.275|
    |         16|   16.0|
    |         17| 29.125|
    |         18|   13.0|
    |         19|   18.0|
    |         21|   26.0|
    |         22|   13.0|
    |         24|   35.5|
    |         25| 21.075|
    |         26|31.3875|
    |         31|27.7208|
    |         34|   10.5|
    |         39|   18.0|
    |         40|11.2417|
    |         42|   21.0|
    +-----------+-------+
    only showing top 20 rows

```python
df_train.select("PassengerID", df_train.Fare.between(10,40)).show()
```

    +-----------+-------------------------------+
    |PassengerID|((Fare >= 10) AND (Fare <= 40))|
    +-----------+-------------------------------+
    |          1|                          false|
    |          2|                          false|
    |          3|                          false|
    |          4|                          false|
    |          5|                          false|
    |          6|                          false|
    |          7|                          false|
    |          8|                           true|
    |          9|                           true|
    |         10|                           true|
    |         11|                           true|
    |         12|                           true|
    |         13|                          false|
    |         14|                           true|
    |         15|                          false|
    |         16|                           true|
    |         17|                           true|
    |         18|                           true|
    |         19|                           true|
    |         20|                          false|
    +-----------+-------------------------------+
    only showing top 20 rows

```python
# contains
df_train.select("PassengerId", "Name").filter(df_train.Name.contains("Mr")).show()
```

    +-----------+--------------------+
    |PassengerId|                Name|
    +-----------+--------------------+
    |          1|Braund, Mr. Owen ...|
    |          2|Cumings, Mrs. Joh...|
    |          4|Futrelle, Mrs. Ja...|
    |          5|Allen, Mr. Willia...|
    |          6|    Moran, Mr. James|
    |          7|McCarthy, Mr. Tim...|
    |          9|Johnson, Mrs. Osc...|
    |         10|Nasser, Mrs. Nich...|
    |         13|Saundercock, Mr. ...|
    |         14|Andersson, Mr. An...|
    |         16|Hewlett, Mrs. (Ma...|
    |         18|Williams, Mr. Cha...|
    |         19|Vander Planke, Mr...|
    |         20|Masselmani, Mrs. ...|
    |         21|Fynney, Mr. Joseph J|
    |         22|Beesley, Mr. Lawr...|
    |         24|Sloper, Mr. Willi...|
    |         26|Asplund, Mrs. Car...|
    |         27|Emir, Mr. Farred ...|
    |         28|Fortune, Mr. Char...|
    +-----------+--------------------+
    only showing top 20 rows

```python
# startswith
df_train.select("PassengerID", 'Sex').filter(df_train.Sex.startswith("fe")).show()
```

    +-----------+------+
    |PassengerID|   Sex|
    +-----------+------+
    |          2|female|
    |          3|female|
    |          4|female|
    |          9|female|
    |         10|female|
    |         11|female|
    |         12|female|
    |         15|female|
    |         16|female|
    |         19|female|
    |         20|female|
    |         23|female|
    |         25|female|
    |         26|female|
    |         29|female|
    |         32|female|
    |         33|female|
    |         39|female|
    |         40|female|
    |         41|female|
    +-----------+------+
    only showing top 20 rows

```python
# endswith
df_train.select("PassengerID", 'Ticket').filter(df_train.Ticket.endswith("50")).show()
```

    +-----------+----------+
    |PassengerID|    Ticket|
    +-----------+----------+
    |          5|    373450|
    |         28|     19950|
    |         89|     19950|
    |        256|      2650|
    |        342|     19950|
    |        439|     19950|
    |        537|    113050|
    |        641|    350050|
    |        671|     29750|
    |        672|F.C. 12750|
    |        685|     29750|
    |        768|    364850|
    |        807|    112050|
    +-----------+----------+

```python
# isin
df_train[df_train.PassengerId.isin([1,2,3])].show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| NULL|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| NULL|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+

```python
# like
df_train[df_train.Name.like("Br%")].show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+---------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+---------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|A/5 21171|   7.25| NULL|       S|
    |        195|       1|     1|Brown, Mrs. James...|female|44.0|    0|    0| PC 17610|27.7208|   B4|       C|
    |        222|       0|     2|Bracken, Mr. James H|  male|27.0|    0|    0|   220367|   13.0| NULL|       S|
    |        478|       0|     3|Braund, Mr. Lewis...|  male|29.0|    1|    0|     3460| 7.0458| NULL|       S|
    |        615|       0|     3|Brocklebank, Mr. ...|  male|35.0|    0|    0|   364512|   8.05| NULL|       S|
    |        671|       1|     2|Brown, Mrs. Thoma...|female|40.0|    1|    1|    29750|   39.0| NULL|       S|
    |        685|       0|     2|Brown, Mr. Thomas...|  male|60.0|    1|    1|    29750|   39.0| NULL|       S|
    |        729|       0|     2|Bryhl, Mr. Kurt A...|  male|25.0|    1|    0|   236853|   26.0| NULL|       S|
    |        767|       0|     1|Brewe, Dr. Arthur...|  male|NULL|    0|    0|   112379|   39.6| NULL|       C|
    +-----------+--------+------+--------------------+------+----+-----+-----+---------+-------+-----+--------+

```python
# substr
df_train.select(df_train.Name.substr(1,5)).show()
```

    +---------------------+
    |substring(Name, 1, 5)|
    +---------------------+
    |                Braun|
    |                Cumin|
    |                Heikk|
    |                Futre|
    |                Allen|
    |                Moran|
    |                McCar|
    |                Palss|
    |                Johns|
    |                Nasse|
    |                Sands|
    |                Bonne|
    |                Saund|
    |                Ander|
    |                Vestr|
    |                Hewle|
    |                Rice,|
    |                Willi|
    |                Vande|
    |                Masse|
    +---------------------+
    only showing top 20 rows

```python
# similarly
df_train[[df_train.Name.substr(1,5)]].show()
```

    +---------------------+
    |substring(Name, 1, 5)|
    +---------------------+
    |                Braun|
    |                Cumin|
    |                Heikk|
    |                Futre|
    |                Allen|
    |                Moran|
    |                McCar|
    |                Palss|
    |                Johns|
    |                Nasse|
    |                Sands|
    |                Bonne|
    |                Saund|
    |                Ander|
    |                Vestr|
    |                Hewle|
    |                Rice,|
    |                Willi|
    |                Vande|
    |                Masse|
    +---------------------+
    only showing top 20 rows

One interesting thing about substr method is that we can't implement the following syntax while working with substr. This syntax is best implemented in a filter when the return values are boolean not a column.

```python
# df_train[df_train.Name.substr(1,5)].show()
```

#### GroupBy

```python
## Let's group by Pclass and get the average fare price per Pclass.
df_train.groupBy("Pclass").mean().toPandas()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>avg(PassengerId)</th>
      <th>avg(Survived)</th>
      <th>avg(Pclass)</th>
      <th>avg(Age)</th>
      <th>avg(SibSp)</th>
      <th>avg(Parch)</th>
      <th>avg(Fare)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>461.597222</td>
      <td>0.629630</td>
      <td>1.0</td>
      <td>38.233441</td>
      <td>0.416667</td>
      <td>0.356481</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>439.154786</td>
      <td>0.242363</td>
      <td>3.0</td>
      <td>25.140620</td>
      <td>0.615071</td>
      <td>0.393075</td>
      <td>13.675550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>445.956522</td>
      <td>0.472826</td>
      <td>2.0</td>
      <td>29.877630</td>
      <td>0.402174</td>
      <td>0.380435</td>
      <td>20.662183</td>
    </tr>
  </tbody>
</table>
</div>

```python
## let's just look at the Pclass and avg(Fare)
df_train.groupBy("Pclass").mean().select('Pclass', 'avg(Fare)').show()
```

    +------+------------------+
    |Pclass|         avg(Fare)|
    +------+------------------+
    |     1| 84.15468749999992|
    |     3|13.675550101832997|
    |     2| 20.66218315217391|
    +------+------------------+

```python
# Alternative way
df_train.groupBy("Pclass").mean("Fare").show()
```

    +------+------------------+
    |Pclass|         avg(Fare)|
    +------+------------------+
    |     1| 84.15468749999992|
    |     3|13.675550101832997|
    |     2| 20.66218315217391|
    +------+------------------+

```python
## What if we want just the average of all fare, we can use .agg with the dataframe.
df_train.agg({'Fare':'mean'}).show()
```

    +----------------+
    |       avg(Fare)|
    +----------------+
    |32.2042079685746|
    +----------------+

```python
## another way this can be done is by importing "mean" funciton from pyspark.sql.functions
from pyspark.sql.functions import mean
df_train.select(mean("Fare")).show()
```

    +----------------+
    |       avg(Fare)|
    +----------------+
    |32.2042079685746|
    +----------------+

```python
## we can also combine the few previous approaches to get similar results.
temp = df_train.groupBy("Pclass")
temp.agg({"Fare": 'mean'}).show()
```

    +------+------------------+
    |Pclass|         avg(Fare)|
    +------+------------------+
    |     1| 84.15468749999992|
    |     3|13.675550101832997|
    |     2| 20.66218315217391|
    +------+------------------+

```python
# What if we want to format the results.
# for example,
# I want to rename the column. this will be accomplished using .alias() method.
# I want to format the number with only two decimals. this can be done using "format_number"
from pyspark.sql.functions import format_number
temp = df_train.groupBy("Pclass")
temp = temp.agg({"Fare": 'mean'})
temp.select('Pclass', format_number("avg(Fare)", 2).alias("average fare")).show()
```

    +------+------------+
    |Pclass|average fare|
    +------+------------+
    |     1|       84.15|
    |     3|       13.68|
    |     2|       20.66|
    +------+------------+

#### OrderBy

There are many built in functions that we can use to do orderby in spark. Let's look at some of those.

```python
## What if I want to order by Fare in ascending order.
df_train.orderBy("Fare").limit(20).toPandas()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>"Parkes, Mr. Francis ""Frank"""</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0000</td>
      <td>B94</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>"Frost, Mr. Anthony Wood ""Archie"""</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0000</td>
      <td>A36</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0000</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>379</td>
      <td>0</td>
      <td>3</td>
      <td>Betros, Mr. Tannous</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>2648</td>
      <td>4.0125</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>16</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>17</th>
      <td>327</td>
      <td>0</td>
      <td>3</td>
      <td>Nysveen, Mr. Johan Hansen</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>345364</td>
      <td>6.2375</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>844</td>
      <td>0</td>
      <td>3</td>
      <td>Lemberopolous, Mr. Peter L</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>2683</td>
      <td>6.4375</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>19</th>
      <td>819</td>
      <td>0</td>
      <td>3</td>
      <td>Holm, Mr. John Fredrik Alexander</td>
      <td>male</td>
      <td>43.0</td>
      <td>0</td>
      <td>0</td>
      <td>C 7075</td>
      <td>6.4500</td>
      <td>None</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

```python
# similarly
df_train.orderBy(df_train.Fare.asc()).show()
```

    +-----------+--------+------+--------------------+----+----+-----+-----+------+------+-----------+--------+
    |PassengerId|Survived|Pclass|                Name| Sex| Age|SibSp|Parch|Ticket|  Fare|      Cabin|Embarked|
    +-----------+--------+------+--------------------+----+----+-----+-----+------+------+-----------+--------+
    |        303|       0|     3|Johnson, Mr. Will...|male|19.0|    0|    0|  LINE|   0.0|       NULL|       S|
    |        278|       0|     2|"Parkes, Mr. Fran...|male|NULL|    0|    0|239853|   0.0|       NULL|       S|
    |        272|       1|     3|Tornquist, Mr. Wi...|male|25.0|    0|    0|  LINE|   0.0|       NULL|       S|
    |        264|       0|     1|Harrison, Mr. Wil...|male|40.0|    0|    0|112059|   0.0|        B94|       S|
    |        482|       0|     2|"Frost, Mr. Antho...|male|NULL|    0|    0|239854|   0.0|       NULL|       S|
    |        180|       0|     3| Leonard, Mr. Lionel|male|36.0|    0|    0|  LINE|   0.0|       NULL|       S|
    |        414|       0|     2|Cunningham, Mr. A...|male|NULL|    0|    0|239853|   0.0|       NULL|       S|
    |        467|       0|     2|Campbell, Mr. Wil...|male|NULL|    0|    0|239853|   0.0|       NULL|       S|
    |        598|       0|     3| Johnson, Mr. Alfred|male|49.0|    0|    0|  LINE|   0.0|       NULL|       S|
    |        634|       0|     1|Parr, Mr. William...|male|NULL|    0|    0|112052|   0.0|       NULL|       S|
    |        675|       0|     2|Watson, Mr. Ennis...|male|NULL|    0|    0|239856|   0.0|       NULL|       S|
    |        733|       0|     2|Knight, Mr. Robert J|male|NULL|    0|    0|239855|   0.0|       NULL|       S|
    |        807|       0|     1|Andrews, Mr. Thom...|male|39.0|    0|    0|112050|   0.0|        A36|       S|
    |        816|       0|     1|    Fry, Mr. Richard|male|NULL|    0|    0|112058|   0.0|       B102|       S|
    |        823|       0|     1|Reuchlin, Jonkhee...|male|38.0|    0|    0| 19972|   0.0|       NULL|       S|
    |        379|       0|     3| Betros, Mr. Tannous|male|20.0|    0|    0|  2648|4.0125|       NULL|       C|
    |        873|       0|     1|Carlsson, Mr. Fra...|male|33.0|    0|    0|   695|   5.0|B51 B53 B55|       S|
    |        327|       0|     3|Nysveen, Mr. Joha...|male|61.0|    0|    0|345364|6.2375|       NULL|       S|
    |        844|       0|     3|Lemberopolous, Mr...|male|34.5|    0|    0|  2683|6.4375|       NULL|       C|
    |        819|       0|     3|Holm, Mr. John Fr...|male|43.0|    0|    0|C 7075|  6.45|       NULL|       S|
    +-----------+--------+------+--------------------+----+----+-----+-----+------+------+-----------+--------+
    only showing top 20 rows

```python
# What about descending order
# df.orderBy(df['Fare'].desc()).limit(5).show()
# dot notation
df_train.orderBy(df_train.Fare.desc()).limit(5).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+--------+--------+-----------+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|  Ticket|    Fare|      Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+--------+-----------+--------+
    |        738|       1|     1|Lesurer, Mr. Gust...|  male|35.0|    0|    0|PC 17755|512.3292|       B101|       C|
    |        680|       1|     1|Cardeza, Mr. Thom...|  male|36.0|    0|    1|PC 17755|512.3292|B51 B53 B55|       C|
    |        259|       1|     1|    Ward, Miss. Anna|female|35.0|    0|    0|PC 17755|512.3292|       NULL|       C|
    |        439|       0|     1|   Fortune, Mr. Mark|  male|64.0|    1|    4|   19950|   263.0|C23 C25 C27|       S|
    |         89|       1|     1|Fortune, Miss. Ma...|female|23.0|    3|    2|   19950|   263.0|C23 C25 C27|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+--------+-----------+--------+

```python
df_train.filter(df_train.Embarked.isNull()).count()
```

    2

```python
df_train.select('PassengerID','Embarked').orderBy(df_train.Embarked.asc_nulls_first()).show()
```

    +-----------+--------+
    |PassengerID|Embarked|
    +-----------+--------+
    |         62|    NULL|
    |        830|    NULL|
    |        204|       C|
    |         74|       C|
    |         66|       C|
    |         97|       C|
    |         65|       C|
    |         98|       C|
    |         31|       C|
    |        112|       C|
    |         35|       C|
    |        115|       C|
    |         40|       C|
    |        119|       C|
    |         44|       C|
    |        123|       C|
    |         53|       C|
    |        126|       C|
    |         58|       C|
    |        129|       C|
    +-----------+--------+
    only showing top 20 rows

```python
df_train.select('PassengerID','Embarked').orderBy(df_train.Embarked.asc_nulls_last()).tail(5)
```

    [Row(PassengerID=887, Embarked='S'),
     Row(PassengerID=888, Embarked='S'),
     Row(PassengerID=889, Embarked='S'),
     Row(PassengerID=62, Embarked=None),
     Row(PassengerID=830, Embarked=None)]

```python
## How do we deal with missing values.
# df.na.drop(how=("any"/"all"), thresh=(1,2,3,4,5...))
df_train.na.drop(how="any").limit(5).toPandas()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

# Advanced Tutorial

### Spark Catalog

```python
# If you have used Spark for a while now, this is a good time to learn about spark Catalog.
# you can also totally skip this section since it is totally independed of what follows.
```

```python
# get all the databases in the database.
spark.catalog.listDatabases()
```

    [Database(name='default', catalog='spark_catalog', description='default database', locationUri='file:/Users/masumrumi/Projects/data_science/pyspark_titanic/working/spark-warehouse')]

```python
# get the name of the current database
spark.catalog.currentDatabase()
```

    'default'

```python
## lists tables
spark.catalog.listTables()
```

    [Table(name='df_test', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True),
     Table(name='mytable', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True)]

```python
# add a table to the catalog
df_train.createOrReplaceTempView("df_train")
```

```python
# list tables
spark.catalog.listTables()
```

    [Table(name='df_test', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True),
     Table(name='df_train', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True),
     Table(name='mytable', catalog=None, namespace=[], description=None, tableType='TEMPORARY', isTemporary=True)]

```python
# Caching
# cached table "df_train"
spark.catalog.cacheTable("df_train")
```

```python
# checks if the table is cached
spark.catalog.isCached("df_train")
```

    True

```python
spark.catalog.isCached("df_test")
```

    False

```python
# lets cahche df_test as well
spark.catalog.cacheTable("df_test")
```

```python
spark.catalog.isCached("df_test")
```

    True

```python
# let's uncache df_train
spark.catalog.uncacheTable("df_train")
```

```python
spark.catalog.isCached("df_train")
```

    False

```python
spark.catalog.isCached("df_test")
```

    True

```python
# How about clearing all cached tables at once.
spark.catalog.clearCache()
```

```python
spark.catalog.isCached("df_train")
```

    False

```python

```

```python
# creating a global temp view
df_train.createGlobalTempView("df_train")
```

```python
# listing all views in global_temp
spark.sql("SHOW VIEWS IN global_temp;").show()
```

    +-----------+--------+-----------+
    |  namespace|viewName|isTemporary|
    +-----------+--------+-----------+
    |global_temp|df_train|       true|
    |           | df_test|       true|
    |           |df_train|       true|
    |           | mytable|       true|
    +-----------+--------+-----------+

```python
# dropping a table.
spark.catalog.dropGlobalTempView("df_train")
```

    True

```python
# checking that global temp view is dropped.
spark.sql("SHOW VIEWS IN global_temp;").show()
```

    +---------+--------+-----------+
    |namespace|viewName|isTemporary|
    +---------+--------+-----------+
    |         | df_test|       true|
    |         |df_train|       true|
    |         | mytable|       true|
    +---------+--------+-----------+

```python
spark.catalog.dropTempView("df_train")
```

    True

```python
# checking that global temp view is dropped.
spark.sql("SHOW VIEWS IN global_temp;").show()
```

    +---------+--------+-----------+
    |namespace|viewName|isTemporary|
    +---------+--------+-----------+
    |         | df_test|       true|
    |         | mytable|       true|
    +---------+--------+-----------+

```python
spark.sql("SHOW VIEWS").show()
```

    +---------+--------+-----------+
    |namespace|viewName|isTemporary|
    +---------+--------+-----------+
    |         | df_test|       true|
    |         | mytable|       true|
    +---------+--------+-----------+

## Dealing with Missing Values

### Cabin

```python
# filling the null values in cabin with "N".
# df.fillna(value, subset=[]);
df_train = df_train.na.fill('N', subset=['Cabin'])
df_test = df_test.na.fill('N', subset=['Cabin'])
```

### Fare

```python
## how do we find out the rows with missing values?
# we can use .where(condition) with .isNull()
df_test.where(df_test['Fare'].isNull()).show()
```

    +-----------+------+------------------+----+----+-----+-----+------+----+-----+--------+
    |PassengerId|Pclass|              Name| Sex| Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+------+------------------+----+----+-----+-----+------+----+-----+--------+
    |       1044|     3|Storey, Mr. Thomas|male|60.5|    0|    0|  3701|NULL|    N|       S|
    +-----------+------+------------------+----+----+-----+-----+------+----+-----+--------+

Here, We can take the average of the **Fare** column to fill in the NaN value. However, for the sake of learning and practicing, we will try something else. We can take the average of the values where **Pclass** is **_3_**, **Sex** is **_male_** and **Embarked** is **_S_**

```python
missing_value = df_test.filter(
    (df_test['Pclass'] == 3) &
    (df_test.Embarked == 'S') &
    (df_test.Sex == "male")
)
## filling in the null value in the fare column using Fare mean.
df_test = df_test.na.fill(
    missing_value.select(mean('Fare')).collect()[0][0],
    subset=['Fare']
)
```

```python
# Checking
df_test.where(df_test['Fare'].isNull()).show()
```

    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    |PassengerId|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+

### Embarked

```python
df_train.where(df_train['Embarked'].isNull()).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+------+----+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+------+----+-----+--------+
    |         62|       1|     1| Icard, Miss. Amelie|female|38.0|    0|    0|113572|80.0|  B28|    NULL|
    |        830|       1|     1|Stone, Mrs. Georg...|female|62.0|    0|    0|113572|80.0|  B28|    NULL|
    +-----------+--------+------+--------------------+------+----+-----+-----+------+----+-----+--------+

```python
## Replacing the null values in the Embarked column with the mode.
df_train = df_train.na.fill('C', subset=['Embarked'])
```

```python
## checking
df_train.where(df_train['Embarked'].isNull()).show()
```

    +-----------+--------+------+----+---+---+-----+-----+------+----+-----+--------+
    |PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+--------+------+----+---+---+-----+-----+------+----+-----+--------+
    +-----------+--------+------+----+---+---+-----+-----+------+----+-----+--------+

```python
df_test.where(df_test.Embarked.isNull()).show()
```

    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    |PassengerId|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+

## Feature Engineering

### Cabin

```python
## this is a code to create a wrapper for function, that works for both python and Pyspark.
from typing import Callable
from pyspark.sql import Column
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType, ArrayType, DataType
class py_or_udf:
    def __init__(self, returnType : DataType=StringType()):
        self.spark_udf_type = returnType

    def __call__(self, func : Callable):
        def wrapped_func(*args, **kwargs):
            if any([isinstance(arg, Column) for arg in args]) or \
                any([isinstance(vv, Column) for vv in kwargs.values()]):
                return udf(func, self.spark_udf_type)(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapped_func


@py_or_udf(returnType=StringType())
def first_char(col):
    return col[0]

```

```python
df_train = df_train.withColumn('Cabin', first_char(df_train['Cabin']))
```

```python
df_test = df_test.withColumn('Cabin', first_char(df_test['Cabin']))
```

```python
df_train.limit(5).toPandas()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>N</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

We can use the average of the fare column We can use pyspark's **_groupby_** function to get the mean fare of each cabin letter.

```python
df_train.groupBy('Cabin').mean("Fare").show()
```

    +-----+------------------+
    |Cabin|         avg(Fare)|
    +-----+------------------+
    |    F| 18.69679230769231|
    |    E|46.026693749999986|
    |    T|              35.5|
    |    B|113.50576382978724|
    |    D| 57.24457575757576|
    |    C|100.15134067796612|
    |    A|39.623886666666664|
    |    N|  19.1573253275109|
    |    G|          13.58125|
    +-----+------------------+

Now, these mean can help us determine the unknown cabins, if we compare each unknown cabin rows with the given mean's above. Let's write a simple function so that we can give cabin names based on the means.

```python
@py_or_udf(returnType=StringType())
def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a
```

```python
## separating data where Cabin == 'N', remeber we used 'N' for Null.
df_withN = df_train.filter(df_train['Cabin'] == 'N')
df2 = df_train.filter(df_train['Cabin'] != 'N')

## replacing 'N' using cabin estimated function.
df_withN = df_withN.withColumn('Cabin', cabin_estimator(df_withN['Fare']))

# putting the dataframe back together.
df_train = df_withN.union(df2).orderBy('PassengerId')
```

```python
#let's do the same for test set
df_testN = df_test.filter(df_test['Cabin'] == 'N')
df_testNoN = df_test.filter(df_test['Cabin'] != 'N')
df_testN = df_testN.withColumn('Cabin', cabin_estimator(df_testN['Fare']))
df_test = df_testN.union(df_testNoN).orderBy('PassengerId')
```

### Name

```python
## creating UDF functions
@py_or_udf(returnType=IntegerType())
def name_length(name):
    return len(name)


@py_or_udf(returnType=StringType())
def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a
```

```python
## getting the name length from name.
df_train = df_train.withColumn("name_length", name_length(df_train['Name']))

## grouping based on name length.
df_train = df_train.withColumn("nLength_group", name_length_group(df_train['name_length']))
```

```python
## Let's do the same for test set.
df_test = df_test.withColumn("name_length", name_length(df_test['Name']))

df_test = df_test.withColumn("nLength_group", name_length_group(df_test['name_length']))
```

### Title

```python
## this function helps getting the title from the name.
@py_or_udf(returnType=StringType())
def get_title(name):
    return name.split('.')[0].split(',')[1].strip()

df_train = df_train.withColumn("title", get_title(df_train['Name']))
df_test = df_test.withColumn('title', get_title(df_test['Name']))
```

```python
## we are writing a function that can help us modify title column
@py_or_udf(returnType=StringType())
def fuse_title1(feature):
    """
    This function helps modifying the title column
    """
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        return 'rare'
    elif feature in ['Ms', 'Mlle']:
        return 'Miss'
    elif feature == 'Mme':
        return 'Mrs'
    else:
        return feature
```

```python
df_train = df_train.withColumn("title", fuse_title1(df_train["title"]))
```

```python
df_test = df_test.withColumn("title", fuse_title1(df_test['title']))
```

```python
print(df_train.toPandas()['title'].unique())
print(df_test.toPandas()['title'].unique())
```

    ['Mr' 'Mrs' 'Miss' 'Master' 'rare']
    ['Mr' 'Mrs' 'Miss' 'Master' 'rare']

### family_size

```python
df_train = df_train.withColumn("family_size", df_train['SibSp']+df_train['Parch'])
df_test = df_test.withColumn("family_size", df_test['SibSp']+df_test['Parch'])
```

```python
## bin the family size.
@py_or_udf(returnType=StringType())
def family_group(size):
    """
    This funciton groups(loner, small, large) family based on family size
    """

    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
```

```python
df_train = df_train.withColumn("family_group", family_group(df_train['family_size']))
df_test = df_test.withColumn("family_group", family_group(df_test['family_size']))

```

### is_alone

```python
@py_or_udf(returnType=IntegerType())
def is_alone(num):
    if num<2:
        return 1
    else:
        return 0
```

```python
df_train = df_train.withColumn("is_alone", is_alone(df_train['family_size']))
df_test = df_test.withColumn("is_alone", is_alone(df_test["family_size"]))
```

### ticket

```python
## dropping ticket column
df_train = df_train.drop('ticket')
df_test = df_test.drop("ticket")
```

### calculated_fare

```python
from pyspark.sql.functions import expr, col, when, coalesce, lit
```

```python
## here I am using a something similar to if and else statement,
#when(condition, value_when_condition_met).otherwise(alt_condition)
df_train = df_train.withColumn(
    "calculated_fare",
    when((col("Fare")/col("family_size")).isNull(), col('Fare'))
    .otherwise((col("Fare")/col("family_size"))))
```

```python
df_test = df_test.withColumn(
    "calculated_fare",
    when((col("Fare")/col("family_size")).isNull(), col('Fare'))
    .otherwise((col("Fare")/col("family_size"))))
```

### fare_group

```python
@py_or_udf(returnType=StringType())
def fare_group(fare):
    """
    This function creates a fare group based on the fare provided
    """

    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a
```

```python
df_train = df_train.withColumn("fare_group", fare_group(col("Fare")))
df_test = df_test.withColumn("fare_group", fare_group(col("Fare")))
```

# That's all for today. Let's come back tomorrow when we will learn how to apply machine learning with Pyspark

```python
# Binarizing, Bucketing & Encoding
```

```python
train = spark.read.csv('../input/titanic/train.csv', header = True, inferSchema=True)
test = spark.read.csv('../input/titanic/test.csv', header = True, inferSchema=True)
```

```python
train.show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| NULL|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| NULL|       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| NULL|       S|
    |          6|       0|     3|    Moran, Mr. James|  male|NULL|    0|    0|          330877| 8.4583| NULL|       Q|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|           17463|51.8625|  E46|       S|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1|          349909| 21.075| NULL|       S|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333| NULL|       S|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708| NULL|       C|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|         PP 9549|   16.7|   G6|       S|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|          113783|  26.55| C103|       S|
    |         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|       A/5. 2151|   8.05| NULL|       S|
    |         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5|          347082| 31.275| NULL|       S|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0|          350406| 7.8542| NULL|       S|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|          248706|   16.0| NULL|       S|
    |         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1|          382652| 29.125| NULL|       Q|
    |         18|       1|     2|Williams, Mr. Cha...|  male|NULL|    0|    0|          244373|   13.0| NULL|       S|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|          345763|   18.0| NULL|       S|
    |         20|       1|     3|Masselmani, Mrs. ...|female|NULL|    0|    0|            2649|  7.225| NULL|       C|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 20 rows

```python
# Binarzing
from pyspark.ml.feature import Binarizer
# Cast the data type to double
train = train.withColumn('SibSp', train['SibSp'].cast('double'))
# Create binarzing transform
bin = Binarizer(threshold=0.0, inputCol='SibSp', outputCol='SibSpBin')
# Apply the transform
train = bin.transform(train)
```

```python
train.select('SibSp', 'SibSpBin').show(10)
```

    +-----+--------+
    |SibSp|SibSpBin|
    +-----+--------+
    |  1.0|     1.0|
    |  1.0|     1.0|
    |  0.0|     0.0|
    |  1.0|     1.0|
    |  0.0|     0.0|
    |  0.0|     0.0|
    |  0.0|     0.0|
    |  3.0|     1.0|
    |  0.0|     0.0|
    |  1.0|     1.0|
    +-----+--------+
    only showing top 10 rows

```python
# Bucketing
from pyspark.ml.feature import Bucketizer
# We are going to bucket the fare column
# Define the split
splits = [0,4,10,20,45, float('Inf')]

# Create bucketing transformer
buck = Bucketizer(splits=splits, inputCol='Fare', outputCol='FareB')

# Apply transformer
train = buck.transform(train)
```

```python
train.toPandas().head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>SibSpBin</th>
      <th>FareB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>None</td>
      <td>Q</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>None</td>
      <td>S</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>None</td>
      <td>S</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>None</td>
      <td>C</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
# One Hot Encoding
# it is a two step process
from pyspark.ml.feature import OneHotEncoder, StringIndexer
# Create indexer transformer for Sex Column

# Step 1: Create indexer for texts
stringIndexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')

# fit transform
model = stringIndexer.fit(train)

# Apply transform
indexed = model.transform(train)
```

```python
# Step 2: One Hot Encode
# Create encoder transformer
encoder = OneHotEncoder(inputCol='SexIndex', outputCol='Sex_Vec')

# fit model
model = encoder.fit(indexed)

# apply transform
encoded_df = model.transform(indexed)

encoded_df.toPandas().head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>SibSpBin</th>
      <th>FareB</th>
      <th>SexIndex</th>
      <th>Sex_Vec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>(1.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>(0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>(0.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>(0.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>(1.0)</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

```python

```

```python

```

<div class="alert alert-info">
    <h1>Resources</h1>
    <ul>
        <li><a href="https://docs.databricks.com/spark/latest/spark-sql/udf-python.html">User-defined functions - Python</a></li>
        <li><a href="https://medium.com/@ayplam/developing-pyspark-udfs-d179db0ccc87">Developing PySpark UDFs</a></li>
    </ul>
        <h1>Credits</h1>
    <ul>
        <li>To DataCamp, I have learned so much from DataCamp.</li>
        <li>To Jose Portilla, Such an amazing teacher with all of his resources</li>
    </ul>
    
</div>

<div class="alert alert-info">
<h4>If you like to discuss any other projects or just have a chat about data science topics, I'll be more than happy to connect with you on:</h4>
    <ul>
        <li><a href="https://www.linkedin.com/in/masumrumi/"><b>LinkedIn</b></a></li>
        <li><a href="https://github.com/masumrumi"><b>Github</b></a></li>
        <li><a href="https://masumrumi.com/"><b>masumrumi.com</b></a></li>
        <li><a href="https://www.youtube.com/channel/UC1mPjGyLcZmsMgZ8SJgrfdw"><b>Youtube</b></a></li>
    </ul>

<p>This kernel will always be a work in progress. I will incorporate new concepts of data science as I comprehend them with each update. If you have any idea/suggestions about this notebook, please let me know. Any feedback about further improvements would be genuinely appreciated.</p>

<h1>If you have come this far, Congratulations!!</h1>

<h1>If this notebook helped you in any way or you liked it, please upvote and/or leave a comment!! :)</h1></div>

<div class="alert alert-info">
    <h1>Versions</h1>
    <ul>
        <li>Version 16</li>
    </ul>
    
</div>

<div class="alert alert-danger">
    <h1>Work Area</h1>
</div>

### Other DataFrame Methods

```python
df_train.show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|  Mrs|          1|       loner|       1|           53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# agg
df_train.agg({"Age" : "min"}).show()
```

    +--------+
    |min(Age)|
    +--------+
    |    0.42|
    +--------+

```python
# agg
from pyspark.sql import functions as F
df_train.groupBy("Sex").agg(
    F.min("Age").name("min_age"),
    F.max("Age").alias("max_age")).show()
```

    +------+-------+-------+
    |   Sex|min_age|max_age|
    +------+-------+-------+
    |female|   0.75|   63.0|
    |  male|   0.42|   80.0|
    +------+-------+-------+

```python
# colRegex
df_train.select(df_train.colRegex("`(Sex)?+.+`")).show(5)
```

    +-----------+--------+------+--------------------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|35.0|    1|    0|   53.1|    C|       S|         44|         good|  Mrs|          1|       loner|       1|           53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    +-----------+--------+------+--------------------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# distinct
df_train[['Pclass', 'Sex']].distinct().show()
```

    +------+------+
    |Pclass|   Sex|
    +------+------+
    |     2|female|
    |     3|  male|
    |     1|  male|
    |     3|female|
    |     1|female|
    |     2|  male|
    +------+------+

```python
# another way
# dropDuplicates
df_train[['Pclass', 'Sex']].dropDuplicates().show()
```

    +------+------+
    |Pclass|   Sex|
    +------+------+
    |     2|female|
    |     3|  male|
    |     1|  male|
    |     3|female|
    |     1|female|
    |     2|  male|
    +------+------+

```python
# beware, this is probably not something we want when we try to do dropDuplicates
df_train.dropDuplicates(subset=['Pclass']).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|30.0708|    T|       C|         35|       medium|  Mrs|          1|       loner|       1|        30.0708|      high|
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+

```python
# drop_dupllicates()
# drop_duplicates() is an alias of dropDuplicates()
df_train[['Pclass', 'Sex']].drop_duplicates().show()
```

    +------+------+
    |Pclass|   Sex|
    +------+------+
    |     2|female|
    |     3|  male|
    |     1|  male|
    |     3|female|
    |     1|female|
    |     2|  male|
    +------+------+

```python
# drop
# dropping a column
df_train.drop('Name').show(5)
```

    +-----------+--------+------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|       0|     3|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|       1|     1|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|       1|     3|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          4|       1|     1|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|  Mrs|          1|       loner|       1|           53.1| very_high|
    |          5|       0|     3|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    +-----------+--------+------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# drop
# dropping multiple columns
df_train.drop("name", "Survived").show(5)
```

    +-----------+------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Pclass|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|     3|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|     1|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|     3|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          4|     1|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|  Mrs|          1|       loner|       1|           53.1| very_high|
    |          5|     3|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    +-----------+------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# dropna
df_train.dropna(how="any", subset=["Age"]).count()
```

    714

```python
#similarly
df_train.na.drop(how="any", subset=['Age']).count()
```

    714

```python
# exceptAll
# temp dataframes
df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
df2 = spark.createDataFrame([("a", 1),("a", 1), ("b", 3)], ["C1", "C2"])
```

```python
df1.show()
```

    +---+---+
    | C1| C2|
    +---+---+
    |  a|  1|
    |  a|  1|
    |  a|  1|
    |  a|  2|
    |  b|  3|
    |  c|  4|
    +---+---+

```python
df2.show()
```

    +---+---+
    | C1| C2|
    +---+---+
    |  a|  1|
    |  a|  1|
    |  b|  3|
    +---+---+

```python
df1.exceptAll(df2).show()
```

    +---+---+
    | C1| C2|
    +---+---+
    |  a|  1|
    |  a|  2|
    |  c|  4|
    +---+---+

```python
# intersect
df1.intersect(df2).show()
```

    +---+---+
    | C1| C2|
    +---+---+
    |  b|  3|
    |  a|  1|
    +---+---+

```python
# intersectAll
# intersectAll preserves the duplicates.
df1.intersectAll(df2).show()
```

    +---+---+
    | C1| C2|
    +---+---+
    |  a|  1|
    |  a|  1|
    |  b|  3|
    +---+---+

```python
# Returns True if the collect() and take() methods can be run locally
df_train.isLocal()
```

    False

```python
## fillna
df_train.fillna("N", subset=['Cabin']).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group| title|family_size|family_group|is_alone|   calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|    Mr|          1|       loner|       1|              7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|   Mrs|          1|       loner|       1|           71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium|  Miss|          0|       loner|       1|             7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|   Mrs|          1|       loner|       1|              53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|    Mr|          0|       loner|       1|              8.05|       low|
    |          6|       0|     3|    Moran, Mr. James|  male|NULL|    0|    0| 8.4583|    G|       Q|         16|        short|    Mr|          0|       loner|       1|            8.4583|       low|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|51.8625|    E|       S|         23|       medium|    Mr|          0|       loner|       1|           51.8625| very_high|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1| 21.075|    F|       S|         30|       medium|Master|          4|       small|       0|           5.26875|      high|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|11.1333|    G|       S|         49|         long|   Mrs|          2|       small|       0|           5.56665|       mid|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|30.0708|    T|       C|         35|       medium|   Mrs|          1|       loner|       1|           30.0708|      high|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|   16.7|    G|       S|         31|       medium|  Miss|          2|       small|       0|              8.35|       mid|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|  26.55|    C|       S|         24|       medium|  Miss|          0|       loner|       1|             26.55|      high|
    |         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|   8.05|    G|       S|         30|       medium|    Mr|          0|       loner|       1|              8.05|       low|
    |         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5| 31.275|    T|       S|         27|       medium|    Mr|          6|       large|       0|5.2124999999999995|      high|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0| 7.8542|    G|       S|         36|         good|  Miss|          0|       loner|       1|            7.8542|       low|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|   16.0|    F|       S|         32|       medium|   Mrs|          0|       loner|       1|              16.0|       mid|
    |         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1| 29.125|    T|       Q|         20|        short|Master|          5|       large|       0|             5.825|      high|
    |         18|       1|     2|Williams, Mr. Cha...|  male|NULL|    0|    0|   13.0|    G|       S|         28|       medium|    Mr|          0|       loner|       1|              13.0|       mid|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|   18.0|    F|       S|         55|         long|   Mrs|          1|       loner|       1|              18.0|       mid|
    |         20|       1|     3|Masselmani, Mrs. ...|female|NULL|    0|    0|  7.225|    G|       C|         23|       medium|   Mrs|          0|       loner|       1|             7.225|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    only showing top 20 rows

```python
# similarly
# dataFrame.na.fill() is alias of dataFrame.fillna()
df_train.na.fill("N", subset=['Cabin']).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group| title|family_size|family_group|is_alone|   calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|    Mr|          1|       loner|       1|              7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|   Mrs|          1|       loner|       1|           71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium|  Miss|          0|       loner|       1|             7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|   Mrs|          1|       loner|       1|              53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|    Mr|          0|       loner|       1|              8.05|       low|
    |          6|       0|     3|    Moran, Mr. James|  male|NULL|    0|    0| 8.4583|    G|       Q|         16|        short|    Mr|          0|       loner|       1|            8.4583|       low|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|51.8625|    E|       S|         23|       medium|    Mr|          0|       loner|       1|           51.8625| very_high|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1| 21.075|    F|       S|         30|       medium|Master|          4|       small|       0|           5.26875|      high|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|11.1333|    G|       S|         49|         long|   Mrs|          2|       small|       0|           5.56665|       mid|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|30.0708|    T|       C|         35|       medium|   Mrs|          1|       loner|       1|           30.0708|      high|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|   16.7|    G|       S|         31|       medium|  Miss|          2|       small|       0|              8.35|       mid|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|  26.55|    C|       S|         24|       medium|  Miss|          0|       loner|       1|             26.55|      high|
    |         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|   8.05|    G|       S|         30|       medium|    Mr|          0|       loner|       1|              8.05|       low|
    |         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5| 31.275|    T|       S|         27|       medium|    Mr|          6|       large|       0|5.2124999999999995|      high|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0| 7.8542|    G|       S|         36|         good|  Miss|          0|       loner|       1|            7.8542|       low|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|   16.0|    F|       S|         32|       medium|   Mrs|          0|       loner|       1|              16.0|       mid|
    |         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1| 29.125|    T|       Q|         20|        short|Master|          5|       large|       0|             5.825|      high|
    |         18|       1|     2|Williams, Mr. Cha...|  male|NULL|    0|    0|   13.0|    G|       S|         28|       medium|    Mr|          0|       loner|       1|              13.0|       mid|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|   18.0|    F|       S|         55|         long|   Mrs|          1|       loner|       1|              18.0|       mid|
    |         20|       1|     3|Masselmani, Mrs. ...|female|NULL|    0|    0|  7.225|    G|       C|         23|       medium|   Mrs|          0|       loner|       1|             7.225|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    only showing top 20 rows

```python
age_mean = df_train.agg({"Age": "mean"}).collect()[0][0]
```

```python
age_mean
```

    29.69911764705882

```python
df_train.fillna({"Age": age_mean, "Cabin": "N"})[['Age', "Cabin"]].show(10)
```

    +-----------------+-----+
    |              Age|Cabin|
    +-----------------+-----+
    |             22.0|    G|
    |             38.0|    C|
    |             26.0|    G|
    |             35.0|    C|
    |             35.0|    G|
    |29.69911764705882|    G|
    |             54.0|    E|
    |              2.0|    F|
    |             27.0|    G|
    |             14.0|    T|
    +-----------------+-----+
    only showing top 10 rows

```python
# first
df_train.first()
```

    Row(PassengerId=1, Survived=0, Pclass=3, Name='Braund, Mr. Owen Harris', Sex='male', Age=22.0, SibSp=1, Parch=0, Fare=7.25, Cabin='G', Embarked='S', name_length=23, nLength_group='medium', title='Mr', family_size=1, family_group='loner', is_alone=1, calculated_fare=7.25, fare_group='low')

```python
def f(passenger):
    print(passenger.Name)
```

```python
# foreach
# this prints out in the terminal.
df_train.foreach(f)
```

    Braund, Mr. Owen Harris
    Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    Heikkinen, Miss. Laina
    Futrelle, Mrs. Jacques Heath (Lily May Peel)
    Allen, Mr. William Henry
    Moran, Mr. James
    McCarthy, Mr. Timothy J
    Palsson, Master. Gosta Leonard
    Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)
    Nasser, Mrs. Nicholas (Adele Achem)
    Sandstrom, Miss. Marguerite Rut
    Bonnell, Miss. Elizabeth
    Saundercock, Mr. William Henry
    Andersson, Mr. Anders Johan
    Vestrom, Miss. Hulda Amanda Adolfina
    Hewlett, Mrs. (Mary D Kingcome)
    Rice, Master. Eugene
    Williams, Mr. Charles Eugene
    Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)
    Masselmani, Mrs. Fatima
    Fynney, Mr. Joseph J
    Beesley, Mr. Lawrence
    "McGowan, Miss. Anna ""Annie"""
    Sloper, Mr. William Thompson
    Palsson, Miss. Torborg Danira
    Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson)
    Emir, Mr. Farred Chehab
    Fortune, Mr. Charles Alexander
    "O'Dwyer, Miss. Ellen ""Nellie"""
    Todoroff, Mr. Lalio
    Uruchurtu, Don. Manuel E
    Spencer, Mrs. William Augustus (Marie Eugenie)
    Glynn, Miss. Mary Agatha
    Wheadon, Mr. Edward H
    Meyer, Mr. Edgar Joseph
    Holverson, Mr. Alexander Oskar
    Mamee, Mr. Hanna
    Cann, Mr. Ernest Charles
    Vander Planke, Miss. Augusta Maria
    Nicola-Yarred, Miss. Jamila
    Ahlin, Mrs. Johan (Johanna Persdotter Larsson)
    Turpin, Mrs. William John Robert (Dorothy Ann Wonnacott)
    Kraeff, Mr. Theodor
    Laroche, Miss. Simonne Marie Anne Andree
    Devaney, Miss. Margaret Delia
    Rogers, Mr. William John
    Lennon, Mr. Denis
    O'Driscoll, Miss. Bridget
    Samaan, Mr. Youssef
    Arnold-Franchi, Mrs. Josef (Josefine Franchi)
    Panula, Master. Juha Niilo
    Nosworthy, Mr. Richard Cater
    Harper, Mrs. Henry Sleeper (Myna Haxtun)
    Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkinson)
    Ostby, Mr. Engelhart Cornelius
    Woolner, Mr. Hugh
    Rugg, Miss. Emily
    Novel, Mr. Mansouer
    West, Miss. Constance Mirium
    Goodwin, Master. William Frederick
    Sirayanian, Mr. Orsen
    Icard, Miss. Amelie
    Harris, Mr. Henry Birkhardt
    Skoog, Master. Harald
    Stewart, Mr. Albert A
    Moubarek, Master. Gerios
    Nye, Mrs. (Elizabeth Ramell)
    Crease, Mr. Ernest James
    Andersson, Miss. Erna Alexandra
    Kink, Mr. Vincenz
    Jenkin, Mr. Stephen Curnow
    Goodwin, Miss. Lillian Amy
    Hood, Mr. Ambrose Jr
    Chronopoulos, Mr. Apostolos
    Bing, Mr. Lee
    Moen, Mr. Sigurd Hansen
    Staneff, Mr. Ivan
    Moutal, Mr. Rahamin Haim
    Caldwell, Master. Alden Gates
    Dowdell, Miss. Elizabeth
    Waelens, Mr. Achille
    Sheerlinck, Mr. Jan Baptist
    McDermott, Miss. Brigdet Delia
    Carrau, Mr. Francisco M
    Ilett, Miss. Bertha
    Backstrom, Mrs. Karl Alfred (Maria Mathilda Gustafsson)
    Ford, Mr. William Neal
    Slocovski, Mr. Selman Francis
    Fortune, Miss. Mabel Helen
    Celotti, Mr. Francesco
    Christmann, Mr. Emil
    Andreasson, Mr. Paul Edvin
    Chaffee, Mr. Herbert Fuller
    Dean, Mr. Bertram Frank
    Coxon, Mr. Daniel
    Shorney, Mr. Charles Joseph
    Goldschmidt, Mr. George B
    Greenfield, Mr. William Bertram
    Doling, Mrs. John T (Ada Julia Bone)
    Kantor, Mr. Sinai
    Petranec, Miss. Matilda
    "Petroff, Mr. Pastcho (""Pentcho"")"
    White, Mr. Richard Frasar
    Johansson, Mr. Gustaf Joel
    Gustafsson, Mr. Anders Vilhelm
    Mionoff, Mr. Stoytcho
    Salkjelsvik, Miss. Anna Kristine
    Moss, Mr. Albert Johan
    Rekic, Mr. Tido
    Moran, Miss. Bertha
    Porter, Mr. Walter Chamberlain
    Zabour, Miss. Hileni
    Barton, Mr. David John
    Jussila, Miss. Katriina
    Attalah, Miss. Malake
    Pekoniemi, Mr. Edvard
    Connors, Mr. Patrick
    Turpin, Mr. William John Robert
    Baxter, Mr. Quigg Edmond
    Andersson, Miss. Ellis Anna Maria
    Hickman, Mr. Stanley George
    Moore, Mr. Leonard Charles
    Nasser, Mr. Nicholas
    Webber, Miss. Susan
    White, Mr. Percival Wayland
    Nicola-Yarred, Master. Elias
    McMahon, Mr. Martin
    Madsen, Mr. Fridtjof Arne
    Peter, Miss. Anna
    Ekstrom, Mr. Johan
    Drazenoic, Mr. Jozef
    Coelho, Mr. Domingos Fernandeo
    Robins, Mrs. Alexander A (Grace Charity Laury)
    Weisz, Mrs. Leopold (Mathilde Francoise Pede)
    Sobey, Mr. Samuel James Hayden
    Richard, Mr. Emile
    Newsom, Miss. Helen Monypeny
    Futrelle, Mr. Jacques Heath
    Osen, Mr. Olaf Elon
    Giglio, Mr. Victor
    Boulos, Mrs. Joseph (Sultana)
    Nysten, Miss. Anna Sofia
    Hakkarainen, Mrs. Pekka Pietari (Elin Matilda Dolck)
    Burke, Mr. Jeremiah
    Andrew, Mr. Edgardo Samuel
    Nicholls, Mr. Joseph Charles
    "Andersson, Mr. August Edvard (""Wennerstrom"")"
    "Ford, Miss. Robina Maggie ""Ruby"""
    "Navratil, Mr. Michel (""Louis M Hoffman"")"
    Byles, Rev. Thomas Roussel Davids
    Bateman, Rev. Robert James
    Pears, Mrs. Thomas (Edith Wearne)
    Meo, Mr. Alfonzo
    van Billiard, Mr. Austin Blyler
    Olsen, Mr. Ole Martin
    Williams, Mr. Charles Duane
    "Gilnagh, Miss. Katherine ""Katie"""
    Corn, Mr. Harry
    Smiljanic, Mr. Mile
    Sage, Master. Thomas Henry
    Cribb, Mr. John Hatfield
    "Watt, Mrs. James (Elizabeth ""Bessie"" Inglis Milne)"
    Bengtsson, Mr. John Viktor
    Calic, Mr. Jovo
    Panula, Master. Eino Viljami
    "Goldsmith, Master. Frank John William ""Frankie"""
    Chibnall, Mrs. (Edith Martha Bowerman)
    Skoog, Mrs. William (Anna Bernhardina Karlsson)
    Baumann, Mr. John D
    Ling, Mr. Lee
    Van der hoef, Mr. Wyckoff
    Rice, Master. Arthur
    Johnson, Miss. Eleanor Ileen
    Sivola, Mr. Antti Wilhelm
    Smith, Mr. James Clinch
    Klasen, Mr. Klas Albin
    Lefebre, Master. Henry Forbes
    Isham, Miss. Ann Elizabeth
    Hale, Mr. Reginald
    Leonard, Mr. Lionel
    Sage, Miss. Constance Gladys
    Pernot, Mr. Rene
    Asplund, Master. Clarence Gustaf Hugo
    Becker, Master. Richard F
    Kink-Heilmann, Miss. Luise Gretchen
    Rood, Mr. Hugh Roscoe
    "O'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)"
    "Romaine, Mr. Charles Hallace (""Mr C Rolmane"")"
    Bourke, Mr. John
    Turcin, Mr. Stjepan
    Pinsky, Mrs. (Rosa)
    Carbines, Mr. William
    Andersen-Jensen, Miss. Carla Christine Nielsine
    Navratil, Master. Michel M
    Brown, Mrs. James Joseph (Margaret Tobin)
    Lurette, Miss. Elise
    Mernagh, Mr. Robert
    Olsen, Mr. Karl Siegwart Andreas
    "Madigan, Miss. Margaret ""Maggie"""
    "Yrois, Miss. Henriette (""Mrs Harbeck"")"
    Vande Walle, Mr. Nestor Cyriel
    Sage, Mr. Frederick
    Johanson, Mr. Jakob Alfred
    Youseff, Mr. Gerious
    "Cohen, Mr. Gurshon ""Gus"""
    Strom, Miss. Telma Matilda
    Backstrom, Mr. Karl Alfred
    Albimona, Mr. Nassef Cassem
    "Carr, Miss. Helen ""Ellen"""
    Blank, Mr. Henry
    Ali, Mr. Ahmed
    Cameron, Miss. Clear Annie
    Perkin, Mr. John Henry
    Givard, Mr. Hans Kristensen
    Kiernan, Mr. Philip
    Newell, Miss. Madeleine
    Honkanen, Miss. Eliina
    Jacobsohn, Mr. Sidney Samuel
    Bazzani, Miss. Albina
    Harris, Mr. Walter
    Sunderland, Mr. Victor Francis
    Bracken, Mr. James H
    Green, Mr. George Henry
    Nenkoff, Mr. Christo
    Hoyt, Mr. Frederick Maxfield
    Berglund, Mr. Karl Ivar Sven
    Mellors, Mr. William John
    "Lovell, Mr. John Hall (""Henry"")"
    Fahlstrom, Mr. Arne Jonas
    Lefebre, Miss. Mathilde
    Harris, Mrs. Henry Birkhardt (Irene Wallach)
    Larsson, Mr. Bengt Edvin
    Sjostedt, Mr. Ernst Adolf
    Asplund, Miss. Lillian Gertrud
    Leyson, Mr. Robert William Norman
    Harknett, Miss. Alice Phoebe
    Hold, Mr. Stephen
    "Collyer, Miss. Marjorie ""Lottie"""
    Pengelly, Mr. Frederick William
    Hunt, Mr. George Henry
    Zabour, Miss. Thamine
    "Murphy, Miss. Katherine ""Kate"""
    Coleridge, Mr. Reginald Charles
    Maenpaa, Mr. Matti Alexanteri
    Attalah, Mr. Sleiman
    Minahan, Dr. William Edward
    Lindahl, Miss. Agda Thorilda Viktoria
    Hamalainen, Mrs. William (Anna)
    Beckwith, Mr. Richard Leonard
    Carter, Rev. Ernest Courtenay
    Reed, Mr. James George
    Strom, Mrs. Wilhelm (Elna Matilda Persson)
    Stead, Mr. William Thomas
    Lobb, Mr. William Arthur
    Rosblom, Mrs. Viktor (Helena Wilhelmina)
    Touma, Mrs. Darwis (Hanne Youssef Razi)
    Thorne, Mrs. Gertrude Maybelle
    Cherry, Miss. Gladys
    Ward, Miss. Anna
    Parrish, Mrs. (Lutie Davis)
    Smith, Mr. Thomas
    Asplund, Master. Edvin Rojj Felix
    Taussig, Mr. Emil
    Harrison, Mr. William
    Henry, Miss. Delia
    Reeves, Mr. David
    Panula, Mr. Ernesti Arvid
    Persson, Mr. Ernst Ulrik
    Graham, Mrs. William Thompson (Edith Junkins)
    Bissette, Miss. Amelia
    Cairns, Mr. Alexander
    Tornquist, Mr. William Henry
    Mellinger, Mrs. (Elizabeth Anne Maidment)
    Natsch, Mr. Charles H
    "Healy, Miss. Hanora ""Nora"""
    Andrews, Miss. Kornelia Theodosia
    Lindblom, Miss. Augusta Charlotta
    "Parkes, Mr. Francis ""Frank"""
    Rice, Master. Eric
    Abbott, Mrs. Stanton (Rosa Hunt)
    Duane, Mr. Frank
    Olsson, Mr. Nils Johan Goransson
    de Pelsmaeker, Mr. Alfons
    Dorking, Mr. Edward Arthur
    Smith, Mr. Richard William
    Stankovic, Mr. Ivan
    de Mulder, Mr. Theodore
    Naidenoff, Mr. Penko
    Hosono, Mr. Masabumi
    Connolly, Miss. Kate
    "Barber, Miss. Ellen ""Nellie"""
    Bishop, Mrs. Dickinson H (Helen Walton)
    Levy, Mr. Rene Jacques
    Haas, Miss. Aloisia
    Mineff, Mr. Ivan
    Lewy, Mr. Ervin G
    Hanna, Mr. Mansour
    Allison, Miss. Helen Loraine
    Saalfeld, Mr. Adolphe
    Baxter, Mrs. James (Helene DeLaudeniere Chaput)
    "Kelly, Miss. Anna Katherine ""Annie Kate"""
    McCoy, Mr. Bernard
    Johnson, Mr. William Cahoone Jr
    Keane, Miss. Nora A
    "Williams, Mr. Howard Hugh ""Harry"""
    Allison, Master. Hudson Trevor
    Fleming, Miss. Margaret
    Penasco y Castellana, Mrs. Victor de Satode (Maria Josefa Perez de Soto y Vallejo)
    Abelson, Mr. Samuel
    Francatelli, Miss. Laura Mabel
    Hays, Miss. Margaret Bechstein
    Ryerson, Miss. Emily Borie
    Lahtinen, Mrs. William (Anna Sylfven)
    Hendekovic, Mr. Ignjac
    Hart, Mr. Benjamin
    Nilsson, Miss. Helmina Josefina
    Kantor, Mrs. Sinai (Miriam Sternin)
    Moraweck, Dr. Ernest
    Wick, Miss. Mary Natalie
    Spedden, Mrs. Frederic Oakley (Margaretta Corning Stone)
    Dennis, Mr. Samuel
    Danoff, Mr. Yoto
    Slayter, Miss. Hilda Mary
    Caldwell, Mrs. Albert Francis (Sylvia Mae Harbaugh)
    Sage, Mr. George John Jr
    Young, Miss. Marie Grice
    Nysveen, Mr. Johan Hansen
    Ball, Mrs. (Ada E Hall)
    Goldsmith, Mrs. Frank John (Emily Alice Brown)
    Hippach, Miss. Jean Gertrude
    McCoy, Miss. Agnes
    Partner, Mr. Austen
    Graham, Mr. George Edward
    Vander Planke, Mr. Leo Edmondus
    Frauenthal, Mrs. Henry William (Clara Heinsheimer)
    Denkoff, Mr. Mitto
    Pears, Mr. Thomas Clinton
    Burns, Miss. Elizabeth Margaret
    Dahl, Mr. Karl Edwart
    Blackwell, Mr. Stephen Weart
    Navratil, Master. Edmond Roger
    Fortune, Miss. Alice Elizabeth
    Collander, Mr. Erik Gustaf
    Sedgwick, Mr. Charles Frederick Waddington
    Fox, Mr. Stanley Hubert
    "Brown, Miss. Amelia ""Mildred"""
    Smith, Miss. Marion Elsie
    Davison, Mrs. Thomas Henry (Mary E Finck)
    "Coutts, Master. William Loch ""William"""
    Dimic, Mr. Jovan
    Odahl, Mr. Nils Martin
    Williams-Lambert, Mr. Fletcher Fellows
    Elias, Mr. Tannous
    Arnold-Franchi, Mr. Josef
    Yousif, Mr. Wazli
    Vanden Steen, Mr. Leo Peter
    Bowerman, Miss. Elsie Edith
    Funk, Miss. Annie Clemmer
    McGovern, Miss. Mary
    "Mockler, Miss. Helen Mary ""Ellie"""
    Skoog, Mr. Wilhelm
    del Carlo, Mr. Sebastiano
    Barbara, Mrs. (Catherine David)
    Asim, Mr. Adola
    O'Brien, Mr. Thomas
    Adahl, Mr. Mauritz Nils Martin
    Warren, Mrs. Frank Manley (Anna Sophia Atkinson)
    Moussa, Mrs. (Mantoura Boulos)
    Jermyn, Miss. Annie
    Aubart, Mme. Leontine Pauline
    Harder, Mr. George Achilles
    Wiklund, Mr. Jakob Alfred
    Beavan, Mr. William Thomas
    Ringhini, Mr. Sante
    Palsson, Miss. Stina Viola
    Meyer, Mrs. Edgar Joseph (Leila Saks)
    Landergren, Miss. Aurora Adelia
    Widener, Mr. Harry Elkins
    Betros, Mr. Tannous
    Gustafsson, Mr. Karl Gideon
    Bidois, Miss. Rosalie
    "Nakid, Miss. Maria (""Mary"")"
    Tikkanen, Mr. Juho
    Holverson, Mrs. Alexander Oskar (Mary Aline Towner)
    Plotcharsky, Mr. Vasil
    Davies, Mr. Charles Henry
    Goodwin, Master. Sidney Leonard
    Buss, Miss. Kate
    Sadlier, Mr. Matthew
    Lehmann, Miss. Bertha
    Carter, Mr. William Ernest
    Jansson, Mr. Carl Olof
    Gustafsson, Mr. Johan Birger
    Newell, Miss. Marjorie
    Sandstrom, Mrs. Hjalmar (Agnes Charlotta Bengtsson)
    Johansson, Mr. Erik
    Olsson, Miss. Elina
    McKane, Mr. Peter David
    Pain, Dr. Alfred
    Trout, Mrs. William H (Jessie L)
    Niskanen, Mr. Juha
    Adams, Mr. John
    Jussila, Miss. Mari Aina
    Hakkarainen, Mr. Pekka Pietari
    Oreskovic, Miss. Marija
    Gale, Mr. Shadrach
    Widegren, Mr. Carl/Charles Peter
    Richards, Master. William Rowe
    Birkeland, Mr. Hans Martin Monsen
    Lefebre, Miss. Ida
    Sdycoff, Mr. Todor
    Hart, Mr. Henry
    Minahan, Miss. Daisy E
    Cunningham, Mr. Alfred Fleming
    Sundman, Mr. Johan Julian
    Meek, Mrs. Thomas (Annie Louise Rowley)
    Drew, Mrs. James Vivian (Lulu Thorne Christian)
    Silven, Miss. Lyyli Karoliina
    Matthews, Mr. William John
    Van Impe, Miss. Catharina
    Gheorgheff, Mr. Stanio
    Charters, Mr. David
    Zimmerman, Mr. Leo
    Danbom, Mrs. Ernst Gilbert (Anna Sigrid Maria Brogren)
    Rosblom, Mr. Viktor Richard
    Wiseman, Mr. Phillippe
    Clarke, Mrs. Charles V (Ada Maria Winfield)
    "Phillips, Miss. Kate Florence (""Mrs Kate Louise Phillips Marshall"")"
    Flynn, Mr. James
    Pickard, Mr. Berk (Berk Trembisky)
    Bjornstrom-Steffansson, Mr. Mauritz Hakan
    Thorneycroft, Mrs. Percival (Florence Kate White)
    Louch, Mrs. Charles Alexander (Alice Adelaide Slow)
    Kallio, Mr. Nikolai Erland
    Silvey, Mr. William Baird
    Carter, Miss. Lucile Polk
    "Ford, Miss. Doolina Margaret ""Daisy"""
    Richards, Mrs. Sidney (Emily Hocking)
    Fortune, Mr. Mark
    Kvillner, Mr. Johan Henrik Johannesson
    Hart, Mrs. Benjamin (Esther Ada Bloomfield)
    Hampe, Mr. Leon
    Petterson, Mr. Johan Emil
    Reynaldo, Ms. Encarnacion
    Johannesen-Bratthammer, Mr. Bernt
    Dodge, Master. Washington
    Mellinger, Miss. Madeleine Violet
    Seward, Mr. Frederic Kimber
    Baclini, Miss. Marie Catherine
    Peuchen, Major. Arthur Godfrey
    West, Mr. Edwy Arthur
    Hagland, Mr. Ingvald Olai Olsen
    Foreman, Mr. Benjamin Laventall
    Goldenberg, Mr. Samuel L
    Peduzzi, Mr. Joseph
    Jalsevac, Mr. Ivan
    Millet, Mr. Francis Davis
    Kenyon, Mrs. Frederick R (Marion)
    Toomey, Miss. Ellen
    O'Connor, Mr. Maurice
    Anderson, Mr. Harry
    Morley, Mr. William
    Gee, Mr. Arthur H
    Milling, Mr. Jacob Christian
    Maisner, Mr. Simon
    Goncalves, Mr. Manuel Estanslas
    Campbell, Mr. William
    Smart, Mr. John Montgomery
    Scanlan, Mr. James
    Baclini, Miss. Helene Barbara
    Keefe, Mr. Arthur
    Cacic, Mr. Luka
    West, Mrs. Edwy Arthur (Ada Mary Worth)
    Jerwan, Mrs. Amin S (Marie Marthe Thuillard)
    Strandberg, Miss. Ida Sofia
    Clifford, Mr. George Quincy
    Renouf, Mr. Peter Henry
    Braund, Mr. Lewis Richard
    Karlsson, Mr. Nils August
    Hirvonen, Miss. Hildur E
    Goodwin, Master. Harold Victor
    "Frost, Mr. Anthony Wood ""Archie"""
    Rouse, Mr. Richard Henry
    Turkula, Mrs. (Hedwig)
    Bishop, Mr. Dickinson H
    Lefebre, Miss. Jeannie
    Hoyt, Mrs. Frederick Maxfield (Jane Anne Forby)
    Kent, Mr. Edward Austin
    Somerton, Mr. Francis William
    "Coutts, Master. Eden Leslie ""Neville"""
    Hagland, Mr. Konrad Mathias Reiersen
    Windelov, Mr. Einar
    Molson, Mr. Harry Markland
    Artagaveytia, Mr. Ramon
    Stanley, Mr. Edward Roland
    Yousseff, Mr. Gerious
    Eustis, Miss. Elizabeth Mussey
    Shellard, Mr. Frederick William
    Allison, Mrs. Hudson J C (Bessie Waldo Daniels)
    Svensson, Mr. Olof
    Calic, Mr. Petar
    Canavan, Miss. Mary
    O'Sullivan, Miss. Bridget Mary
    Laitinen, Miss. Kristina Sofia
    Maioni, Miss. Roberta
    Penasco y Castellana, Mr. Victor de Satode
    Quick, Mrs. Frederick Charles (Jane Richards)
    "Bradley, Mr. George (""George Arthur Brayton"")"
    Olsen, Mr. Henry Margido
    Lang, Mr. Fang
    Daly, Mr. Eugene Patrick
    Webber, Mr. James
    McGough, Mr. James Robert
    Rothschild, Mrs. Martin (Elizabeth L. Barrett)
    Coleff, Mr. Satio
    Walker, Mr. William Anderson
    Lemore, Mrs. (Amelia Milley)
    Ryan, Mr. Patrick
    "Angle, Mrs. William A (Florence ""Mary"" Agnes Hughes)"
    Pavlovic, Mr. Stefo
    Perreault, Miss. Anne
    Vovk, Mr. Janko
    Lahoud, Mr. Sarkis
    Hippach, Mrs. Louis Albert (Ida Sophia Fischer)
    Kassem, Mr. Fared
    Farrell, Mr. James
    Ridsdale, Miss. Lucy
    Farthing, Mr. John
    Salonen, Mr. Johan Werner
    Hocking, Mr. Richard George
    Quick, Miss. Phyllis May
    Toufik, Mr. Nakli
    Elias, Mr. Joseph Jr
    Peter, Mrs. Catherine (Catherine Rizk)
    Cacic, Miss. Marija
    Hart, Miss. Eva Miriam
    Butt, Major. Archibald Willingham
    LeRoy, Miss. Bertha
    Risien, Mr. Samuel Beard
    Frolicher, Miss. Hedwig Margaritha
    Crosby, Miss. Harriet R
    Andersson, Miss. Ingeborg Constanzia
    Andersson, Miss. Sigrid Elisabeth
    Beane, Mr. Edward
    Douglas, Mr. Walter Donald
    Nicholson, Mr. Arthur Ernest
    Beane, Mrs. Edward (Ethel Clarke)
    Padro y Manent, Mr. Julian
    Goldsmith, Mr. Frank John
    Davies, Master. John Morgan Jr
    Thayer, Mr. John Borland Jr
    Sharp, Mr. Percival James R
    O'Brien, Mr. Timothy
    "Leeni, Mr. Fahim (""Philip Zenni"")"
    Ohman, Miss. Velin
    Wright, Mr. George
    "Duff Gordon, Lady. (Lucille Christiana Sutherland) (""Mrs Morgan"")"
    Robbins, Mr. Victor
    Taussig, Mrs. Emil (Tillie Mandelbaum)
    de Messemaeker, Mrs. Guillaume Joseph (Emma)
    Morrow, Mr. Thomas Rowan
    Sivic, Mr. Husein
    Norman, Mr. Robert Douglas
    Simmons, Mr. John
    Meanwell, Miss. (Marion Ogden)
    Davies, Mr. Alfred J
    Stoytcheff, Mr. Ilia
    Palsson, Mrs. Nils (Alma Cornelia Berglund)
    Doharr, Mr. Tannous
    Jonsson, Mr. Carl
    Harris, Mr. George
    Appleton, Mrs. Edward Dale (Charlotte Lamson)
    "Flynn, Mr. John Irwin (""Irving"")"
    Kelly, Miss. Mary
    Rush, Mr. Alfred George John
    Patchett, Mr. George
    Garside, Miss. Ethel
    Silvey, Mrs. William Baird (Alice Munger)
    Caram, Mrs. Joseph (Maria Elias)
    Jussila, Mr. Eiriik
    Christy, Miss. Julie Rachel
    Thayer, Mrs. John Borland (Marian Longstreth Morris)
    Downton, Mr. William James
    Ross, Mr. John Hugo
    Paulner, Mr. Uscher
    Taussig, Miss. Ruth
    Jarvis, Mr. John Denzil
    Frolicher-Stehli, Mr. Maxmillian
    Gilinski, Mr. Eliezer
    Murdlin, Mr. Joseph
    Rintamaki, Mr. Matti
    Stephenson, Mrs. Walter Bertram (Martha Eustis)
    Elsbury, Mr. William James
    Bourke, Miss. Mary
    Chapman, Mr. John Henry
    Van Impe, Mr. Jean Baptiste
    Leitch, Miss. Jessie Wills
    Johnson, Mr. Alfred
    Boulos, Mr. Hanna
    "Duff Gordon, Sir. Cosmo Edmund (""Mr Morgan"")"
    Jacobsohn, Mrs. Sidney Samuel (Amy Frances Christy)
    Slabenoff, Mr. Petco
    Harrington, Mr. Charles H
    Torber, Mr. Ernst William
    "Homer, Mr. Harry (""Mr E Haven"")"
    Lindell, Mr. Edvard Bengtsson
    Karaic, Mr. Milan
    Daniel, Mr. Robert Williams
    Laroche, Mrs. Joseph (Juliette Marie Louise Lafargue)
    Shutes, Miss. Elizabeth W
    Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren)
    Jardin, Mr. Jose Neto
    Murphy, Miss. Margaret Jane
    Horgan, Mr. John
    Brocklebank, Mr. William Alfred
    Herman, Miss. Alice
    Danbom, Mr. Ernst Gilbert
    Lobb, Mrs. William Arthur (Cordelia K Stanlick)
    Becker, Miss. Marion Louise
    Gavey, Mr. Lawrence
    Yasbeck, Mr. Antoni
    Kimball, Mr. Edwin Nelson Jr
    Nakid, Mr. Sahid
    Hansen, Mr. Henry Damsgaard
    "Bowen, Mr. David John ""Dai"""
    Sutton, Mr. Frederick
    Kirkland, Rev. Charles Leonard
    Longley, Miss. Gretchen Fiske
    Bostandyeff, Mr. Guentcho
    O'Connell, Mr. Patrick D
    Barkworth, Mr. Algernon Henry Wilson
    Lundahl, Mr. Johan Svensson
    Stahelin-Maeglin, Dr. Max
    Parr, Mr. William Henry Marsh
    Skoog, Miss. Mabel
    Davis, Miss. Mary
    Leinonen, Mr. Antti Gustaf
    Collyer, Mr. Harvey
    Panula, Mrs. Juha (Maria Emilia Ojala)
    Thorneycroft, Mr. Percival
    Jensen, Mr. Hans Peder
    Sagesser, Mlle. Emma
    Skoog, Miss. Margit Elizabeth
    Foo, Mr. Choong
    Baclini, Miss. Eugenie
    Harper, Mr. Henry Sleeper
    Cor, Mr. Liudevit
    Simonius-Blumer, Col. Oberst Alfons
    Willey, Mr. Edward
    Stanley, Miss. Amy Zillah Elsie
    Mitkoff, Mr. Mito
    Doling, Miss. Elsie
    Kalvik, Mr. Johannes Halvorsen
    "O'Leary, Miss. Hanora ""Norah"""
    "Hegarty, Miss. Hanora ""Nora"""
    Hickman, Mr. Leonard Mark
    Radeff, Mr. Alexander
    Bourke, Mrs. John (Catherine)
    Eitemiller, Mr. George Floyd
    Newell, Mr. Arthur Webster
    Frauenthal, Dr. Henry William
    Badt, Mr. Mohamed
    Colley, Mr. Edward Pomeroy
    Coleff, Mr. Peju
    Lindqvist, Mr. Eino William
    Hickman, Mr. Lewis
    Butler, Mr. Reginald Fenton
    Rommetvedt, Mr. Knud Paust
    Cook, Mr. Jacob
    Taylor, Mrs. Elmer Zebley (Juliet Cummins Wright)
    Brown, Mrs. Thomas William Solomon (Elizabeth Catherine Ford)
    Davidson, Mr. Thornton
    Mitchell, Mr. Henry Michael
    Wilhelms, Mr. Charles
    Watson, Mr. Ennis Hastings
    Edvardsson, Mr. Gustaf Hjalmar
    Sawyer, Mr. Frederick Charles
    Turja, Miss. Anna Sofia
    Goodwin, Mrs. Frederick (Augusta Tyler)
    Cardeza, Mr. Thomas Drake Martinez
    Peters, Miss. Katie
    Hassab, Mr. Hammad
    Olsvigen, Mr. Thor Anderson
    Goodwin, Mr. Charles Edward
    Brown, Mr. Thomas William Solomon
    Laroche, Mr. Joseph Philippe Lemercier
    Panula, Mr. Jaako Arnold
    Dakic, Mr. Branko
    Fischer, Mr. Eberhard Thelander
    Madill, Miss. Georgette Alexandra
    Dick, Mr. Albert Adrian
    Karun, Miss. Manca
    Lam, Mr. Ali
    Saad, Mr. Khalil
    Weir, Col. John
    Chapman, Mr. Charles Henry
    Kelly, Mr. James
    "Mullens, Miss. Katherine ""Katie"""
    Thayer, Mr. John Borland
    Humblen, Mr. Adolf Mathias Nicolai Olsen
    Astor, Mrs. John Jacob (Madeleine Talmadge Force)
    Silverthorne, Mr. Spencer Victor
    Barbara, Miss. Saiide
    Gallagher, Mr. Martin
    Hansen, Mr. Henrik Juul
    "Morley, Mr. Henry Samuel (""Mr Henry Marshall"")"
    "Kelly, Mrs. Florence ""Fannie"""
    Calderhead, Mr. Edward Pennington
    Cleaver, Miss. Alice
    "Moubarek, Master. Halim Gonios (""William George"")"
    "Mayne, Mlle. Berthe Antonine (""Mrs de Villiers"")"
    Klaber, Mr. Herman
    Taylor, Mr. Elmer Zebley
    Larsson, Mr. August Viktor
    Greenberg, Mr. Samuel
    Soholt, Mr. Peter Andreas Lauritz Andersen
    Endres, Miss. Caroline Louise
    "Troutt, Miss. Edwina Celia ""Winnie"""
    McEvoy, Mr. Michael
    Johnson, Mr. Malkolm Joackim
    "Harper, Miss. Annie Jessie ""Nina"""
    Jensen, Mr. Svend Lauritz
    Gillespie, Mr. William Henry
    Hodges, Mr. Henry Price
    Chambers, Mr. Norman Campbell
    Oreskovic, Mr. Luka
    Renouf, Mrs. Peter Henry (Lillian Jefferys)
    Mannion, Miss. Margareth
    Bryhl, Mr. Kurt Arnold Gottfrid
    Ilmakangas, Miss. Pieta Sofia
    Allen, Miss. Elisabeth Walton
    Hassan, Mr. Houssein G N
    Knight, Mr. Robert J
    Berriman, Mr. William John
    Troupiansky, Mr. Moses Aaron
    Williams, Mr. Leslie
    Ford, Mrs. Edward (Margaret Ann Watson)
    Lesurer, Mr. Gustave J
    Ivanoff, Mr. Kanio
    Nankoff, Mr. Minko
    Hawksford, Mr. Walter James
    Cavendish, Mr. Tyrell William
    "Ryerson, Miss. Susan Parker ""Suzette"""
    McNamee, Mr. Neal
    Stranden, Mr. Juho
    Crosby, Capt. Edward Gifford
    Abbott, Mr. Rossmore Edward
    Sinkkonen, Miss. Anna
    Marvin, Mr. Daniel Warner
    Connaghton, Mr. Michael
    Wells, Miss. Joan
    Moor, Master. Meier
    Vande Velde, Mr. Johannes Joseph
    Jonkoff, Mr. Lalio
    Herman, Mrs. Samuel (Jane Laver)
    Hamalainen, Master. Viljo
    Carlsson, Mr. August Sigfrid
    Bailey, Mr. Percy Andrew
    Theobald, Mr. Thomas Leonard
    Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)
    Garfirth, Mr. John
    Nirva, Mr. Iisakki Antino Aijo
    Barah, Mr. Hanna Assi
    Carter, Mrs. William Ernest (Lucile Polk)
    Eklund, Mr. Hans Linus
    Hogeboom, Mrs. John C (Anna Andrews)
    Brewe, Dr. Arthur Jackson
    Mangan, Miss. Mary
    Moran, Mr. Daniel J
    Gronnestad, Mr. Daniel Danielsen
    Lievens, Mr. Rene Aime
    Jensen, Mr. Niels Peder
    Mack, Mrs. (Mary)
    Elias, Mr. Dibo
    Hocking, Mrs. Elizabeth (Eliza Needs)
    Myhrman, Mr. Pehr Fabian Oliver Malkolm
    Tobin, Mr. Roger
    Emanuel, Miss. Virginia Ethel
    Kilgannon, Mr. Thomas J
    Robert, Mrs. Edward Scott (Elisabeth Walton McMillan)
    Ayoub, Miss. Banoura
    Dick, Mrs. Albert Adrian (Vera Gillespie)
    Long, Mr. Milton Clyde
    Johnston, Mr. Andrew G
    Ali, Mr. William
    Harmer, Mr. Abraham (David Lishin)
    Sjoblom, Miss. Anna Sofia
    Rice, Master. George Hugh
    Dean, Master. Bertram Vere
    Guggenheim, Mr. Benjamin
    "Keane, Mr. Andrew ""Andy"""
    Gaskell, Mr. Alfred
    Sage, Miss. Stella Anna
    Hoyt, Mr. William Fisher
    Dantcheff, Mr. Ristiu
    Otter, Mr. Richard
    Leader, Dr. Alice (Farnham)
    Osman, Mrs. Mara
    Ibrahim Shawah, Mr. Yousseff
    Van Impe, Mrs. Jean Baptiste (Rosalie Paula Govaert)
    Ponesell, Mr. Martin
    Collyer, Mrs. Harvey (Charlotte Annie Tate)
    Carter, Master. William Thornton II
    Thomas, Master. Assad Alexander
    Hedman, Mr. Oskar Arvid
    Johansson, Mr. Karl Johan
    Andrews, Mr. Thomas Jr
    Pettersson, Miss. Ellen Natalia
    Meyer, Mr. August
    Chambers, Mrs. Norman Campbell (Bertha Griggs)
    Alexander, Mr. William
    Lester, Mr. James
    Slemen, Mr. Richard James
    Andersson, Miss. Ebba Iris Alfrida
    Tomlin, Mr. Ernest Portage
    Fry, Mr. Richard
    Heininen, Miss. Wendla Maria
    Mallet, Mr. Albert
    Holm, Mr. John Fredrik Alexander
    Skoog, Master. Karl Thorsten
    Hays, Mrs. Charles Melville (Clara Jennings Gregg)
    Lulic, Mr. Nikola
    Reuchlin, Jonkheer. John George
    Moor, Mrs. (Beila)
    Panula, Master. Urho Abraham
    Flynn, Mr. John
    Lam, Mr. Len
    Mallet, Master. Andre
    McCormack, Mr. Thomas Joseph
    Stone, Mrs. George Nelson (Martha Evelyn)
    Yasbeck, Mrs. Antoni (Selini Alexander)
    Richards, Master. George Sibley
    Saad, Mr. Amin
    Augustsson, Mr. Albert
    Allum, Mr. Owen George
    Compton, Miss. Sara Rebecca
    Pasic, Mr. Jakob
    Sirota, Mr. Maurice
    Chip, Mr. Chang
    Marechal, Mr. Pierre
    Alhomaki, Mr. Ilmari Rudolf
    Mudd, Mr. Thomas Charles
    Serepeca, Miss. Augusta
    Lemberopolous, Mr. Peter L
    Culumovic, Mr. Jeso
    Abbing, Mr. Anthony
    Sage, Mr. Douglas Bullen
    Markoff, Mr. Marin
    Harper, Rev. John
    Goldenberg, Mrs. Samuel L (Edwiga Grabowska)
    Andersson, Master. Sigvard Harald Elias
    Svensson, Mr. Johan
    Boulos, Miss. Nourelain
    Lines, Miss. Mary Conover
    Carter, Mrs. Ernest Courtenay (Lilian Hughes)
    Aks, Mrs. Sam (Leah Rosen)
    Wick, Mrs. George Dennick (Mary Hitchcock)
    Daly, Mr. Peter Denis
    Baclini, Mrs. Solomon (Latifa Qurban)
    Razi, Mr. Raihed
    Hansen, Mr. Claus Peter
    Giles, Mr. Frederick Edward
    Swift, Mrs. Frederick Joel (Margaret Welles Barron)
    "Sage, Miss. Dorothy Edith ""Dolly"""
    Gill, Mr. John William
    Bystrom, Mrs. (Karolina)
    Duran y More, Miss. Asuncion
    Roebling, Mr. Washington Augustus II
    van Melkebeke, Mr. Philemon
    Johnson, Master. Harold Theodor
    Balkic, Mr. Cerin
    Beckwith, Mrs. Richard Leonard (Sallie Monypeny)
    Carlsson, Mr. Frans Olof
    Vander Cruyssen, Mr. Victor
    Abelson, Mrs. Samuel (Hannah Wizosky)
    "Najib, Miss. Adele Kiamie ""Jane"""
    Gustafsson, Mr. Alfred Ossian
    Petroff, Mr. Nedelio
    Laleff, Mr. Kristo
    Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)
    Shelley, Mrs. William (Imanita Parrish Hall)
    Markun, Mr. Johann
    Dahlberg, Miss. Gerda Ulrika
    Banfield, Mr. Frederick James
    Sutehall, Mr. Henry Jr
    Rice, Mrs. William (Margaret Norton)
    Montvila, Rev. Juozas
    Graham, Miss. Margaret Edith
    "Johnston, Miss. Catherine Helen ""Carrie"""
    Behr, Mr. Karl Howell
    Dooley, Mr. Patrick

```python
# freqItems
# this function is meant for exploratory data analysis.
df_train.freqItems(cols=["Cabin"]).show()
```

    +--------------------+
    |     Cabin_freqItems|
    +--------------------+
    |[D, A, B, E, T, C...|
    +--------------------+

```python
# groupBy
# pandas value_counts() equivalent.
df_train.groupBy("Fare").count().orderBy("count", ascending=False).show()
```

    +------+-----+
    |  Fare|count|
    +------+-----+
    |  8.05|   43|
    |  13.0|   42|
    |7.8958|   38|
    |  7.75|   34|
    |  26.0|   31|
    |  10.5|   24|
    | 7.925|   18|
    | 7.775|   16|
    |   0.0|   15|
    |7.2292|   15|
    | 26.55|   15|
    |  7.25|   13|
    |7.8542|   13|
    |8.6625|   13|
    | 7.225|   12|
    |   9.5|    9|
    |  16.1|    9|
    | 24.15|    8|
    |  15.5|    8|
    |31.275|    7|
    +------+-----+
    only showing top 20 rows

```python
df_train.groupBy(['Sex', 'Pclass']).count().show()
```

    +------+------+-----+
    |   Sex|Pclass|count|
    +------+------+-----+
    |  male|     3|  347|
    |female|     3|  144|
    |female|     1|   94|
    |female|     2|   76|
    |  male|     2|  108|
    |  male|     1|  122|
    +------+------+-----+

```python
df_train.hint("broadcast").show()
```

    24/05/30 18:56:29 WARN HintErrorLogger: A join hint (strategy=broadcast) is specified but it is not part of a join relation.


    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group| title|family_size|family_group|is_alone|   calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|    Mr|          1|       loner|       1|              7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|   Mrs|          1|       loner|       1|           71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium|  Miss|          0|       loner|       1|             7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|   Mrs|          1|       loner|       1|              53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|    Mr|          0|       loner|       1|              8.05|       low|
    |          6|       0|     3|    Moran, Mr. James|  male|NULL|    0|    0| 8.4583|    G|       Q|         16|        short|    Mr|          0|       loner|       1|            8.4583|       low|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|51.8625|    E|       S|         23|       medium|    Mr|          0|       loner|       1|           51.8625| very_high|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1| 21.075|    F|       S|         30|       medium|Master|          4|       small|       0|           5.26875|      high|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|11.1333|    G|       S|         49|         long|   Mrs|          2|       small|       0|           5.56665|       mid|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|30.0708|    T|       C|         35|       medium|   Mrs|          1|       loner|       1|           30.0708|      high|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|   16.7|    G|       S|         31|       medium|  Miss|          2|       small|       0|              8.35|       mid|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|  26.55|    C|       S|         24|       medium|  Miss|          0|       loner|       1|             26.55|      high|
    |         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|   8.05|    G|       S|         30|       medium|    Mr|          0|       loner|       1|              8.05|       low|
    |         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5| 31.275|    T|       S|         27|       medium|    Mr|          6|       large|       0|5.2124999999999995|      high|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0| 7.8542|    G|       S|         36|         good|  Miss|          0|       loner|       1|            7.8542|       low|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|   16.0|    F|       S|         32|       medium|   Mrs|          0|       loner|       1|              16.0|       mid|
    |         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1| 29.125|    T|       Q|         20|        short|Master|          5|       large|       0|             5.825|      high|
    |         18|       1|     2|Williams, Mr. Cha...|  male|NULL|    0|    0|   13.0|    G|       S|         28|       medium|    Mr|          0|       loner|       1|              13.0|       mid|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|   18.0|    F|       S|         55|         long|   Mrs|          1|       loner|       1|              18.0|       mid|
    |         20|       1|     3|Masselmani, Mrs. ...|female|NULL|    0|    0|  7.225|    G|       C|         23|       medium|   Mrs|          0|       loner|       1|             7.225|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+------------------+----------+
    only showing top 20 rows

```python
# isStreaming
# Returns True if this DataFrame contains one or more sources that continuously return data as it arrives.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.isStreaming.html
df_train.isStreaming
```

    False

```python
# sort/orderBy
df_train.sort('Survived', ascending = False).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group| title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+---------------+----------+
    |         33|       1|     3|Glynn, Miss. Mary...|female|NULL|    0|    0|   7.75|    G|       Q|         24|       medium|  Miss|          0|       loner|       1|           7.75|       low|
    |        124|       1|     2| Webber, Miss. Susan|female|32.5|    0|    0|   13.0|    E|       S|         19|        short|  Miss|          0|       loner|       1|           13.0|       mid|
    |        108|       1|     3|Moss, Mr. Albert ...|  male|NULL|    0|    0|  7.775|    G|       S|         22|       medium|    Mr|          0|       loner|       1|          7.775|       low|
    |        299|       1|     1|Saalfeld, Mr. Ado...|  male|NULL|    0|    0|   30.5|    C|       S|         21|       medium|    Mr|          0|       loner|       1|           30.5|      high|
    |         37|       1|     3|    Mamee, Mr. Hanna|  male|NULL|    0|    0| 7.2292|    G|       C|         16|        short|    Mr|          0|       loner|       1|         7.2292|       low|
    |        129|       1|     3|   Peter, Miss. Anna|female|NULL|    1|    1|22.3583|    F|       C|         17|        short|  Miss|          2|       small|       0|       11.17915|      high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium|  Miss|          0|       loner|       1|          7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|   Mrs|          1|       loner|       1|           53.1| very_high|
    |         40|       1|     3|Nicola-Yarred, Mi...|female|14.0|    1|    0|11.2417|    G|       C|         27|       medium|  Miss|          1|       loner|       1|        11.2417|       mid|
    |        137|       1|     1|Newsom, Miss. Hel...|female|19.0|    0|    2|26.2833|    D|       S|         28|       medium|  Miss|          2|       small|       0|       13.14165|      high|
    |         57|       1|     2|   Rugg, Miss. Emily|female|21.0|    0|    0|   10.5|    G|       S|         17|        short|  Miss|          0|       loner|       1|           10.5|       mid|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|   16.7|    G|       S|         31|       medium|  Miss|          2|       small|       0|           8.35|       mid|
    |         59|       1|     2|West, Miss. Const...|female| 5.0|    1|    2|  27.75|    T|       S|         28|       medium|  Miss|          3|       small|       0|           9.25|      high|
    |        152|       1|     1|Pears, Mrs. Thoma...|female|22.0|    1|    0|   66.6|    C|       S|         33|       medium|   Mrs|          1|       loner|       1|           66.6| very_high|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|11.1333|    G|       S|         49|         long|   Mrs|          2|       small|       0|        5.56665|       mid|
    |         22|       1|     2|Beesley, Mr. Lawr...|  male|34.0|    0|    0|   13.0|    D|       S|         21|       medium|    Mr|          0|       loner|       1|           13.0|       mid|
    |         66|       1|     3|Moubarek, Master....|  male|NULL|    1|    1|15.2458|    G|       C|         24|       medium|Master|          2|       small|       0|         7.6229|       mid|
    |        167|       1|     1|Chibnall, Mrs. (E...|female|NULL|    0|    1|   55.0|    E|       S|         38|         good|   Mrs|          1|       loner|       1|           55.0| very_high|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|   16.0|    F|       S|         32|       medium|   Mrs|          0|       loner|       1|           16.0|       mid|
    |        195|       1|     1|Brown, Mrs. James...|female|44.0|    0|    0|27.7208|    B|       C|         41|         good|   Mrs|          0|       loner|       1|        27.7208|      high|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+---------------+----------+
    only showing top 20 rows

```python
# randomSplit
# randomly splits the dataframe into two based on the given weights.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html
splits = df_train.randomSplit([1.0, 2.0], seed=42)
```

```python
splits[0].count()
```

    315

```python
splits[0].show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group| title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+---------------+----------+
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|   Mrs|          1|       loner|       1|           53.1| very_high|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1| 21.075|    F|       S|         30|       medium|Master|          4|       small|       0|        5.26875|      high|
    |         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1| 29.125|    T|       Q|         20|        short|Master|          5|       large|       0|          5.825|      high|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|   18.0|    F|       S|         55|         long|   Mrs|          1|       loner|       1|           18.0|       mid|
    |         26|       1|     3|Asplund, Mrs. Car...|female|38.0|    1|    5|31.3875|    T|       S|         57|         long|   Mrs|          6|       large|       0|        5.23125|      high|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+------+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
splits[1].count()
```

    576

```python
splits[1].show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    |          6|       0|     3|    Moran, Mr. James|  male|NULL|    0|    0| 8.4583|    G|       Q|         16|        short|   Mr|          0|       loner|       1|         8.4583|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# replace
df_train.replace("male", "Man").show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|   Man|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|  Mrs|          1|       loner|       1|           53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|   Man|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# similarly
df_train.na.replace("male", "Man").show(5)
```

    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|   Fare|Cabin|Embarked|name_length|nLength_group|title|family_size|family_group|is_alone|calculated_fare|fare_group|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    |          1|       0|     3|Braund, Mr. Owen ...|   Man|22.0|    1|    0|   7.25|    G|       S|         23|       medium|   Mr|          1|       loner|       1|           7.25|       low|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|71.2833|    C|       C|         51|         long|  Mrs|          1|       loner|       1|        71.2833| very_high|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|  7.925|    G|       S|         22|       medium| Miss|          0|       loner|       1|          7.925|       low|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|   53.1|    C|       S|         44|         good|  Mrs|          1|       loner|       1|           53.1| very_high|
    |          5|       0|     3|Allen, Mr. Willia...|   Man|35.0|    0|    0|   8.05|    G|       S|         24|       medium|   Mr|          0|       loner|       1|           8.05|       low|
    +-----------+--------+------+--------------------+------+----+-----+-----+-------+-----+--------+-----------+-------------+-----+-----------+------------+--------+---------------+----------+
    only showing top 5 rows

```python
# cube
# the following stack overflow explains cube better than official spark page.
# https://stackoverflow.com/questions/37975227/what-is-the-difference-between-cube-rollup-and-groupby-operators
df = spark.createDataFrame([("foo", 1), ("foo", 2), ("bar", 2), ("bar", 2)]).toDF("x", "y")
df.show()
```

    +---+---+
    |  x|  y|
    +---+---+
    |foo|  1|
    |foo|  2|
    |bar|  2|
    |bar|  2|
    +---+---+

```python
temp_df.show()
```

    +-----------+--------+------+--------------------+----+----+-----+-----+---------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name| Sex| Age|SibSp|Parch|         Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+----+----+-----+-----+---------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|male|22.0|    1|    0|      A/5 21171|   7.25| NULL|       S|
    |          5|       0|     3|Allen, Mr. Willia...|male|35.0|    0|    0|         373450|   8.05| NULL|       S|
    |          6|       0|     3|    Moran, Mr. James|male|NULL|    0|    0|         330877| 8.4583| NULL|       Q|
    |          8|       0|     3|Palsson, Master. ...|male| 2.0|    3|    1|         349909| 21.075| NULL|       S|
    |         13|       0|     3|Saundercock, Mr. ...|male|20.0|    0|    0|      A/5. 2151|   8.05| NULL|       S|
    |         14|       0|     3|Andersson, Mr. An...|male|39.0|    1|    5|         347082| 31.275| NULL|       S|
    |         17|       0|     3|Rice, Master. Eugene|male| 2.0|    4|    1|         382652| 29.125| NULL|       Q|
    |         18|       1|     2|Williams, Mr. Cha...|male|NULL|    0|    0|         244373|   13.0| NULL|       S|
    |         21|       0|     2|Fynney, Mr. Joseph J|male|35.0|    0|    0|         239865|   26.0| NULL|       S|
    |         22|       1|     2|Beesley, Mr. Lawr...|male|34.0|    0|    0|         248698|   13.0|  D56|       S|
    |         27|       0|     3|Emir, Mr. Farred ...|male|NULL|    0|    0|           2631|  7.225| NULL|       C|
    |         30|       0|     3| Todoroff, Mr. Lalio|male|NULL|    0|    0|         349216| 7.8958| NULL|       S|
    |         31|       0|     1|Uruchurtu, Don. M...|male|40.0|    0|    0|       PC 17601|27.7208| NULL|       C|
    |         34|       0|     2|Wheadon, Mr. Edwa...|male|66.0|    0|    0|     C.A. 24579|   10.5| NULL|       S|
    |         37|       1|     3|    Mamee, Mr. Hanna|male|NULL|    0|    0|           2677| 7.2292| NULL|       C|
    |         38|       0|     3|Cann, Mr. Ernest ...|male|21.0|    0|    0|     A./5. 2152|   8.05| NULL|       S|
    |         43|       0|     3| Kraeff, Mr. Theodor|male|NULL|    0|    0|         349253| 7.8958| NULL|       C|
    |         46|       0|     3|Rogers, Mr. Willi...|male|NULL|    0|    0|S.C./A.4. 23567|   8.05| NULL|       S|
    |         47|       0|     3|   Lennon, Mr. Denis|male|NULL|    1|    0|         370371|   15.5| NULL|       Q|
    |         49|       0|     3| Samaan, Mr. Youssef|male|NULL|    2|    0|           2662|21.6792| NULL|       C|
    +-----------+--------+------+--------------------+----+----+-----+-----+---------------+-------+-----+--------+
    only showing top 20 rows

```python
df.cube("x", "y").count().show()
```

    +----+----+-----+
    |   x|   y|count|
    +----+----+-----+
    | foo|   1|    1|
    |NULL|NULL|    4|
    |NULL|   1|    1|
    | foo|NULL|    2|
    | foo|   2|    1|
    |NULL|   2|    3|
    | bar|   2|    2|
    | bar|NULL|    2|
    +----+----+-----+

Here is what cube returns

````
// +----+----+-----+
// |   x|   y|count|
// +----+----+-----+
// | foo|   1|    1|   <- count of records where x = foo AND y = 1
// | foo|   2|    1|   <- count of records where x = foo AND y = 2
// | bar|   2|    2|   <- count of records where x = bar AND y = 2
// |null|null|    4|   <- total count of records
// |null|   2|    3|   <- count of records where y = 2
// |null|   1|    1|   <- count of records where y = 1
// | bar|null|    2|   <- count of records where x = bar
// | foo|null|    2|   <- count of records where x = foo
// +----+----+-----+```


```python
# rollup
df.rollup("x", "y").count().show()
````

    +----+----+-----+
    |   x|   y|count|
    +----+----+-----+
    | foo|   1|    1|
    |NULL|NULL|    4|
    | foo|NULL|    2|
    | foo|   2|    1|
    | bar|   2|    2|
    | bar|NULL|    2|
    +----+----+-----+

Here is what rollup's look like

````
// +----+----+-----+
// |   x|   y|count|
// +----+----+-----+
// | foo|null|    2|   <- count where x is fixed to foo
// | bar|   2|    2|   <- count where x is fixed to bar and y is fixed to  2
// | foo|   1|    1|   ...
// | foo|   2|    1|   ...
// |null|null|    4|   <- count where no column is fixed
// | bar|null|    2|   <- count where x is fixed to bar
// +----+----+-----+```


```python
# sameSemantics
# Returns True when the logical query plans inside both DataFrames are equal and therefore return same results.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.sameSemantics.html
df1 = spark.range(10)
df2 = spark.range(10)
````

```python
df1.show()
```

    +---+
    | id|
    +---+
    |  0|
    |  1|
    |  2|
    |  3|
    |  4|
    |  5|
    |  6|
    |  7|
    |  8|
    |  9|
    +---+

```python
df2.show()
```

    +---+
    | id|
    +---+
    |  0|
    |  1|
    |  2|
    |  3|
    |  4|
    |  5|
    |  6|
    |  7|
    |  8|
    |  9|
    +---+

```python
df1.withColumn("col1", df1.id * 2).sameSemantics(df2.withColumn("col1", df2.id * 2))
```

    True

```python
df1.withColumn("col1", df1.id * 2).sameSemantics(df2.withColumn("col1", df2.id + 2))
```

    False

```python
df1.withColumn("col1", df1.id * 2).sameSemantics(df2.withColumn("col0", df2.id * 2))
```

    True

```python
df_train.schema
```

    StructType([StructField('PassengerId', IntegerType(), True), StructField('Survived', IntegerType(), True), StructField('Pclass', IntegerType(), True), StructField('Name', StringType(), True), StructField('Sex', StringType(), True), StructField('Age', DoubleType(), True), StructField('SibSp', IntegerType(), True), StructField('Parch', IntegerType(), True), StructField('Fare', DoubleType(), True), StructField('Cabin', StringType(), True), StructField('Embarked', StringType(), False), StructField('name_length', IntegerType(), True), StructField('nLength_group', StringType(), True), StructField('title', StringType(), True), StructField('family_size', IntegerType(), True), StructField('family_group', StringType(), True), StructField('is_alone', IntegerType(), True), StructField('calculated_fare', DoubleType(), True), StructField('fare_group', StringType(), True)])

```python
df_train.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = false)
     |-- name_length: integer (nullable = true)
     |-- nLength_group: string (nullable = true)
     |-- title: string (nullable = true)
     |-- family_size: integer (nullable = true)
     |-- family_group: string (nullable = true)
     |-- is_alone: integer (nullable = true)
     |-- calculated_fare: double (nullable = true)
     |-- fare_group: string (nullable = true)
