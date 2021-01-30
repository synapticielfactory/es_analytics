
# Part 5: Predicting Next Purchase Day

Most of the actions we explained in Data Driven Growth series have the same mentality behind:

**Treat your customers in a way they deserve before they expect that (e.g., LTV prediction) and act before something bad happens (e.g., churn)****.**

Predictive analytics helps us a lot on this one. One of the many opportunities it can provide is predicting the next purchase day of the customer. What if you know if a customer is likely to make another purchase in 7 days?

We can build our strategy on top of that and come up with lots of tactical actions like:

-   No promotional offer to this customer since s/he will make a purchase anyways
-   Nudge the customer with inbound marketing if there is no purchase in the predicted time window (or fire the guy who did the prediction 🦹‍♀️ 🦹‍♂️ )

In this article, we will be using  [online retail dataset](https://github.com/synapticielfactory/eland_es_analytics/blob/master/invoices.7z)  and follow the steps below:

-   Data Wrangling (creating previous/next datasets and calculate purchase day differences)
-   Feature Engineering
-   Selecting a Machine Learning Model
-   Multi-Classification Model
-   Hyperparameter Tuning

## Data Wrangling

Let’s start with importing our data and do the preliminary data work:

**Importing CSV file and date field transformation**

We have imported the CSV file, converted the date field from string to DateTime to make it workable like we did in [this article](https://github.com/synapticielfactory/eland_es_analytics)

```python
import eland as ed
import pandas as pd
import matplotlib.pyplot as plt

# import elasticsearch-py client
from elasticsearch import Elasticsearch

# Function for pretty-printing JSON
def json(raw):
    import json
    print(json.dumps(raw, indent=2, sort_keys=True))


# Connect to an Elasticsearch instance
# here we use the official Elastic Python client
# check it on https://github.com/elastic/elasticsearch-py
es = Elasticsearch(
  ['http://localhost:9200'],
  http_auth=("es_kbn", "changeme")
)
# print the connection object info (same as visiting http://localhost:9200)
# make sure your elasticsearch node/cluster respond to requests
json(es.info())

# Load the dataset from the local csv file of call logs
pd_df = pd.read_csv("./invoices.csv", sep=';', encoding = 'unicode_escape')
pd_df.info()

#converting the type of Invoice Date Field from string to datetime.
pd_df['invoice_date'] = pd.to_datetime(pd_df['invoice_date'])

# Arrange prices for phones
pd_df['unit_price'] = pd_df['unit_price'] * 10.00

# Rename the columns to be snake_case
pd_df.columns = [x.lower().replace(" ", "_") for x in pd_df.columns]

# Combine the 'latitude' and 'longitude' columns into one column 'location' for 'geo_point'
pd_df["country_location"] = pd_df[["country_latitude", "country_longitude"]].apply(lambda x: ",".join(str(item) for item in x), axis=1)

# Drop the old columns in favor of 'location'
pd_df.drop(["country_latitude", "country_longitude"], axis=1, inplace=True)

# Load the data into elasticsearch
ed_df = ed.pandas_to_eland(
    pd_df=pd_df,
    es_client=es,

    # Where the data will live in Elasticsearch
    es_dest_index="es-invoices",

    # Type overrides for certain columns, this can be used to customize index mapping before ingest
    es_type_overrides={
        "invoice_id": "keyword",
        "item_id": "keyword",
        "item_model": "keyword",
        "item_name": "keyword",     
        "item_brand": "keyword",
        "item_vendor": "keyword",   
        "order_qty": "integer",
        "invoice_date": "date",
        "unit_price": "float",  
        "customer_id": "keyword",
        "country_name": "keyword",
        "country_location": "geo_point"  
    },

    # If the index already exists what should we do?
    es_if_exists="replace",

    # Wait for data to be indexed before returning
    es_refresh=True,
)
```

To build our model, we should split our data into two parts:

![Image for post](/src-5/capture_01.png)

**Data structure for training the model**

![Image for post](/src-5/capture_02.png)

We use six months of behavioral data to predict customers’ first purchase date in the next three months. If there is no purchase, we will predict that too. Let’s assume our cut off date is June 1st ’19 and split the data:
```json
POST _reindex/?wait_for_completion=false
{
  "source": {
    "index": "es-invoices",
    "query": {
      "range": {
        "invoice_date": {
          "gte": "2019-06-01||-6M/d",
          "lt": "2019-06-01"
        }
      }
    }
  },
  "dest": {
    "index": "es-invoices-6m"
  }
}


POST _reindex/?wait_for_completion=false
{
  "source": {
    "index": "es-invoices",
    "query": {
      "range": {
        "invoice_date": {
          "gte": "2019-06-01",
          "lt": "2019-06-01||+3M/d"
        }
      }
    }
  },
  "dest": {
    "index": "es-invoices-3m"
  }
}


```
**es-invoices-6m**  represents the six months performance whereas we will use  **es-invoices-3m** for the find out the days between the last purchase date in es-invoices-6m and the first one in es-invoices-3m.

Also, we will create a dataframe called  **tx_user**  to possess a user-level feature set for the prediction model:
```
tx_user = pd.DataFrame(tx_6m['CustomerID'].unique())  
tx_user.columns = ['CustomerID']
```
By using the data in tx_next, we need the calculate our  **label**  (days between last purchase before cut off date and first purchase after that):

First, we start with Elastic queries :
```json
PUT _transform/es-customers-6m
{
  "id": "es-customers-6m",
  "source": {
    "index": [
      "es-invoices-6m"
    ]
  },
  "dest": {
    "index": "es-customers-6m"
  },
  "pivot": {
    "group_by": {
      "customer_id": {
        "terms": {
          "field": "customer_id"
        }
      }
    },
    "aggregations": {
      "max_purchase_date":{
        "max":{
          "field": "invoice_date"
        }
      }
    }
  }
}
POST _transform/es-customers-6m/_start

PUT _transform/es-customers-3m
{
  "id": "es-customers-3m",
  "source": {
    "index": [
      "es-invoices-3m"
    ]
  },
  "dest": {
    "index": "es-customers-3m"
  },
  "pivot": {
    "group_by": {
      "customer_id": {
        "terms": {
          "field": "customer_id"
        }
      }
    },
    "aggregations": {
      "min_purchase_date":{
        "min":{
          "field": "invoice_date"
        }
      }
    }
  }
}
POST _transform/es-customers-3m/_start
```
Then, we continue with Eland to make our labels:
```python
tx_6m = ed.DataFrame(es, es_index_pattern="es-invoices-6m").to_pandas()

tx_user = pd.DataFrame(tx_6m['customer_id'].unique())
tx_user.columns = ['customer_id']

#create a dataframe with customer id and first purchase date in tx_next
tx_next_first_purchase = ed.DataFrame(es, es_index_pattern="es-customers-3m").to_pandas()

#create a dataframe with customer id and last purchase date in tx_6m
tx_last_purchase = ed.DataFrame(es, es_index_pattern="es-customers-6m").to_pandas()

#merge two dataframes
tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='customer_id',how='left')

#calculate the time difference in days:
tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['min_purchase_date'] - tx_purchase_dates['max_purchase_date']).dt.days

#merge with tx_user 
tx_user = pd.merge(tx_user, tx_purchase_dates[['customer_id','NextPurchaseDay']],on='customer_id',how='left')

#print tx_user
tx_user.head()
```

Now, tx_user looks like below:

![Image for post](/src-5/capture_03.png)


As you can easily notice, we have NaN values because those customers haven’t made any purchase yet. We fill NaN with 999 to quickly identify them later.

```python
#fill NA values with 999
tx_user = tx_user.fillna(999)
```

We have customer ids and corresponding labels in a dataframe. Let’s enrich it with our feature set to build our machine learning model.

## Feature Engineering

For this project, we have selected our feature candidates like below:

-   RFM scores & clusters
-   Days between the last three purchases
-   Mean & standard deviation of the difference between purchases in days

After adding these features, we need to deal with the categorical features by applying  segmentation pipeline.

For RFM, to not repeat  [Part 2](https://github.com/synapticielfactory/es_analytics/blob/master/README-2.md), we share the code block and move forward:

**RFM Scores & Clustering**

Let’s focus on how we can add the next two features. We will be using  **shift()** method a lot in this part.

First, we create a dataframe with Customer ID and Invoice Day (not datetime). Then we will remove the duplicates since customers can do multiple purchases in a day and difference will become 0 for those.
```python
#create a dataframe with CustomerID and Invoice Date  
tx_day_order = tx_6m[['CustomerID','InvoiceDate']]

#convert Invoice Datetime to day  
tx_day_order['InvoiceDay'] = tx_6m['InvoiceDate'].dt.date
tx_day_order = tx_day_order.sort_values(['CustomerID','InvoiceDate'])

#drop duplicates  
tx_day_order = tx_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')
```
Next, by using shift, we create new columns with the dates of last 3 purchases and see how our dataframe looks like:
```python
#shifting last 3 purchase dates
tx_day_order['prev_invoice_date'] = tx_day_order.groupby('customer_id')['invoice_day'].shift(1)
tx_day_order['t2_invoice_date'] = tx_day_order.groupby('customer_id')['invoice_day'].shift(2)
tx_day_order['t3_invoice_date'] = tx_day_order.groupby('customer_id')['invoice_day'].shift(3)
tx_day_order
```
Output:

![Image for post](/src-5/capture_04.png)

Let’s begin calculating the difference in days for each invoice date:
```python
tx_day_order['day_diff'] = (tx_day_order['invoice_day'] - tx_day_order['prev_invoice_date']).dt.days
tx_day_order['day_diff_2'] = (tx_day_order['invoice_day'] - tx_day_order['t2_invoice_date']).dt.days
tx_day_order['day_diff_3'] = (tx_day_order['invoice_day'] - tx_day_order['t3_invoice_date']).dt.days
tx_day_order
```
Output:

![Image for post](/src-5/capture_05.png)

For each customer ID, we utilize  **.agg()** method to find out the mean and standard deviation of the difference between purchases in days:
```python
tx_day_diff = tx_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
tx_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']
```
Now we are going to make a tough decision. The calculation above is quite useful for customers who have many purchases. But we can’t say the same for the ones with 1–2 purchases. For instance, it is too early to tag a customer as  **_frequent_** who has only 2 purchases but back to back.

We only keep customers who have > 3 purchases by using the following line:
```python
tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerID'],keep='last')
```
Finally, we drop NA values, merge new dataframes with tx_user and let machine learning module take charge of converting categorical values:
```python
tx_day_order_last = tx_day_order_last.dropna()
tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='CustomerID')
tx_user = pd.merge(tx_user, tx_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')
```

## Selecting a Machine Learning Model

Before jumping into choosing the model, we need to take two actions. First, we need to identify the classes in our label. Generally, percentiles give the right for that. Let’s use  the **percentiles** aggregation on  **NextPurchaseDay:**

```json
GET es-users/_search
{
  "size": 0,
  "aggs": {
    "describe": {
      "percentiles": {
        "field": "next_purchase_day"
      }
    }
  }
}
```

![Image for post](/src-5/capture_06.png)

Deciding the boundaries is a question for both statistics and business needs. It should make sense in terms of the first one and be easy to take action and communicate. Considering these two, we will have three classes:

-   0–64: Customers that will purchase in 0–64 days —  **Class name: 2**
-   65–196: Customers that will purchase in 65–196 days —  **Class name: 1**
-   ≥ 197: Customers that will purchase in more than 197 days —  **Class name: 0**
```json
PUT /_ingest/pipeline/npd_segmentation
{
  "description": "",
  "processors": [
    {
      "script": {
        "lang": "painless",
        "source": "if (ctx.next_purchase_day>=197) { ctx.next_purchase_day_range=2; } else if (ctx.next_purchase_day>=65) { ctx.next_purchase_day_range=1; } else { ctx.next_purchase_day_range=0; }"
      }
    }
  ]
}

POST es-users/_update_by_query?pipeline=npd_segmentation
{
  "query": {
    "match_all": {}
  }
}
```
The last step is to see the correlation between our features and label. The  **correlation matrix** is one of the cleanest ways to show this [vega visualization](/src-5/vega_visualization.json). :

![Image for post](/src-5/capture_07.png)

Looks like  **Overall Score** has the highest positive correlation and  **Recency**  has the highest negative.

For this particular problem, we want to use the model which gives the highest accuracy. Let’s split train and test tests and measure the accuracy of our model.

## Multi-Classification Model


To fit Data frame analytics to our data, we should prepare features (X) and label(y) sets and do the train & test split.

To do so, let's create new job :
```json
PUT _ml/data_frame/analytics/next_purchase_analysis
{
  "description": "",
  "source": {
    "index": "es-users",
    "query": {
      "match_all": {}
    }
  },
  "dest": {
    "index": "ml-users"
  },
  "analyzed_fields": {
    "includes": [
      "count_orders",
      "customer_id",
      "day_diff",
      "day_diff_2",
      "day_diff_3",
      "day_diff_mean",
      "day_diff_std",
      "frequency_cluster",
      "next_purchase_day_range",
      "overall_score",
      "recency",
      "recency_cluster",
      "revenue_cluster",
      "total_revenue"
    ]
  },
  "analysis": {
    "classification": {
      "dependent_variable": "next_purchase_day_range",
      "num_top_feature_importance_values": 0,
      "training_percent": 95,
      "num_top_classes": 3
    }
  },
  "model_memory_limit": "22mb",
  "max_num_threads": 1
}
```

To start the analysis we use the following command :

```
POST _ml/data_frame/analytics/next_purchase_analysis/_start
```

also, we can use automatic [feature encoding](https://www.elastic.co/guide/en/machine-learning/7.10/ml-feature-encoding.html) for all categorical features which allows to process our features as we did in **Feature Engineering** section. However, it is allowed to define costume Feature Labelling.

query for accuracy :
```json
POST ml-users/_search?size=0
{
  "query": {
    "match_all": {}
  },
  "aggs": {
    "training_accuracy": {
      "scripted_metric": {
        "init_script": "state.samples = []", 
        "map_script": "if(doc['ml.is_training'].value == true){state.samples.add(doc['ml.next_purchase_day_range_prediction'].value == doc.next_purchase_day_range.value ? 1 : 0)}",
        "combine_script": "double correct = 0; double total = 0; for (t in state.samples) { total++; correct += t } return correct/total",
        "reduce_script": "double accuracy = 0; for (a in states) { accuracy += a } return accuracy"
      }
    },
    "test_accuracy": {
      "scripted_metric": {
        "init_script": "state.samples = []", 
        "map_script": "if(doc['ml.is_training'].value == false){state.samples.add(doc['ml.next_purchase_day_range_prediction'].value == doc.next_purchase_day_range.value ? 1 : 0)}",
        "combine_script": "double correct = 0; double total = 0; for (t in state.samples) { total++; correct += t } return correct/total",
        "reduce_script": "double accuracy = 0; for (a in states) { accuracy += a } return accuracy"
      }
    }
  }
}
```
![Image for post](/src-5/capture_08.png)

In this version, our accuracy on the test set is 71%:

You can find the Jupyter Notebook, elastic requests, and vega visualization for this article  [here](/src-5/).
