# OLIST

# SCOPE
You’ve just joined Olist as a Junior Data Analyst on the Sales Optimization team. Olist is a major e-commerce platform in Brazil, known for connecting businesses with customers online. To stay ahead in a competitive market, Olist needs to enhance customer experience, optimize seller performance, improve logistics, and maximize marketing effectiveness. You’ve been given historical orders, customers, products, sellers and geolocation data. Your goal is to uncover data-driven insights that support these key focus areas—such as analyzing buyer behavior, identifying top-performing products, evaluating seller KPIs, optimizing inventory, and measuring campaign ROI. But first,Olist executives must approve your recommendations, so they must be backed up with compeling data insights and professional data visualizations. 

## Project Objective
The project aims to ensure the objectives of the business analysis project for sales management—such as optimizing the sales process, increasing revenue and profit, gaining deeper customer insights, enhancing advertising effectiveness, improving user experience, optimizing inventory management, and forecasting market trends. Key objectives include:
-**Identify potential issues**: Propose improvements that enhance performance and reduce operational costs.
-**Analyze customer purchasing behavior** : Include purchase frequency, average order value, and popular product categories.
-**Evaluate sales performance**: Assess overall revenue, revenue by individual products or product groups, and revenue over time (daily, weekly, monthly).
-**Enhance customer experience**: Analyzing customer service metrics such as response time, satisfaction rate, and the number of complaints.

## About the company
OLIST is a prominent e-commerce platform in Brazil, offering solutions that support businesses in selling online and managing their e-commerce operations. As an integrated multi-channel platform, OLIST enables businesses to manage and sell products across multiple e-commerce marketplaces simultaneously. This includes major platforms in Brazil such as Mercado Livre, Americanas, and others, helping businesses expand their customer reach.
Moreover, the platform facilitates product listing and inventory management by allowing users to easily upload and manage products across different sales channels from a single interface. OLIST also provides tools to synchronize product information and track inventory in real time.

## Dataset

- Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers, a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.
Here's the Entity-Relationship Diagram of the resulting SQLite database:
<center>
      <img src="schema.png" width="900" />
  </center>

# DATA LOADING

  ```python
import pandas as pd
import numpy as np
from datetime import timedelta
```
```python
orders = pd.read_csv("/content/drive/MyDrive/archive/olist_orders_dataset.csv")
order_items = pd.read_csv("/content/drive/MyDrive/archive/olist_order_items_dataset.csv")
payments = pd.read_csv("/content/drive/MyDrive/archive/olist_order_payments_dataset.csv")
reviews = pd.read_csv("/content/drive/MyDrive/archive/olist_order_reviews_dataset.csv")
customers = pd.read_csv("/content/drive/MyDrive/archive/olist_customers_dataset.csv")
sellers = pd.read_csv("/content/drive/MyDrive/archive/olist_sellers_dataset.csv")
products = pd.read_csv("/content/drive/MyDrive/archive/olist_products_dataset.csv")
geolocation = pd.read_csv("/content/drive/MyDrive/archive/olist_geolocation_dataset.csv")
categories = pd.read_csv("/content/drive/MyDrive/archive/product_category_name_translation.csv")

dfs = {
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "reviews": reviews,
    "customers": customers,
    "sellers": sellers,
    "products": products,
    "geolocation": geolocation,
    "categories": categories
}

for name, df in dfs.items():
    print(f"\n\n  ---  for table «{name.upper()}» shape is {df.shape}  ---\n")
    print(f"First 2 rows:\n{df.head(2)}\n")
    print(f"Column Names:\n{df.columns.tolist()}\n")
    print(f"Data Types:\n{df.dtypes}")
```
```python
df.info()
```
# DATA CLEANING

## Missing Values:
- Check the missing values. columns that are missing by more than 50 percent will be considered to be removed.
```python
def missing_data_summary(df, threshold=0):
    """
    Summarizes missing data, showing count and percentage of missing values for each column.
    Filters columns based on a missing percentage threshold.
    
    Parameters:
        df (pd.DataFrame): The dataframe to analyze.
        threshold (float): The minimum percentage of missing data to include in the summary.
    
    Returns:
        pd.DataFrame: A summary of missing data.
    """
    return (pd.DataFrame(df.isna().sum())
            .reset_index()
            .rename(columns={'index': 'Column', 0: 'mis_count'})
            .query('mis_count > 0')  # Only include columns with missing values
            .assign(Missing_Percentage=lambda x: x['mis_count'] / df.shape[0] * 100)
            .query(f'Missing_Percentage > {threshold}')  # Filter by threshold
            .sort_values('mis_count', ascending=False)
            .reset_index(drop=True))

missing_data_summaries = {}

for name, df in dfs.items():
    missing_data_summaries[name] = missing_data_summary(df,51)

# Print the results for each DataFrame
for name, summary in missing_data_summaries.items():
    print(f"Missing data summary for {name}:")
    print(summary)
    print("\n")
```
```python
reviews = reviews.drop('review_comment_title', axis=1)
```
```python
for name, df in dfs.items():
    print(f"\n{name.upper()} - NULL values:")
    print(df.isnull().sum(),'\n')
```

1. Handle missing data in **ORDERS** table:

```python
date_cols = ['order_purchase_timestamp', 'order_approved_at', 
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')
```

- Column **order_approved_at**

```python
orders[orders["order_approved_at"].isnull()]["order_status"].value_counts()
```
```python
orders[orders["order_approved_at"].isnull() & (orders["order_status"] == "delivered")]["order_id"].values
```
```python
payments[payments["order_id"].isin(['e04abd8149ef81b95221e88f6ed9ab6a',
       '8a9adc69528e1001fc68dd0aaebbb54a',
       '7013bcfc1c97fe719a7b5e05e61c12db',
       '5cf925b116421afa85ee25e99b4c34fb',
       '12a95a3c06dbaec84bcfb0e2da5d228a',
       'c1d4211b3dae76144deccd6c74144a88',
       'd69e5d356402adc8cf17e08b5033acfb',
       'd77031d6a3c8a52f019764e68f211c69',
       '7002a78c79c519ac54022d4f8a65e6e8',
       '2eecb0d85f281280f79fa00f9cec1a95',
       '51eb2eebd5d76a24625b31c33dd41449',
       '88083e8f64d95b932164187484d90212',
       '3c0b8706b065f9919d0505d3b3343881',
       '2babbb4b15e6d2dfe95e2de765c97bce'])]
```
```python
# Calculate average time from purchase to approval
avg_approval_time = (orders['order_approved_at'] - orders['order_purchase_timestamp']).mean()

# Impute missing approval times with purchase time + average
orders['order_approved_at'] = orders['order_approved_at'].fillna(
    orders['order_purchase_timestamp'] + avg_approval_time
)
```

- Column **order_delivered_carrier_date**

```python
orders[orders["order_delivered_carrier_date"].isnull()]["order_status"].value_counts()
```
```python
orders[(orders["order_delivered_carrier_date"].isnull()) & (orders["order_status"] == "delivered")]
```
```python
orders = orders[orders['order_id'] != '2d858f451373b04fb5c984a1cc2defaf']
delivered1 = orders[orders['order_status'] == 'delivered']
avg_approval_to_carrier = (pd.to_datetime(delivered['order_delivered_carrier_date']) - 
                          pd.to_datetime(delivered['order_approved_at'])).mean()
orders['order_delivered_carrier_date'].fillna(pd.to_datetime(orders['order_approved_at']) + avg_approval_to_carrier, inplace=True)
```

- Column **order_delivered_carrier_date**

```python
orders[orders["order_delivered_customer_date"].isnull()]["order_status"].value_counts()
```
```python
orders[(orders["order_delivered_customer_date"].isnull()) & (orders["order_status"] == "delivered")]
```
```python
delivered2 = orders[orders['order_status'] == 'delivered']
avg_carrier_to_delivery = (pd.to_datetime(delivered['order_delivered_customer_date']) - 
                          pd.to_datetime(delivered['order_delivered_carrier_date'])).mean()
orders['order_delivered_customer_date'].fillna(pd.to_datetime(orders['order_delivered_carrier_date']) + avg_carrier_to_delivery, inplace=True)
```
```
orders.isnull().sum()
```

2. Handle missing data in **PRODUCTS** table:

```python
products = products.drop(['product_name_lenght', 'product_description_lenght', 'product_photos_qty'], axis=1)
products['product_category_name'] = products['product_category_name'].fillna('unknown')
products['product_weight_g'] = products['product_weight_g'].fillna(products['product_weight_g'].median())
products['product_length_cm'] = products['product_length_cm'].fillna(products['product_length_cm'].median())
products['product_height_cm'] = products['product_height_cm'].fillna(products['product_height_cm'].median())
products['product_width_cm'] = products['product_width_cm'].fillna(products['product_width_cm'].median())
```
```
products.isnull().sum()
```
## Duplicates:

```python
for name, df in dfs.items():
    print(f"\n{name.upper()} - Duplicates: {df.duplicated().sum()}")
```
```python
geolocation.duplicated().value_counts()
```
```python
geolocation = geolocation.drop_duplicates()
geolocation.duplicated().sum()
```

## Merge Dataframe

```python
df = orders.merge(order_items, on='order_id', how='inner')
df = df.merge(payments, on='order_id', how='inner', validate='m:m')
df = df.merge(reviews, on='order_id', how='inner')
df = df.merge(products, on='product_id', how='inner')
df = df.merge(customers, on='customer_id', how='inner')
df = df.merge(sellers, on='seller_id', how='inner')
df = df.merge(categories, on='product_category_name', how='inner')
df.info()
```
```python
# Create useful features from order_purchase_timestamp
df['day_of_week_int'] = df['order_purchase_timestamp'].dt.weekday + 1  # Day of week as integer (1 = Monday, etc.)
df['hour'] = df['order_purchase_timestamp'].dt.hour                    # Hour of day
df['month'] = df['order_purchase_timestamp'].dt.month                  # Month as integer
df['year'] = df['order_purchase_timestamp'].dt.year                    # Year as integer
df['date'] = df['order_purchase_timestamp'].dt.to_period('M')          # Monthly period for time series analysis

# Calculate delivery time in days
df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
```
```python
geo_avg = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()

df = df.merge(geo_avg, how='left', left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')

df.isna().sum()
```
```python
median_coords = df.groupby('customer_city')[['geolocation_lat', 'geolocation_lng']].transform('median')
df[['geolocation_lat', 'geolocation_lng']] = df[['geolocation_lat', 'geolocation_lng']].fillna(median_coords)
df.isna().sum()
```
```python
median_lat = df['geolocation_lat'].median()
median_lng = df['geolocation_lng'].median()

df['geolocation_lat'] = df['geolocation_lat'].fillna(median_lat)
df['geolocation_lng'] = df['geolocation_lng'].fillna(median_lng)
df.isnull().sum()
```
```python
df.duplicated().sum()
```

# EXPLORATORY DATA ANALYSIS (EDA)



  
