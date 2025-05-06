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
import os
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
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

## Initial Data Overview 

```python
df.info()
```
```python
df.nunique()
```
```python
df['price'].describe()
```
```python
df['payment_value'].describe()
```
```python
df['payment_type'].value_counts()
```
```python
df['review_score'].value_counts()
```
```python
print(df['order_purchase_timestamp'].min(), df['order_purchase_timestamp'].max())
```
```python
# Select relevant features for correlation analysis
correlation_features = df[['price', 'review_score', 'delivery_time',
                            'payment_value', 'freight_value',
                            'payment_installments', 'order_item_id',
                            'hour', 'month']]

# Calculate the correlation matrix
correlation = correlation_features.corr()

# Define the custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ['green', 'white', 'darkblue'])

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap=custom_cmap, vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```

## Analyze customer purchasing behavior

### Monthly sales trend
```python
# Monthly sales trend
monthly_sales = df.groupby(df['order_purchase_timestamp'].dt.to_period('M')).agg({'price': 'sum'})

# Set the figure size for better visibility
plt.figure(figsize=(12, 6))

# Plot the data with custom colors
plt.plot(monthly_sales.index.astype(str), monthly_sales['price'],
         marker='o', linestyle='-', color='blue', linewidth=2)

# Adding titles and labels
plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)

# Customize the ticks
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Add gridlines for better readability
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
```
- From **September 2016** to around **mid-2018**, there's a clear **upward trend** in **total sales**, indicating **consistent growth** over time.
- The **highest peak** occurs around **November 2017**, where **sales** exceed **1 million units**. This could be linked to **seasonal events** like **Black Friday** or **holiday shopping**.
- There's a **dramatic drop** in sales in **September 2018**. Possible reason is **missing data**.

### The frequency of orders per day

```python
# Group by timestamp and aggregate order counts
daily_orders = df.groupby(df['order_purchase_timestamp'].dt.floor('D'))['order_id'].count().reset_index()

# Convert 'order_purchase_timestamp' to matplotlib dates
dates = mdates.date2num(daily_orders['order_purchase_timestamp'])

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates, daily_orders['order_id'],color='blue')

# Format x-axis to display dates
date_format = mdates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis_date()

# Set labels and title
plt.ylabel('Number of orders')
plt.title('Number of orders per day', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=14)

# Add grid and show the plot
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```
- There's a **large spike** of **orders** around **Christmas**, specifically on the **24th of December**.
- The **number of orders** also seems to be **increasing steadily over time** as Olist's business grows. Also notice how there's very **few data** at the **start and end** of the **timespan** covered by the dataset, so I'll exclude these dates from some of my next queries.

### Heatmap of order volume by Day of Week and Hour of Day
How are orders distributed over the week? Are more orders placed on weekends? How about during the day? Do customers place more orders in the evening? We can answer these questions using a heatmap that plots the distribution of both on a grid. 
To plot a heatmap we need a matrix of counts of orders, where rows represent days of the week and columns represent hours of the day:
```python
# Convert day_of_week_int to day names
day_map = {
    1: 'Mon',
    2: 'Tue',
    3: 'Wed',
    4: 'Thu',
    5: 'Fri',
    6: 'Sat',
    7: 'Sun'
}

df['day_of_week_name'] = df['day_of_week_int'].map(day_map)

# Create a pivot table: count number of orders by day and hour
pivot_df = df.pivot_table(
    index='day_of_week_name',
    columns='hour',
    values='order_id',
    aggfunc='count',
    fill_value=0
)

# Reorder the days properly (Sun -> Sat)
ordered_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
pivot_df = pivot_df.reindex(ordered_days)

# Display the result table
print(pivot_df)
```
With the data matrix ready, we can use seaborn to create the heatmap:
```python
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot_df, cmap='YlGnBu', cbar=False)
mean_orders = pivot_df.mean().mean()
for i in range(len(pivot_df)):
    for j in range(len(pivot_df.columns)):
        text_color = 'white' if pivot_df.iloc[i, j] > mean_orders else 'black'
        ax.text(j+0.5, i+0.5, int(pivot_df.iloc[i, j]),
            color=text_color, fontsize=10, ha="center", va="center")
plt.title("Number of orders by day of the week and hour of the day")
plt.xlabel("Hour of the day")
plt.ylabel("")
plt.show()
```
- **Most orders** were placed during the **weekdays** from **10 AM to 4 PM** with a **small dip** in sales around **12 PM** due to **lunchtime**.
- Customers also order through Olist in the **evenings** around **9 PM** from **Sunday to Thursday**. **Saturday** is the day with the **fewest orders**, although it's still quite busy.
- The **least busy** time of the day is from **3 to 5 AM**, although there are still a few orders during that time.

### Payment behavior analysis

```python
df['payment_type'].nunique()
```

Boleto is a widely used payment method in Latin America, particularly in Brazil. Introduced in 1993, it was designed to facilitate cash payments. It functions similarly to a proforma invoice and can be paid at various locations, including ATMs, bank branches, internet banking, post offices, lottery agents, and some supermarkets, as long as it is within the due date.

For better understanding, you could compare Boleto to systems like direct debit or bank transfer in many other countries, where payments are made outside the traditional credit card system, often using a pre-issued invoice or bill. A more specific comparison might be with BillPay systems in the U.S. or SEPA Direct Debit in Europe, where payments are processed through a network of banks, but Boleto is more widely accessible due to its integration with multiple payment points.

```python
payment_type_share = df['payment_type'].value_counts(normalize=True).nlargest(4)
payment_type_share = (payment_type_share*100).round(2)
payment_type_share
```

```python
# Group by payment_type and aggregate
agg_df = df.groupby('payment_type').agg(total_revenue=('payment_value', 'sum'),payment_count=('payment_value', 'count')).reset_index()

# Sort by total_revenue
agg_df = agg_df.sort_values(by='total_revenue')

# Create figure with bar and line traces
fig = go.Figure()

# Bar chart for total_revenue
fig.add_trace(go.Bar(x=agg_df['payment_type'],y=agg_df['total_revenue'],name='Total revenue',marker_color='darkblue'))

# Line chart for payment_count
fig.add_trace(go.Scatter(x=agg_df['payment_type'],y=agg_df['payment_count'],name='Payment count',yaxis='y2',mode='lines+markers',marker=dict(color='dodgerblue', size=8),line=dict(width=3)))

# Layout with dual y-axes
fig.update_layout(title=dict(text='<b>Total revenue and Payment count by Payment type</b>',x=0.5,xanchor='center'),
    xaxis=dict(title='Payment type', showgrid=False),
    yaxis=dict(title='Total revenue', showgrid=False),
    yaxis2=dict(title='Payment count',overlaying='y',side='right'),
    legend=dict(x=0.5, xanchor='center', orientation='h'),
    bargap=0.3,width = 900, height  =700)

fig.show()
```
- **Credit cards dominate** the landscape, accounting for a **striking 73.67%** of all **payments** and achieve the **highest total revenue** **(≈ 15M+)**. The **second most popular method, boleto bancário**, trails far behind at **19.46%** and generates **moderate revenue (≈ 4M)**, followed by **vouchers (5.43%)** and **debit cards (1.43%)** with **low revenue (less than 1M each)**. This **preference** speaks not only to **access** but also to the **flexibility** Brazilian **consumers expect**.

```python
installment_ratio = (df['payment_installments'] > 1).mean() * 100
print(f'The proportion of installment orders: {installment_ratio:.2f}%')
```
- **Installments** are the **norm** — not the exception. Over **half of all credit card transactions (50.10%)** are paid in **installments**, effectively blurring the line between **affordability** and **deferred spending**.
  
```python
installment_by_category = df.groupby('product_category_name_english')['payment_installments'].apply(lambda x: (x > 1).mean()).nlargest(10)
installment_top_categories = (installment_by_category*100).round()
plt.figure(figsize=(12, 6))
bars = plt.barh(installment_top_categories['product_category_name_english'], installment_top_categories['payment_installments'], color='darkblue')
plt.xlabel('Percentage of Installment Payments (%)', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
plt.title('Top 10 Categories by Installment Usage (%)', fontsize=14, pad=20)
plt.xlim(0, 100)
for bar in bars:
    width = bar.get_width()
    plt.text(width - 5, bar.get_y() + bar.get_height()/2,
             f'{width}%',
             ha='right', va='center', color='white', fontweight='bold')
plt.tight_layout()
plt.show()
```
- This pattern is especially pronounced in **high-ticket** or **aspirational categories** like **PCs (75% installment rate), kitchenware, mattresses**, and **home appliances**. Even in **fashion** and **gift** categories, **installment** usage often **exceeds 60%**.

In short, Brazil’s payment structure is heavily credit-driven, with strong reliance on installment plans for both essentials and lifestyle goods. Merchants and platforms that accommodate — or incentivize — this behavior are better positioned to drive conversion and basket size. Conversely, any limitations on credit usage or installment options could act as a major friction point in the purchase journey.








  
