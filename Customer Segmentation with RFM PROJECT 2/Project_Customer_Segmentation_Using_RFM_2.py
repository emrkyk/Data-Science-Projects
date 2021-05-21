# ==========================================
# Customer Segmentation with RFM / Project 2
# ==========================================

# Customer Segmentation with RFM in 6 Steps  (The CRISP-DM METHODOLOGY)
# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Naming & Analysing RFM Segments

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II  or
# https://www.kaggle.com/nathaniel/uci-online-retail-ii-data-set
# Applying RFM analysis to the sheet named "Year 2010-2011" of the online_retail_II.xlsx data set.
# Online Retail II dataset includes the sales of a UK-based online retail store between Year 2010-2011.
# The company aims define the behavior of customers and create groups according to clustering in these behaviors,
# segment its customers and determine marketing strategies according to these segments.

"""  VARIABLES
InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction.
If this code starts with the letter 'c', it indicates a cancellation.
StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal. The name of the country where a customer resides."""

import pandas as pd
import numpy as np
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

data = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()
df.head()
df.columns
# Index(['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'Price', 'Customer ID', 'Country'],
#       dtype='object')

df.info()
df.isnull().sum()

df["Description"].nunique()  # the unique number of products?
df["Description"].value_counts()  # how many of which product are there?

# Which is the most ordered product?
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False)

df["Invoice"].nunique()  # How many invoices in total?

# If Invoice starts with "C", it means the product has been returned
df = df.loc[~df["Invoice"].str.contains("C", na=False)]  # removing returned products
df.shape

# total price variable
df["TotalPrice"] = df["Quantity"] * df["Price"]

df.sort_values("Price", ascending=False)  # the most expensive products?

df["Country"].value_counts()  # How many orders from which country?
df.groupby("Country").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False)

# =================
# Data Preparation
# =================
df.isnull().sum()
df.dropna(inplace=True)  # Missing values do not affect the main purpose, so dropped!

df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# =======================
# Calculating RFM Metrics  //// Recency, Frequency, Monetary
# =======================

df["InvoiceDate"].max()  # '2011-12-09"

today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                     "Invoice": lambda num: len(num),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm

rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]
rfm.head()
#              Recency  Frequency  Monetary
# Customer ID
# 12346.0          326          1  77183.60
# 12347.0            3        182   4310.00
# 12348.0           76         31   1797.24
# 12349.0           19         73   1757.55
# 12350.0          311         17    334.40


# ====================================
# Converting RFM Metrics to RFM Scores
# ====================================

rfm["Recency Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["Frequency Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm[["Recency", "Frequency", "Monetary", "Recency Score", "Frequency Score", "Monetary Score"]]

rfm["RFM_SCORE"] = (rfm["Recency Score"].astype(str) + rfm["Frequency Score"].astype(str) + rfm["Monetary Score"].astype(str))

rfm[rfm["RFM_SCORE"] == "555"]      # Champions group
rfm.loc[rfm["RFM_SCORE"] == "111"]  # Hibernating group


# ==========================================
# Naming & Analysing RFM Segments with REGEX
# ==========================================

# REGEX: An important tool to capture and manipulate pattern matches in any text.

# Naming RFM segments    # segment map  => built for Recency and Frequency values
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'}

rfm

rfm["Segment"] = rfm["Recency Score"].astype(str) + rfm["Frequency Score"].astype(str)

rfm["Segment"] = rfm["Segment"].replace(seg_map, regex=True)

rfm["Segment"]

# Customer ID
# 12346.0        Hibernating
# 12347.0          Champions
# 12348.0            At_Risk
# 12349.0    Loyal_Customers
# 12350.0        Hibernating


rfm.head()

rfm

df["Customer ID"].nunique()  # 4339
rfm.groupby("Segment").agg({"Recency": ["mean", "count"], "Frequency": ["mean", "count"], "Monetary": ["mean", "count"]})
# or  rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])


#                              Recency         Frequency           Monetary
#                            mean count        mean count         mean count
# Segment
# About_to_Sleep        53.192547   322   15.987578   322   450.997174   322
# At_Risk              166.435852   569   56.859402   569   996.916872   569
# Cant_Loose           144.226190    84  181.666667    84  2370.705012    84
# Champions              6.400990   606  289.031353   606  6960.915446   606
# Hibernating          210.251397  1074   13.589385  1074   536.535672  1074
# Loyal_Customers       34.206854   817  157.116279   817  2845.732852   817
# Need_Attention        52.985366   205   41.736585   205   856.195854   205
# New_Customers          7.421053    57    7.578947    57  3618.697018    57
# Potential_Loyalists   16.668699   492   34.945122   492   915.486994   492
# Promising             23.437500   112    7.767857   112   429.433929   112


# Let's identify the Customers that belong to "Loyal Customers" segment

loyal_customers = pd.DataFrame()
loyal_customers["loyal_customers"] = rfm[rfm["Segment"] == "Loyal_Customers"].index

loyal_customers.to_csv("Loyal_Customers.csv")

# FINAL EVALUATION

"""    ⭐⭐⭐          
When customers were segmented into groups with RFM analysis, There are labelled segments that reflect the customers' 
RFM behaviours. Among these segments, "champions", "need-attention" and "at-risk" groups were selected.

MARKETING!  
Since acquiring new customers is more costly than retaining existing customers, the main goal is to retain existing 
customers with actions towards pleasing customers by getting to know them better.

⭐Need-Attention Segment: 
The Need Attention group is a group that requires attention. The purchasing frequency of 
customers in this segment is low, but the time that has passed since their shopping time is closer to the "At-Risk" 
group. Customers in this customer segment can go either side, positive or negative. For this reason, action must be 
taken to include them in the loyal customer's segmentation. For this reason, actions such as making special campaigns 
for this customer group, offering special discounts via e-mails and messages, sending gifts or trial products can be taken.

⭐Champions Segment: 
Customers in this group have a high shopping frequency and spending. For this reason, retaining these
customers is crucial. By taking actions such as special discounts, special customer cards for this segment, it can be
aimed to retain these customers and to direct other customers to the company.

⭐At-risk Segment: 
Customers in this segment seem to be at risk of losing. Shopping frequency is high, but the time 
elapsed after the shopping date is long. Actions such as making special discounts for these customers, communicating
by phone or email, gifting cards that earn points for customer spending can be applied.
"""









