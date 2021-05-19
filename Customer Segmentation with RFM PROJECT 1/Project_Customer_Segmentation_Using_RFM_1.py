# ================================
# Customer Segmentation with RFM
# ================================

# Customer Segmentation with RFM in 6 Steps  (The CRISP-DM METHODOLOGY)
# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Naming & Analysing RFM Segments

""" Recency - Frequency- Monetary (RFM) - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Why do we segment Customers? => Let's imagine that a company has 10 employees and 10000 customers.
Labor, resource and time are always scarce/limited, so optimum use of resource is a must!
How can 10 employees deal effectively with 10k customers? We need to divide it into groups, evaluate and manage.
It is necessary to divide/segment customers into groups based on how often customers purchase, when they last
purchased and how much they spent!
Main goal: direct our effort and time to the right targeted customers! - - - - - - - - - - - - - - -
"""

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II dataset includes the sales of a UK-based online retail store between 01/12/2009 - 09/12/2011.
# The company aims define the behavior of customers and create groups according to clustering in these behaviors,
# segment its customers and determine marketing strategies according to these segments.

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

data = pd.read_excel("Datasets/online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")

df = data.copy()
df.head()
df.columns
df.isnull().sum()

df["Description"].nunique()                                 # the unique number of products?
df["Description"].value_counts().head()                     # how many of which product are there?

df.groupby("Description").agg({"Quantity": "sum"}).head()   # Which is the most ordered product?
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

df["Invoice"].nunique()                                     # How many invoices in total?

# If Invoice starts with "C", it means the product has been returned
df = df[~df["Invoice"].str.contains("C", na=False)]         # removing returned products

df["TotalPrice"] = df["Quantity"] * df["Price"]             # total price variable

df.sort_values("Price", ascending=False).head()             # the most expensive products?

df["Country"].value_counts()                                # How many orders from which country?

df.groupby("Country").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False).head()

df.head()

# =================
# Data Preparation
# =================

df.isnull().sum()
df.dropna(inplace=True)                             # Missing values do not affect the main purpose, so dropped!

df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

"""It's not necessary to deal with outliers in RFM, because i already convert them into scores, 
even if they have large monetary values"""

# =======================
# Calculating RFM Metrics  //// Recency, Frequency, Monetary
# =======================

df["InvoiceDate"].max()  # 2010-12-09

today_date = dt.datetime(2010, 12, 11)  # reasonable to assume today's date as 2 or 1 days later than last date

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: len(num),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm = rfm[(rfm["Monetary"] > 0) & (rfm["Frequency"] > 0)]
rfm.head()

#              Recency  Frequency  Monetary
# Customer ID
# 12346.0          165         33    372.86
# 12347.0            3         71   1323.32
# 12348.0           74         20    222.16
# 12349.0           43        102   2671.14
# 12351.0           11         21    300.93

# =======================
# Calculating RFM Scores
# =======================

# Recency
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

# Frecuency
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])

# Monetary
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))

rfm[rfm["RFM_SCORE"] == "555"].head()  # Champions group!!!

rfm[rfm["RFM_SCORE"] == "111"]         # Hibernating group

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

rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)  

df[["Customer ID"]].nunique()
rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])

#                            Recency         Frequency         Monetary
#                          mean  count      mean  count       mean   count
# Segment
# About_to_Sleep       53.54360   344   16.10756   344   447.83983   344
# At_Risk             165.05546   577   59.56499   577  1180.62517   577
# Cant_Loose          128.86869    99  220.64646    99  3002.42698    99
# Champions             7.03956   632  273.35443   632  6964.07719   632
# Hibernating         206.06134  1027   14.51607  1027   461.18677  1027
# Loyal_Customers      37.40103   773  168.27684   773  2662.04686   773
# Need_Attention       53.68072   166   45.27108   166   935.62663   166
# New_Customers         7.75758    66    7.81818    66   482.08712    66
# Potential_Loyalists  18.43992   516   37.20349   516  1024.27688   516
# Promising            24.99107   112    8.61607   112   456.50821   112


# Let's now focus on Customers labeled "Need_Attention" - - - - - - - - -

rfm[rfm["Segment"] == "Need_Attention"].head()
rfm[rfm["Segment"] == "Need_Attention"].index

new_df = pd.DataFrame()

new_df["Need_Attention"] = rfm[rfm["Segment"] == "Need_Attention"].index

new_df.to_csv("Need_Attention_Customers.csv")

"""The company's potentially losing customers in need of attention are named in the segment as such."""

# Let's now focus on Customers labeled "Loyal_Customers" - - - - - - - - -

loyal_customers = pd.DataFrame()

loyal_customers["Loyal_Customers"] = rfm[rfm["Segment"] == "Loyal_Customers"].index

loyal_customers.to_csv("Loyal_Customers.csv")


# Let's now focus on Customers labeled "Can't Lose"  - - - - - - - - -

cant_lose = pd.DataFrame()

cant_lose["Cant_Loose"] = rfm[rfm["Segment"] == "Cant_Loose"].index
cant_lose.to_csv("Cant_Lose_Customers.csv")

