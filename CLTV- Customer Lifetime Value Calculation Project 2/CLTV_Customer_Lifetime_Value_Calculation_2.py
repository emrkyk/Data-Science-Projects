####################################################
# CLTV Customer Lifetime Value Calculation Project 2
####################################################

# CLTV Calc. will be applied to the same dataset as it is in the previous CLTV project, but for the sheet of the 2010-2011 year.

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II dataset includes the sales of a UK-based online retail store between the Year 2010-2011.
# The company aims to calculate its Customers' lifetime value and would like to have insights into their overall value.

# ⭐
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin
# ⭐

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", 20)
# pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

data = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()
df.head()
df.shape   #  (541910, 8)
df.info()

# RangeIndex: 541910 entries, 0 to 541909
# Data columns (total 8 columns):
#  #   Column       Non-Null Count   Dtype
# ---  ------       --------------   -----
#  0   Invoice      541910 non-null  object
#  1   StockCode    541910 non-null  object
#  2   Description  540456 non-null  object
#  3   Quantity     541910 non-null  int64
#  4   InvoiceDate  541910 non-null  datetime64[ns]
#  5   Price        541910 non-null  float64
#  6   Customer ID  406830 non-null  float64
#  7   Country      541910 non-null  object
# dtypes: datetime64[ns](1), float64(2), int64(1), object(4)


# =================
# Data Preparation
# =================

df = df.loc[~df["Invoice"].str.contains("C", na=False)]
df.shape  # (532622, 8)

df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]  # calculating the total price, total price column added to dataset
df.head()

cltv_df = df.groupby("Customer ID").agg({"Invoice": lambda x: len(x), "Quantity": lambda x: x.sum(), "TotalPrice": lambda x: x.sum()})
cltv_df.columns = ["total_transaction", "total_unit", "total_price"]
cltv_df.head()

#              total_transaction  total_unit  total_price
# Customer ID
# 12346.00000                  1       74215  77183.60000
# 12347.00000                182        2458   4310.00000
# 12348.00000                 31        2341   1797.24000
# 12349.00000                 73         631   1757.55000
# 12350.00000                 17         197    334.40000


# ======================================
# 1. Calculating the Average Order Value
# ======================================
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders

cltv_df["average_order_value"] = cltv_df["total_price"] / cltv_df["total_transaction"]

# =====================================
# 2. Calculating the Purchase Frequency
# =====================================
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
#  Total_Number_of_Orders => cltv_df['total_transaction']

cltv_df.shape[0]  # (4339)
cltv_df["purchase_frequency"] = cltv_df["total_transaction"] / cltv_df.shape[0]
cltv_df.head()

#               total_transaction  total_unit  total_price   average_order_value   purchase_frequency
# Customer ID
# 12346.00000                  1       74215    77183.60000          77183.60000      0.00023
# 12347.00000                182        2458     4310.00000             23.68132      0.04195
# 12348.00000                 31        2341     1797.24000             57.97548      0.00714
# 12349.00000                 73         631     1757.55000             24.07603      0.01682
# 12350.00000                 17         197      334.40000             19.67059      0.00392

# =============================================
# 3. Calculating the Repeat Rate and Churn Rate
# =============================================
# Repeat Rate = the number of customers who has purchased at least once /(divide by) total number of customers

repeat_rate = cltv_df[cltv_df["total_transaction"] > 1].shape[0] / cltv_df.shape[0]
# 0.9834063148190827

churn_rate = 1 - repeat_rate
# 0.016593685180917306

# ===================================
# 4. Calculating the Profit Margin
# ===================================
# let's assume %5 profitings out of the total price
cltv_df["profit_margin"] = cltv_df["total_price"] * 0.05

# ==========================================
# 5. Calculating the Customer Lifetime Value
# ==========================================
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

cltv_df["CV"] = (cltv_df["average_order_value"] * cltv_df["purchase_frequency"]) / churn_rate
cltv_df["CLTV"] = cltv_df["CV"] * cltv_df["profit_margin"]
cltv_df.sort_values("CLTV", ascending=False)


# ===================
# SCALING CLTV SCORES
# ===================
# Let's standardize CLTV values between 1 and 100 with MinMaxScaler in order to have a better evaluation

scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])

cltv_df.sort_values("CLTV", ascending=False)

cltv_df[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV", ascending=False).head()
#              total_transaction   total_unit    total_price        CLTV        SCALED_CLTV
# Customer ID
# 14646.00000               2080      197491    280206.02000   54524592.80850   100.00000
# 18102.00000                431       64124    259657.30000   46820773.22451    86.01222
# 17450.00000                337       69993    194550.79000   26284729.09002    48.72504
# 16446.00000                  3       80997    168472.50000   19710405.03906    36.78807
# 14911.00000               5677       80515    143825.06000   14365033.25278    27.08251

cltv_df.sort_values("total_price", ascending=False)

# ===================================
# SEGMENTATION OF SCALED CLTV SCORES
# ===================================

cltv_df["segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])

# Let's observe the 5 MOST valuable customers!
cltv_df[["segment", "total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV", ascending=False).head()

#               segment   total_transaction  total_unit  total_price           CLTV          SCALED_CLTV
# Customer ID
# 14646.00000       A               2080      197491    280206.02000        54524592.80850    100.00000
# 18102.00000       A                431       64124    259657.30000        46820773.22451     86.01222
# 17450.00000       A                337       69993    194550.79000        26284729.09002     48.72504
# 16446.00000       A                  3       80997    168472.50000        19710405.03906     36.78807
# 14911.00000       A               5677       80515    143825.06000        14365033.25278     27.08251


cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg({"count", "mean", "sum"})

# ⭐ Final Evaluation:
# The customers have been segmented by dividing into 4 equal number of groups based on their Scaled CLTV.
# Class A delivers the highest Customer Lifetime Value, whereas Class D the lowest.
# When examining the results, there is an obvious, direct effect of the average order value and purchase frequency on CLTV.
# In segment A, purchase frequency is higher with the average order value compared to other segments.
# Although the purchase frequency of customers in the B segment is decreasing, it is higher than customers in the C and D segments.

