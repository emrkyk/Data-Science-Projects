###################################################
# CLTV Customer Lifetime Value Calculation Project
###################################################

# ⭐
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin 
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin
# ⭐

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II dataset includes the sales of a UK-based online retail store between 01/12/2009 - 09/12/2011.
# The company aims to calculate its Customers' lifetime value and would like to have insights into their overall value.

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
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", 20)
# pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

data = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = data.copy()
df.head()
df.shape        # (525461, 8)
df.info()


# =================
# Data Preparation
# =================

df = df.loc[~df["Invoice"].str.contains("C", na=False)]             # removing returned products
df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)
df.shape

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby("Customer ID").agg({"Invoice": lambda x: len(x),
                                         "Quantity": lambda x: x.sum(),
                                         "TotalPrice": lambda x: x.sum()})

cltv_df.columns = ["total_transaction", "total_unit", "total_price"]

cltv_df.head()
#              total_transaction  total_unit  total_price
# Customer ID
# 12346.00000                 33          70    372.86000
# 12347.00000                 71         828   1323.32000
# 12348.00000                 20         373    222.16000
# 12349.00000                102         993   2671.14000
# 12351.00000                 21         261    300.93000

# ======================================
# 1. Calculating the Average Order Value
# ======================================
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders

cltv_df["avg_order_value"] = cltv_df["total_price"] / cltv_df["total_transaction"]

# =====================================
# 2. Calculating the Purchase Frequency
# =====================================
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
#  Total_Number_of_Orders => cltv_df['total_transaction']

cltv_df["purchase_frequency"] = cltv_df["total_transaction"] / cltv_df.shape[0]

# =============================================
# 3. Calculating the Repeat Rate and Churn Rate
# =============================================
# Repeat Rate = the number of customers who has purchased at least once /(divide by) total number of customers

repeat_rate = cltv_df[cltv_df.total_transaction > 1].shape[0] / cltv_df.shape[0]
repeat_rate   # 0.978

churn_rate = 1 - repeat_rate
churn_rate    # 0.021

# ===================================
# 4. Calculating the Profit Margin
# ===================================
# let's assume %5 profitings out of the total price

cltv_df["profit_margin"] = cltv_df['total_price'] * 0.05

# ==========================================
# 5. Calculating the Customer Lifetime Value
# ==========================================
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

cltv_df["CV"] = (cltv_df["avg_order_value"] * cltv_df["purchase_frequency"]) / churn_rate

cltv_df["CLTV"] = cltv_df["CV"] * cltv_df["profit_margin"]

cltv_df.sort_values("CLTV", ascending=False)
cltv_df[["CLTV"]].head()

#                   CLTV
# Customer ID
# 12346.00000   74.74440
# 12347.00000  941.49238
# 12348.00000   26.53498
# 12349.00000 3836.01554
# 12351.00000   48.68756

# Let's standardize CLTV values between 1 and 100 with MinMaxScaler in order to have a better evaluation

scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])

cltv_df.sort_values("CLTV", ascending=False)

cltv_df[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV", ascending=False).head()

#               SCALED_CLTV
# Customer ID
# 18102.00      100.00
# 14646.00      51.103
# 14156.00      32.375
# 14911.00      19.797
# 13694.00      15.029

cltv_df.sort_values("total_price", ascending=False)

cltv_df["segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg({"count", "mean", "sum"})

#                         total_transaction          total_unit                total_price
#                      mean count     sum       mean    count     sum        mean   count     sum
# segment
# D                17.12407  1080   18494   109.01389   1080   117735    178.30841   1080  192573.08000   . . . . . .
# C                39.21913  1077   42239   283.20891   1077   305016    476.05246   1077  512708.50300   . . . . . .
# B                81.59091  1078   87955   680.15863   1078   733211   1131.35918   1078 1219605.20000   . . . . . .
# A               240.04356  1079  259007   4062.33735  1079  4383262   6401.40546   1079 6907116.49100   . . . . . .

#                         CLTV                                 SCALED_CLTV
#                  mean   count        sum            mean       count    sum
# segment
# D             20.20300  1080     21819.23798     1.00003    1080    1080.03296
# C            128.94417  1077    138872.87636     1.00019    1077    1077.20975
# B            733.85778  1078    791098.68675     1.00111    1078    1079.19487
# A         178832.53238  1079 192960302.43825     1.27011    1079    1370.44481



#⭐Let's observe the 5 MOST valuable customers!

cltv_df[["segment", "total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(
    by="SCALED_CLTV", ascending=False).head(5)

#               segment  total_transaction  total_unit   total_price       CLTV          SCALED_CLTV
# Customer ID
# 18102.00000       A                627      124216    349164.35000   65546098.55426    100.00000
# 14646.00000       A               1774      170342    248396.50000   33172484.52272     51.10330
# 14156.00000       A               2648      108107    196566.74000   20773378.10442     32.37585
# 14911.00000       A               5570       69722    152147.57000   12445636.05210     19.79773
# 13694.00000       A                957      125893    131443.19000   9288877.52547      15.02980
