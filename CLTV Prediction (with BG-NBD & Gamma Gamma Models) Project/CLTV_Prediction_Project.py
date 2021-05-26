# ##################################################################
#   CLTV PREDICTION PROJECT  (using BG/NBD & GAMMA-GAMMA MODEL)
# ##################################################################

# ⭐ CLTV Prediction in 4 steps  
# 1. Data Preparation
# 2. Calculating the Expected Sale Forecasting with BG/NBD Model.
# 3. Calculating the Expected Average Profit with Gamma-Gamma Model.
# 4. Calculating the CLTV for a specific time period with BG/NBD and Gamma-Gamma models.


⭐ # BG/NBD (BETA GEOMETRIC NEGATIVE BINOMIAL DISTRIBUTION) ⇒ (EXPECTED SALES FORECASTING: predicts how many purchases customers can make in a given time period)
 # BG/NBD Model stochastic model learns the distribution of customers' purchase behaviour structure. Thereby, it enables to predict the expected number of sales 
 # by taking into account both the overall distribution and an individual's own purchasing behaviour. Model executes probability distribution by taking into consideration
 # all customers' purchase frequency. Model learns a pattern of customers' purchase frequency and predict.
    
⭐ # Gamma gamma model => Expected Average Profit

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II dataset includes the sales of a UK-based online retail store between the years 2010 - 2011.
# The company aims to predict its Customers' lifetime value for a specicific time period.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter  # BG/NBD
from lifetimes import GammaGammaFitter  # Gamma Gamma
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Detecting Outliers (In literature, quantiles are usually used with 0.25 & 0.75) I used 0.01 & 0.99 in order to trim outliers (below) by replacing them with up_limit)
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


data = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()
df.head()
df.shape
df.info()
df.describe().T
#                count        mean      std           min           25%       50%          75%          max
# Quantity     541910.00     9.55      218.08     -80995.00       1.00       3.00         10.00       80995.00
# Price        541910.00     4.61      96.76      -11062.06       1.25       2.08         4.13        38970.00
# Customer ID  406830.00   15287.68   1713.603    12346.00      13953.00     15152.00    16791.00     18287.00

# Min values of Quantity and Price cannot be negative.

# =============
# DATA PREP
# =============

df.dropna(inplace=True)
df = df.loc[~df["Invoice"].str.contains("C", na=False)]                # removing returned products
df = df[df["Quantity"] > 0]

replace_with_thresholds(df, "Quantity")                                # Outliers in Quantity and Price trimmed! (replaced with up_limit)
replace_with_thresholds(df, "Price")
df.describe().T
#                 count        mean       std        min         25%       50%           75%           max
# Quantity      397925.00    11.84      25.54       1.00        2.00        6.00        12.00       298.50
# Price         397925.00     2.90      3.23        0.00        1.25        1.95        3.75        37.06
# Customer ID   397925.00   15294.31   1713.18     12346.00     13969.00   15159.00     16795.00    18287.00


df["TotalPrice"] = df["Quantity"] * df["Price"]

df["InvoiceDate"].max()  # '2011-12-09'
today_date = dt.datetime(2011, 12, 11)

# ==========
# RFM Table
# ==========

rfm = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: (today_date - date.max()).days,
                                                     lambda date: (today_date - date.min()).days],
                                     "Invoice": lambda num: num.nunique(),
                                     "TotalPrice": lambda price: price.sum()})

rfm.columns = rfm.columns.droplevel(0)
rfm.columns = ["Recency", "T", "Frequency", "Monetary"]
rfm.head()

#                Recency       T     Frequency     Monetary
# Customer ID
# 12346.00000      326       326          1       310.44000
# 12347.00000        3       368          7      4310.00000
# 12348.00000       76       359          4      1770.78000
# 12349.00000       19        19          1      1491.72000
# 12350.00000      311       311          1       331.46000

# Gamma gamma model requires the average of monetary values for each transaction.
# Gamma gamma model => Expected Average Profit

temp_df = df.groupby(["Customer ID", "Invoice"]).agg({"Invoice": "count", "TotalPrice": ["mean"]})
temp_df.columns = temp_df.columns.droplevel(0)
temp_df.reset_index(inplace=True)
temp_df.columns = ["Customer ID", "Invoice", "total_price_count", "total_price_mean"]
temp_df.head()
#    Customer ID  Invoice  total_price_count  total_price_mean
# 0  12346.00000   541431                  1         310.44000
# 1  12347.00000   537626                 31          22.96097
# 2  12347.00000   542237                 29          16.39276
# 3  12347.00000   549222                 24          26.51042
# 4  12347.00000   556201                 18          21.25111

temp_df2 = temp_df.groupby(["Customer ID"], as_index=False).agg({"total_price_mean": ["mean"]})
temp_df2.columns = temp_df2.columns.droplevel(0)
temp_df2.columns = ["Customer ID", "monetary_avg"]
temp_df2.head()
#    Customer ID  monetary_avg
# 0  12346.00000     310.44000
# 1  12347.00000      23.09700
# 2  12348.00000      66.01550
# 3  12349.00000      20.43452
# 4  12350.00000      19.49765

# ===> This is what Gamma Gamma model requires!

rfm.index.isin(temp_df2["Customer ID"]).all()  # Checked if the indexes of rfm table and temp_df2 table are the same!
# True!

rfm = rfm.merge(temp_df2, how="left", on="Customer ID")
rfm.head()
rfm.set_index("Customer ID", inplace=True)  # Customer ID has been assigned to the index!
rfm.index = rfm.index.astype(int)
rfm.head()
#              Recency        T    Frequency     Monetary    monetary_avg
# Customer ID
# 12346            326      326         1       310.44000     310.44000
# 12347              3      368         7      4310.00000      23.09700
# 12348             76      359         4      1770.78000      66.01550
# 12349             19      19          1      1491.72000      20.43452
# 12350            311      311         1       331.46000      19.49765

# Converting tenure and recency values to weekly values.
rfm["Recency_weekly"] = rfm["Recency"] / 7
rfm["T_weekly"] = rfm["T"] / 7
rfm = rfm.loc[rfm["monetary_avg"] > 0]
rfm_cltv = rfm.copy()  # CHECKPOINT!

rfm_cltv.head()

# BG/NBD model assumes that there should be no correlation between the Monetary and Recency Value!
rfm_cltv[["monetary_avg", "Recency_weekly"]].corr()
#                 monetary_avg  Recency_weekly
# monetary_avg         1.00000         0.01836
# Recency_weekly       0.01836         1.00000

rfm_cltv["Frequency"] = rfm_cltv["Frequency"].astype(int)



# #############################
# Establishing the BG/NBD MODEL  ==> Expected Sales Forecasting!
# #############################

# pip install lifetimes

bgf = BetaGeoFitter(penalizer_coef=0.001)  # initiating the model object

bgf.fit(rfm_cltv["Frequency"],
        rfm_cltv["Recency_weekly"],
        rfm_cltv["T_weekly"])

# Out[62]: <lifetimes.BetaGeoFitter: fitted with 4338 subjects, a: 1.52, alpha: 0.07, b: 5.69, r: 0.28>
"""In BG/NBD model, there are alpha and beta models that execute probability distribution by taking into consideration
all customers' purchase frequency. Model learns a pattern of customers' purchase frequency and predict. 
"""

# ==================================================================================
# What are the 10 customers to be expected to make the purchase the most in 1 week?
# ==================================================================================
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        rfm_cltv["Frequency"],
                                                        rfm_cltv["Recency_weekly"],
                                                        rfm_cltv["T_weekly"]).sort_values(ascending=False).head(10)

# Customer ID
# 16000   3.47241
# 12713   2.61763
# 15520   1.87669
# 13298   1.87669
# 14569   1.87669
# 13436   1.87669
# 15060   1.82989
# 18139   1.64053
# 14087   1.47115
# 15471   1.47115
# dtype: float64


rfm_cltv["expected_number_of_purchases"] = bgf.predict(1,
                                                       rfm_cltv["Frequency"],
                                                       rfm_cltv["Recency_weekly"],
                                                       rfm_cltv["T_weekly"])
rfm_cltv.head()

#              Recency    T    Frequency   Monetary    monetary_avg    Recency_weekly    T_weekly    expected_number_of_purchases
# Customer ID
# 12346         326     326          1     310.44000     310.44000        46.57143       46.57143            0.02158
# 12347           3     368          7    4310.00000      23.09700         0.42857       52.57143            0.00000
# 12348          76     359          4    1770.78000      66.01550        10.85714       51.28571            0.00062
# 12349          19      19          1    1491.72000      20.43452         2.71429       2.71429             0.33805
# 12350         311     311          1     331.46000      19.49765        44.42857       44.42857            0.02261

rfm_cltv.sort_values("expected_number_of_purchases", ascending=False).head(10)

# ==================================================================================
# ⭐ What are the 10 customers to be expected to make the purchase the most in 1 month?
# ==================================================================================

bgf.predict(4,
            rfm_cltv["Frequency"],
            rfm_cltv["Recency_weekly"],
            rfm_cltv["T_weekly"]).sort_values(ascending=False).head(10)

# Customer ID
# 16000   7.13233
# 12713   5.10115
# 15060   4.84868
# 18139   4.81851
# 13298   4.14386
# 15520   4.14386
# 14569   4.14386
# 13436   4.14386
# 15195   3.52802
# 15471   3.52802
# dtype: float64

rfm_cltv["expected_number_of_purchases"] = bgf.predict(4,
                                                       rfm_cltv["Frequency"],
                                                       rfm_cltv["Recency_weekly"],
                                                       rfm_cltv["T_weekly"])

rfm_cltv.head()
rfm_cltv.sort_values("expected_number_of_purchases", ascending=False).head(20)

⭐""" What makes BG/NBD MODEL and CLTV very valuable is that they provide predictively valuable information about 
new customers although new customers' purchasing patterns are not yet known.
Model investigated the purchasing patterns of all past customers and estimated that the customers with
 high recency and high purchasing frequency and high monetary as high potential customers."""

# ==================================================================================
# What are the total expected sales in 1 month?
# ==================================================================================

bgf.predict(4,
            rfm_cltv["Frequency"],
            rfm_cltv["Recency_weekly"],
            rfm_cltv["T_weekly"]).sum()

# 883.60

"""How to validate this value? What should the reference point be? ==> the average of past purchases
The mean of past purchases was around 1400, and std around 400-500, it's close! """

# ==================================================================================
# What are the total expected sales in 3 month?
# ==================================================================================

bgf.predict(4 * 3,
            rfm_cltv["Frequency"],
            rfm_cltv["Recency_weekly"],
            rfm_cltv["T_weekly"]).sum()

# 2082.05

# ==================================================================================
# Evaluating the Predictive Results?
# ==================================================================================

plot_period_transactions(bgf)
plt.show()

# ##################################
# Establishing the GAMMA-GAMMA MODEL  ==> Expected Average Profit!
# ##################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm_cltv["Frequency"], rfm_cltv["monetary_avg"])

#  <lifetimes.GammaGammaFitter: fitted with 4338 subjects, p: 3.54, q: 1.00, v: 3.25>


# =============================================
# ⭐ 10 most expected average profitable customers 
# =============================================
ggf.conditional_expected_average_profit(rfm_cltv["Frequency"],
                                        rfm_cltv["monetary_avg"]).sort_values(ascending=False).head(10)

# Customer ID
# 16000   1188.52359
# 16532   1123.69478
# 15749    970.89486
# 15098    853.82762
# 15195    824.67825
# 18102    634.92178
# 13270    593.64789
# 18080    568.96565
# 17291    554.42152
# 16698    530.14009
# dtype: float64


rfm_cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm_cltv["Frequency"],
                                                                              rfm_cltv["monetary_avg"])

rfm_cltv.sort_values("expected_average_profit", ascending=False).head(20)

# ##################################################################
#   CLTV PREDICTION by combining  BG/NBD and GAMMA-GAMMA MODEL
# ##################################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   rfm_cltv["Frequency"],
                                   rfm_cltv["Recency_weekly"],
                                   rfm_cltv["T_weekly"],
                                   rfm_cltv["monetary_avg"],
                                   time=3,  # for 3 months
                                   freq="W",  # weekly frequency
                                   discount_rate=0.01)

cltv.shape       # (4338,)
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(10)

# ⭐ 10 Most Valuable Customers! 

#       Customer ID         clv
# 2678        16000    11794.07113
# 2087        15195     4641.70350
# 715         13298     1140.24615
# 2011        15098      988.92748
# 3403        16986      895.74984
# 1523        14424      864.20530
# 514         13017      828.99144
# 208         12603      588.07853
# 3216        16737      530.40716
# 1310        14126      496.62311

cltv_100 = cltv.sort_values(by="clv", ascending=False).head(100)
cltv_100.to_csv("Cltv_100_most_valuable_customers.csv")

# VALIDATION!!!

rfm_cltv_final = rfm_cltv.merge(cltv, how="left", on="Customer ID")
rfm_cltv_final.head()
