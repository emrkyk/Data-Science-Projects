# ###################################################################
#   CLTV PREDICTION PROJECT 2 (with BG/NBD & GAMMA-GAMMA MODEL)
# ###################################################################

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II dataset includes the sales of a UK-based online retail store between the years 2010 - 2011.
# ⭐ The aim of the project is to predict CLTV for customers in U.K for 6 months of period between 2010-2011.

# ⭐ CLTV Prediction in 4 steps
# 1. Data Preparation
# 2. Calculating the Expected Sale Forecasting with BG/NBD Model.
# 3. Calculating the Expected Average Profit with Gamma-Gamma Model.
# 4. Calculating the CLTV for a specific time period with BG/NBD and Gamma-Gamma models.

# BG/NBD (BETA GEOMETRIC NEGATIVE BINOMIAL DISTRIBUTION) ⇒ (EXPECTED SALES FORECASTING: predicts how many purchases customers can make in a given time period)
# BG/NBD Model stochastic model learns the distribution of customers' purchase behaviour structure. Thereby, it enables to predict the expected number of sales
# by taking into account both the overall distribution and an individual's own purchasing behaviour. Model executes probability distribution by taking into consideration
# all customers' purchase frequency. Model learns a pattern of customers' purchase frequency and predict.

# Gamma gamma model => Expected Average Profit

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter  # BG/NBD
from lifetimes import GammaGammaFitter  # Gamma Gamma
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


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


##########################
#  Data Preperation
##########################

data = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()
df.head()
df.info()

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max()    # Timestamp('2011-12-09 12:50:00')
today_date = dt.datetime(2011, 12, 11)

df_uk = df[df["Country"] == "United Kingdom"]

##########################
# RFM Table
##########################

rfm = df_uk.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                        lambda x: (today_date - x.min()).days],
                                        "Invoice": lambda x: x.nunique(),
                                        "TotalPrice": lambda x: x.sum()})

rfm.head()

rfm.columns = rfm.columns.droplevel(0)
rfm.columns = ["recency", "tenure", "frequency", "monetary"]

rfm.head()
#              recency  tenure  frequency    monetary
# Customer ID
# 12346.00000        0     326          1   310.44000
# 12747.00000      366     370         11  4196.01000
# 12748.00000      372     374        210 32380.41000
# 12749.00000      209     214          5  4077.94000
# 12820.00000      323     327          4   942.34000

# monetary_avg
rfm["monetary"] = rfm["monetary"] / rfm["frequency"]
rfm.rename(columns={"monetary": "avg_monetary"}, inplace=True)

# recency_weekly_p
rfm["recency_weekly"] = rfm["recency"] / 7
rfm["T_weekly"] = rfm["tenure"] / 7

rfm = rfm[rfm["avg_monetary"] > 0]
rfm = rfm[rfm["frequency"] > 1]
rfm["frequency"].dtype  # int
rfm.head()

# ==============================
# Establishing the BG/NBD MODEL  ==> Expected Sales Forecasting!
# ==============================

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(rfm["frequency"], rfm["recency_weekly"], rfm["T_weekly"])
# <lifetimes.BetaGeoFitter: fitted with 2570 subjects, a: 0.12, alpha: 11.66, b: 2.51, r: 2.21>


# ===================================
# Establishing the GAMMA-GAMMA MODEL  ==> Expected Average Profit!
# ===================================
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm["frequency"], rfm["avg_monetary"])

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm["frequency"], rfm["avg_monetary"])
rfm.sort_values("expected_average_profit", ascending=False).head()

#               recency   tenure  frequency  avg_monetary   recency_weekly   T_weekly    expected_average_profit
# Customer ID
# 14088.00000      312     323         13    3864.55462        44.57143     46.14286             3916.38316
# 18102.00000      366     368         60    3859.73908        52.28571     52.57143             3870.83874
# 15749.00000       97     333          3    3028.78000        13.85714     47.57143             3213.27392
# 14096.00000       97     102         17    3163.58824        13.85714     14.57143             3195.97186
# 17511.00000      370     374         31    2933.94306        52.85714     53.42857             2950.34628



# ===========================================================================================
# ⭐ CLTV Prediction for customers in U.K for 6 months of projection between the years 2010-2011
# ===========================================================================================

cltv = ggf.customer_lifetime_value(bgf,
                                   rfm["frequency"],
                                   rfm["recency_weekly"],
                                   rfm["T_weekly"],
                                   rfm["avg_monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head()

rfm_cltv = rfm.merge(cltv, on="Customer ID", how="left")
rfm_cltv.sort_values(by="clv", ascending=False).head(5)

CLV = rfm_cltv.sort_values(by="clv", ascending=False)
CLV.to_excel("CLV_6_months.xlsx")

# Customer ID	  recency	tenure	frequency   avg_monetary  recency_weekly	T_weekly	expected_average_profit	      clv
# 18102.00000	    366	    368	        60	    3859.73908	    52.28571	    52.57143	    3870.83874	           92217.93935
# 14096.00000	    97	    102	        17	    3163.58824	    13.85714	    14.57143	    3195.97186	           55730.60119
# 17450.00000	    359	    368	        46	    2863.27489	    51.28571	    52.57143	    2874.04545	           52847.58548
# 17511.00000	    370	    374	        31	    2933.94306	    52.85714	    53.42857	    2950.34628	           36948.18585
# 16684.00000	    353	    359	        28	    2209.96911	    50.42857	    51.28571	    2223.68989	           26146.99798


# The highest CLV value customer is 18102 for 6 months of projection.

# ================================================
# ⭐ CLTV Prediction for customers in U.K for 1 month
# ================================================

cltv_1_months = ggf.customer_lifetime_value(bgf,
                                            rfm["frequency"],
                                            rfm["recency_weekly"],
                                            rfm["T_weekly"],
                                            rfm["avg_monetary"],
                                            time=1,
                                            freq="W",
                                            discount_rate=0.01)
cltv_1_months.head()
cltv_1_months = cltv_1_months.reset_index()

rfm_cltv_1_months = rfm.merge(cltv_1_months, on="Customer ID", how="left")
CLV1 = rfm_cltv_1_months.sort_values(by="clv", ascending=False)
CLV1.to_csv("CLV_1_month.csv")

# ==================================================
# ⭐ CLTV Prediction for customers in U.K for 12 months
# ==================================================

cltv_12_months = ggf.customer_lifetime_value(bgf,
                                             rfm["frequency"],
                                             rfm["recency_weekly"],
                                             rfm["T_weekly"],
                                             rfm["avg_monetary"],
                                             time=1,
                                             freq="W",
                                             discount_rate=0.01)
cltv_12_months.head()
cltv_12_months = cltv_12_months.reset_index()

rfm_cltv_12_months = rfm.merge(cltv_12_months, on="Customer ID", how="left")
CLV2 = rfm_cltv_12_months.sort_values(by="clv", ascending=False)
CLV2.to_excel("CLV_12_months.xlsx")

# #cltv_1_month
# Customer ID	  clv
# 12747.00000	336.78338
# 12748.00000	2153.46419
# 12749.00000	604.90454
# 12820.00000	110.12903
# 12822.00000	286.94537

# cltv_12_months
# Customer ID	   clv
# 12747.00000	336.78338
# 12748.00000	2153.46419
# 12749.00000	604.90454
# 12820.00000	110.12903
# 12822.00000	286.94537


# ==================================================
# ⭐ Segmenting Customers based on 6 months of CLTV
# ==================================================

rfm_cltv["segment"] = pd.qcut(rfm_cltv["clv"], 3, labels=["C", "B", "A"])

rfm_cltv_final = rfm_cltv.sort_values(by="clv", ascending=False)
rfm_cltv_final.to_csv("6_months_CLTV_Pred_with_Segments.csv")

rfm_cltv_final = rfm_cltv_final.reset_index()
rfm_cltv_final.head()


# Selecting the best of %20 customers based on CLV and mark them 1 and others 0 (Pareto)
rfm_cltv_final.shape
rfm_cltv_final.shape[0] * 0.2  # 514


rfm_cltv_final["top_flag"] = 0
rfm_cltv_final["top_flag"].iloc[0:515] = 1
rfm_cltv_final

rfm_cltv_final
