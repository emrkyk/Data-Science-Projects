# ======================================
#  ARL RECOMMENDATION PROJECT
# ======================================

"""CUSTOMIZED RECOMMENDATION SYSTEM => Segment-specific customize and customized according to association rules
 and make product recommendations
Using ARL, generate association rules for locations & segments and make suggestions according to these rules.

-- for Germany (2010-2011)
-- For which segments? Potential customers who will be on CLTV for cltv_p segments.

*-*-*-* CRUCIAL POINT *-*-*-*
If assuming that the behaviours of the A segments in Germany and the A segments in the UK are similar, then I can fulfil
the unfruitful suggestions in Germany from the UK. In that case, it is necessary to model from the A segment in the whole data,
extract the association rules, and it will suggest something to (let say) the A-segment of Germany, which it learned from all the data.
If the model learned the association rules only from Germany's A segment, then it'd end up experiencing unfruitfulness somewhere.
"""

# DATASET:   https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II dataset includes the sales of a UK-based online retail store.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import *

pd.set_option("display.max_columns", None)

df_ = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.info()

# <class 'pandas.core.frame.DataFrame'>
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


# ===================
# Data Preprocessing
# ===================

df_prep = crm_data_prep(df)  # from helpers
check_df(df_prep)

# =========================================================
# Creating Predictive CLTV Segments with create_cltv_p function
# =========================================================

"""The function I utilized to automate the stuff I did in the CLTV prediction project(You can see in the repository)"""


def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)

    # recency user-specific
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
    rfm.columns = rfm.columns.droplevel(0)

    # recency_cltv_p
    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

    # Simplified monetary_avg (since Gamma-Gamma model requires this way)
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]
    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)

    # Calculating WEEKLY RECENCY VE WEEKLY T for BG/NBD MODEL
    # recency_weekly_cltv_p
    rfm["recency_weekly_cltv_p"] = rfm["recency_cltv_p"] / 7
    rfm["T_weekly"] = rfm["T"] / 7

    # CHECK IT OUT! Monetary avg must be positive
    rfm = rfm[rfm["monetary_avg"] > 0]

    # recency filter
    rfm = rfm[(rfm["frequency"] > 1)]
    rfm["frequency"] = rfm["frequency"].astype(int)  # converting it to integer just in case!

    # Establishing the BGNBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm["frequency"],
            rfm["recency_weekly_cltv_p"],
            rfm["T_weekly"])

    # exp_sales_1_month
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm["frequency"],
                                           rfm["recency_weekly_cltv_p"],
                                           rfm["T_weekly"])
    # exp_sales_3_month
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm["frequency"],
                                           rfm["recency_weekly_cltv_p"],
                                           rfm["T_weekly"])

    # Establishing Gamma-Gamma Model  calculates=> Expected Average Profit
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm["frequency"], rfm["monetary_avg"])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm["frequency"],
                                                                             rfm["monetary_avg"])
    # CLTV Pred for 6 months
    cltv = ggf.customer_lifetime_value(bgf,
                                       rfm["frequency"],
                                       rfm["recency_weekly_cltv_p"],
                                       rfm["T_weekly"],
                                       rfm["monetary_avg"],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    rfm["cltv_p"] = cltv

    # Minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    # rfm.fillna(0, inplace=True)

    # cltv_p_segment
    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    # recency_cltv_p, recency_weekly_cltv_p
    rfm = rfm[["recency_cltv_p", "T", "monetary_avg", "recency_weekly_cltv_p", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit",
               "cltv_p", "cltv_p_segment"]]

    return rfm


cltv_p = create_cltv_p(df_prep)
check_df(cltv_p)
cltv_p.head()

cltv_p.groupby("cltv_p_segment").agg({"count", "mean"})

# =========================================================
# Reduction of the data set according to the user IDs of the desired segments.
# =========================================================
# Get id's
a_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "A"].index
b_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "B"].index
c_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "C"].index

# Reduction df according to these id's
a_segment_df = df_prep[df_prep["Customer ID"].isin(a_segment_ids)]
b_segment_df = df_prep[df_prep["Customer ID"].isin(b_segment_ids)]
c_segment_df = df_prep[df_prep["Customer ID"].isin(c_segment_ids)]
a_segment_df.head()

# =========================================================
# Creating Association Rules with APRIORI Algorithm for Each Segment
# =========================================================

"""ASSOCIATION RULE LEARNING ANALYSIS FUNCTION:
   I aggregated the operations in the first ARL project (in the repository) and turned it into a function."""


def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True, low_memory=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True, low_memory=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules


# a
rules_a = create_rules(a_segment_df)
rules_a.head()

#   antecedents consequents  antecedent support  consequent support   support     confidence   lift    leverage  conviction
# 0    (85099B)     (20685)            0.112441            0.040000  0.010079      0.089636  2.240896  0.005581    1.054523
# 1     (20685)    (85099B)            0.040000            0.112441  0.010079      0.251969  2.240896  0.005581    1.186526
# 2    (85099B)     (20711)            0.112441            0.019318  0.010289      0.091503  4.736786  0.008117    1.079456
# 3     (20711)    (85099B)            0.019318            0.112441  0.010289      0.532609  4.736786  0.008117    1.898964
# 4     (20712)     (20713)            0.036850            0.031391  0.010814      0.293447  9.348112  0.009657    1.370894

product_a = int(rules_a["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# ------------------------------------------------------

# b
rules_b = create_rules(b_segment_df)

#     antecedents consequents  antecedent support  consequent support   support       confidence    lift     leverage  conviction
# 481     (22917)     (22916)            0.011040            0.010819  0.010378       0.940000    86.882857  0.010258   16.486347
# 480     (22916)     (22917)            0.010819            0.011040  0.010378       0.959184    86.882857  0.010258   24.229521
# 487     (22919)     (22917)            0.010598            0.011040  0.010157       0.958333    86.805833  0.010040   23.735041
# 486     (22917)     (22919)            0.011040            0.010598  0.010157       0.920000    86.805833  0.010040   12.367520
# 482     (22916)     (22918)            0.010819            0.010819  0.010157       0.938776    86.769679  0.010040   16.156620

product_b = int(rules_b["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])

# ------------------------------------------------------

# c

rules_c = create_rules(c_segment_df)

#    antecedents consequents  antecedent support  consequent support   support       confidence       lift  leverage  conviction
# 67     (22748)     (22745)            0.013722            0.013387  0.010375        0.756098    56.480488  0.010191    4.045114
# 66     (22745)     (22748)            0.013387            0.013722  0.010375        0.775000    56.480488  0.010191    4.383460
# 87    (47590B)    (47590A)            0.016399            0.014726  0.011379        0.693878    47.120594  0.011137    3.218563
# 86    (47590A)    (47590B)            0.014726            0.016399  0.011379        0.772727    47.120594  0.011137    4.327845
# 71     (23301)     (23300)            0.019076            0.015730  0.010375        0.543860    34.575588  0.010075    2.157824

product_c = int(rules_c["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])


def check_id(stock_code):
    product_name = df_prep[df_prep["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return print(product_name)


check_id(71053)  # ['WHITE METAL LANTERN']

# ===============================================================
#  RECOMMENDATIONS FOR GERMAN CUSTOMERS ACCORDING TO THEIR SEGMENT
# ===============================================================
"""I added a variable named recommended_product to the dataframe, which is the output of cltv_p.
    and added 1 product for each segment. In other words, I added one of the rules from above for which segment the customer is in"""

cltv_p.head()

#              recency_cltv_p   T     monetary_avg      recency_weekly_cltv_p  \
# Customer ID
# 12347.0                 365  368    615.714286              52.142857
# 12348.0                 282  359    442.695000              40.285714
# 12352.0                 260  297    219.542500              37.142857
# 12356.0                 302  326    937.143333              43.142857
# 12358.0                 149  151    575.210000              21.285714
#
#
#               T_weekly      exp_sales_1_month       exp_sales_3_month  \
# Customer ID
# 12347.0      52.571429           0.561684           1.678069
# 12348.0      51.285714           0.364322           1.087951
# 12352.0      42.428571           0.739160           2.206857
# 12356.0      46.571429           0.333040           0.993934
# 12358.0      21.571429           0.474048           1.408955
#
#              expected_average_profit    cltv_p      cltv_p_segment
# Customer ID
# 12347.0                   631.911974    2.933517              A
# 12348.0                   463.745539    1.919391              B
# 12352.0                   224.886669    1.904146              B
# 12356.0                   995.997679    2.802492              A
# 12358.0                   631.900951    2.612984              A


germany_ids = df_prep[df_prep["Country"] == "Germany"]["Customer ID"].drop_duplicates()
cltv_p["recommended_product"] = ""

cltv_p.head()

cltv_p.loc[cltv_p.index.isin(germany_ids)]
cltv_p[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A")]
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A"), "recommended_product"] = product_a

cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B"), "recommended_product"] = product_b
cltv_p[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B")]

cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C"), "recommended_product"] = product_c
cltv_p[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C")]

# LET'S TRY!

cltv_p[cltv_p.index == 12500]  # => SEGMENT A!   One of the most valuable Customer!

#              recency_cltv_p    T        monetary_avg    recency_weekly_cltv_p    T_weekly    exp_sales_1_month   exp_sales_3_month
# Customer ID
# 12500.0            310        334        387.074545         44.285714          47.714286        0.884459          2.64212
#
#
#                 expected_average_profit        cltv_p          cltv_p_segment     recommended_product
# Customer ID
# 12500.0                   393.623083          2.896085             A                   20685


cltv_p[cltv_p.index == 12708]  # => SEGMENT B

#              recency_cltv_p      T      monetary_avg     recency_weekly_cltv_p   T_weekly      exp_sales_1_month     exp_sales_3_month
# Customer ID
# 12708.0           338           369       271.291              48.285714        52.714286           0.748012            2.235132
#
#
#              expected_average_profit         cltv_p          cltv_p_segment      recommended_product
# Customer ID
# 12708.0                 276.461302          2.127007              B                 20719
