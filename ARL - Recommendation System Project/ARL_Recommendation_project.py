# ======================================
#  ARL RECOMMENDATION PROJECT
# ======================================

"""⭐ CUSTOMIZED RECOMMENDATION SYSTEM => Segment-specific customized Association Rule Learning (using Apriori Algorithm) and make product recommendations.
Using ARL, generate association rules for locations & segments and make recommendations according to these rules. 

-- for Germany (2010-2011) in Online Retail II data set
-- For which segments? Potential customers & segments that will be determined as a result of the prediction in CLTV (using BG/NBD & Gamma Gamma Models).

⭐ *-*-*-* CRUCIAL POINT *-*-*-*
If assuming that the behaviours of the A segments in Germany and the A segments in the UK are similar, then I can fulfil the unfruitful suggestions in Germany from the UK.
In that case, it is necessary to model from the A segment in the whole data,extract the association rules, and it will suggest something to (let say) the A-segment of Germany, 
which it learned from all the data. If the model learned the association rules only from Germany's A segment, then it'd end up experiencing unfruitfulness regarding 
recommendation a point sooner or later.

This project includes combining CLTV Prediction and ARL projects that I have conducted before and turning them into customized product recommendations.
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

df_prep = crm_data_prep(df)  # (from helpers module)  # Specialized function I utilized when I conducted CRM & CLTV projects
check_df(df_prep)

# ===============================================================
# Creating Predictive CLTV Segments with create_cltv_p function
# ===============================================================

"""The function I utilized to automate the stuff I did in the CLTV prediction project(You can see in the repository). The outcome of the function contains the predictions of 
   "recency_cltv_p", "T", "monetary_avg", "recency_weekly_cltv_p", "T_weekly", "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit", "cltv_p", "cltv_p_segment"
   as the results of  CLTV Prediction (BG/NBD & Gamma Gamma Models)
"""


def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)

    # recency user-specific
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,       # "recency_cltv_p"
                                                                lambda date: (today_date - date.min()).days],      # "T"
                                                'Invoice': lambda num: num.nunique(),                              # "frequency"
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})                # "monetary"
    rfm.columns = rfm.columns.droplevel(0)

    # recency_cltv_p
    rfm.columns = ["recency_cltv_p", "T", "frequency", "monetary"]

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
"""
##################### Shape #####################
(2845, 10)
##################### Types #####################
recency_cltv_p                int64
T                             int64
monetary_avg                float64
recency_weekly_cltv_p       float64
T_weekly                    float64
exp_sales_1_month           float64
exp_sales_3_month           float64
expected_average_profit     float64
cltv_p                      float64
cltv_p_segment             category
dtype: object
##################### Head #####################
             recency_cltv_p    T  monetary_avg  recency_weekly_cltv_p  \
Customer ID                                                             
12347.0                 365  368    615.714286              52.142857   
12348.0                 282  359    442.695000              40.285714   
12352.0                 260  297    219.542500              37.142857   
              T_weekly  exp_sales_1_month  exp_sales_3_month  \
Customer ID                                                    
12347.0      52.571429           0.561684           1.678069   
12348.0      51.285714           0.364322           1.087951   
12352.0      42.428571           0.739160           2.206857   
             expected_average_profit    cltv_p cltv_p_segment  
Customer ID                                                    
12347.0                   631.911974  2.933517              A  
12348.0                   463.745539  1.919391              B  
12352.0                   224.886669  1.904146              B  
##################### Tail #####################
             recency_cltv_p    T  monetary_avg  recency_weekly_cltv_p  \
Customer ID                                                             
18282.0                 118  127     89.025000              16.857143   
18283.0                 333  338    130.930000              47.571429   
18287.0                 158  202    612.426667              22.571429   
              T_weekly  exp_sales_1_month  exp_sales_3_month  \
Customer ID                                                    
18282.0      18.142857           0.530590           1.575547   
18283.0      48.285714           1.226968           3.665906   
18287.0      28.857143           0.478113           1.423970   
             expected_average_profit    cltv_p cltv_p_segment  
Customer ID                                                    
18282.0                    99.524747  1.283796              C  
18283.0                   132.601200  1.886451              B  
18287.0                   651.345353  2.684413              A  
##################### NA #####################
recency_cltv_p             0
T                          0
monetary_avg               0
recency_weekly_cltv_p      0
T_weekly                   0
exp_sales_1_month          0
exp_sales_3_month          0
expected_average_profit    0
cltv_p                     0
cltv_p_segment             0
dtype: int64
##################### Quantiles #####################
                                 0.00        0.05        0.50        0.95  \
recency_cltv_p           0.000000e+00   16.000000  207.000000  361.800000   
T                        3.000000e+00   55.000000  283.000000  373.000000   
monetary_avg             3.450000e+00  101.978667  300.487308  867.951667   
recency_weekly_cltv_p    0.000000e+00    2.285714   29.571429   51.685714   
T_weekly                 4.285714e-01    7.857143   40.428571   53.285714   
exp_sales_1_month        4.945689e-27    0.128856    0.510695    1.435971   
exp_sales_3_month        1.478339e-26    0.383725    1.523013    4.274179   
expected_average_profit  5.819483e+00  109.425403  316.538801  917.205040   
cltv_p                   1.000000e+00    1.115096    1.881909    4.897332   
                                0.99         1.00  
recency_cltv_p            369.000000   373.000000  
T                         374.000000   374.000000  
monetary_avg             1732.183600  5724.302619  
recency_weekly_cltv_p      52.714286    53.285714  
T_weekly                   53.428571    53.428571  
exp_sales_1_month           2.566856    13.469161  
exp_sales_3_month           7.672579    40.266173  
expected_average_profit  1839.299523  5772.177190  
cltv_p                     12.469683   100.000000  
"""

cltv_p.head()

cltv_p.groupby("cltv_p_segment").agg({"count", "mean"})

#                     recency_cltv_p               T             monetary_avg           recency_weekly_cltv_p       T_weekly             exp_sales_1_month
#                      mean     count        mean   count         mean    count              mean   count        mean     count            mean     count
# cltv_p_segment                                                             
# C                  169.920969   949    280.121180   949     198.315161   949             24.274424   949      40.017311    949          0.323827   949
# B                  213.163502   948    257.175105   948     317.029688   948             30.451929   948      36.739301    948          0.566315   948
# A                  213.662447   948    239.033755   948     601.261390   948             30.523207   948      34.147679    948          1.027486   948

                   
#                          exp_sales_3_month            expected_average_profit              cltv_p
#                              mean   count                  mean    count                mean    count
# cltv_p_segment                                                               
# C                          0.964801   949              214.541431   949               1.318969   949
# B                          1.686891   948              335.687124   948               1.899350   948
# A                          3.061011   948              626.565366   948               4.311039   948


                          

# =========================================================
# Reduction of the data set according to the user IDs of the desired segments.
# =========================================================
# Get id's for each segment by keeping their indexes! 
a_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "A"].index
b_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "B"].index
c_segment_ids = cltv_p[cltv_p["cltv_p_segment"] == "C"].index

"""I got the segment infos from CLTV P because I created the segments for deduplicated users in that dataframe."""
# for example: Customer ID's in Segment A ⭐ (PREMIUM CUSTOMER WITH HIGHEST POTENTIAL)
a_segment_ids 
# Float64Index([12347.0, 12356.0, 12358.0, 12359.0, 12360.0, 12362.0, 12364.0,
#               12370.0, 12371.0, 12380.0,
#               ...
#               18225.0, 18226.0, 18228.0, 18229.0, 18230.0, 18235.0, 18242.0,
#               18259.0, 18272.0, 18287.0],
#              dtype='float64', name='Customer ID', length=948)




# Reduction df according to these segments
a_segment_df = df_prep[df_prep["Customer ID"].isin(a_segment_ids)]
b_segment_df = df_prep[df_prep["Customer ID"].isin(b_segment_ids)]
c_segment_df = df_prep[df_prep["Customer ID"].isin(c_segment_ids)]
a_segment_df.head()

"""" Now, going back to the whole data(df_prep) and reducting it in for each segment, I created the bulk dataframes of the users in segments 
A, B, C in the whole data for each segment."""

# ===============================================================================
# Creating Association Rules with APRIORI Algorithm for Each Segment
# ===============================================================================

""" ASSOCIATION RULE LEARNING ANALYSIS FUNCTION:
   I aggregated the operations in the first ARL project (in the repository) and turned it into a function. This is a specialized function for ARL analysis 
   for same retail dataset that creates invoice_product basket, derived support values and association rules."""

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

""" Derived rules for segment A,B,C in the whole data, Germany has not been selected yet.
    In summary, I found out 3 groups of products for each segments from the whole data (product_a, product_b, product_c)"""

# a
rules_a = create_rules(a_segment_df)
rules_a.head()

#   antecedents consequents  antecedent support  consequent support   support     confidence   lift    leverage  conviction
# 0    (85099B)     (20685)            0.112441            0.040000  0.010079      0.089636  2.240896  0.005581    1.054523
# 1     (20685)    (85099B)            0.040000            0.112441  0.010079      0.251969  2.240896  0.005581    1.186526
# 2    (85099B)     (20711)            0.112441            0.019318  0.010289      0.091503  4.736786  0.008117    1.079456
# 3     (20711)    (85099B)            0.019318            0.112441  0.010289      0.532609  4.736786  0.008117    1.898964
# 4     (20712)     (20713)            0.036850            0.031391  0.010814      0.293447  9.348112  0.009657    1.370894

product_a = int(rules_a["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])   # in order to get rid off frozen set data type

# ------------------------------------------------------

# b
rules_b = create_rules(b_segment_df)

#     antecedents consequents  antecedent support  consequent support   support       confidence    lift     leverage  conviction
# 481     (22917)     (22916)            0.011040            0.010819  0.010378       0.940000    86.882857  0.010258   16.486347
# 480     (22916)     (22917)            0.010819            0.011040  0.010378       0.959184    86.882857  0.010258   24.229521
# 487     (22919)     (22917)            0.010598            0.011040  0.010157       0.958333    86.805833  0.010040   23.735041
# 486     (22917)     (22919)            0.011040            0.010598  0.010157       0.920000    86.805833  0.010040   12.367520
# 482     (22916)     (22918)            0.010819            0.010819  0.010157       0.938776    86.769679  0.010040   16.156620

product_b = int(rules_b["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])   # in order to get rid off frozen set data type

# ------------------------------------------------------

# c

rules_c = create_rules(c_segment_df)

#    antecedents consequents  antecedent support  consequent support   support       confidence       lift  leverage  conviction
# 67     (22748)     (22745)            0.013722            0.013387  0.010375        0.756098    56.480488  0.010191    4.045114
# 66     (22745)     (22748)            0.013387            0.013722  0.010375        0.775000    56.480488  0.010191    4.383460
# 87    (47590B)    (47590A)            0.016399            0.014726  0.011379        0.693878    47.120594  0.011137    3.218563
# 86    (47590A)    (47590B)            0.014726            0.016399  0.011379        0.772727    47.120594  0.011137    4.327845
# 71     (23301)     (23300)            0.019076            0.015730  0.010375        0.543860    34.575588  0.010075    2.157824

product_c = int(rules_c["consequents"].apply(lambda x: list(x)[0]).astype("unicode")[0])    # in order to get rid off frozen set data type


def check_id(stock_code):
    product_name = df_prep[df_prep["StockCode"] == stock_code][["Description"]].values[0].tolist()
    return print(product_name)


check_id(71053)  # ['WHITE METAL LANTERN']

# ===================================================================
# ⭐ RECOMMENDATIONS FOR GERMAN CUSTOMERS ACCORDING TO THEIR SEGMENT
# ===================================================================
""" I added a variable named recommended_product to the dataframe, which is the output of cltv_p, and added 1 product for each segment. 
    In other words, I added one of the rules from above for which segment the customer is in"""

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


germany_ids = df_prep[df_prep["Country"] == "Germany"]["Customer ID"].drop_duplicates()       # duplicates have been dropped!
cltv_p["recommended_product"] = ""

cltv_p.head()

cltv_p.loc[cltv_p.index.isin(germany_ids)]
cltv_p[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A")]
cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "A"), "recommended_product"] = product_a

cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B"), "recommended_product"] = product_b
cltv_p[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "B")]

cltv_p.loc[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C"), "recommended_product"] = product_c
cltv_p[(cltv_p.index.isin(germany_ids)) & (cltv_p["cltv_p_segment"] == "C")]





# ⭐ LET'S TRY!

cltv_p[cltv_p.index == 12500]     # => SEGMENT A   One of the most valuable Customer!

#              recency_cltv_p    T        monetary_avg    recency_weekly_cltv_p    T_weekly    exp_sales_1_month   exp_sales_3_month
# Customer ID
# 12500.0            310        334        387.074545         44.285714          47.714286        0.884459          2.64212
#
#
#                 expected_average_profit        cltv_p          cltv_p_segment     recommended_product
# Customer ID
# 12500.0                   393.623083          2.896085             A                   20685


cltv_p[cltv_p.index == 12708]     # => SEGMENT B

#              recency_cltv_p      T      monetary_avg     recency_weekly_cltv_p   T_weekly      exp_sales_1_month     exp_sales_3_month
# Customer ID
# 12708.0           338           369       271.291              48.285714        52.714286           0.748012            2.235132
#
#
#              expected_average_profit         cltv_p          cltv_p_segment      recommended_product
# Customer ID
# 12708.0                 276.461302          2.127007              B                 20719
