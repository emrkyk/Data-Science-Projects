############################################
# PROJECT: LEVEL BASED PERSONA & SIMPLE CUSTOMER SEGMENTATION
#############################################

""" The Purpose of the Project:
- Consider, investigate the concept of Persona.
- Information obtained from the dataset is aimed to use in order to make new customer definitions
  based on level by separating them into certain levels or categories and considering each breakout point
  as a persona. Accordingly, when a new customer shows up, determining which segment the new future
  customer may belong to.
"""
################# Before #####################
#
# country device gender age  price
# USA     and    M      15   61550
# BRA     and    M      19   45392
# DEU     iOS    F      16   41602
# USA     and    F      17   40004
#                M      23   39802

################# After #####################
#
#   customers_level_based      price groups
# 0        USA_AND_M_0_18 157120.000      A
# 1        USA_AND_F_0_18 151121.000      A
# 2        BRA_AND_M_0_18 149544.000      A
# 3        USA_IOS_F_0_18 133773.000      A
# 4       USA_AND_F_19_23 133645.000      A


import numpy as np
import pandas as pd
import seaborn as sns

users = pd.read_csv("Datasets/users.csv")
users.head()

#         uid              reg_date device gender country  age
# 0  54030035  2017-06-29T00:00:00Z    and      M     USA   19
# 1  72574201  2018-03-05T00:00:00Z    iOS      F     TUR   22
# 2  64187558  2016-02-07T00:00:00Z    iOS      M     USA   16
# 3  92513925  2017-05-25T00:00:00Z    and      M     BRA   41
# 4  99231338  2017-03-26T00:00:00Z    iOS      M     FRA   59


purchases = pd.read_csv("Datasets/purchases.csv")
purchases.head()

#          date       uid  price
# 0  2017-07-10  41195147    499
# 1  2017-07-15  41195147    499
# 2  2017-11-12  41195147    599
# 3  2017-09-26  91591874    299
# 4  2017-12-01  91591874    599


# 1. Step: Merging the datasets according to the "uid" variable with an inner join.
df = purchases.merge(users, how="inner", on="uid")
df.shape  # (9006, 8)
df.head()

#          date       uid  price              reg_date device gender country  age
# 0  2017-07-10  41195147    499  2017-06-26T00:00:00Z    and      M     BRA   17
# 1  2017-07-15  41195147    499  2017-06-26T00:00:00Z    and      M     BRA   17
# 2  2017-11-12  41195147    599  2017-06-26T00:00:00Z    and      M     BRA   17
# 3  2017-09-26  91591874    299  2017-01-05T00:00:00Z    and      M     TUR   17
# 4  2017-12-01  91591874    599  2017-01-05T00:00:00Z    and      M     TUR   17


# 2. Step:  What are the total earnings in the breakdown of “country”, “device”, “gender”, “age”?
df.groupby(["country", "device", "gender", "age"]).agg({"price": "sum"})

#                            price
# country device gender age
# BRA     and    F      15   33824
#                       16   31619
#                       17   20352
#                       18   20047
#                       19   21352
# ..     ..     ..     ..     ...


agg_df = df.groupby(["country", "device", "gender", "age"]).agg({"price": "sum"}).sort_values("price", ascending=False)
agg_df.head()

agg_df.reset_index(inplace=True)
agg_df

#     country device gender  age  price
# 0       USA    and      M   15  61550
# 1       BRA    and      M   19  45392
# 2       DEU    iOS      F   16  41602
# 3       USA    and      F   17  40004
# 4       USA    and      M   23  39802
# ..      ...    ...    ...  ...    ...


# 3. Step: Converting the age variable to a categorical variable and adding to the dataset as a new variable.

agg_df["age"].dtype  # => int
agg_df["age"].value_counts()

# num to cat!
bins = [0, 19, 24, 31, 41, agg_df["age"].max()]
labels = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["age"].max())]
agg_df["age_cat"] = pd.cut(agg_df["age"], bins=bins, labels=labels)
agg_df["age_cat"]

agg_df.head()
#   country device gender  age  price age_cat
# 0     USA    and      M   15  61550    0_18
# 1     BRA    and      M   19  45392    0_18
# 2     DEU    iOS      F   16  41602    0_18
# 3     USA    and      F   17  40004    0_18
# 4     USA    and      M   23  39802   19_23


# 4. Step: Considering the categorical breakdowns as customer groups and defining
# new level-based customers by combining these groups.

agg_df["customers_level_based"] = [col[0] + "_" + col[1].upper() + "_" + col[2] + "_" + col[-1] for col in agg_df.values]

#  alternative way:
# for index, column in agg_df.iterrows():
#     agg_df.loc[index, "customers_level_based"] = column["country"].upper() + "_" + column["device"].upper() + "_" + column["gender"].upper()
#     + "_" + column["age_cat"].upper()

agg_df[["customers_level_based", "price"]]

#   customers_level_based  price
# 0        USA_AND_M_0_18  61550
# 1        BRA_AND_M_0_18  45392
# 2        DEU_IOS_F_0_18  41602
# 3        USA_AND_F_0_18  40004
# 4       USA_AND_M_19_23  39802


""" The variable "customers_level_based" is now our new customer definition. For example "USA_AND_M_0_18". 
The USA-ANDROID-MALE-0-18 class is a single customer representing one class of customers for us.
"""

# 5. Step: Segmenting the new customers according to price

agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"])

agg_df[["customers_level_based", "price", "segment"]].head()

# describing the segments
agg_df[["segment", "price"]]
agg_df.groupby("segment").agg({"price": "mean"})

#                 price
# segment
# D         1335.096491
# C         3675.504505
# B         7447.812500
# A        20080.150442

# 6. Step: What segment is a 42-year-old Turkish woman who uses IOS device in?
# Express the segment (group) of this person according to the final analysis?

new_user = "TUR_IOS_F_41_75"

agg_df.loc[agg_df["customers_level_based"] == new_user]

#       country device gender  ...   age_cat   customers_level_based segment
# 377     TUR    iOS      F   ...    41_75       TUR_IOS_F_41_75       D

""" 
As a result of the analysis, a female new user between the ages of 41-75  who uses an IOS device from Turkey, 
belongs to the segment D.
"""


