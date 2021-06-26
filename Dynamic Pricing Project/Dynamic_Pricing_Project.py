###############################
# DYNAMIC PRICING PROJECT      ---> What should the item price be?
###############################

"""A game company gave gift coins to its users for item purchases in a game. Users buy various tools for their characters of game by using these virtual coins.
   The game company did not specify a price for any item and allowed users to buy related item at the price they wanted. For example, for the item named shield, 
   users can buy the shield by paying the amounts they are willing to pay. A user can pay with 30 units of the virtual money given to him/her, whereas other user 
   with 45 units. Therefore, users can buy any item with the amounts they are willing to pay for themselves.

    # Problems to be solved:
    # Does the price of the item differ according to the categories? Express it statistically.
    # What should the price of the item be based on the first question? Explain why?
    # It is desired to be "movable" in terms of price. Create a decision support system for the price strategy.
    # Simulate item purchases and income for possible price changes.
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import statsmodels.stats.api as sms
import itertools
from itertools import combinations
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, levene
from scipy import stats
from helpers.eda import *
from helpers.data_prep import *



pd.set_option("display.max_columns", None)

df = pd.read_csv("Datasets/pricing.csv", sep=";")
df.head()

#    category_id      price
# 0       489756  32.117753
# 1       361254  30.711370
# 2       361254  31.572607
# 3       489756  34.543840
# 4       489756  47.205824


check_df(df)
df.info()

df["category_id"].value_counts()
# 489756    1705
# 874521     750
# 361254     620
# 326584     145
# 675201     131
# 201436      97


# -------------------------
#    OUTLIERS ?
# -------------------------

low_limit, up_limit = outlier_thresholds(df, "price")  # (-64.46732638496243, 187.44554397493738)

check_outlier(df, "price")  # True   ---> There are outliers!

outlier_index = grab_outliers(df, "price", index=True)
#      category_id          price
# 12        361254  201436.887637
# 100       489756     441.638389
# 234       489756  201436.957996
# 289       361254  201436.427631
# 314       874521     200.000000

replace_with_thresholds(df, "price")

df.groupby("category_id").agg({"price": ["mean", "median", "count", "nunique"]})
#                  price
#                   mean    median   count nunique
# category_id
# 201436       36.175498  33.534678    97      81
# 326584       36.739739  31.748242   145      69
# 361254       36.702812  34.459195   620     531
# 489756       47.569117  35.635784  1705    1482
# 675201       39.733698  33.835566   131     109
# 874521       43.619565  34.400860   750     607

df["category_id"].nunique()  # 6 unique


# QUESTION 1:  Does the price of the item differ according to the categories? Express it statistically?

# A/B tests should be done between different categories. Assumptions (normality and homogeneity of variance) need to be
# checked in order to decide whether to use parametric or nonparametric tests.

# HYPOTHESES:
# H0: There is no statistically significant difference between the categories
# H1: There is  statistically significant difference between the categories


# --------------------
# Normality Assumption
# --------------------
# H0 = Normality assumption is provided
# H1 = Normality assumption is NOT provided


def normality_test(i):
    test_stats, p_value = shapiro(df.loc[df["category_id"] == i, "price"])
    print("Category ID: ", i, "-->" " Test Statistic= %.5f, p-value= %.5f" % (test_stats, p_value))


for i in df["category_id"].unique():
    normality_test(i)
# Category ID:  489756 --> Test Statistic= 0.55251, p-value= 0.00000
# Category ID:  361254 --> Test Statistic= 0.30580, p-value= 0.00000
# Category ID:  874521 --> Test Statistic= 0.45945, p-value= 0.00000
# Category ID:  326584 --> Test Statistic= 0.39809, p-value= 0.00000
# Category ID:  675201 --> Test Statistic= 0.41619, p-value= 0.00000
# Category ID:  201436 --> Test Statistic= 0.61898, p-value= 0.00000


"""All categories' p-values less than 0.05 ---> reject H0 hypothesis. Normality is not ensured.
The Assumption of Normality is NOT provided in any of the above categories. 
Therefore, a non-parametric test which is mannwhitneyu test should be applied.
Since normality is not provided, no need to look at the homogeneity of variance."""

# --------------------
# HYPOTHESES TESTING     NON-PARAMETRIC -----> MANNWHITNEYU
# --------------------
# HYPOTHESES:
# H0: There is no statistically significant difference between the categories
# H1: There is  statistically significant difference between the categories

# FIRST, combinations of id's are derived:
for hyp in list(itertools.combinations(df["category_id"].unique(), 2)):
    print(hyp)


# (489756, 361254)
# (489756, 874521)
# (489756, 326584)
# (489756, 675201)
# (489756, 201436)
# (361254, 874521)
# (361254, 326584)
# (361254, 675201)
# (361254, 201436)
# (874521, 326584)
# (874521, 675201)
# (874521, 201436)
# (326584, 675201)
# (326584, 201436)
# (675201, 201436)


def mann_whit_u(hypo):
    test_stats, pvalue = stats.mannwhitneyu(df.loc[df["category_id"] == hypo[0], "price"],
                                            df.loc[df["category_id"] == hypo[1], "price"])
    print(hypo)
    print('Test Statistic = %.4f, p-value = %.4f' % (test_stats, pvalue))


for hypo in list(itertools.combinations(df["category_id"].unique(), 2)):
    mann_whit_u(hypo)

# (489756, 361254)
# Test Statistic = 380060.0000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (489756, 874521)
# Test Statistic = 519398.0000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (489756, 326584)
# Test Statistic = 69998.5000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (489756, 675201)
# Test Statistic = 86723.5000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (489756, 201436)
# Test Statistic = 60158.0000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (361254, 874521)
# Test Statistic = 218106.0000, p-value = 0.0241
# H0 is REJECTED. There is a significant difference between the two related categories.

# (361254, 326584)
# Test Statistic = 33158.5000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (361254, 675201)
# Test Statistic = 39586.0000, p-value = 0.3249
# H0 CANNOT BE REJECTED. There is NO significant difference between the two related categories  p > 0.05

# (361254, 201436)
# Test Statistic = 30006.0000, p-value = 0.4866
# H0 CANNOT BE REJECTED. There is NO significant difference between the two related categories  p > 0.05

# (874521, 326584)
# Test Statistic = 38748.0000, p-value = 0.0000
# H0 is REJECTED. There is a significant difference between the two related categories.

# (874521, 675201)
# Test Statistic = 47522.0000, p-value = 0.2752
# H0 CANNOT BE REJECTED. There is NO significant difference between the two related categories  p > 0.05

# (874521, 201436)
# Test Statistic = 34006.0000, p-value = 0.1478
# H0 CANNOT BE REJECTED. There is NO significant difference between the two related categories  p > 0.05

# (326584, 675201)
# Test Statistic = 6963.5000, p-value = 0.0001
# H0 is REJECTED. There is a significant difference between the two related categories.

# (326584, 201436)
# Test Statistic = 5301.0000, p-value = 0.0005
# H0 is REJECTED. There is a significant difference between the two related categories.

# (675201, 201436)
# Test Statistic = 6121.0000, p-value = 0.3185
# H0 CANNOT BE REJECTED. There is NO significant difference between the two related categories  p > 0.05


# There is NO significant difference between the price of two related categories as follows:
# (361254, 675201)
# (361254, 201436)
# (874521, 675201)
# (874521, 201436)
# (675201, 201436)


# ========================================================================================

# QUESTION2: Based on the first question, what should the price of the item be? Explain why?

# According to MannWhitney U test, results indicate that there is no significant difference between the means of
# some items. A fixed price can be determined by looking at the mean and median of the Category Id & prices.
# Any desired price can be selected among these products.
# Combinations of items that there is no significant difference between the mean of items as follows : (H0 cannot be rejected)
# (361254, 675201)
# (361254, 201436)
# (874521, 675201)
# (874521, 201436)
# (675201, 201436)

df.groupby('category_id').agg({'price': ['mean', 'median', "count"]})

#                  price
#                   mean     median count
# category_id
# 201436*      36.175498  33.534678    97
# 326584       36.739739  31.748242   145
# 361254*      36.702812  34.459195   620
# 489756       47.569117  35.635784  1705
# 675201*      39.733698  33.835566   131
# 874521       43.619565  34.400860   750

# Let's decide the mean of medians --->   33.93

# ========================================================================================

# QUESTION3: It is desired to be "movable- dynamic" in terms of price. Create a decision support system for
# the price strategy.

# Price flexibility can be created by focusing on 95% confidence intervals in Category Id items.
# In addition, a wide scale can be selected by looking at the confidence intervals between those who do not have a
# significant difference between each other's means.

for i in df["category_id"].unique():
    print('{0}: {1}'.format(i, sms.DescrStatsW(df.loc[df["category_id"] == i, "price"]).tconfint_mean()))

# 489756 price range for category decision-support system: (46.08434746302928, 49.05388670944087)
# 361254 price range for category decision-support system: (35.42887870193408, 37.97674480809039)
# 874521 price range for category decision-support system: (41.37178582892473, 45.86734400455721)
# 326584 price range for category decision-support system: (33.88356818130745, 39.595908835042025)
# 675201 price range for category decision-support system: (36.01515731082091, 43.45223940145658)
# 201436 price range for category decision-support system: (34.381720084633564, 37.96927659690045)

# =========================================================================================

# QUESTION4 : Simulate item purchases and income for possible price changes (for each category)
