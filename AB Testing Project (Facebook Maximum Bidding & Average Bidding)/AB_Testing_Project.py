###############################################################
#   A/B TESTING PROJECT  ( AVERAGE BIDDING & MAXIMUM BIDDING)
###############################################################
"""
  *** BUSINESS PROBLEM ***
- Facebook recently introduced a new type of bidding named "Average Bidding," as an alternative to the existing bidding system
  called "Maximum Bidding".
- Untitled_Company.com has decided to test this new feature. The company aims to measure whether the new bid type
  "Average Bidding" yields better results than current bid type "Maximum Bidding" using A/B testing.

- Research Question: Is "Average Bidding" different/ better than "Maximum Bidding"?

- "Untitled_Company.com" company's target audience has been randomly divided into two groups of equal size.
  Control Group : Maximum Bidding Version.
  Test Group: Average Bidding Version

- The ultimate measure of success for "Untitled_Company.com" company is "Purchase" metric.
- So, Mainly focused on the "Purchase" metric for statistical testing first,
- Later, out of curiosity, I also created a new variable for "Conversation Rate" and did research on and conducted related tests for "Earning" and  "Conversation Rate" 
  metrics and came up with interesting results!
- You can find the presentation in the repository.
  """

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import statsmodels.stats.api as sms
from helpers.eda import *
from helpers.helpers import *
from helpers.data_prep import *
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, levene
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

control_group = pd.read_excel("Datasets/ab_testing_data.xlsx", sheet_name="Control Group")
test_group = pd.read_excel("Datasets/ab_testing_data.xlsx", sheet_name="Test Group")

##################################################
#             DATA EXPLORATION                   #
##################################################

check_df(control_group)
# =======================
# CONTROL GROUP
# =======================
# Number of Observations = 40
# Shape (40, 4)
# No null values
# No outlier
# ##################### Types #####################
# Impression    float64
# Click         float64
# Purchase      float64
# Earning       float64
# ##################### Head #####################
#    Impression   Click  Purchase  Earning
# 0    82529.46 6090.08    665.21  2311.28
# 1    98050.45 3382.86    315.08  1742.81
# 2    82696.02 4167.97    458.08  1797.83
# ##################### Quantiles #####################
#                0.00     0.05     0.50      0.95      0.99      1.00
# Impression 45475.94 79412.02 99790.70 132950.53 143105.79 147539.34
# Click       2189.75  3367.48  5001.22   7374.36   7761.80   7959.13
# Purchase     267.03   328.66   531.21    748.27    790.19    801.80
# Earning     1253.99  1329.58  1975.16   2318.53   2481.31   2497.30


check_df(test_group)
# =======================
# TEST GROUP
# =======================
# Number of Observations = 40
# Shape (40, 4)
# No null values
# No outlier
# ##################### Types #####################
# Impression    float64
# Click         float64
# Purchase      float64
# Earning       float64
# dtype: object
# ##################### Head #####################
#    Impression   Click  Purchase  Earning
# 0   120103.50 3216.55    702.16  1939.61
# 1   134775.94 3635.08    834.05  2929.41
# 2   107806.62 3057.14    422.93  2526.24
# ##################### Quantiles #####################
#                0.00     0.05      0.50      0.95      0.99      1.00
# Impression 79033.83 83150.50 119291.30 153178.69 158245.26 158605.92
# Click       1836.63  2600.36   3931.36   5271.19   6012.88   6019.70
# Purchase     311.63   356.70    551.36    854.21    876.58    889.91
# Earning     1939.61  2080.98   2544.67   2931.31   3091.94   3171.49


# Confidence Intervals for Purchase Variable (%95)
sms.DescrStatsW(control_group["Purchase"]).tconfint_mean()
# (508.0041754264924, 593.7839421139709)

sms.DescrStatsW(test_group["Purchase"]).tconfint_mean()
# (530.5670226990063, 633.645170597929)


# VISUALIZATION
sns.boxplot(x=control_group['Purchase'])
plt.xlabel("'Purchase' in control group")
plt.show()

sns.boxplot(x=test_group["Purchase"])
plt.xlabel("'Purchase' in test group")
plt.show()

# Concatenate purchase variables from the both group to compare and draw KDE plot
A_purchase = control_group[["Purchase"]]
B_purchase = test_group[["Purchase"]]
AB_purchase = pd.concat([A_purchase, B_purchase], axis=1)
AB_purchase.columns = ["Control_Purchase", "Test_Purchase"]
AB_purchase.head()
#    Control_Purchase  Test_Purchase
# 0            665.21         702.16
# 1            315.08         834.05
# 2            458.08         422.93
# 3            487.09         429.03
# 4            441.03         749.86

AB_purchase.mean()
# Control_Purchase   550.89
# Test_Purchase      582.11

sns.kdeplot(data=AB_purchase, shade=True)
plt.show()

#######################################################################
#        DEFINING A/B TESTING FUNCTIONS AND IMPLEMENTATION            #
#######################################################################

# Since I aim to conduct the tests multiple times, I've created functions.

# ==============================================
# 1. TESTING ASSUMPTIONS
# ==============================================
""" In order to conduct the A/B test, first of all, 2 statistical assumptions must be ensured beforehand. 
    1. The Assumption of Normality
    2. The Assumption of Homogeneity of Variance"""

# --------------------------------
# 1.1 THE ASSUMPTION OF NORMALITY           ----> Shapiro-Wilk Test
# --------------------------------
""" It is necessary to test whether the distribution of a variable is the same as the theoretical normal distribution.
    The Shapiro-Wilk test is conducted to examine if the variable has normal distribution.
    
    Hypothesis 
    H0: The assumption of the normality is provided.
    H1: The assumption of the normality is NOT provided. 
    """


def normality_test(dataframe, col_name, plot=False):
    """The Assumptions of normality: The first theoretical assumption of AB testing to be tested.
    Most of the parametric tests require that the assumption of normality be met.
    The Shapiro-Wilk test is used to examine whether the distribution of the data as a whole
    deviates from a comparable normal distribution.

    H0: The assumption of the normal distribution is provided.
    H1: The assumption of the normal distribution is NOT provided."""

    # from scipy.stats import shapiro
    test_stats, p_value = shapiro(dataframe[col_name])
    print("Test Statistic= %.4f, p-value = %.4f" % (test_stats, p_value))

    if plot:
        stats.probplot(dataframe[col_name], dist="norm", plot=pylab)
        pylab.title("Q-Q Plot")
        pylab.show()


normality_test(control_group, "Purchase", True)
# Test Statistic= 0.9773, p-value = 0.5891
# p-value = 0.5891 > 0.05 => H0 cannot be rejected
# The assumption of the normality is provided for Purchase variable in Control Group


normality_test(test_group, "Purchase", True)
# Test Statistic= 0.9589, p-value = 0.1541
# p-value = 0.1541 > 0.05 => H0 cannot be rejected
# The assumption of the normality is provided for Purchase variable in Test Group

# FIRST ASSUMPTION IS PASSED!


# ---------------------------------------------
# 1.2 THE ASSUMPTION OF HOMOGENEITY OF VARIANCE           ----> Levene Test
# ---------------------------------------------
"""
    H0: Variances are homogenous
    H1: Variances are not homogenous"""


def testing_variance_homogeneity(arg1, arg2):
    """
    The Assumption of Homogeneity of Variance: The second theoretical assumption of AB testing to be tested.
    Levene's test is used to test whether the variances of the distributions of two variables are similar.
    (whether variances are homogenous)

    H0: Variances are homogenous
    H1: Variances are not homogenous
    """
    # from scipy import stats
    test_stats, p_value = stats.levene(arg1, arg2)
    print("Test Statistic= %.4f, p-value = %.4f" % (test_stats, p_value))


testing_variance_homogeneity(control_group["Purchase"], test_group["Purchase"])
# Test Statistic= 2.6393, p-value = 0.1083
# p-value = 0.10 > 0.05 => H0 cannot be rejected
# The assumption of homogeneity of variance is provided which means that variances are homogenous, the variances of
# the distributions of two variables are similar.


# ==============================================
# 2.          A/B TESTING for Purchase
# ==============================================

""" The parametric test(A/B Testing-Independent two-sample t-test)  will be applied since two assumptions are 
    ensured beforehand.
    
    # control_group["Purchase"].mean() => 550.89
    # test_group["Purchase"].mean()   =>  582.11

    HYPOTHESES
    # H0: M1 = M2 (There is no statistically significant difference between the Purchase means of the Control 
                   and Test Groups.
    # H1: M1 != M2 (There is a statistically significant difference between the Purchase means of the Control
                   and Test Groups.
    """


def ab_testing(arg1, arg2):
    """
    If the assumptions of normality & variance homogeneity are met=> AB TESTING(Independent two-sample t-test- parametric)
    If the assumptions are not met => MannWhitneyU (nonparametric)"""
    # H0: M1 = M2  (There is no statistically significant difference between the groups with % 95 confidence)
    # H1: M1 != M2 (There is a statistically significant difference between the groups with % 95 confidence)

    test_stats, p_value = stats.ttest_ind(arg1, arg2, equal_var=True)
    print("Test Statistic= %.4f, p-value = %.4f" % (test_stats, p_value))


ab_testing(control_group["Purchase"], test_group["Purchase"])
# Test Statistic= -0.9416, p-value = 0.3493
# Since p-value = 0.34 > 0.05 ===> H0 cannot be rejected!
# There is no statistically significant difference between the Purchase means of the Control and Test Groups.

""" SUMMARY: 
    When evaluated statistically according to the Purchase variable, there is no statistically significant 
    difference  between the means of Purchase in the Control and Test Groups. 
    All in all, it turns out that there is no statistically significant difference between the maximum bidding 
    version(control) and averaged bidding version(test) with % 95 confidence in terms of Purchase variable"""


# I also defined the MannWhitnetU test function just in case, I might need later.

def mann_whitney_u_test(arg1, arg2):
    """
    If the assumptions of normality & variance homogeneity are not met=> MannWhitneyU (nonparametric)"""
    # H0: M1 = M2  (There is no statistically significant difference between the groups with % 95 confidence)
    # H1: M1 != M2 (There is a statistically significant difference between the groups with % 95 confidence)

    test_stats, p_value = stats.mannwhitneyu(arg1, arg2)
    print("Test Statistic= %.4f, p-value = %.4f" % (test_stats, p_value))


# =======================================
# EARNING
# =======================================

sns.boxplot(x=control_group["Earning"])
plt.xlabel("'Earning' in control group")
plt.show()

sns.boxplot(x=test_group["Earning"])
plt.xlabel("'Earning' in test group")
plt.show()

A_earning = control_group[['Earning']]
B_earning = test_group[['Earning']]
AB_earning = pd.concat([A_earning, B_earning], axis=1)
AB_earning.columns = ['Control_Earning', 'Test_Earning']
AB_earning.head()
#    Control_Earning  Test_Earning
# 0      2311.277143   1939.611243
# 1      1742.806855   2929.405820
# 2      1797.827447   2526.244877
# 3      1696.229178   2281.428574
# 4      1543.720179   2781.697521


AB_earning.mean()
# Control_Earning   1908.57
# Test_Earning      2514.89

sns.kdeplot(data=AB_earning, shade=True)
plt.show()

# ========================================
# AB TESTING FOR EARNING
# ========================================
"""  HYPOTHESES
    # H0: M1 = M2   (There is no statistically significant difference between the means of Earning variable in 
                    the Control and Test Groups)
    # H1: M1 != M2  (There is a statistically significant difference between the means of Earning variable in 
    #                the Control and Test Groups))"""
# ------------------------------------------
# 1.1 THE ASSUMPTION OF NORMALITY  (Earning)
# ------------------------------------------

#  Hypotheses
#  H0: The assumption of the normality is provided.
#  H1: The assumption of the normality is NOT provided.

normality_test(control_group, "Earning", True)
# Test Statistic= 0.9756, p-value = 0.5306
# p-value = 0.5306 > 0.05 => H0 cannot be rejected
# The assumption of the normality is provided for Earning variable in Control Group


normality_test(test_group, "Earning", True)
# Test Statistic= 0.9780, p-value = 0.6163
# p-value = 0.6163 > 0.05 => H0 cannot be rejected
# The assumption of the normality is provided for Earning variable in Test Group


# FIRST ASSUMPTION IS PASSED FOR EARNING

# ------------------------------------------
# 1.2 THE ASSUMPTION OF HOMOGENEITY OF VARIANCE (Earning)
# ------------------------------------------
# Hypotheses
# H0: Variances are homogenous
# H1: Variances are not homogenous

testing_variance_homogeneity(control_group["Earning"], test_group["Earning"])
# Test Statistic= 0.3532, p-value = 0.5540
# p-value = 0.5540 > 0.05 => H0 cannot be rejected
# The assumption of the homogeneity of variance is provided for Earning variable. Variances are homogenous.

# The assumptions are ensured for A/B Testing, so parametric test will be conducted!

# ========================================
# 2. A/B TESTING (Earning)
# ========================================

ab_testing(control_group["Earning"], test_group["Earning"])
# Test Statistic= -9.2545, p-value = 0.0000

"""Since p-value = 0.0000  < 0.05 ===>  H0 is rejected.
 There is a statistically significant difference between the means of Earning variable in the Control and Test Groups """


# ===========================
# SIMPLE FEATURE ENGINEERING
# ===========================
# Let's calculate the Conversion rate and observe!

# Conversion Rate
control_group["Conversion Rate"] = (control_group["Purchase"] / control_group["Click"]) * 100
test_group["Conversion Rate"] = (test_group["Purchase"] / test_group["Click"]) * 100

control_group.head()
#    Impression   Click  Purchase  Earning  Conversion Rate
# 0    82529.46 6090.08    665.21  2311.28            10.92
# 1    98050.45 3382.86    315.08  1742.81             9.31
# 2    82696.02 4167.97    458.08  1797.83            10.99
# 3   109914.40 4910.88    487.09  1696.23             9.92
# 4   108457.76 5987.66    441.03  1543.72             7.37

test_group.head()
#    Impression   Click  Purchase  Earning  Conversion Rate
# 0   120103.50 3216.55    702.16  1939.61            21.83
# 1   134775.94 3635.08    834.05  2929.41            22.94
# 2   107806.62 3057.14    422.93  2526.24            13.83
# 3   116445.28 4650.47    429.03  2281.43             9.23
# 4   145082.52 5201.39    749.86  2781.70            14.42


# ========================================
# AB TESTING FOR CONVERSION RATE
# ========================================

"""  HYPOTHESES
    # H0: M1 = M2   (There is no statistically significant difference between the means of "Conversion Rate" variable in 
                    the Control and Test Groups)
    # H1: M1 != M2  (There is a statistically significant difference between the means of "Conversion Rate" variable in 
    #                the Control and Test Groups))"""

# --------------------------------------------------
# 1.1 THE ASSUMPTION OF NORMALITY  (Conversion Rate)
# --------------------------------------------------
#  Hypotheses
#  H0: The assumption of the normality is provided.
#  H1: The assumption of the normality is NOT provided.

normality_test(control_group, "Conversion Rate", True)
# Test Statistic= 0.8720, p-value = 0.0003

# p_value for control group is: 0.0003 < 0.05  => H0 is rejected!
# Conversion Rate variable in control group does not seem to have a normal distribution.

normality_test(test_group, "Conversion Rate", True)
# Test Statistic= 0.8381, p-value = 0.0000

# p_value for test group is: 0.0000 < 0.05  => H0 is rejected!
# Conversion Rate variable in test group does not seem to have a normal distribution.


# THE ASSUMPTION OF NORMALITY IS NOT PASSED!  ===> NONPARAMETRIC

# --------------------------------------------------
# 1.2 THE ASSUMPTION OF HOMOGENEITY OF VARIANCE (Conversion Rate)
# --------------------------------------------------
# Hypotheses
# H0: Variances are homogenous
# H1: Variances are not homogenous

testing_variance_homogeneity(control_group["Conversion Rate"], test_group["Conversion Rate"])
# Test Statistic= 2.0759, p-value = 0.1536

# p_value for homogenous variances: 0.1536  > 0.05   => H0 cannot be rejected!
# Variances are homogenous. The second assumption is ensured.

# -----------------------------------------------------
# 2. A/B TESTING (Conversion Rate)  ===>> NONPARAMETRIC
# -----------------------------------------------------

""" HYPOTHESES
    H0: M1 = M2   (There is no statistically significant difference between the means of "Conversion Rate" variable in 
                    the Control and Test Groups)
    H1: M1 != M2  (There is a statistically significant difference between the means of "Conversion Rate" variable in 
                   the Control and Test Groups))"""

mann_whitney_u_test(control_group["Conversion Rate"], test_group["Conversion Rate"])
# Test Statistic= 459.0000, p-value = 0.0005

""" p-value for mann-whitney-u non-parametric test = 0.0005  < 0.05  ==> H0 is rejected!
    There is a statistically significant difference between the means of "Conversion Rate" variable in the Control and 
    Test Groups with 95 % confidence. Test group's Conversation Rate is higher than the control group."""





##############################################################
#     -------    FINAL REMARKS       -------                #
##############################################################

# 1. Purchase Variable
""" There is no statistically significant difference between the maximum bidding 
    version(control) and averaged bidding version(test) with % 95 confidence in terms of Purchase variable"""

# 2. Earning Variable
"""There is a statistically significant difference between the means of Earning variable in the Control and Test Groups"""

# 3. Conversation Rate Variable
"""There is a statistically significant difference between the means of "Conversion Rate" variable in the Control and 
  Test Groups with 95 % confidence. Test group's Conversation Rate is higher than the control group."""





