#############################################################
#   HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES - LIGHTGBM   RMSE = 0.14
#############################################################

""" https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
    Data Description file is available in the repository.
    SalePrice is the target variable & 79 explanatory variables describing (almost) every aspect of residential homes"""
# The functions I optimized such as check_df() in the Functional_EDA_Data_Prep repository

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from helpers.data_prep import *
from helpers.eda import *
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Uniting train and test sets.
train = pd.read_csv("Datasets/train.csv")
test = pd.read_csv("Datasets/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()

check_df(df)  # (2919, 81)               # 81 variable
df.isnull().any().sum()                  # 35 null values in total

#########################################
#                EDA                    #
#########################################

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
print("cat_cols: ", len(cat_cols))

num_cols = [col for col in df.columns if df[col].dtypes != "O"]
print("num_cols: ", len(num_cols))

check_df(df)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)  # cardinality threshold = 10

# cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
# Observations: 2919
# Variables: 81
# cat_cols: 50  --> ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'YrSold']
# num_cols: 28  --> ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'SalePrice']
# cat_but_car: 3  --> ['Neighborhood', 'Exterior1st', 'Exterior2nd']
# num_but_cat: 10    <---   (already included in "cat_cols". Just given for reporting purposes)
# {cat_cols + num_cols + cat_but_car = all variables}


# -------------------------------
# Categorical Variables Analysis
# -------------------------------

""" Real CATEGORICAL VARIABLES => cat_cols. 
    (cat_but_car excluded --> since it looks categorical but cardinality is high 
    (num_but_cat ------> looks numerical but cardinality is so less that measurability exhibits categorical features!  
    """

for col in cat_cols:
    cat_summary(df, col, True)
#          MSZoning  Ratio
# RL           2265 77.595
# RM            460 15.759
# FV            139  4.762
# RH             26  0.891
# C (all)        25  0.856
# ========================
#          Street   Ratio
# Pave      2907    99.589
# Grvl        12     0.411
# ========================
#         Alley    Ratio
# Grvl      120      4.111
# Pave       78      2.672
# ..........
# ........
# ......


# CAT BUT CAR           -----> CATEGORICAL but cardinality is high!

for col in cat_but_car:
    cat_summary(df, col, True)

for col in ["Neighborhood", "Exterior1st", "Exterior2nd"]:
    print(df[col].value_counts())

#          Neighborhood  Ratio
# NAmes             443 15.176
# CollgCr           267  9.147
# OldTown           239  8.188
# Edwards           194  6.646
# Somerst           182  6.235
# NridgHt           166  5.687
# Gilbert           165  5.653
# Sawyer            151  5.173
# NWAmes            131  4.488
# ......           ....  .....
# =============================
#          Exterior1st  Ratio
# VinylSd         1025 35.115
# MetalSd          450 15.416
# HdBoard          442 15.142
# Wd Sdng          411 14.080
# Plywood          221  7.571
# CemntBd          126  4.317
# ......           ....  .....
# =============================
#          Exterior2nd  Ratio
# VinylSd         1014 34.738
# MetalSd          447 15.313
# HdBoard          406 13.909
# Wd Sdng          391 13.395
# Plywood          270  9.250
# # ......        ....  .....


# NUM BUT CAT
"""Variables that are numerical as its data type but exhibit categorical properties due to the low cardinality
 num_but_cat variables are evaluated as categorical variables due to its measurability. 
 that's why num_but_cat variables take place in cat_cols"""

for col in num_but_cat:
    cat_summary(df, col, True)

#    OverallCond  Ratio
# 5         1645 56.355
# 6          531 18.191
# 7          390 13.361
# 8          144  4.933
# 4          101  3.460
# 3           50  1.713
# 9           41  1.405
# 2           10  0.343
# 1            7  0.240
# ===================
#        BsmtFullBath  Ratio
# 0.000          1705 58.410
# 1.000          1172 40.151
# 2.000            38  1.302
# 3.000             2  0.069
# ===================
#        BsmtHalfBath  Ratio
# 0.000          2742 93.936
# 1.000           171  5.858
# 2.000             4  0.137
# .....          .... .....
# .....          .... .....
# .....          .... .....


# -------------------------------
# Numerical Variables Analysis
# -------------------------------

num_summary(df, num_cols, plot=True)

num_hist_boxplot(df, num_cols)

# -------------------------------
# Target Variable Analysis
# -------------------------------

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])
# count     1460.000
# mean    180921.196
# std      79442.503
# min      34900.000
# 5%       88000.000
# 10%     106475.000
# 25%     129975.000
# 50%     163000.000
# 75%     214000.000
# 80%     230000.000
# 90%     278000.000
# 95%     326100.000
# 99%     442567.010
# max     755000.000
# Name: SalePrice, dtype: float64

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)  # --->  df.groupby("MSZoning").agg({"SalePrice": "mean"})
    #  --->  df.groupby("Street").agg({"SalePrice": "mean"})
    # for each cat_cols


#           TARGET_MEAN
# MSZoning
# C (all)     74528.000
# FV         214014.062
# RH         131558.375
# RL         191004.995
# RM         126316.830

#         TARGET_MEAN
# Street
# Grvl     130190.500
# Pave     181130.539
# .....          ....
# .....          ....
# .....          ....


# =============================================================
# Correlations of dependent variable and independent variables
# =============================================================

def correlation_heatmap(dataframe):
    _, ax = plt.subplots(figsize=(36, 16))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(dataframe.corr(), annot=True, cmap=colormap)
    plt.show()


correlation_heatmap(df)


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

high_corrs
# High Correlated variables   according to correlation threshold =  0.60
# 'OverallQual: 0.7909816005838047',
#  'TotalBsmtSF: 0.6135805515591944',
#  '1stFlrSF: 0.6058521846919166',
#  'GrLivArea: 0.7086244776126511',
#  'GarageArea: 0.6234314389183598'

high_correlated_list = ["OverallQual", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageArea"]

for i in high_correlated_list:
    sns.boxplot(x=i, y="SalePrice", data=df,
                whis=[0, 100], width=.6, palette="vlag")
    plt.show()

correlation_heatmap(df[high_correlated_list])

# =======================================================
#                   FEATURE ENGİNEERİNG                 #
# =======================================================

# 1stFlrSF: First Floor square feet
df["1stFlrSF_cat"] = pd.qcut(x=df["1stFlrSF"], q=3, labels=["small", "normal", "big"])
df[["1stFlrSF", "1stFlrSF_cat"]].head(10)

# GrLivArea: Above grade (ground) living area square feet
df["GrLivArea_cat"] = pd.qcut(x=df["GrLivArea"], q=3, labels=["small", "normal", "big"])
df[["GrLivArea", "GrLivArea_cat"]].head(10)

# Yearbuilt
df["Yearbuilt_cat"] = pd.qcut(x=df["YearBuilt"], q=3, labels=["ancient", "old", "new"])
df[["YearBuilt", "Yearbuilt_cat"]].head(10)

df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["Restoration_interval"] = df["YearRemodAdd"] - df["YearBuilt"]

df["OutArea"] = df["OpenPorchSF"] + df["PoolArea"] + df["WoodDeckSF"] + df["GarageArea"]
df["OverallMultp"] = df["OverallCond"] * df["OverallQual"]

df[["TotalSF", "OutArea"]].head()

# ExterQual: Evaluates the quality of the material on the exterior
# HeatingQC: Heating quality and condition
# GarageQual: Garage quality

quality_categories = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
df["ExterQual"] = df["ExterQual"].map(quality_categories).astype("int")
df["HeatingQC"] = df["HeatingQC"].map(quality_categories).astype("int")
df[["ExterQual", "HeatingQC"]].head()

df["Total_bath"] = df["FullBath"] + (df["HalfBath"] * 0.5)
df["GarageQual"] = df["GarageQual"].map(quality_categories)
df["GarageQual"].fillna(0, inplace=True)
df["BsmtQual"] = df["BsmtQual"].map(quality_categories)
df["BsmtQual"].fillna(0, inplace=True)
df["GarageQual"] = df["GarageQual"].astype("int")
df["BsmtQual"] = df["BsmtQual"].astype("int")

missing_values_table(df)

years_cat = {"new": 3, "old": 2, "ancient": 1}
df["Yearbuilt_cat"] = df["Yearbuilt_cat"].map(years_cat).astype("int")

classes = {'big': 3, 'normal': 2, 'small': 1}
df["1stFlrSF_cat"] = df["1stFlrSF_cat"].map(classes).astype("int")
df["GrLivArea_cat"] = df["GrLivArea_cat"].map(classes).astype("int")

df["Foundation"].value_counts()

df.groupby("Foundation").agg({"SalePrice": sum})
#                SalePrice
# Foundation
# BrkTil      19314497.000
# CBlock      94976823.000
# PConc      145724096.000
# Slab         2576775.000
# Stone         995755.000
# Wood          557000.000

types = {"Wood": 1, "Stone": 2, "Slab": 4, "BrkTil": 16, "CBlock": 160, "PConc": 320}
df["Foundation_types"] = df["Foundation"].map(types).astype("int")

# ======================================================= #
#                    MISSING_VALUES                       #
# ======================================================= #

missing_values_table(df)

#               n_miss  ratio
# PoolQC          2909 99.660
# MiscFeature     2814 96.400
# Alley           2721 93.220
# Fence           2348 80.440
# SalePrice       1459 49.980
# FireplaceQu     1420 48.650
# LotFrontage      486 16.650
# GarageYrBlt      159  5.450
# GarageCond       159  5.450
# ..........       ...  .....
# ........          ..   ....


na_cols = missing_values_table(df, True)  # keep variables that have null values

[col for col in na_cols if df[col].dtype == "O"]
df[na_cols].head()

upd_na_cols = ["BsmtFinType1", "BsmtFinType2", "MasVnrType", "Alley", "PoolQC", "MiscFeature", "Fence", "FireplaceQu",
               "GarageType", "GarageFinish", "GarageCond", "BsmtCond", "BsmtExposure"]

for col in upd_na_cols:
    df[col].replace(np.nan, 0, inplace=True)

keep_list = ["GrLivArea_cat", "1stFlrSF_cat", "Total_bath", "ExterQual", "GarageQual", "Foundationtipe", "BsmtQual",
             "HeatingQC", "Yearbuilt_cat"]

check_df(df)

# UPDATED VARIABLES!
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

# Observations: 2919
# Variables: 90
# cat_cols: 56  --> ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'OverallCond', 'ExterQual', 'BsmtQual', 'HeatingQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'GarageQual', 'YrSold', '1stFlrSF_cat', 'GrLivArea_cat', 'Yearbuilt_cat', 'Foundation_types']
# num_cols: 33  --> ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'SalePrice', 'TotalSF', 'Restoration_interval', 'OutArea', 'OverallMultp', 'Total_bath']
# cat_but_car: 1  --> ['Neighborhood']
# num_but_cat: 18    <---   (already included in "cat_cols". Just given for reporting purposes)
# {cat_cols + num_cols + cat_but_car = all variables}


# ==================
#   RARE ENCODING
# ==================

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)

drop_these = ["PoolQC", "MiscFeature", "Street", "Utilities", "LandSlope"]

# Rare Encoder
# PoolQC : 4
#     COUNT  RATIO  TARGET_MEAN
# 0    2909  0.997   180404.663
# Gd      4  0.001   201990.000
# Ex      4  0.001   490000.000
# Fa      2  0.001   215500.000

# MiscFeature : 5
#       COUNT  RATIO  TARGET_MEAN
# 0      2814  0.964   182046.410
# Shed     95  0.033   151187.612
# Gar2      5  0.002   170750.000
# Othr      4  0.001    94000.000
# TenC      1  0.000   250000.000

# Street : 2
#       COUNT  RATIO  TARGET_MEAN
# Grvl     12  0.004   130190.500
# Pave   2907  0.996   181130.539


cat_colss = [col for col in cat_cols if col not in drop_these]
cat_cols = [col for col in cat_colss if col not in keep_list]

for col in drop_these:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", 0.01)

# =================================
# LABEL ENCODING & ONE-HOT ENCODING
# =================================

cat_cols2 = cat_cols + cat_but_car
cat_cols = [col for col in cat_cols2 if col not in keep_list]

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)
df.shape  # (2919, 232)

# =================================
#      MISSING_VALUES             #
# =================================

missing_values_table(df)
#              n_miss  ratio
# SalePrice      1459 49.980
# LotFrontage     486 16.650
# GarageYrBlt     159  5.450
# MasVnrArea       23  0.790
# OutArea           1  0.030
# TotalSF           1  0.030
# GarageArea        1  0.030
# TotalBsmtSF       1  0.030
# BsmtUnfSF         1  0.030
# BsmtFinSF2        1  0.030
# BsmtFinSF1        1  0.030


null_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]
df[null_cols] = df[null_cols].apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

df[null_cols].head()

# # categorical variables have been filled by the mode of these variables
# df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
# df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
#
missing_values_table(df)


# =====================
#       OUTLIERS
# =====================

"""Outliers have been replaced with thresholds"""

len(num_cols)  # 33

for col in num_cols:
    print(col, check_outlier(df, col))

a = check_outlier(df, col)

for col in num_cols:
    if a == True:
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

######################################
# SPLIT TRAIN & TEST SET
######################################

train_df = df[df["SalePrice"].notnull()]
test_df = df[df["SalePrice"].isnull()].drop("SalePrice", axis=1)

train_df.to_pickle("Datasets/train_df.pkl")
test_df.to_pickle("Datasets/test_df.pkl")

#######################################
# MODEL: LGBM
#######################################

X = train_df.drop(["SalePrice", "Id"], axis=1)
y = np.log1p(train_df["SalePrice"])

y = train_df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=32)

lgbm_model = LGBMRegressor(random_state=32).fit(X_train, y_train)
y_pred = lgbm_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y.mean()

y_pred = lgbm_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))


#######################################
#      Model Tuning
#######################################

lgbm_params = {"learning_rate": [0.0007, 0.01, 0.1],
               "n_estimators": [1200, 2000, 2500],
               "max_depth": [5, 8, 10],
               "num_leaves": [8, 15],
               "colsample_bytree": [0.8, 0.5]}

lgbm_model = LGBMRegressor(random_state=42)
lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
lgbm_cv_model.best_params_

# {'colsample_bytree': 0.5,
#  'learning_rate': 0.01,
#  'max_depth': 8,
#  'n_estimators': 2000,
#  'num_leaves': 8}

#######################################
#        Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))


y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# RMSE: 0.14

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_tuned, X_train, 50)


# UPLOADING THE RESULTS

submission_df = pd.DataFrame()
submission_df["Id"] = test_df["Id"]
submission_df["Id"].head()

y_pred_sub = lgbm_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub
submission_df.head()

#            Id  SalePrice
# 1460 1461.000 131258.261
# 1461 1462.000 163659.464
# 1462 1463.000 180725.488
# 1463 1464.000 194537.526
# 1464 1465.000 199979.552

submission_df.to_csv("houseprice_lgbm.csv", index=False)
