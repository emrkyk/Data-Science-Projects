# =============================
# HITTERS LIGHTGBM
# =============================

# Summary of Dataset
# AtBat: Number of times at bat in 1986
# Hits: Number of hits in 1986
# HmRun: Number of home runs in 1986
# Runs: Number of runs in 1986
# RBI: Number of runs batted in in 1986
# Walks: Number of walks in 1986
# Years: Number of years in the major leagues
# CAtBat: Number of times at bat during his career
# CHits: Number of hits during his career
# CHmRun: Number of home runs during his career
# CRuns: Number of runs during his career
# CRBI: Number of runs batted in during his career
# CWalks: Number of walks during his career
# League: A factor with levels A and N indicating player's league at the end of 1986
# Division: A factor with levels E and W indicating player's division at the end of 1986
# PutOuts: Number of put outs in 1986
# Assists: Number of assists in 1986
# Errors: Number of errors in 1986
# Salary: 1987 annual salary on opening day in thousands of dollars    ====>>> TARGET VARIABLE
# NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987

import pandas as pd
import numpy as np
from helpers.helpers import *
from helpers.eda import *
from helpers.data_prep import *
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option("display.expand_frame_repr", False)

data = pd.read_csv("Datasets/hitters.csv")
df = data.copy()
df.head()
df.info()

check_df(df)

# -----------------
# # Missing Values - Eksik Gözlemler
# -----------------

missing_values_table(df)
#         n_miss  ratio
# Salary      59 18.320


# VARIABLES!
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

# ---------------------------
# Outliers / Aykırı Gözlemler
# ---------------------------
df.describe([0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

for col in num_cols:
    print(col, check_outlier(df, col))

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

corr = df.corr()
plt.figure(figsize=(18, 10))
sns.heatmap(corr, annot=True)
plt.show()

check_df(df)  # 20
df.head()

df.loc[(df['Years'] < 5), 'Experience'] = 'inexperienced'
df.loc[(df['Years'] >= 5) & (df['Years'] < 10), 'Experience'] = 'experienced'
df.loc[(df['Years'] >= 10), 'Experience'] = 'senior'

df["Ratio_CAtBat"] = df["AtBat"] / df["CAtBat"]
df["Ratio_CHits"] = df["Hits"] / df["CHits"]
df["Ratio_CHmRun"] = df["HmRun"] / df["CHmRun"]
df["Ratio_Cruns"] = df["Runs"] / df["CRuns"]
df["Ratio_CRBI"] = df["RBI"] / df["CRBI"]
df["Ratio_CWalks"] = df["Walks"] / df["CWalks"]

df['CAtBat_average'] = df['CAtBat'] / df['Years']
df['CHits_average'] = df['CHits'] / df['Years']
df['CHmRun_average'] = df['CHmRun'] / df['Years']
df['CRun_average'] = df['CRuns'] / df['Years']
df['CRBI_average'] = df['CRBI'] / df['Years']
df['CWalks_average'] = df['CWalks'] / df['Years']

df["General Performance"] = df["PutOuts"] + df["Assists"] - df["Errors"]

df.loc[(df["Years"] > df["Years"].mean()) & (
        df["General Performance"] > df["General Performance"].mean()), "PERFORMANCE"] = "Good"
df.loc[(df["Years"] < df["Years"].mean()) & (
        df["General Performance"] > df["General Performance"].mean()), "PERFORMANCE"] = "Good"
df.loc[(df["Years"] > df["Years"].mean()) & (
        df["General Performance"] > df["General Performance"].mean()), "PERFORMANCE"] = "Reasonable"
df.loc[(df["Years"] < df["Years"].mean()) & (
        df["General Performance"] < df["General Performance"].mean()), "PERFORMANCE"] = "Bad"

df.columns = [col.upper() for col in df.columns]

df.head()
df.shape

# --------------------------
# LABEL ENCODING                 # Conversions related to representation of variables
# --------------------------
# Expressing categorical variables ===> numerical
#  Neden? Bazı fonk. kategorik tipte değişkenler yerine bunları sayısal olarak temsil edebilecek bir versiyonunu ister.
# Özellikle 2 sınıflı değişkenleri labellarını değiştiriyoruz, binary encoding de denebilir.

binary_cols = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique() == 2]

# ['LEAGUE', 'DIVISION', 'NEWLEAGUE']

for col in binary_cols:
    df = label_encoder(df, col)

check_df(df)

# --------------------------
# ONE-HOT ENCODING
# --------------------------

# İkiden fazla sınıfa sahip olan kategorik değişkenlerin 1-0 olarak encode edilmesi.
# Sadece 2 sınıfı olan değişkenlere de uygulanabilir.

onehot_e_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

df = one_hot_encoder(df, onehot_e_cols)

df.head()
df.shape
check_df(df)

# --------------------------
#  MINMAXSCALER
# --------------------------

from sklearn.preprocessing import MinMaxScaler

scal_cols = [col for col in df.columns if df[col].nunique() > 20
             and df[col].dtype != 'O'
             and col not in "SALARY"]

scaler = MinMaxScaler(feature_range=(0, 1))

df[scal_cols] = scaler.fit_transform(df[scal_cols])

df.head()
df.shape
check_df(df)

df.dropna(inplace=True)

y = df["SALARY"]  # dependent variable
X = df.drop(["SALARY"], axis=1)  # independent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

#######################################
# LightGBM: Model & Tahmin
#######################################

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # 313

#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # 302


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


plot_importance(lgbm_tuned, X_train)
