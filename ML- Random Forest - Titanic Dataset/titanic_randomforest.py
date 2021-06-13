#############################################
# TITANIC Random Forest
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from helpers.data_prep import *
from helpers.eda import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

train_data = pd.read_csv("Datasets/train.csv")
test_data = pd.read_csv("Datasets/test.csv")
df = train_data.copy()

check_df(df)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

missing_values_table(df)

#CABIN, NAME, CABIN
# ===================
# FEATURE ENGINEERING
# ===================

def tita_data_prep(dataframe):
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["Age"] * dataframe["Pclass"]

    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) > 0), "NEW_ALONE"] = "NO"
    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) == 0), "NEW_ALONE"] = "YES"

    dataframe.loc[(dataframe['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 18) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['Sex'] == 'male') & (
            (dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (
            (dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Missing Values
    dataframe.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)  # cat_but_car
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,
                                axis=0)

    # Outliers
    num_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) > 20
                and dataframe[col].dtypes != 'O'
                and col not in "PASSENGERID"]

    # for col in num_cols:
    #    print(col, check_outlier(titanic_df, col))
    # print(check_df(titanic_df))

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # Label Encoding
    binary_cols = [col for col in dataframe.columns if
                   len(dataframe[col].unique()) == 2 and dataframe[col].dtypes == 'O']

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    dataframe = rare_encoder(dataframe, 0.01)

    ohe_cols = [col for col in dataframe.columns if 10 >= len(dataframe[col].unique()) > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    return dataframe


df_train = tita_data_prep(df)

df_test = tita_data_prep(test_data)

y = df_train["SURVIVED"]
X = df_train.drop(["SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=21)
rf_model.fit(X_train, y_train)

# train hatasÄ±
train_pred = rf_model.predict(X_train)
train_prob = rf_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, train_pred))
roc_auc_score(y_train, train_prob)  # 1.0

# test hatasÄ±
test_pred = rf_model.predict(X_test)
test_prob = rf_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, test_pred))
roc_auc_score(y_test, test_prob)  # 0.83984

#               precision    recall  f1-score   support
#            0       0.80      0.87      0.83       157
#            1       0.79      0.68      0.73       111
#     accuracy                           0.79       268
#    macro avg       0.79      0.78      0.78       268
# weighted avg       0.79      0.79      0.79       268

# Model Tuning

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 8, 15],
             "n_estimators": [200, 300, 500],
             "min_samples_split": [2, 5, 8]}

rf_model = RandomForestClassifier(random_state=21)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_

# {'max_depth': 8,
#  'max_features': 3,
#  'min_samples_split': 8,
#  'n_estimators': 200}

# Tuned Model

rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)
test_pred = rf_tuned.predict(X_test)
test_prob = rf_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, test_pred))
roc_auc_score(y_test, test_prob)  # 0.8363


#               precision    recall  f1-score   support
#            0       0.79      0.87      0.83       157
#            1       0.79      0.68      0.73       111
#     accuracy                           0.79       268
#    macro avg       0.79      0.77      0.78       268
# weighted avg       0.79      0.79      0.79       268


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


plot_importance(rf_tuned, X)

df_test = df_test.fillna(df_test.mean())

# titanic_rf_results = pd.DataFrame()

# titanic_rf_results["PassengerId"] = df_test["PASSENGERID"]

# titanic_rf_results["Survived"] = rf_tuned.predict(df_test)

# titanic_rf_results.to_csv("titanic_pred.csv", index=False)

