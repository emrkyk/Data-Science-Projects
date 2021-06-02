import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_control = pd.read_excel("Datasets/ab_testing_data.xlsx", sheet_name='Control Group')
df_test = pd.read_excel("Datasets/ab_testing_data.xlsx", sheet_name='Test Group')

df_control.head()
df_control.describe().T
df_test.describe().T

df_control.info()
df_test.info()

############################
# 1. Varsayım Kontrolü
############################

# 1.1 Normallik Varsayımı
# 1.2 Varyans Homojenliği

############################
# 1.1 Normallik Varsayımı
############################
# df_control için
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:.Normal dağılım varsayımı sağlanmamaktadır.

test_istatistigi, pvalue = shapiro(df_control["Purchase"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05'ten HO REDDEDILEBİLİR.
# p-value < değilse 0.05 H0 REDDEDİLEMEZ.

# Test İstatistiği = 0.9773, p-değeri = 0.5891
# p-değeri > 0.05 olduğundan HO reddedilemez, dolayısıyla Normal dağılım varsayımı sağlanmaktadır.

################################################################
###############################################################

# df_test için
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

test_istatistigi, pvalue = shapiro(df_test["Purchase"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Test İstatistiği = 0.9589, p-değeri = 0.1541
# p-değeri > 0.05 olduğundan HO reddedilemez, dolayısıyla Normal dağılım varsayımı sağlanmaktadır.

#####################################################


############################
# 1.2 Varyans Homojenligi Varsayımı
############################


# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_istatistigi, pvalue = levene(df_control['Purchase'], df_test['Purchase'])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Test İstatistiği = 2.6393, p-değeri = 0.1083 , p_value değeri > 0.05 olduğu için HO reddedilemez, Varyanslar homojendir, varsayım
# dogrudur.


# 1.1 Yukarıdaki varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 1.2 Yukarıdaki varsayımlar sağlanmıyorsa MannWhitneyU testi (non-parametrik test uygulanır)

# bağımsız iki örneklem t testi (parametrik test)
# H0: M1 = M2 ( iki grup ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 ( anlamlı fark vardır)

test_istatistigi, pvalue = ttest_ind(df_control['Purchase'], df_test['Purchase'], equal_var=True)
print('Test Statistics = %.4f, p-value = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Sonuc olarak:
# Test Statistics = -0.9416, p-value = 0.3493, p değeri > 0.05 , dolayısıyla HO reddedilemez
# iki grup ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.