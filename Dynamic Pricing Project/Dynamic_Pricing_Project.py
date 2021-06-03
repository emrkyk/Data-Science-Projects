# PRICING: Item fiyatı ne olmalı?

"""Bir oyun şirketi bir oyununda kullanıcılarına item satın alımları için hediye paralar vermiştir.
Kullanıcılar bu sanal paraları kullanarak karakterlerine çeşitli araçlar satın almaktadır.
Oyun şirketi bir item için fiyat belirtmemiş ve kullanıcılardan bu item'ı istedikleri fiyattan almalarını sağlamış.
Örneğin kalkan isimli item için kullanıcılar kendi uygun gördükleri miktarları ödeyerek bu kalkanı satın alacaklar.
Örneğin bir kullanıcı kendisine verilen sanal paralardan 30 birim, diğer kullanıcı 45 birim ile ödeme yapabilir.
Dolayısıyla kullanıcılar kendilerine göre ödemeyi göze aldıkları miktarlar ile bu item'ı satın alabilirler."""

# Çözülmesi gereken problemler:
# Item'in fiyatı kategorilere göre farklılık göstermekte midir? İstatistiki olarak ifade ediniz.
# İlk soruya bağlı olarak item'ın fiyatı ne olmalıdır? Nedenini açıklayınız?
# Fiyat konusunda "hareket edebilir olmak" istenmektedir. Fiyat stratejisi için karar destek sistemi oluşturunuz.
# Olası fiyat değişiklikleri için item satın almalarını ve gelirlerini simüle ediniz.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import shapiro
from scipy.stats.stats import pearsonr
from scipy.stats import stats
import statsmodels.stats.api as sms
from scipy import stats
import itertools

data = pd.read_csv("Week_5/pricing.csv", sep=';')
df = data.copy()
df.head()

df.info()
df.describe().T

df["category_id"].unique()  # array([489756, 361254, 874521, 326584, 675201, 201436]
df["category_id"].nunique()  # 6

from dsmlbc4.helpers.eda import *
from dsmlbc4.helpers.data_prep import *
from dsmlbc4.helpers import *

# OUTLIERS?
check_df(df)

outlier_thresholds(df, "price")  # (-64.46732638496243, 187.44554397493738)

replace_with_thresholds(df, "price")

df["category_id"].value_counts()
df["category_id"].unique()
df.groupby("category_id").agg({"price": ["mean", "median", "count", "nunique"]})


# SORU!!!! Item'in fiyatı kategorilere göre farklılık göstermekte midir? İstatistiki olarak ifade ediniz. ???


# Farklı kategoriler arasında A/B testleri yapılmalıdır. Parametrik veya nonparametrik testlerinden hangisi olacağına karar
# verebilmek  için varsayımların (normallik ve varyans homojenliği) kontrol edilmesi gerekmektedir.

# Hipotez:
# H0: Kategoriler arasında anlamlı bir farklılık yoktur
# H1: Kategoriler arasında anlamlı bir farklılık vardır

# Normallik Varsayımı
# H0 = Normal dagılım varsayımı saglanmaktadır.
# H1 = Normal dagılım varsayımı saglanmamaktadır.

# öncelikle normallik varsayımını kontrol edelim:
# Normallik Varsayımı

def normallik_kontrolü(i):
    test_istatistigi, p_value = shapiro(df.loc[df["category_id"] == i, "price"])
    print("Test İstatistiği= %.5f, p-value= %.5f" % (test_istatistigi, p_value))


for i in df["category_id"].unique():
    normallik_kontrolü(i)


# Test İstatistiği= 0.55251, p-value= 0.00000
# Test İstatistiği= 0.30580, p-value= 0.00000
# Test İstatistiği= 0.45945, p-value= 0.00000
# Test İstatistiği= 0.39809, p-value= 0.00000
# Test İstatistiği= 0.41619, p-value= 0.00000
# Test İstatistiği= 0.61898, p-value= 0.00000

# Tüm kategorilerin p-value degerleri 0.05 den küçük =>> reject Ho hypothesis. Normallik saglanmamaktadir.
# Yukarıdaki kategorilerin hiçbirinde normallik varsayımı sağlanmamaktadır. Bu yüzden non-parametrik bir test olan
# mannwhitneyu testi uygulanmalıdır.
# Normallik sağlanmadığından Varyans homojenligine de bakmamiza gerek kalmamıştır.


# Hypothesis:
# H0: Katogoriler arasında anlamlı bir farklılık yokur
# H1: Katogoriler arasında anlamlı bir farklılık vardır

# Hipotez Testi
def mann_whit_u(hypo):
    test_istatistigi, pvalue = stats.mannwhitneyu(df.loc[df['category_id'] == hypo[0], 'price'],
                                                  df.loc[df['category_id'] == hypo[1], 'price'])
    print(hypo)
    print('Test İstatistiği = %.4f, p-value = %.4f' % (test_istatistigi, pvalue))


for hypo in list(itertools.combinations(df['category_id'].unique(), 2)):
    mann_whit_u(hypo)

# (489756, 361254)
# Test İstatistiği = 380060.0000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (489756, 874521)
# Test İstatistiği = 519398.0000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (489756, 326584)
# Test İstatistiği = 69998.5000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (489756, 675201)
# Test İstatistiği = 86723.5000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (489756, 201436)
# Test İstatistiği = 60158.0000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (361254, 874521)
# Test İstatistiği = 218106.0000, p-value = 0.0241
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (361254, 326584)
# Test İstatistiği = 33158.5000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (361254, 675201)
# Test İstatistiği = 39586.0000, p-value = 0.3249
# H0 hipotezi REDDEDILEMEZ. İlgili iki kategori arasında anlamlı bir fark YOKTUR. p > 0.05

# (361254, 201436)
# Test İstatistiği = 30006.0000, p-value = 0.4866
# H0 hipotezi REDDEDILEMEZ. İlgili iki kategori arasında anlamlı bir fark YOKTUR. p > 0.05


# (874521, 326584)
# Test İstatistiği = 38748.0000, p-value = 0.0000
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (874521, 675201)
# Test İstatistiği = 47522.0000, p-value = 0.2752
# H0 hipotezi REDDEDILEMEZ. İlgili iki kategori arasında anlamlı bir fark YOKTUR. p > 0.05

# (874521, 201436)
# Test İstatistiği = 34006.0000, p-value = 0.1478
# H0 hipotezi REDDEDILEMEZ. İlgili iki kategori arasında anlamlı bir fark YOKTUR. p > 0.05


# (326584, 675201)
# Test İstatistiği = 6963.5000, p-value = 0.0001
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.


# (326584, 201436)
# Test İstatistiği = 5301.0000, p-value = 0.0005
# H0 hipotezi reddedilir. İlgili iki kategori arasında anlamlı bir fark vardır.

# (675201, 201436)
# Test İstatistiği = 6121.0000, p-value = 0.3185
# H0 hipotezi REDDEDILEMEZ. İlgili iki kategori arasında anlamlı bir fark YOKTUR. p > 0.05


# ========================================================================================


# SORU: İlk soruya bağlı olarak item'ın fiyatı ne olmalıdır? Nedenini açıklayınız?

# Category Id fiyatlarının mean ve medyanlarina bakarak sabit bir fiyat belirlenebilir. Yukarıdaki testlerde,
# itemlarin ortalamalari arasinda anlamli bir fark olmadıgı test etmiştik. Bu urunler arasinda istenilen herhangi
# bir fiyat seçilebilir. (Ho hipotezi reddedilemezleri getirdik)

# (361254, 675201)
# (361254, 201436)
# (874521, 675201)
# (874521, 201436)
# (675201, 201436)

df.groupby('category_id').agg({'price': ['mean', 'median', "count"]})


# ========================================================================================


# SORU: Fiyat konusunda "hareket edebilir olmak" istenmektedir. Fiyat stratejisi için karar destek sistemi oluşturunuz.

# Category Id itemlarda %95 lik güven aralıklarina odaklanarak fiyat esnekliği yaratılır.
# Ayrıca birbirlerinin ortalamalari arasinda anlamli bir fark olmayanlar arasinda guven aralıklarına bakarak
# genis bir skala seçilebilir.

for i in df['category_id'].unique():
    print('{0}: {1}'.format(i, sms.DescrStatsW(df.loc[df['category_id'] == i, 'price']).tconfint_mean()))

# 489756: (46.08434746302928, 49.05388670944087)
# 361254: (35.42887870193408, 37.97674480809039)
# 874521: (41.37178582892473, 45.86734400455721)
# 326584: (33.88356818130745, 39.595908835042025)
# 675201: (36.01515731082091, 43.45223940145658)
# 201436: (34.381720084633564, 37.96927659690045)

# ========================================================================================


# SORU: Olası fiyat değişiklikleri için item satın almalarını ve gelirlerini simüle ediniz.

df["price"].mean() # 43.68

df.groupby('category_id').agg({'price': ['mean', 'count']})


