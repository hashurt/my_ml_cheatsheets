import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme: 'eksikveriler.csv' dosyasını okur ve bir DataFrame'e yükler.
# Burada 'eksikveriler.csv' yerine kendi veri dosyanızın yolunu kullanmalısınız.
veriler = pd.read_csv("eksikveriler.csv")

# Eksik verilerle başa çıkmak için gerekli kütüphaneyi içe aktarır.
from sklearn.impute import SimpleImputer

# SimpleImputer sınıfından bir nesne oluşturur:
# - `missing_values=np.nan` eksik verilerin `NaN` (Not a Number) olarak işaretlendiğini belirtir.
# - `strategy='mean'` eksik değerlerin ortalama ile doldurulacağını belirtir.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Verileri seçer ve `Yas` adında bir numpy dizisi oluşturur:
# - `veriler.iloc[:,1:4].values` DataFrame'in 1. ile 3. sütunları arasındaki verileri alır.
Yas = veriler.iloc[:,1:4].values

# İmputer nesnesini `Yas` dizisinde öğrenme (fit) aşamasına geçirir.
# Bu, eksik değerlerin ortalama ile nasıl doldurulacağını öğrenir.
imputer = imputer.fit(Yas[:,0:3])

# Öğrenilen stratejiyi uygulayarak eksik verileri doldurur.
# `transform` metodu, eksik değerleri ortalama ile doldurur.
Yas[:,0:3] = imputer.transform(Yas[:,0:3])

# Ülke verilerini seçer ve `ulke` adında bir numpy dizisi oluşturur:
# - `veriler.iloc[:,0:1].values` DataFrame'in ilk sütunundaki verileri alır.
ulke = veriler.iloc[:,0:1].values

# Kategorik verileri sayısal verilere dönüştürmek için gerekli kütüphaneyi içe aktarır.
from sklearn import preprocessing

# LabelEncoder sınıfından bir nesne oluşturur:
# - `le` nesnesi, kategorik verileri sayısal verilere dönüştürür.
le = preprocessing.LabelEncoder()

# Ülke isimlerini sayısal değerlere dönüştürür.
# `fit_transform` metodu, kategorileri sayısal değerlere dönüştürür.
ulke[:,0] = le.fit_transform(ulke[:,0])

# OneHotEncoder sınıfından bir nesne oluşturur:
# - `ohe` nesnesi, kategorik verileri one-hot encoding (tek sıcak kodlama) ile dönüştürür.
ohe = preprocessing.OneHotEncoder()

# OneHotEncoder kullanarak ülke verilerini one-hot kodlamaya dönüştürür.
# `toarray` metodu, sonucu bir numpy dizisine çevirir.
ulke = ohe.fit_transform(ulke).toarray()

# `ulke` ve `Yas` dizilerini pandas DataFrame'lere dönüştürür:
# - `sonuc`, one-hot kodlama yapılmış ülke verilerini içerir.
# - `sonuc2`, eksik veriler doldurulmuş yaş, boy ve kilo verilerini içerir.
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])

# Cinsiyet verilerini seçer ve `cinsiyet` adında bir numpy dizisi oluşturur:
# - `veriler.iloc[:,-1].values` DataFrame'in son sütunundaki verileri alır.
cinsiyet = veriler.iloc[:,-1].values

# Cinsiyet verilerini pandas DataFrame'e dönüştürür.
sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])

# DataFrame'leri birleştirir:
# - `s` DataFrame'i, `sonuc` (one-hot kodlama yapılmış ülke verileri) ve `sonuc2` (eksik veriler doldurulmuş yaş, boy ve kilo) verilerini birleştirir.
# - `s2` DataFrame'i, `s` ve `sonuc3` (cinsiyet verileri) DataFrame'lerini birleştirir.
s = pd.concat([sonuc, sonuc2], axis=1)
s2 = pd.concat([s, sonuc3], axis=1)

# Verileri eğitim ve test kümelerine ayırmak için gerekli kütüphaneyi içe aktarır.
from sklearn.model_selection import train_test_split

# `train_test_split` fonksiyonunu kullanarak verileri eğitim ve test kümelerine böler:
# - `s` (özellikler) ve `sonuc3` (hedef değişken) verilerini böler.
# - `test_size=0.33` test kümesinin verilerin %33'ünü içereceğini belirtir.
# - `random_state=0` sonuçların tekrarlanabilirliğini sağlar.
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

# Özellik ölçekleme işlemi için gerekli kütüphaneyi içe aktarır.
from sklearn.preprocessing import StandardScaler

# StandardScaler sınıfından bir nesne oluşturur:
# - `sc` nesnesi, verileri standardize eder (ortalama 0, standart sapma 1).
sc = StandardScaler()

# Eğitim ve test verilerini standardize eder:
# - `fit_transform` metodu, eğitim verilerine göre ölçekleme yapar ve bu ölçekleme bilgilerini test verilerine uygular.
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
