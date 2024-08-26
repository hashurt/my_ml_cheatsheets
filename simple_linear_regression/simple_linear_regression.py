import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme: 'eksikveriler.csv' dosyasını okur ve bir DataFrame'e yükler.
# Burada 'eksikveriler.csv' yerine kendi veri dosyanızın yolunu kullanmalısınız.
veriler = pd.read_csv("satislar.csv")


aylar =veriler[['Aylar']]
satislar=veriler[['Satislar']]






# Verileri eğitim ve test kümelerine ayırmak için gerekli kütüphaneyi içe aktarır.
from sklearn.model_selection import train_test_split

# `train_test_split` fonksiyonunu kullanarak verileri eğitim ve test kümelerine böler:
# - `s` (özellikler) ve `sonuc3` (hedef değişken) verilerini böler.
# - `test_size=0.33` test kümesinin verilerin %33'ünü içereceğini belirtir.
# - `random_state=0` sonuçların tekrarlanabilirliğini sağlar.
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)


"""
# Özellik ölçekleme işlemi için gerekli kütüphaneyi içe aktarır.
from sklearn.preprocessing import StandardScaler

# StandardScaler sınıfından bir nesne oluşturur:
# - `sc` nesnesi, verileri standardize eder (ortalama 0, standart sapma 1).
sc = StandardScaler()

# Eğitim ve test verilerini standardize eder:
# - `fit_transform` metodu, eğitim verilerine göre ölçekleme yapar ve bu ölçekleme bilgilerini test verilerine uygular.
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

"""

#model (linear regresyon)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#eğitim
lr.fit(x_train, y_train)
#tahmin
tahmin=lr.predict(x_test)

#görselleştirme
#tabloda düzgün sıralanması için sıraya dizme
x_train=x_train.sort_index()
y_train=y_train.sort_index()


#zaten bir doğru çizeceğinden test çıktınısını sıralamaya çok da gerek yok ama yine de sıraladım
x_test=x_test.sort_index()

#sırasıyla train verileri(verinin kendisi) ve tahmin doğrusunu çizdirme
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("tabloya başlık")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")