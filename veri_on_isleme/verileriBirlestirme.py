import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri.yukleme
#veriler.csv'yi kendi doayanla degistir
veriler = pd.read_csv("eksikveriler.csv")

#verileri kullanma

#print(veriler)




#eksik veriler 
from sklearn.impute import SimpleImputer


#SimpleiImputer class'ından obje tanımlama: burada hangi değerlerin değiştireleceği ve strategy tanımlaması yapıldı
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#veriler bölünüyor(integerLocation fonksiyonu)
Yas = veriler.iloc[:,1:4].values
print(Yas)


#öğrenme aşaması(belirlenmiş strategye göre), fit fonksiyonu öğrenme aşamasında kullanılıyor
imputer=imputer.fit(Yas[:,0:3])

#öğrenilen değerin uygulanma/transfprm aşaması
Yas[:,0:3]= imputer.transform(Yas[:,0:3])

print(Yas)


ulke= veriler.iloc[:,0:1].values
print(ulke)

#bu aşamada kategorik veriler numerik veriye dönüştürülüyor
from sklearn import preprocessing
le =preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke =ohe.fit_transform(ulke).toarray()
print(ulke)



#print(list(range(22)))
#verilerin birleştirilmesi
sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

