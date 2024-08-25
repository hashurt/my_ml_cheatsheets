import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri.yukleme
#veriler.csv'yi kendi doayanla degistir
veriler = pd.read_csv("eksikveriler.csv")

#eksik veriler 
from sklearn.impute import SimpleImputer

#SimpleiImputer class'ından obje tanımlama: burada hangi değerlerin değiştireleceği ve strategy tanımlaması yapıldı
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#veriler bölünüyor(integerLocation fonksiyonu)
Yas = veriler.iloc[:,1:4].values
#print(Yas)

#öğrenme aşaması(belirlenmiş strategye göre), fit fonksiyonu öğrenme aşamasında kullanılıyor
imputer=imputer.fit(Yas[:,0:3])

#öğrenilen değerin uygulanma/transfprm aşaması
Yas[:,0:3]= imputer.transform(Yas[:,0:3])
#print(Yas)


ulke= veriler.iloc[:,0:1].values
#print(ulke)

#bu aşamada kategorik veriler numerik veriye dönüştürülüyor
from sklearn import preprocessing
le =preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])
#print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke =ohe.fit_transform(ulke).toarray()
#print(ulke)


#print(list(range(22)))
#verilerin birleştirilmesi
#bu kısımda veri grupları data frameler olarak derlendi
sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
#print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index=range(22), columns= ['boy','kilo','yas'])
#print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
sonuc3=pd.DataFrame(data= cinsiyet, index= range(22),columns=['cinsiyet'])
#print(sonuc3)


#dataframelerin birleştirilmesi

s=pd.concat([sonuc,sonuc2], axis=1)
#print(s)

s2=pd.concat([s,sonuc3], axis=1)
#print(s2)


#Verilerin eğitim ve Test Kümelerine Bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)




