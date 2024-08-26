import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri.yukleme
#veriler.csv'yi kendi doayanla degistir
veriler = pd.read_csv("veriler.csv")

#verileri kullanma
print(veriler)

boy =veriler[['boy']]
print(boy)