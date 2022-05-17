import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
dosyakonumu = pd.read_csv('heart.csv')
ilk5veri=dosyakonumu.head()
print(ilk5veri)
bilgi=dosyakonumu.info()
print(bilgi)
tanım=dosyakonumu.describe()
print(tanım)
yasi48_67arasıolanhastasayilari=dosyakonumu.age.value_counts()[:20]
print(yasi48_67arasıolanhastasayilari)
sns.barplot(x= dosyakonumu.age.value_counts()[:20].index, y= dosyakonumu.age.value_counts()[:20].values  )
plt.xlabel('Yaş')
plt.ylabel("Yaş sayıları")
plt.title("Yaş Analizi")
plt.show()
hastaolanlarinsayisi = len(dosyakonumu[dosyakonumu.target == 0])
hastaolmayanlarinsayisi = len(dosyakonumu[dosyakonumu.target == 1])
print("Kalp hastalığı olmayan hastaların yüzdesi: {:.2f}%".format((hastaolanlarinsayisi/(len(dosyakonumu.target)))*100))
print("Kalp hastalığı olan hastaların yüzdesi: {:.2f}%".format((hastaolmayanlarinsayisi/(len(dosyakonumu.target)))*100))
kadinSayisi= len(dosyakonumu[dosyakonumu.sex == 0])
erkekSayisi = len(dosyakonumu[dosyakonumu.sex == 1])
print("Kadın Hasta Yüzdesi: {:.2f}%".format((kadinSayisi/(len(dosyakonumu.sex))*100)))
print("Erkek Hasta Yüzdesi: {:.2f}%".format((erkekSayisi/(len(dosyakonumu.sex))*100)))

renkler1 = ['blue','green',]
explode= [1,1,1]
plt.figure(figsize= (8,8))
plt.pie([kadinSayisi, erkekSayisi ], labels=['Kadın', 'Erkek'])
plt.show()
Aclik_kan_sekeri_120ustu=dosyakonumu[(dosyakonumu.fbs==1)]
Aclik_kan_sekeri_120alti=dosyakonumu[(dosyakonumu.fbs==0)]
print("Aclik_kan_sekeri_120ustu", len(Aclik_kan_sekeri_120ustu))
print("Aclik_kan_sekeri_120alti", len(Aclik_kan_sekeri_120alti))
renkler2 = ['blue','green',]
explode= [1,1,1]
plt.figure(figsize= (8,8))
plt.pie([len(Aclik_kan_sekeri_120ustu), len(Aclik_kan_sekeri_120alti), ], labels=['Aclik_kan_sekeri_120ustu', 'Aclik_kan_sekeri_120alti'])
plt.show()

gogus_agrisi_tipi0=dosyakonumu[(dosyakonumu.cp==0)]
gogus_agrisi_tipi1=dosyakonumu[(dosyakonumu.cp==1)]
gogus_agrisi_tipi2=dosyakonumu[(dosyakonumu.cp==2)]
gogus_agrisi_tipi3=dosyakonumu[(dosyakonumu.cp==3)]
print("gogus_agrisi_tipi0", len(gogus_agrisi_tipi0))
print("gogus_agrisi_tipi1", len(gogus_agrisi_tipi1))
print("gogus_agrisi_tipi2", len(gogus_agrisi_tipi2))
print("gogus_agrisi_tipi3", len(gogus_agrisi_tipi3))

renkler3 = ['blue','green', 'red','yellow']
explode= [1,1,1]
plt.figure(figsize= (8,8))
plt.pie([len(gogus_agrisi_tipi0), len(gogus_agrisi_tipi1), len(gogus_agrisi_tipi2),len(gogus_agrisi_tipi3)], labels=['gogus_agrisi_tipi0', 'gogus_agrisi_tipi1', 'gogus_agrisi_tipi2','gogus_agrisi_tipi3'])
plt.show()

genc_yas = dosyakonumu[(dosyakonumu.age>=29)&(dosyakonumu.age<40)]
orta_yas = dosyakonumu[(dosyakonumu.age>=40)&(dosyakonumu.age<55)]
yasli = dosyakonumu[(dosyakonumu.age>=55)]
print("genc_yas", len(genc_yas))
print("orta_yas", len(orta_yas))
print("yasli", len(yasli))

renkler4 = ['blue','green', 'red']
explode= [1,1,1]
plt.figure(figsize= (8,8))
plt.pie([len(genc_yas), len(orta_yas), len(yasli)], labels=['genc_yas', 'orta_yas', 'yasli'])
plt.show()

dosyakonumu.corr()
# Logistic Regression
from sklearn.linear_model import LogisticRegression
x_data = dosyakonumu.drop(['target'], axis = 1)
y = dosyakonumu.target.values

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.2, random_state= 0)


lr = LogisticRegression()

lr.fit(x_train, y_train)

predict=lr.predict(x_test)
print(y_test)
print(predict)
n1=np.array(predict)
n2=np.array(y_test)
line1=np.array([0,1])
line2=np.array([0,1])
plt.xlabel("predict")
plt.ylabel("y_test")
plt.scatter(n1,n2)
plt.plot(line1,line2,color='red')
plt.show()
dogru=0
yanlis=0
for i in range(len(y_test)):
  if(y_test[i]==predict[i]):
    print("y_test[",i,"]=",y_test[i],"tahmin[",i,"]=",predict[i]," dogru tahmin etmis")
    dogru+=1
  else:
    print("y_test[",i,"]=",y_test[i],"tahmin[",i,"]=",predict[i]," yanlis tahmin etmis")
    yanlis+=1
print("Dogru Tahmin adedi:",dogru)
print("Yanlis Tahmin adedi:",yanlis)
print('Test Accuracy {:.2f}%'.format(lr.score(x_test, y_test)*100))