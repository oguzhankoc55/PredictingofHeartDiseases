# Predicting of Heart Diseases 

Bu projede Makine öğrenmesi modellerinden olan logis tic regrasyon Kalp Hastalıklarının varlığını tahmin edilmiştir. 
Logistic Regresyonun Matematiği ve Python da uygulanması:
İlk olarak : Veri kodda kullanılabilecek şekilde hazırlanır ve koda aktarılır 2.olarak : Sınıflandırma işlemlerinin yapılabilmesi için eğitim yapmak gerekir 3. olarak Lojistik regresyon kullanılarak tahmin işlemi gerçekleştirilir.
Lojistik regresyon, verilere koordinat sisteminde en mükemmel eğriyi çizer. Bu eğrinin adı Sigmoid eğrisidir. Bu eğriyi Sigmoid Fonksiyonu başlığı ile inceleyeceğiz.
# Sigmoid Fonksiyonu
Sigmoid Fonksiyonu, gerçek değerli sayıyı alır ve onu 0 ile 1 arasında birdeğer ile eşleştiren bir eğri verir.Bu fonksiyon Logaritmik bir fonksiyondur.:
## Sigmoid Fonksiyonu Formüller

![image](https://user-images.githubusercontent.com/58952369/180183588-76053027-41bd-4719-83fa-56085c562ee2.png)


Q(t) fonksiyonu t ‘nin farklı değerleri için [0 ,1] arasında hangi kategoride olduğuna dairolasılık bilgilerinin tümünü içerir.
Lojistik regresyonda çizgi arasındaki mesafe hata miktarı bulan aşağıdaki formülle hesaplanır. Bu fonksiyon logaritmiktir.Logaritmik birfonksiyonu kullanılırken amaç eğrideki değişimlere daha kolay uyum sağlaması ve noktalar ile eğriler arasındaki mesafeyi minimum değerde elde edebilmektir.
## Hata miktar toplamı formulü:

![image](https://user-images.githubusercontent.com/58952369/180183635-2938aa39-316c-487c-a5ae-0591a75178f3.png)


 Yaptığım Projede öncelikle kullanılması gerekli fonksiyonları import ettim:

![image](https://user-images.githubusercontent.com/58952369/180183678-4ac01294-8599-4517-8103-28fac9eec628.png)


Sonrasında ise verileri heart.csv uzantılı dosyadan çektik:

![image](https://user-images.githubusercontent.com/58952369/180183709-096a1c9c-3582-4b87-bec5-eda848447bc4.png)


Sonrasında ise bazı verileri görselleştirerek verilerin kullanıcı tarafından görülmesini sağladım:

![image](https://user-images.githubusercontent.com/58952369/180183743-21070121-d028-4508-a0ae-a5e0fae075db.png)

![image](https://user-images.githubusercontent.com/58952369/180183843-4e9e0d7a-8bde-48cb-bf9c-bdede119097e.png)


Ardından verilerin eğitimin gerçekleştirdim:

![image](https://user-images.githubusercontent.com/58952369/180183866-a40bf7a8-7b9d-4262-8e3f-29c7478645ee.png)


Bunu ardından ise tahmin işlemini gerçekleştirdim:

![image](https://user-images.githubusercontent.com/58952369/180183878-f18ec24d-782d-4089-9229-ecdd5c95943f.png)


Bunların doğruluğunu kontrol etmek amacıyla verileri terminalde yazdırdım:

![image](https://user-images.githubusercontent.com/58952369/180183913-9b97b142-938b-44ec-9579-d796f3e5b8b7.png)


# Problem : 

## Temel vektör ve matris iş lemleri ile ilgili örnek olmalı

Boyutlarının ve değerlerinin kullanıcı tarafından girildiği iki matrisin çarpımının yapıldığı bir program hazırlayınız.İki matrisin çarpılacağını varsayalım.

## Matris Çarpımının Tanımı

Eğer A matrisi n × m boyutlu ve B matrisi m × p boyutlu ise;

![image](https://user-images.githubusercontent.com/58952369/180184515-d84756f4-51e2-4ad6-9de7-d6485efff2b4.png)


AB matrisi için çarpma n × p matrisi şeklinde gösterilir.

![image](https://user-images.githubusercontent.com/58952369/180184553-1a21b71c-b729-4df1-a151-e257d6b58373.png)


Şekilsel gösterim

Aşağıdaki şekilde, A ve B iki matrisininçarpımı gösteriliyor. Sonuç olarak yeni matrisin boyutu 4’e 3 matrisi oluyor.

![image](https://user-images.githubusercontent.com/58952369/180184575-82a7765c-bc09-4cc9-80e6-3065aa96897f.png)


Şekilde, çemberle işaretlenen hücrelerin değerleri şunlardır:

![image](https://user-images.githubusercontent.com/58952369/180184619-4d0da058-cc8f-4d8c-b776-f0ed5a33c296.png)


Yukarıdakiler, X matrisinin belirlenen girişleridir.
Programımızda matematiksel işlemlerden önce kullanıcıdan veri girişi alınır:


![image](https://user-images.githubusercontent.com/58952369/180184681-f5055dd3-5434-4d16-ae48-d382b574ff35.png)

Bu aldığımız verileri önceden hazırladığımız matematiksel işlemlerin yapıldığı matriscarpimfonk adlı fonksiyonumuza gönderiyoruz:


![image](https://user-images.githubusercontent.com/58952369/180184704-a2e6fd5f-a46f-4d29-8bb8-018ac6d759d6.png)


Bu fonksuyonumuzda öncelikle gönderilen matrislerin çarpım yapmaya uygun olup olmadığını kontrol ediyoruz.Bunun ilk matrisin sütun değerinin ikinci matrisinde satır kısmının eşit olması kontrol edilir:


![image](https://user-images.githubusercontent.com/58952369/180184745-90d437fd-d14b-4e15-8b35-1f1319dd74c9.png)


Eğer uygunsa matrislerin içini kullanıcıya doldurtuyoruz:

![image](https://user-images.githubusercontent.com/58952369/180184776-70ae9e59-5dba-4554-ab54-7b536eeac988.png)

Bu işlem bittikten sonra kullanıcının toplam sonucunu görmesi için toplam işlemini yapıyoruz:

![image](https://user-images.githubusercontent.com/58952369/180184797-f4a8fc99-a31e-4316-9993-3d498c027b7d.png)




