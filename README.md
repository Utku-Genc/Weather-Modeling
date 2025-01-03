# Zaman Serisi Tahmin Modelleri 📈

## 1. Genel Bakış 🔍
Bu depo, zaman serisi tahmini için geliştirilmiş birden fazla derin öğrenme modelinin implementasyonlarını içermektedir. Modeller şunlardır:

1. **Informer**
2. **Reformer**
3. **Bayesian Reformer**
4. **Temporal Fusion Transformer (TFT)**
5. **Vanilla Transformer**
6. **Time Series Transformer (TST)**
7. **Autoformer**
   
Modeller, **Sıcaklık**, **Çiğ Noktası**, **Nem**, **Rüzgar Hızı** ve **Basınç** gibi günlük hava durumu verilerini içeren bir veri seti üzerinde eğitilmiş ve değerlendirilmiştir. Her model, sıralı verilerle etkili bir şekilde çalışmak için dikkat mekanizmaları ve zaman serisi odaklı iyileştirmelerle tasarlanmıştır.

## 2. Modeller ve Özellikler 🚀

 ### 1. Informer
   
****Açıklama****: ProbSparse kendi kendine dikkat mekanizması ile zaman serisi tahmini için optimize edilmiş bir Transformer.

****Öne Çıkan Özellikler****:
- Seyrek dikkat mekanizması ile artırılmış verimlilik.
- Uzun dönemli bağımlılıkların daha iyi ele alınması.

 ### 2. Reformer
   
****Açıklama****: Hafıza verimli bir Transformer varyantı, kendi kendine dikkat (self-attention) hesaplamaları için locality-sensitive hashing (LSH) kullanır.

****Öne Çıkan Özellikler****:
- Uzun sekansları verimli bir şekilde işler.
- Otoregresif tahmin için nedensel dikkat (causal attention).
- Dikkat hesaplamaları için yapılandırılabilir bucket boyutu.

 ### 3. Bayesian Reformer
   
****Açıklama****:Belirsizlik tahmini için Bayesian bileşenleri ile zenginleştirilmiş bir Reformer adaptasyonu.

****Öne Çıkan Özellikler****:
- Belirsizlik tahmini ile olasılıksal tahminler.
- Risk odaklı karar verme için faydalı.

 ### 4. Temporal Fusion Transformer (TFT)
   
****Açıklama****: Çok değişkenli zaman serisi tahmini için özel olarak tasarlanmış bir model.

****Öne Çıkan Özellikler****:
- Dinamik giriş gömme (embedding).
- Zaman ve özellik boyutlarında dikkat mekanizması.
- Entegre açıklanabilirlik özellikleri.

 ### 5. Vanilla Transformer
   
****Açıklama****: Zaman serisine uyarlanmış bir temel Transformer uygulaması.

****Öne Çıkan Özellikler****:
-Ölçeklenebilir ve esnek mimari.
- Karşılaştırma için sağlam bir temel.

### 6. Time Series Transformer (TST)
   
****Açıklama****: Zaman serileri için optimize edilmiş bir Transformer modeli.

****Öne Çıkan Özellikler****:
- Klasik Transformer mimarisine dayalı.
- Multivaryant zaman serilerinde yüksek performans.
- Giriş verilerini dikkatle işlemek için düzenlenmiş katmanlar.

 ### 7. Autoformer
   
****Açıklama****: Zaman serisi tahmini için Otomatik Korelasyon temelli bir model.

****Öne Çıkan Özellikler****:
- Mevsimsel ve eğilim bilgilerini etkili bir şekilde yakalar.
- Hafif ve hızlı.


## 3. Veri Seti 🗂️
Modeller günlük hava durumu veri seti üzerinde eğitilmiştir. Veri seti şu bilgileri içerir:
- *Özellikler*: Sıcaklık, Çiğ Noktası, Nem, Rüzgar Hızı, Basınç
- *Hedef*: Çok değişkenli tahmin.
- *Sekans Uzunluğu*: 30 gün.
- *Eğitim-Test Ayrımı*: %80-%20.
- *Ölçeklendirme*: Normalizasyon için StandardScaler kullanılmıştır.


## 4. Kullanım 🌟
   
1. **Google Colab Ortamına Giriş Yapın:**
(https://colab.research.google.com/)

3. **Depoyu Klonlayın:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>

4. **Bağımlılıkları Yükleyin:**
   ```bash
   pip install -r requirements.txt

5. **Belirli bir model için kod dosyasını çalıştırın (örnek):**
     ```bash
      reformer_model.py


## 5. Performans Mertrikleri  📊
1. Modeller aşağıdaki metrikler kullanılarak değerlendirilmiştir:
- *Ortalama Kare Hata (MSE)*: Tahmin edilen ve gerçek değerler arasındaki farkların karelerinin ortalamasıdır. Daha düşük bir değer daha iyi bir performansı gösterir.
- *Ortalama Mutlak Hata (MAE)*: Tahmin edilen ve gerçek değerler arasındaki farkların mutlak değerlerinin ortalamasıdır. Hata büyüklüğünü ifade eder.
- *Karekök Ortalama Kare Hata (RMSE)*: MSE'nin kareköküdür ve büyük hataları daha fazla cezalandırır. Tahminin doğruluğunu ifade eder.
- *R-Kare (R²)*: Modelin açıklayıcı gücünü ifade eder. 1'e yakın değerler, modelin veriyi iyi açıkladığını gösterir.
- *Ortalama Mutlak Yüzde Hata (MAPE)*: Tahmin edilen ve gerçek değerler arasındaki farkın gerçek değere oranının yüzde cinsinden ortalamasıdır. Yüzde bazında hata oranını gösterir.

2. **⏱️ Çıkarım ve Eğitim Zamanı Hesabı**:
- Modellerin çıkarım ve eğitim zamanı hesaplanarak performansı değerlendirilmiştir.
  
## 6. Görselleştirme  🚦
-  📉 Eğitim ve doğrulama kayıpları, aşırı öğrenmeyi izlemek için çizilmiştir.
-  🧐 Test verisindeki tahminlerle gerçek değerler karşılaştırılmıştır.
-  🔮 Gelecekteki tahminler sezgisel bir anlayış için görselleştirilmiştir.

## 7. Model Performansı İçin Ekstra Özellikler
**⏸️ Early Stopping** 
- Doğrulama kaybını izleyerek aşırı öğrenmeyi önlemek için uygulanmıştır. Doğrulama kaybı belirli bir süre boyunca iyileşmezse (varsayılan: 7 epoch), eğitim durdurulur.
  
## 8. Ekler
Google Colab Bağlantısı:[Google Colab](https://colab.research.google.com/drive/19ggo0-2GEgF3kroaaDlKgYJUeE5F2Y7z)
<br>
Proje Raporu:[Rapor](https://github.com/Utku-Genc/Weather-Modeling/blob/main/Proje%20Raporu.pdf)
<br>
Proje Kapsamında Toplanan Veriler:[Veri Seti](https://drive.google.com/drive/u/0/folders/1eSPkRUiUx6AkxZYrtghkakOASbia3wuu)
<br>

  


