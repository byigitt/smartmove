# Ankara Metro Yolcu Yoğunluğu Tahmin Sistemi

*Pandora AI - Ostimtech AI Yarışması 2024*

Bu proje, Ankara Metro sistemindeki yolcu yoğunluğunu tahmin etmek için geliştirilmiş yapay zeka tabanlı bir sistemdir. Sistem, çeşitli faktörleri (saat, hava durumu, istasyon konumu, vb.) göz önünde bulundurarak gerçek zamanlı yoğunluk tahminleri yapar.

## 🚇 Özellikler

- Ankara Metro'nun tüm hatlarında (M1-2-3, M4, A1) yoğunluk tahmini
- Dikkate alınan faktörler:
  - Gün içi saat ve haftanın günü
  - Hava durumu ve etkisi
  - İstasyon konumu ve tipi
  - Sefer sıklığı
  - Geçmiş yolcu desenleri
- Özel durumlar için optimize edilmiş:
  - Terminal istasyonları
  - Yoğun saatler
  - Hava durumu aksaklıkları
  - Merkezi istasyonlara uzaklık
- Detaylı tahmin sonuçları ve açıklamalar

## 🛠️ Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/your-username/ankara-metro-predictor.git
cd ankara-metro-predictor
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## 📊 Veri Üretimi ve Model Eğitimi

### Sentetik Veri Üretimi

Gerçekçi metro yolcu verisi üretmek için:

```bash
python generate.py --start-date 2023-01-01 --end-date 2023-12-31 --freq 5min
```

Parametreler:
- `--start-date`: Başlangıç tarihi (varsayılan: 2023-01-01)
- `--end-date`: Bitiş tarihi (varsayılan: 2023-12-31)
- `--freq`: Örnekleme sıklığı (varsayılan: 5min)
- `--output`: Çıktı dosyası adı
- `--out-dir`: Çıktı dizini

### Model Eğitimi

Yeni bir model eğitmek için:

```bash
python predict.py --train --model-type rf
```

Parametreler:
- `--model-type`: Model tipi (rf: Random Forest, gb: Gradient Boosting)
- `--model-path`: Model kayıt/yükleme yolu
- `--data-path`: Özel veri yolu

## 🎯 Tahmin Yapma

Eğitilmiş model ile tahmin yapmak için:

```bash
python predict.py \
    --metro-line M1-2-3 \
    --station "15 Temmuz Kızılay Millî İrade" \
    --hour 8 \
    --weather Sunny \
    --weekend
```

Parametreler:
- `--metro-line`: Metro hattı
- `--station`: İstasyon adı
- `--hour`: Saat (0-23)
- `--weather`: Hava durumu
- `--weekend`: Hafta sonu tahmini için

## 📁 Proje Yapısı

```
.
├── data/
│   └── generate_data.py     # Veri üretim modülü
├── train/
│   ├── __init__.py         # Paket başlatıcı
│   ├── cli.py             # Komut satırı arayüzü
│   ├── predictor.py       # Ana tahmin sınıfı
│   ├── data_preprocessing.py # Veri ön işleme
│   ├── station_config.py  # İstasyon yapılandırmaları
│   ├── evaluation.py      # Model değerlendirme
│   └── utils.py          # Yardımcı fonksiyonlar
├── generate.py           # Veri üretim betiği
├── predict.py           # Tahmin betiği
└── requirements.txt     # Bağımlılıklar
```

## 🔍 Veri Formatı

Eğitim verisi aşağıdaki sütunları içerir:
- Timestamp: Zaman damgası
- Metro_Line: Metro hattı
- Station_ID: İstasyon adı
- Station_Type: İstasyon tipi
- Weather_Condition: Hava durumu
- Time_Period: Zaman dilimi
- Is_Weekend: Hafta sonu mu?
- Weather_Disruption: Hava kaynaklı aksama
- Service_Frequency: Sefer sıklığı
- Trains_Per_Hour: Saatlik tren sayısı
- Boarding_Passengers: Binen yolcu
- Alighting_Passengers: İnen yolcu
- Transfer_Out: Transfer yapan yolcu
- Capacity_Utilization: Kapasite kullanımı
- Occupancy_Rate: Doluluk oranı

## 🎓 İstasyon Tipleri

- `central_hub`: Merkezi aktarma istasyonları (örn. Kızılay)
- `major_destination`: Ana varış noktaları (örn. üniversiteler, AŞTİ)
- `industrial`: Sanayi bölgesi istasyonları
- `residential`: Yerleşim bölgesi istasyonları
- `standard`: Standart istasyonlar

## 🌤️ Hava Durumu Etkileri

- Güneşli: Normal yoğunluk
- Bulutlu: Normal yoğunluk
- Yağmurlu: %10 artış
- Karlı: %20 artış
- Fırtınalı: %25 artış ve olası aksamalar

## 📈 Model Performansı

Model, aşağıdaki metrikler üzerinden değerlendirilir:
- RMSE (Root Mean Square Error)
- R² Skoru
- MAPE (Mean Absolute Percentage Error)

## 👥 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen bir Pull Request göndermekten çekinmeyin.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

## 🏆 Yarışma Bilgileri

Bu proje, Pandora AI - Ostimtech AI Yarışması 2024 için geliştirilmiştir. Proje, Ankara Metro sisteminde yolcu yoğunluğunu tahmin ederek:
- Yolcuların seyahat planlamasını kolaylaştırmayı
- Metro işletmesinin optimizasyonunu sağlamayı
- Yoğun saatlerde alternatif rotalar önermeyi
hedeflemektedir.

## ✨ Teşekkürler

Bu proje, Pandora AI ekibi tarafından geliştirilmiştir. Katkıda bulunan herkese teşekkür ederiz.
