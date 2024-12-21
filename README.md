# smartmove - Metro Yolcu Analiz Sistemi

*cyberia ~ Ostimtech AI Competition 2024*

Bu proje, Ankara Metro sistemindeki yolcu yoğunluğunu tahmin etmek ve analiz etmek için geliştirilmiş yapay zeka tabanlı bir sistemdir. Sistem, çeşitli faktörleri göz önünde bulundurarak gerçek zamanlı yoğunluk tahminleri yapar ve yolcu sayımı gerçekleştirir.

## 🚇 Özellikler

### Yolcu Yoğunluğu Tahmini
- Ankara Metro'nun tüm hatlarında (M1-2-3, M4, A1) yoğunluk tahmini
- Dikkate alınan faktörler:
  - Gün içi saat ve haftanın günü
  - Hava durumu ve etkisi
  - İstasyon konumu ve tipi
  - Sefer sıklığı
  - Geçmiş yolcu desenleri

### Gerçek Zamanlı Yolcu Sayımı
- Video tabanlı yolcu giriş/çıkış sayımı
- YOLOv8 nesne tespiti ile hassas insan tespiti
- Çizgi geçiş analizi ile yön tespiti
- Gerçek zamanlı sayaçlar:
  - Giren yolcu sayısı
  - Çıkan yolcu sayısı
- Görsel analiz ve raporlama

## 🛠️ Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/byigitt/smartmove.git
cd smartmove
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## 📊 Kullanım

### Yolcu Sayımı

Video üzerinden yolcu sayımı yapmak için:

```bash
python counter.py
```

### Yoğunluk Tahmini

Sentetik veri üretimi:
```bash
python generate.py --start-date 2023-01-01 --end-date 2023-12-31 --freq 5min
```

Model eğitimi:
```bash
python predict.py --train --model-type rf
```

Tahmin yapma:
```bash
python predict.py \
    --metro-line M1-2-3 \
    --station "15 Temmuz Kızılay Millî İrade" \
    --hour 8 \
    --weather Sunny \
    --weekend
```

## 📁 Proje Yapısı

```
.
├── data/                  # Veri dosyaları
│   └── demo/             # Demo videoları
├── generator/            # Veri üretim modülleri
├── train/               # Model eğitim modülleri
├── visualize/           # Görselleştirme modülleri
├── generate.py          # Veri üretim betiği
├── predict.py           # Tahmin betiği
├── people_counter.py    # Yolcu sayım betiği
└── requirements.txt     # Bağımlılıklar
```

## 🎯 Özellikler ve Kullanım Senaryoları

### Yolcu Sayımı
- Gerçek zamanlı insan tespiti ve sayımı
- Giriş/çıkış yönü analizi
- Görsel raporlama ve istatistikler
- Video kaydı ve analiz

### Yoğunluk Tahmini
- İstasyon bazlı yoğunluk tahminleri
- Hava durumu etkisi analizi
- Zaman bazlı yoğunluk desenleri
- Özel durum ve etkinlik analizi

## 📈 Görselleştirmeler

- Gerçek zamanlı sayım grafikleri
- İstasyon yoğunluk haritaları
- Zaman bazlı analiz grafikleri
- Hat karşılaştırma grafikleri

## 👥 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen bir Pull Request göndermekten çekinmeyin.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

## 🏆 Yarışma Bilgileri

Bu proje, Pandora AI - Ostimtech AI Yarışması 2024 için Cyberia ekibi tarafından geliştirilmiştir. Proje, Ankara Metro sisteminde:
- Yolcu yoğunluğunu tahmin etmeyi
- Gerçek zamanlı yolcu sayımı yapmayı
- Metro işletmesinin optimizasyonunu sağlamayı
- Yoğun saatlerde alternatif rotalar önermeyi
hedeflemektedir.

## ✨ Teşekkür

Bu projeyi geliştirmemize yardımcı olan tüm katılımcılara ve destekçilere teşekkür ederiz. Özellikle, Pandora AI - Ostimtech AI Yarışması 2024 organizatörlerine özel teşekkürlerimizi sunuyoruz.
