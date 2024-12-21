"""
Station configurations and constants for the Ankara Metro prediction system
"""

# Define terminal stations for each line
TERMINAL_STATIONS = {
    'M1-2-3': ['Koru', 'OSB-Törekent'],  # Terminal stations for M1-2-3 line
    'M4': ['15 Temmuz Kızılay Millî İrade', 'Şehitler'],  # Terminal stations for M4
    'A1': ['AŞTİ', 'Dikimevi']  # Terminal stations for A1
}

# Define station order according to the metro map
STATION_ORDER = {
    'M1-2-3': [
        'Koru', 'Çayyolu', 'Ümitköy', 'Beytepe', 'Tarım Bakanlığı-Danıştay',
        'Bilkent', 'Orta Doğu Teknik Üniversitesi', 'Maden Tetkik ve Arama',
        'Söğütözü', 'Millî Kütüphane', 'Necatibey', '15 Temmuz Kızılay Millî İrade',
        'Sıhhiye', 'Ulus', 'Atatürk Kültür Merkezi', 'Akköprü', 'İvedik',
        'Yenimahalle', 'Demetevler', 'Hastane', 'Macunköy',
        'Orta Doğu Sanayi ve Ticaret Merkezi', 'Batıkent', 'Batı Merkez',
        'Mesa', 'Botanik', 'İstanbul Yolu', 'Eryaman 1-2', 'Eryaman 5',
        'Devlet Mahallesi/1910 Ankaragücü', 'Harikalar Diyarı', 'Fatih',
        'Gaziosmanpaşa', 'OSB-Törekent'
    ],
    'M4': [
        '15 Temmuz Kızılay Millî İrade', 'Adliye', 'Gar', 'Atatürk Kültür Merkezi',
        'Ankara Su ve Kanalizasyon İdaresi', 'Dışkapı', 'Meteoroloji', 'Belediye',
        'Mecidiye', 'Kuyubaşı', 'Dutluk', 'Şehitler'
    ],
    'A1': [
        'AŞTİ', 'Emek', 'Bahçelievler', 'Beşevler', 'Anadolu/Anıtkabir',
        'Maltepe', 'Demirtepe', '15 Temmuz Kızılay Millî İrade', 'Kolej',
        'Kurtuluş', 'Dikimevi'
    ]
}

# Service frequency based on time period and metro line
SERVICE_FREQUENCIES = {
    'A1': {'peak': 3, 'regular': 5, 'off_peak': 10},
    'M1-2-3': {'peak': 3, 'regular': 5, 'off_peak': 10},
    'M4': {'peak': 4, 'regular': 7, 'off_peak': 15}
}

# Weather impact factors
WEATHER_FACTORS = {
    'Sunny': {'factor': 0.9, 'disruption_prob': 0.0},
    'Cloudy': {'factor': 1.0, 'disruption_prob': 0.0},
    'Rainy': {'factor': 1.1, 'disruption_prob': 0.02},
    'Snowy': {'factor': 1.2, 'disruption_prob': 0.10},
    'Stormy': {'factor': 1.25, 'disruption_prob': 0.15}
} 