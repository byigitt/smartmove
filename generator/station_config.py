"""
Station and line configurations for the Ankara Metro system
"""

import numpy as np

class StationConfig:
    def __init__(self):
        self._initialize_metro_lines()
        self.train_capacity = 1200  # Updated capacity for Ankara Metro trains
        
    def _initialize_metro_lines(self):
        """Initialize metro lines with their stations and characteristics"""
        # Define special locations near stations that affect passenger flow
        self.station_features = {
            # M1-2-3 Line Stations (Koru - OSB-Törekent)
            'Koru': ['residential_high_income', 'terminal', 'park_and_ride'],
            'Çayyolu': ['residential_high_income', 'shopping_district', 'weekend_active'],
            'Ümitköy': ['residential_high_income', 'shopping_district'],
            'Beytepe': ['university_hacettepe', 'education_zone', 'student_residential'],
            'Tarım Bakanlığı-Danıştay': ['government_offices', 'office_district'],
            'Bilkent': ['university_bilkent', 'education_zone', 'student_residential', 'shopping_mall'],
            'Orta Doğu Teknik Üniversitesi': ['university_metu', 'education_zone', 'student_residential', 'research_center'],
            'Maden Tetkik ve Arama': ['government_offices', 'research_center'],
            'Söğütözü': ['business_district', 'shopping_mall', 'office_district'],
            'Millî Kütüphane': ['education_zone', 'cultural_center', 'government_offices'],
            'Necatibey': ['business_district', 'office_district'],
            '15 Temmuz Kızılay Millî İrade': ['central_hub', 'shopping_district', 'business_district', 'transfer_hub', 
                                             'entertainment_district', 'restaurant_district', 'youth_center'],
            'Sıhhiye': ['hospital_zone', 'education_zone', 'government_offices', 'shopping_district'],
            'Ulus': ['historic_center', 'shopping_district', 'traditional_market', 'tourist_attraction'],
            'Atatürk Kültür Merkezi': ['cultural_center', 'transfer_hub', 'government_offices', 'museum_district'],
            'Akköprü': ['shopping_mall', 'residential'],
            'İvedik': ['industrial_zone', 'business_park', 'manufacturing'],
            'Yenimahalle': ['residential_mixed', 'local_shopping'],
            'Demetevler': ['residential_high_density', 'local_market'],
            'Hastane': ['hospital_zone', 'medical_center'],
            'Macunköy': ['industrial_zone', 'manufacturing'],
            'Orta Doğu Sanayi ve Ticaret Merkezi': ['industrial_zone', 'business_park', 'wholesale_market'],
            'Batıkent': ['residential_mixed', 'shopping_district', 'transfer_hub'],
            'Batı Merkez': ['residential', 'local_shopping'],
            'Mesa': ['residential_planned', 'park'],
            'Botanik': ['park', 'recreation_area', 'residential'],
            'İstanbul Yolu': ['residential', 'industrial_mixed'],
            'Eryaman 1-2': ['residential_planned', 'local_shopping'],
            'Eryaman 5': ['residential_planned', 'education_zone'],
            'Devlet Mahallesi/1910 Ankaragücü': ['sports_complex', 'stadium', 'residential'],
            'Harikalar Diyarı': ['park', 'recreation_area', 'family_entertainment'],
            'Fatih': ['residential_mixed', 'local_market'],
            'Gaziosmanpaşa': ['residential_high_income', 'embassy_district'],
            'OSB-Törekent': ['industrial_zone', 'terminal', 'manufacturing_hub'],

            # M4 Line Stations
            'Adliye': ['government_offices', 'courthouse', 'office_district'],
            'Gar': ['transfer_hub', 'historic_station', 'transport_hub'],
            'Ankara Su ve Kanalizasyon İdaresi': ['government_offices', 'utility_services'],
            'Dışkapı': ['hospital_zone', 'medical_center', 'education_zone'],
            'Meteoroloji': ['government_offices', 'research_center'],
            'Belediye': ['government_offices', 'civic_center'],
            'Mecidiye': ['residential_mixed', 'local_shopping'],
            'Kuyubaşı': ['residential', 'local_market'],
            'Dutluk': ['residential', 'park'],
            'Şehitler': ['residential', 'terminal'],

            # A1 Line Stations
            'AŞTİ': ['transport_hub', 'terminal', 'shopping_center'],
            'Emek': ['residential_mixed', 'office_district'],
            'Bahçelievler': ['residential_high_income', 'shopping_district', 'education_zone', 'restaurant_district'],
            'Beşevler': ['university_area', 'education_zone', 'student_residential'],
            'Anadolu/Anıtkabir': ['historic_monument', 'tourist_attraction', 'cultural_center'],
            'Maltepe': ['residential_mixed', 'business_district', 'shopping_district'],
            'Demirtepe': ['business_district', 'office_district'],
            'Kolej': ['education_zone', 'student_area'],
            'Kurtuluş': ['residential_mixed', 'local_shopping'],
            'Dikimevi': ['hospital_zone', 'education_zone', 'terminal']
        }
        
        self.metro_lines = {
            'A1': {'stations': ['AŞTİ', 'Emek', 'Bahçelievler', 'Beşevler', 'Anadolu/Anıtkabir', 'Maltepe', 'Demirtepe', '15 Temmuz Kızılay Millî İrade', 'Kolej', 'Kurtuluş', 'Dikimevi'],
                   'terminal_stations': ['AŞTİ', 'Dikimevi'],
                   'junction_stations': ['15 Temmuz Kızılay Millî İrade'],
                   'frequency_minutes': {
                       'peak': 3,
                       'regular': 5,
                       'off_peak': 10
                   }},
            'M1-2-3': {'stations': ['Koru', 'Çayyolu', 'Ümitköy', 'Beytepe', 'Tarım Bakanlığı-Danıştay', 'Bilkent', 'Orta Doğu Teknik Üniversitesi', 'Maden Tetkik ve Arama', 'Söğütözü', 'Millî Kütüphane', 'Necatibey', '15 Temmuz Kızılay Millî İrade', 'Sıhhiye', 'Ulus', 'Atatürk Kültür Merkezi', 'Akköprü', 'İvedik', 'Yenimahalle', 'Demetevler', 'Hastane', 'Macunköy', 'Orta Doğu Sanayi ve Ticaret Merkezi', 'Batıkent', 'Batı Merkez', 'Mesa', 'Botanik', 'İstanbul Yolu', 'Eryaman 1-2', 'Eryaman 5', 'Devlet Mahallesi/1910 Ankaragücü', 'Harikalar Diyarı', 'Fatih', 'Gaziosmanpaşa', 'OSB-Törekent'],
                      'terminal_stations': ['Koru', 'OSB-Törekent'],
                      'junction_stations': ['15 Temmuz Kızılay Millî İrade', 'Batıkent'],
                      'frequency_minutes': {
                          'peak': 3,
                          'regular': 5,
                          'off_peak': 10
                      }},
            'M4': {'stations': ['15 Temmuz Kızılay Millî İrade', 'Adliye', 'Gar', 'Atatürk Kültür Merkezi', 'Ankara Su ve Kanalizasyon İdaresi', 'Dışkapı', 'Meteoroloji', 'Belediye', 'Mecidiye', 'Kuyubaşı', 'Dutluk', 'Şehitler'],
                   'terminal_stations': ['15 Temmuz Kızılay Millî İrade', 'Şehitler'],
                   'junction_stations': ['15 Temmuz Kızılay Millî İrade', 'Atatürk Kültür Merkezi'],
                   'frequency_minutes': {
                       'peak': 4,
                       'regular': 7,
                       'off_peak': 15
                   }}
        }
    
    def get_station_capacity(self, station):
        """Get the capacity of a station"""
        if station == 'Kızılay':
            return 2000
        elif station in ['Bahçelievler', 'Milli Kütüphane']:
            return 1200  # Higher capacity for these busy stations
        elif station in sum([details['junction_stations'] for details in self.metro_lines.values()], []):
            return 1500
        else:
            return 800
    
    def generate_station_characteristics(self):
        """Generate characteristics for all stations"""
        station_data = {}
        for line, details in self.metro_lines.items():
            stations = details['stations']
            terminal_stations = details['terminal_stations']
            junction_stations = details['junction_stations']

            for station in stations:
                if station not in station_data:
                    if station == 'Kızılay':
                        base_popularity = 1.0
                        capacity = 2000
                    elif station in ['Bahçelievler', 'Milli Kütüphane']:
                        base_popularity = 0.95  # Higher base popularity
                        capacity = 1200
                    elif station in junction_stations:
                        base_popularity = 0.9
                        capacity = 1500
                    else:
                        # Just a slight decay based on position
                        station_idx = stations.index(station)
                        center_idx = 0 if 'Kızılay' in stations else len(stations)//2
                        distance = abs(station_idx - center_idx)
                        base_popularity = 0.85 * np.exp(-0.03 * distance)
                        capacity = 800

                    station_data[station] = {
                        'base_popularity': base_popularity,
                        'type': 'central' if station == 'Kızılay' else
                               'high_traffic' if station in ['Bahçelievler', 'Milli Kütüphane'] else
                               'junction' if station in junction_stations else
                               'terminal' if station in terminal_stations else 'regular',
                        'capacity': capacity
                    }

        return station_data 