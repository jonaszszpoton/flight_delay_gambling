import pandas as pd

df = pd.read_csv('flights_generated.csv',header=0,sep=',')

print(df.head())


# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
#
# # --- KONFIGURACJA I STAŁE ---
# NUM_ROWS = 50000
# FILENAME = 'flights_dataset_eu_final_2022_2025.csv'
# np.random.seed(42)  # Dla powtarzalności wyników
#
# # 1. Lista tras (Origin, Dest, Dystans, Czas(min), Linia, Samolot, Typ Linii)
# # Wszystkie trasy spełniają warunek: Min. 1 lotnisko w UE
# ROUTES = [
#     # --- HUBY POLSKA ---
#     ('WAW', 'JFK', 6850, 570, 'LOT', 'Boeing 787-9', 'LEGACY'),
#     ('WAW', 'LHR', 1450, 165, 'LOT', 'Boeing 737-800', 'LEGACY'),
#     ('WAW', 'CDG', 1370, 145, 'LOT', 'Embraer 195', 'LEGACY'),
#     ('KRK', 'BGY', 950, 110, 'Ryanair', 'Boeing 737-8200', 'LCC'),
#     ('GDN', 'STN', 1300, 140, 'Wizz Air', 'Airbus A321', 'LCC'),
#     ('WMI', 'ALC', 2300, 205, 'Ryanair', 'Boeing 737-800', 'LCC'),
#     ('KRK', 'MUC', 600, 85, 'Lufthansa', 'Airbus A319', 'LEGACY'),
#
#     # --- HUBY ZACHODNIE ---
#     ('FRA', 'SFO', 9150, 690, 'Lufthansa', 'Boeing 747-8', 'LEGACY'),
#     ('FRA', 'HND', 9400, 780, 'Lufthansa', 'Airbus A350', 'LEGACY'),
#     ('AMS', 'DXB', 5150, 390, 'KLM', 'Boeing 787-10', 'LEGACY'),
#     ('CDG', 'JFK', 5840, 500, 'Air France', 'Boeing 777', 'LEGACY'),
#     ('MAD', 'BOG', 8000, 610, 'Iberia', 'Airbus A350', 'LEGACY'),
#     ('FCO', 'EWR', 6900, 580, 'United', 'Boeing 777', 'LEGACY'),
#
#     # --- INBOUND DO UE ---
#     ('DXB', 'WAW', 4200, 340, 'Emirates', 'Boeing 777-300ER', 'LEGACY'),
#     ('DOH', 'CDG', 4900, 400, 'Qatar Airways', 'Airbus A380', 'LEGACY'),
#     ('IST', 'BER', 1750, 170, 'Turkish Airlines', 'Airbus A321', 'LEGACY'),
#     ('JFK', 'FRA', 6200, 460, 'Delta', 'Airbus A330', 'LEGACY'),
#     ('ZRH', 'WAW', 1050, 115, 'Swiss', 'Airbus A220', 'LEGACY'),
# ]
#
# # Wagi dla tras (korygujemy, aby idealnie sumowały się do 1.0)
# raw_weights = [0.08, 0.06, 0.10, 0.12, 0.08, 0.08, 0.06, 0.03, 0.02, 0.03, 0.04, 0.02, 0.03, 0.03, 0.03, 0.05, 0.03,
#                0.03]
# route_weights = np.array(raw_weights)
# route_weights = route_weights / route_weights.sum()  # Automatyczna normalizacja
#
# print(">>> Krok 1: Generowanie szkieletu danych...")
#
# route_indices = np.random.choice(len(ROUTES), NUM_ROWS, p=route_weights)
# data_routes = [ROUTES[i] for i in route_indices]
# df = pd.DataFrame(data_routes,
#                   columns=['origin', 'dest', 'dist_km', 'sched_duration', 'airline', 'aircraft', 'airline_type'])
#
# # Unikalny kod lotu
# df['flight_code'] = df['airline'].str[:2].str.upper() + np.random.randint(100, 9999, size=NUM_ROWS).astype(str)
#
# # Daty i Czasy
# start_date = datetime(2022, 1, 1)
# end_date = datetime(2025, 10, 31)
# total_days = (end_date - start_date).days
#
# days_offset = np.random.randint(0, total_days + 1, size=NUM_ROWS)
# base_dates = np.array([start_date + timedelta(days=int(d)) for d in days_offset])
#
# # Godziny z uwzględnieniem szczytów (znormalizowane)
# raw_hours_prob = [0.005, 0.005, 0.005, 0.005, 0.01, 0.05, 0.08, 0.09, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07,
#                   0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01, 0.005]
# hours_prob = np.array(raw_hours_prob)
# hours_prob = hours_prob / hours_prob.sum()  # Bezpieczna normalizacja do 1.0
#
# hours = np.random.choice(np.arange(24), size=NUM_ROWS, p=hours_prob)
# minutes = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], size=NUM_ROWS)
#
# df['sched_dep_time'] = [pd.Timestamp(d.year, d.month, d.day, h, m) for d, h, m in zip(base_dates, hours, minutes)]
# df.sort_values('sched_dep_time', inplace=True)
# df.reset_index(drop=True, inplace=True)
#
# # Atrybuty daty
# df['year'] = df['sched_dep_time'].dt.year
# df['month'] = df['sched_dep_time'].dt.month
# df['day_of_week'] = df['sched_dep_time'].dt.day_of_week + 1  # 1=Poniedziałek
# df['hour'] = df['sched_dep_time'].dt.hour
# df['sched_arr_time'] = df['sched_dep_time'] + pd.to_timedelta(df['sched_duration'], unit='m')
#
# print(">>> Krok 2: Symulacja pogody...")
# weather_types = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow', 'Storm']
#
#
# # Funkcja pomocnicza do normalizacji prawdopodobieństw (żeby uniknąć błędu < 1.0)
# def normalize_probs(probs):
#     p = np.array(probs)
#     return p / p.sum()
#
#
# weather_col = []
# for m in df['month']:
#     if m in [12, 1, 2]:  # Zima
#         p = normalize_probs([0.15, 0.30, 0.10, 0.20, 0.20, 0.05])
#     elif m in [6, 7, 8]:  # Lato (burze)
#         p = normalize_probs([0.45, 0.25, 0.10, 0.05, 0.00, 0.15])
#     else:  # Przejściowe
#         p = normalize_probs([0.30, 0.30, 0.25, 0.10, 0.00, 0.05])
#     weather_col.append(np.random.choice(weather_types, p=p))
#
# df['weather_origin'] = weather_col
#
# print(">>> Krok 3: Obliczanie opóźnień...")
# prob_delay = 0.14
# # Wpływ czynników na prawdopodobieństwo opóźnienia
# weather_factor = np.select([df['weather_origin'].isin(['Fog', 'Snow', 'Storm']), df['weather_origin'] == 'Rain'],
#                            [0.30, 0.10], default=0.0)
# time_factor = (df['hour'] - 6).clip(0) * 0.01  # Kumulacja w ciągu dnia
# season_factor = np.where(df['month'].isin([7, 8, 12, 1]), 0.05, 0.0)  # Wakacje/Święta
#
# final_prob = (prob_delay + weather_factor + time_factor + season_factor).clip(0, 0.95)
# is_dep_delayed_event = np.random.rand(NUM_ROWS) < final_prob
#
# # Minuty opóźnienia (Rozkład Gamma dla opóźnionych + Szum normalny dla reszty)
# dep_delay_minutes = np.where(
#     is_dep_delayed_event,
#     np.random.gamma(2.0, 25.0, NUM_ROWS) + 10,  # Minimum 10 min poślizgu przy incydencie
#     np.random.normal(0, 5, NUM_ROWS)
# ).astype(int)
#
# df['dep_delay'] = dep_delay_minutes
# df['act_dep_time'] = df['sched_dep_time'] + pd.to_timedelta(df['dep_delay'], unit='m')
#
# # Wariancja czasu lotu
# air_time_variance = np.random.normal(0, df['sched_duration'] * 0.08, NUM_ROWS)
# df['act_duration'] = df['sched_duration'] + air_time_variance
# df['act_arr_time'] = df['act_dep_time'] + pd.to_timedelta(df['act_duration'], unit='m')
#
# # Kluczowa zmienna analityczna: Całkowite opóźnienie na przylocie
# df['delay_minutes'] = ((df['act_arr_time'] - df['sched_arr_time']).dt.total_seconds() / 60).astype(int)
# df['is_delayed'] = df['delay_minutes'] > 15
#
# print(">>> Krok 4: Przyczyny opóźnień...")
# random_causes = np.random.choice(['Carrier Fault', 'External Factors'], size=NUM_ROWS, p=[0.65, 0.35])
# conditions = [
#     (~df['is_delayed']),
#     (df['is_delayed'] & df['weather_origin'].isin(['Snow', 'Fog', 'Storm'])),
#     (df['is_delayed'])  # Pozostałe opóźnione
# ]
# choices = ['NA', 'External Factors', random_causes]
# df['delay_reason'] = np.select(conditions, choices, default='NA')
#
# print(">>> Krok 5: Obliczanie cen biletów (z Inflacją)...")
# base_rate = np.where(df['dist_km'] < 1500, 0.14, 0.09)
# airline_factor = np.where(df['airline_type'] == 'LCC', 0.7, 1.0)
# base_price = (50 + (df['dist_km'] * base_rate)) * airline_factor
#
# month_mult = np.where(df['month'].isin([7, 8, 12]), 1.4, 1.0)
# month_mult = np.where(df['month'].isin([1, 2, 11]), 0.85, month_mult)
# dow_mult = np.where(df['day_of_week'].isin([5, 7]), 1.2, np.where(df['day_of_week'].isin([2, 3]), 0.9, 1.0))
#
# # Mapa inflacji
# inflation_map = {2022: 1.0, 2023: 1.12, 2024: 1.20, 2025: 1.25}
# inflation_mult = df['year'].map(inflation_map)
#
# booking_variance = np.random.lognormal(0, 0.25, NUM_ROWS)
# df['ticket_price_eur'] = (base_price * month_mult * dow_mult * inflation_mult * booking_variance).clip(
#     lower=15.0).round(2)
#
# print(">>> Krok 6: Czyszczenie i Zapis...")
# final_columns = [
#     'flight_code', 'airline', 'origin', 'dest',
#     'sched_dep_time', 'act_dep_time',
#     'sched_arr_time', 'act_arr_time',
#     'dist_km', 'ticket_price_eur', 'aircraft',
#     'day_of_week', 'weather_origin',
#     'is_delayed', 'delay_minutes', 'delay_reason'
# ]
# df_final = df[final_columns].copy()
# df_final.columns = [
#     'flight_code', 'airline', 'origin_airport', 'dest_airport',
#     'sched_dep_time', 'act_dep_time',
#     'sched_arr_time', 'act_arr_time',
#     'distance_km', 'ticket_price_eur', 'aircraft_type',
#     'day_of_week', 'weather_origin',
#     'is_delayed', 'delay_minutes', 'delay_reason'
# ]
#
# # Zaokrąglanie sekund do minut (estetyka)
# time_cols = ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']
# for col in time_cols:
#     df_final[col] = df_final[col].dt.round('1min')
#
# # Ochrona przed datami spoza zakresu
# limit_date = pd.Timestamp('2025-10-31 23:59:00')
# df_final = df_final[df_final['act_arr_time'] <= limit_date]
#
# df_final.to_csv(FILENAME, index=False)
# print(f"Zapisano {len(df_final)} wierszy w pliku: {FILENAME}")