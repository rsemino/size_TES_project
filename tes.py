import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from scipy import interpolate
import requests
from datetime import datetime

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Simulatore i-TES Pro", layout="wide")
st.title("🚰 Simulatore i-TES: Site-Specific & PVGIS")

# --- GESTIONE STATO ---
if 'qty_6' not in st.session_state: st.session_state.qty_6 = 0
if 'qty_12' not in st.session_state: st.session_state.qty_12 = 0
if 'qty_20' not in st.session_state: st.session_state.qty_20 = 1
if 'qty_40' not in st.session_state: st.session_state.qty_40 = 0
if 'lat' not in st.session_state: st.session_state.lat = 41.9028 # Default Roma
if 'lon' not in st.session_state: st.session_state.lon = 12.4964
if 'address_found' not in st.session_state: st.session_state.address_found = "Roma, Italia (Default)"

# Inizializzazione Slider Strategia PV
if 'pv_start_sun' not in st.session_state: st.session_state.pv_start_sun = 95
if 'pv_start_night' not in st.session_state: st.session_state.pv_start_night = 30
if 'pv_stop' not in st.session_state: st.session_state.pv_stop = 98
if 'std_start' not in st.session_state: st.session_state.std_start = 70
if 'std_stop' not in st.session_state: st.session_state.std_stop = 98

# ==========================================
#  SEZIONE FUNZIONI
# ==========================================

def manage_qty(key_name, label):
    col_minus, col_val, col_plus = st.columns([1, 2, 1])
    with col_minus:
        if st.button("➖", key=f"dec_{key_name}"):
            if st.session_state[key_name] > 0:
                st.session_state[key_name] -= 1
                st.rerun()
    with col_val:
        st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 5px;'>{label}<br><span style='font-size: 20px;'>{st.session_state[key_name]}</span></div>", unsafe_allow_html=True)
    with col_plus:
        if st.button("➕", key=f"inc_{key_name}"):
            st.session_state[key_name] += 1
            st.rerun()

def get_solar_window(zone):
    windows = {
        "A": (9, 17), "B": (9, 17),
        "C": (10, 16), "D": (10, 16),
        "E": (11, 15), "F": (11, 15)
    }
    return windows.get(zone, (10, 16))

def get_coordinates(address):
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
    headers = {'User-Agent': 'iTES_Simulator/1.0'}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200 and r.json():
            data = r.json()[0]
            return float(data['lat']), float(data['lon']), data['display_name']
    except:
        return None, None, None
    return None, None, None

def get_pvgis_data(lat, lon, month_idx=0):
    url = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
    params = {
        'lat': lat, 'lon': lon, 'peakpower': 1, 'loss': 14,
        'outputformat': 'json', 'angle': 35, 'aspect': 0 
    }
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            monthly_data = data['outputs']['monthly']['fixed']
            ed_kwh = monthly_data[month_idx]['E_d']
            return ed_kwh
    except:
        pass
    return 3.5 

def create_solar_curve_site_specific(lat, lon, total_daily_kwh_per_kwp):
    day_length = 12
    minutes = np.arange(1440)
    center_min = 12 * 60
    sigma = (day_length * 60) / 6 
    curve = np.exp(-((minutes - center_min)**2) / (2 * sigma**2))
    curve[curve < 0.01] = 0
    return curve

def calcola_qd_en806(total_lu, max_single_lu):
    max_single_qa = max_single_lu * 0.1 
    conv_lu = 300.0; conv_qd = 1.7
    if total_lu <= 0: return 0.0
    if total_lu >= conv_lu: return 0.1 * math.sqrt(total_lu)
    else:
        if total_lu < max_single_lu: total_lu = max_single_lu
        x1, y1 = math.log(max_single_lu), math.log(max_single_qa)
        x2, y2 = math.log(conv_lu), math.log(conv_qd)
        m = 0 if x2 == x1 else (y2 - y1) / (x2 - x1)
        return math.exp(y1 + m * (math.log(total_lu) - x1))

def get_system_curves(config, t_pcm, dt_calc, p_hp_tot, e_tot_e0):
    interpolators_p = {}
    interpolators_t = {}
    limits = {}
    for qty, size in config:
        if qty > 0:
            raw = CURVES_DB[size][t_pcm]
            flows = [r[0] for r in raw]
            powers = [r[1] for r in raw]
            temps = [r[2] for r in raw]
            limits[size] = max(flows)
            interpolators_p[size] = interpolate.interp1d(flows, powers, kind='linear', fill_value="extrapolate")
            interpolators_t[size] = interpolate.interp1d(flows, temps, kind='linear', fill_value="extrapolate")
    if not interpolators_p: return [], [], [], []
    alpha_values = np.linspace(0.01, 1.1, 50) 
    sys_flows, sys_powers, sys_temps, sys_v40_volumes = [], [], [], []
    for alpha in alpha_values:
        f_tot, p_tot_batt, weighted_temp_sum = 0, 0, 0
        for qty, size in config:
            if qty > 0:
                f_single = limits[size] * alpha
                p_single = float(interpolators_p[size](f_single))
                t_single = float(interpolators_t[size](f_single))
                f_block = f_single * qty
                f_tot += f_block
                p_tot_batt += (p_single * qty)
                weighted_temp_sum += (t_single * f_block)
        sys_flows.append(f_tot)
        sys_powers.append(p_tot_batt)
        t_mix = weighted_temp_sum / f_tot if f_tot > 0 else 0
        sys_temps.append(t_mix)
        p_req_at_flow = f_tot * dt_calc * 0.0697
        p_deficit = p_req_at_flow - p_hp_tot
        v40_vol = (e_tot_e0 / p_deficit) * 60 * f_tot if p_deficit > 0 else 99999 
        sys_v40_volumes.append(v40_vol)
    return sys_flows, sys_powers, sys_temps, sys_v40_volumes

def get_daily_profile_curve(n_people=4, building_type="Residenziale"):
    liters_per_person = 50 
    if building_type == "Ufficio": liters_per_person = 15
    elif building_type == "Hotel": liters_per_person = 80
    total_daily_vol = n_people * liters_per_person
    profiles = {
        "Residenziale": [0.5, 0.2, 0.1, 0.1, 0.5, 2.0, 8.0, 12.0, 9.0, 6.0, 5.0, 4.0, 5.0, 4.0, 3.0, 3.0, 4.0, 6.0, 10.0, 11.0, 5.0, 2.0, 1.0, 0.6],
        "Ufficio":      [0, 0, 0, 0, 0, 0, 2, 8, 15, 12, 10, 15, 12, 10, 8, 5, 3, 0, 0, 0, 0, 0, 0, 0],
        "Hotel":        [1, 0.5, 0.5, 0.5, 1, 3, 10, 15, 12, 8, 5, 4, 3, 3, 3, 4, 6, 8, 12, 8, 5, 3, 2, 1]
    }
    selected_profile = profiles.get(building_type, profiles["Residenziale"])
    factor = 100.0 / sum(selected_profile)
    hourly_flow = [(p * factor / 100.0) * total_daily_vol for p in selected_profile]
    hourly_flow_lmin = [val / 60.0 for val in hourly_flow]
    return list(range(24)), hourly_flow_lmin, total_daily_vol

# ==========================================
#  DATABASE PARAMETRI E HP
# ==========================================
prices = { 6: 3300.0, 12: 5100.0, 20: 7400.0, 40: 13200.0 }
NOMINAL_FLOWS = { 6: 10.0, 12: 20.0, 20: 25.0, 40: 50.0 }
params_v40 = {
    6:  {'V0': 130.0,  'max_lmin': 24.0}, 
    12: {'V0': 260.0,  'max_lmin': 32.0},
    20: {'V0': 525.0,  'max_lmin': 40.0},
    40: {'V0': 1050.0, 'max_lmin': 80.0}
}

CURVES_DB = {
    6: {
        48: [[0.0, 0.55, 47.5], [3.0, 5.52, 46.3], [6.0, 10.48, 45.0], [9.0, 15.45, 43.8], [12.0, 20.42, 42.6], [15.0, 25.38, 41.3], [18.0, 30.35, 40.1], [21.0, 35.32, 38.9], [24.0, 40.28, 37.6]],
        58: [[0.0, 1.26, 51.3], [3.0, 7.82, 49.8], [6.0, 14.38, 48.3], [9.0, 20.94, 46.8], [12.0, 27.50, 45.3], [15.0, 34.05, 43.8], [18.0, 40.61, 42.4], [21.0, 47.17, 40.9], [24.0, 53.73, 39.4]],
        74: [[0.0, 1.64, 68.9], [3.0, 10.21, 67.1], [6.0, 18.77, 65.3], [9.0, 27.34, 63.5], [12.0, 35.90, 61.7], [15.0, 44.47, 59.9], [18.0, 53.03, 58.1], [21.0, 61.60, 56.3], [24.0, 70.17, 54.5]],
    },
    12: {
        48: [[0.0, 0.0, 47.8], [4.0, 6.27, 47.1], [8.0, 14.23, 46.5], [12.0, 22.19, 45.8], [16.0, 30.15, 45.2], [20.0, 38.11, 44.5], [24.0, 46.07, 43.9], [28.0, 54.03, 43.2], [32.0, 61.99, 42.6]],
        58: [[0.0, 0.61, 57.8], [4.0, 10.97, 57.0], [8.0, 21.34, 56.2], [12.0, 31.71, 55.5], [16.0, 42.07, 54.7], [20.0, 52.44, 53.9], [24.0, 62.81, 53.1], [28.0, 73.17, 52.3], [32.0, 83.54, 51.5]],
        74: [[0.0, 3.94, 66.9], [4.0, 17.13, 66.3], [8.0, 30.32, 65.6], [12.0, 43.51, 65.0], [16.0, 56.70, 64.3], [20.0, 69.89, 63.7], [24.0, 83.08, 63.0], [28.0, 96.27, 62.3], [32.0, 109.46, 61.7]],
    },
    20: {
        48: [[0.0, 1.42, 47.2], [5.0, 11.66, 46.8], [10.0, 21.90, 46.5], [15.0, 32.14, 46.2], [20.0, 42.38, 45.9], [25.0, 52.62, 45.6], [30.0, 62.86, 45.3], [35.0, 73.10, 45.0], [40.0, 83.35, 44.6]],
        58: [[0.0, 1.58, 56.9], [5.0, 15.11, 56.5], [10.0, 28.64, 56.2], [15.0, 42.17, 55.8], [20.0, 55.70, 55.5], [25.0, 69.23, 55.1], [30.0, 82.76, 54.8], [35.0, 96.29, 54.5], [40.0, 109.82, 54.1]],
        74: [[0.0, 1.87, 68.6], [5.0, 19.34, 68.2], [10.0, 36.81, 67.8], [15.0, 54.28, 67.4], [20.0, 71.75, 67.0], [25.0, 89.22, 66.7], [30.0, 106.69, 66.3], [35.0, 124.16, 65.9], [40.0, 141.63, 65.5]],
    },
    40: {
        48: [[0.0, 2.84, 47.2], [10.0, 23.32, 46.8], [20.0, 43.80, 46.5], [30.0, 64.29, 46.2], [40.0, 84.77, 45.9], [50.0, 105.25, 45.6], [60.0, 125.73, 45.3], [70.0, 146.21, 45.0], [80.0, 166.69, 44.6]],
        58: [[0.0, 3.15, 56.9], [10.0, 30.21, 56.5], [20.0, 57.27, 56.2], [30.0, 84.33, 55.8], [40.0, 111.39, 55.5], [50.0, 138.45, 55.1], [60.0, 165.51, 54.8], [70.0, 192.57, 54.5], [80.0, 219.63, 54.1]],
        74: [[0.0, 3.66, 68.5], [10.0, 38.63, 68.2], [20.0, 73.60, 67.8], [30.0, 108.57, 67.4], [40.0, 143.54, 67.1], [50.0, 178.51, 66.7], [60.0, 213.48, 66.3], [70.0, 248.45, 65.9], [80.0, 283.42, 65.6]],
    },
}

HP_DATABASE = [
    # --- CLIMER ECOFLEX ---
    {"brand": "Climer", "model": "EcoFlex 150", "type": "Aria/Acqua", "kw": 1.8, "gas": "R290", "price": 2800},
    {"brand": "Climer", "model": "EcoFlex 200", "type": "Aria/Acqua", "kw": 2.2, "gas": "R290", "price": 3100},
    {"brand": "Climer", "model": "EcoFlex 300", "type": "Aria/Acqua", "kw": 3.0, "gas": "R290", "price": 3500},
    {"brand": "Climer", "model": "EcoFlex Plus", "type": "Aria/Acqua", "kw": 4.5, "gas": "R290", "price": 4200},
    {"brand": "Climer", "model": "EcoFlex EF02", "type": "Aria/Acqua", "kw": 2.2, "gas": "R134a/R513A", "price": 2600},
    {"brand": "Climer", "model": "EcoFlex EF04", "type": "Aria/Acqua", "kw": 3.8, "gas": "R134a/R513A", "price": 3100},

    # --- ALTRE PICCOLE TAGLIE (< 4 kW) ---
    {"brand": "Panasonic", "model": "Aquarea (J Gen)", "type": "Aria/Acqua", "kw": 3.2, "gas": "R32", "price": 3900},
    {"brand": "Climer", "model": "EcoHeat", "type": "Aria/Acqua", "kw": 3.5, "gas": "R290", "price": 3500},
    
    # --- STANDARD (4 - 200kW) ---
    {"brand": "Climer", "model": "EcoPlus", "type": "Aria/Acqua", "kw": 8.0, "gas": "R290", "price": 5200},
    {"brand": "Climer", "model": "EcoPlus", "type": "Aria/Acqua", "kw": 12.0, "gas": "R290", "price": 6800},
    {"brand": "Daikin", "model": "Altherma 3 R", "type": "Aria/Acqua", "kw": 4.0, "gas": "R32", "price": 4200},
    {"brand": "Daikin", "model": "Altherma 3 R", "type": "Aria/Acqua", "kw": 8.0, "gas": "R32", "price": 4500},
    {"brand": "Daikin", "model": "Altherma 3 H HT", "type": "Aria/Acqua", "kw": 14.0, "gas": "R32", "price": 9800},
    {"brand": "Mitsubishi", "model": "Ecodan Split", "type": "Aria/Acqua", "kw": 8.0, "gas": "R32", "price": 4200},
    {"brand": "Mitsubishi", "model": "Zubadan", "type": "Aria/Acqua", "kw": 12.0, "gas": "R32", "price": 6800},
    {"brand": "Vaillant", "model": "aroTHERM plus", "type": "Aria/Acqua", "kw": 12.0, "gas": "R290", "price": 7500},
    {"brand": "Wolf", "model": "CHA-10", "type": "Aria/Acqua", "kw": 10.0, "gas": "R290", "price": 8200},
    {"brand": "Viessmann", "model": "Vitocal 250-A", "type": "Aria/Acqua", "kw": 10.0, "gas": "R290", "price": 7900},
    {"brand": "Panasonic", "model": "Aquarea T-Cap", "type": "Aria/Acqua", "kw": 9.0, "gas": "R32", "price": 5400},
    {"brand": "Panasonic", "model": "Aquarea T-Cap", "type": "Aria/Acqua", "kw": 16.0, "gas": "R32", "price": 8500},
    {"brand": "Samsung", "model": "EHS TDM Plus", "type": "Aria/Acqua", "kw": 14.0, "gas": "R32", "price": 6200},
    {"brand": "LG", "model": "Therma V Split", "type": "Aria/Acqua", "kw": 16.0, "gas": "R32", "price": 6500},
    {"brand": "Stiebel Eltron", "model": "WPL 25", "type": "Aria/Acqua", "kw": 14.0, "gas": "R410A", "price": 9500},
    {"brand": "Climer", "model": "CA Series", "type": "Aria/Acqua", "kw": 18.0, "gas": "R410A", "price": 9500},
    {"brand": "Climer", "model": "CA Series", "type": "Aria/Acqua", "kw": 30.0, "gas": "R410A", "price": 13000},
    {"brand": "Daikin", "model": "EWYT-B", "type": "Aria/Acqua", "kw": 25.0, "gas": "R32", "price": 11000},
    {"brand": "Daikin", "model": "EWYT-B", "type": "Aria/Acqua", "kw": 50.0, "gas": "R32", "price": 18500},
    {"brand": "Mitsubishi", "model": "CAHV-R", "type": "Aria/Acqua", "kw": 40.0, "gas": "R454B", "price": 16000},
    {"brand": "Aermec", "model": "NRK", "type": "Aria/Acqua", "kw": 35.0, "gas": "R410A", "price": 14500},
    {"brand": "Aermec", "model": "NRK", "type": "Aria/Acqua", "kw": 55.0, "gas": "R410A", "price": 21000},
    {"brand": "Clivet", "model": "ELFOEnergy Sheen", "type": "Aria/Acqua", "kw": 45.0, "gas": "R32", "price": 17000},
    {"brand": "Clivet", "model": "ELFOEnergy Sheen", "type": "Aria/Acqua", "kw": 70.0, "gas": "R32", "price": 24000},
    {"brand": "Carrier", "model": "AquaSnap 30RB", "type": "Aria/Acqua", "kw": 40.0, "gas": "R32", "price": 15500},
    {"brand": "Carrier", "model": "AquaSnap 30RB", "type": "Aria/Acqua", "kw": 70.0, "gas": "R32", "price": 25000},
    {"brand": "Rhoss", "model": "WinPACK", "type": "Aria/Acqua", "kw": 30.0, "gas": "R410A", "price": 12000},
    {"brand": "Climer", "model": "H Series", "type": "Aria/Acqua", "kw": 90.0, "gas": "R410A", "price": 28000},
    {"brand": "Daikin", "model": "EWYT-B", "type": "Aria/Acqua", "kw": 85.0, "gas": "R32", "price": 29000},
    {"brand": "Daikin", "model": "EWAT-B", "type": "Aria/Acqua", "kw": 110.0, "gas": "R32", "price": 36000},
    {"brand": "Daikin", "model": "EWAT-B", "type": "Aria/Acqua", "kw": 150.0, "gas": "R32", "price": 45000},
    {"brand": "Daikin", "model": "EWAT-B", "type": "Aria/Acqua", "kw": 200.0, "gas": "R32", "price": 58000},
    {"brand": "Carrier", "model": "AquaSnap 30RBP", "type": "Aria/Acqua", "kw": 100.0, "gas": "R32", "price": 34000},
    {"brand": "Carrier", "model": "AquaSnap 30RBP", "type": "Aria/Acqua", "kw": 150.0, "gas": "R32", "price": 46000},
    {"brand": "Carrier", "model": "AquaSnap 30RBP", "type": "Aria/Acqua", "kw": 200.0, "gas": "R32", "price": 59000},
    {"brand": "Clivet", "model": "SpinChiller4", "type": "Aria/Acqua", "kw": 120.0, "gas": "R32", "price": 38000},
    {"brand": "Clivet", "model": "SpinChiller4", "type": "Aria/Acqua", "kw": 180.0, "gas": "R32", "price": 52000},
    {"brand": "Aermec", "model": "NRB", "type": "Aria/Acqua", "kw": 100.0, "gas": "R410A", "price": 31000},
    {"brand": "Aermec", "model": "NRG", "type": "Aria/Acqua", "kw": 150.0, "gas": "R32", "price": 47000},
    {"brand": "Aermec", "model": "NRG", "type": "Aria/Acqua", "kw": 200.0, "gas": "R32", "price": 59000},
    {"brand": "Mitsubishi", "model": "NX2-G02", "type": "Aria/Acqua", "kw": 110.0, "gas": "R454B"},
    {"brand": "Mitsubishi", "model": "NX2-G02", "type": "Aria/Acqua", "kw": 160.0, "gas": "R454B"},
    {"brand": "Mitsubishi", "model": "NX2-G02", "type": "Aria/Acqua", "kw": 210.0, "gas": "R454B"},
    {"brand": "Daikin", "model": "Altherma 3 GEO", "type": "Acqua/Acqua", "kw": 10.0, "gas": "R32", "price": 9500},
    {"brand": "Nibe", "model": "S1155", "type": "Acqua/Acqua", "kw": 12.0, "gas": "R407C", "price": 11000},
    {"brand": "Viessmann", "model": "Vitocal 300-G", "type": "Acqua/Acqua", "kw": 17.0, "gas": "R410A", "price": 13000},
    {"brand": "Daikin", "model": "EWWD", "type": "Acqua/Acqua", "kw": 50.0, "gas": "R410A", "price": 22000},
    {"brand": "Daikin", "model": "EWWD", "type": "Acqua/Acqua", "kw": 90.0, "gas": "R410A", "price": 32000},
    {"brand": "Daikin", "model": "EWWD", "type": "Acqua/Acqua", "kw": 150.0, "gas": "R134a"},
    {"brand": "Daikin", "model": "EWWD", "type": "Acqua/Acqua", "kw": 190.0, "gas": "R134a"},
    {"brand": "Aermec", "model": "WRL", "type": "Acqua/Acqua", "kw": 80.0, "gas": "R410A", "price": 28000},
    {"brand": "Aermec", "model": "WRL", "type": "Acqua/Acqua", "kw": 150.0, "gas": "R410A"},
    {"brand": "Clivet", "model": "WSH-XEE", "type": "Acqua/Acqua", "kw": 100.0, "gas": "R410A", "price": 35000},
]

# ==========================================
#  DEFINIZIONE FUNZIONE DI RICERCA (ORA CHE IL DB ESISTE)
# ==========================================
def get_suggested_hp(target_kw):
    if target_kw <= 0: return [], []
    air_water = []
    water_water = []
    # Logica di ricerca flessibile
    search_min = target_kw if target_kw > 3.0 else 0.0
    limit_upper = max(target_kw + 10.0, 8.0)
    
    for hp in HP_DATABASE:
        if search_min <= hp['kw'] < limit_upper:
             if hp['type'] == "Aria/Acqua":
                 air_water.append(hp)
             else:
                 water_water.append(hp)
    
    air_water.sort(key=lambda x: x['kw'])
    water_water.sort(key=lambda x: x['kw'])
    return air_water, water_water

# --- SIDEBAR UI ---
st.sidebar.header("Parametri Progetto")
t_in = st.sidebar.number_input("Temp. Acqua Rete (°C)", value=12.5, step=0.5, format="%.1f")
dt_target = 40.0 - t_in
if dt_target <= 0: dt_target = 27.5

with st.sidebar.expander("🏗️ 1. Utenze (LU)", expanded=False):
    inputs = {}
    inputs['LU1'] = {'qty': st.number_input("Lavabo (1 LU)", 0, value=0), 'val': 1}
    inputs['LU2'] = {'qty': st.number_input("Doccia (2 LU)", 0, value=2), 'val': 2}
    inputs['LU3'] = {'qty': st.number_input("Orinatoio (3 LU)", 0, value=0), 'val': 3}
    inputs['LU4'] = {'qty': st.number_input("Vasca (4 LU)", 0, value=0), 'val': 4}
    inputs['LU5'] = {'qty': st.number_input("Giardino (5 LU)", 0, value=0), 'val': 5}
    inputs['LU8'] = {'qty': st.number_input("Comm. (8 LU)", 0, value=0), 'val': 8}
    inputs['LU15'] = {'qty': st.number_input("Valvola (15 LU)", 0, value=0), 'val': 15}
    
    lu_totali = sum(i['qty']*i['val'] for i in inputs.values()) or 1
    max_lu_unit = max([i['val'] for i in inputs.values() if i['qty']>0] or [1])
    
    qd_ls_target = calcola_qd_en806(lu_totali, max_lu_unit)
    qp_lmin_target = qd_ls_target * 60 
    st.info(f"Target (EN 806): **{qp_lmin_target:.1f} L/min**")

st.sidebar.subheader("🔥 2. Generazione")
recharge_min = st.sidebar.number_input("Tempo Reintegro Target (min)", value=60, step=10, min_value=1)

with st.sidebar.expander("🔋 3. Batterie", expanded=True):
    manage_qty('qty_6', "i-6")
    st.divider()
    manage_qty('qty_12', "i-12")
    st.divider()
    manage_qty('qty_20', "i-20")
    st.divider()
    manage_qty('qty_40', "i-40")
    st.markdown("---")
    t_pcm = st.radio("Temp. PCM (°C)", [48, 58, 74], horizontal=True)

    # --- OTTIMIZZATORE ---
    st.markdown("### 🛠️ Ottimizzatore")
    flow_correction_pct = st.number_input("Coeff. Correzione Portata (%)", value=0, step=1)
    
    if st.button("🚀 OTTIMIZZA (Min. Batterie)"):
        st.toast("Ricerca configurazione...", icon="⏳")
        req_flow = qp_lmin_target
        best_count = float('inf'); best_cost = float('inf'); best_cfg = None
        max_search = 6
        for q40, q20, q12, q6 in itertools.product(range(max_search), range(max_search), range(max_search), range(max_search)):
            if q40==0 and q20==0 and q12==0 and q6==0: continue
            nom_flow_tot = (q6 * NOMINAL_FLOWS[6]) + (q12 * NOMINAL_FLOWS[12]) + (q20 * NOMINAL_FLOWS[20]) + (q40 * NOMINAL_FLOWS[40])
            effective_flow = nom_flow_tot * (1 + flow_correction_pct / 100.0)
            if effective_flow >= req_flow:
                curr_count = q6 + q12 + q20 + q40
                curr_cost = (q6*prices[6]) + (q12*prices[12]) + (q20*prices[20]) + (q40*prices[40])
                if curr_count < best_count:
                    best_count = curr_count; best_cost = curr_cost; best_cfg = (q6, q12, q20, q40)
                elif curr_count == best_count:
                    if curr_cost < best_cost: best_cost = curr_cost; best_cfg = (q6, q12, q20, q40)
        if best_cfg:
            st.session_state.qty_6, st.session_state.qty_12 = best_cfg[0], best_cfg[1]
            st.session_state.qty_20, st.session_state.qty_40 = best_cfg[2], best_cfg[3]
            st.success(f"Trovato: {sum(best_cfg)} Batterie (Cost: €{best_cost:,.0f})")
            st.rerun()
        else:
            st.error("Nessuna combinazione trovata.")

# --- CALCOLI PRINCIPALI ---
# (Spostati QUI, prima della sezione di simulazione che li usa)
config = [(st.session_state.qty_6, 6), (st.session_state.qty_12, 12), 
          (st.session_state.qty_20, 20), (st.session_state.qty_40, 40)]
total_cost = sum(q*prices[s] for q,s in config)

total_energy_e0 = 0
for qty, size in config:
    v0_nominal = params_v40[size]['V0']
    if t_pcm >= 58: v0_nominal /= 0.76 
    e0_single = (v0_nominal * 4.186 * dt_target) / 3600.0
    total_energy_e0 += (e0_single * qty)

recharge_hours = recharge_min / 60.0
p_hp_tot_input = total_energy_e0 / recharge_hours if recharge_hours > 0 and total_energy_e0 > 0 else 0.0

p_load = qp_lmin_target * dt_target * 0.0697
p_req_batt = max(0, p_load - p_hp_tot_input)

net_power_deficit = p_load - p_hp_tot_input
if net_power_deficit > 0.1:
    total_v40_liters = (total_energy_e0 / net_power_deficit) * 60 * qp_lmin_target
    autonomy_min = total_v40_liters / qp_lmin_target if qp_lmin_target > 0 else 0
    autonomy_str = f"{autonomy_min:.1f} min"
else:
    total_v40_liters = 99999
    autonomy_str = "∞ (Illimitata)"

# --- SIMULAZIONE 24H INPUTS ---
with st.sidebar.expander("📈 Simulazione 24h & PV", expanded=True):
    sim_people = st.number_input("Numero Utenti", min_value=1, value=4, step=1)
    sim_type = st.selectbox("Tipo Edificio", ["Residenziale", "Ufficio", "Hotel"])
    
    st.markdown("---")
    st.markdown("**Strategia di Reintegro**")
    recharge_strategy = st.selectbox("Modalità", ["Smart Hysteresis (Standard)", "Autoconsumo Fotovoltaico"])
    
    st.markdown("📍 **Localizzazione Impianto**")
    address_input = st.text_input("Indirizzo (es. Via Roma 1, Milano)", value="")
    if st.button("Cerca Indirizzo"):
        if address_input:
            lat, lon, disp_name = get_coordinates(address_input)
            if lat:
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.address_found = disp_name
                st.success("Indirizzo trovato!")
            else:
                st.error("Indirizzo non trovato.")
    
    st.caption(f"Posizione: {st.session_state.address_found}")
    df_map = pd.DataFrame({'lat': [st.session_state.lat], 'lon': [st.session_state.lon]})
    st.map(df_map, zoom=10, use_container_width=True)
    
    pv_coverage = 0
    val_start_std = 70
    
    if recharge_strategy == "Autoconsumo Fotovoltaico":
        pv_coverage = st.number_input("Copertura PV della PdC (%)", 0, 100, 50, step=10)
        
        # NUOVO SELETTORE PRIORITA'
        pv_priority = st.radio("Priorità Strategia PV", ["🛡️ Massimo Comfort (Batteria sempre carica)", "⚙️ Salvaguardia PdC (Minimizza ON/OFF)"])
        
        if st.button("✨ Ottimizza Strategia"):
            if pv_priority == "🛡️ Massimo Comfort (Batteria sempre carica)":
                st.session_state.pv_start_sun = 98
                st.session_state.pv_start_night = 30
                st.session_state.pv_stop = 100
            else:
                # Strategia Long Cycle: Scarica profonda prima di partire
                st.session_state.pv_start_sun = 50 
                st.session_state.pv_start_night = 15
                st.session_state.pv_stop = 100
            st.rerun()
            
        st.markdown("**Soglie Intervento (%)**")
        start_soc_sun = st.slider("Avvio con Sole se carica < (%)", 0, 100, key='pv_start_sun')
        start_soc_night = st.slider("Avvio Notturno se carica < (%)", 0, 100, key='pv_start_night')
        stop_soc = st.slider("Stop Reintegro se carica > (%)", 0, 100, key='pv_stop')
    else:
        st.markdown("**Soglie Intervento (%)**")
        
        if st.button("✨ Ottimizza Soglie (Min. Energia)"):
            best_kwh = float('inf')
            best_start = 70
            best_stop = 98
            
            # Parametri temporanei per simulazione
            # Usa i valori calcolati globalmente
            tank_cap_opt = total_energy_e0 / (4.186 * dt_target) * 3600.0 if dt_target > 0 else 999
            if net_power_deficit > 0.1:
                tank_cap_opt = (total_energy_e0 / net_power_deficit) * 60 * qp_lmin_target
            
            hp_flow_opt = (p_hp_tot_input * 60) / (4.186 * dt_target) if dt_target > 0 else 0
            
            # Get profile
            _, hourly_flow_opt, _ = get_daily_profile_curve(sim_people, sim_type)
            x_h = np.linspace(0, 1440, 25)
            y_f = hourly_flow_opt + [hourly_flow_opt[0]]
            f_int = interpolate.interp1d(x_h, y_f, kind='linear')
            cons_curve_opt = f_int(np.arange(1440))
            
            # Grid Search
            for start_test in range(10, 90, 10):
                for stop_test in range(start_test + 10, 101, 10):
                    # Simulation Logic (Simplified for optimization)
                    # Warmup
                    curr_v = tank_cap_opt * 0.5
                    is_on = False
                    for _ in range(1440): # Warmup
                        st_th = tank_cap_opt * (start_test/100.0)
                        sp_th = tank_cap_opt * (stop_test/100.0)
                        cons = cons_curve_opt[_]
                        if not is_on: 
                            if curr_v < st_th: is_on = True
                        else:
                            if curr_v >= sp_th: is_on = False
                        prod = hp_flow_opt if is_on else 0
                        curr_v = curr_v - cons + prod
                        if curr_v > tank_cap_opt: curr_v = tank_cap_opt
                        if curr_v < 0: curr_v = 0
                    
                    # Real Run
                    kwh_run = 0
                    min_soc = tank_cap_opt
                    for i in range(1440):
                        st_th = tank_cap_opt * (start_test/100.0)
                        sp_th = tank_cap_opt * (stop_test/100.0)
                        cons = cons_curve_opt[i]
                        if not is_on: 
                            if curr_v < st_th: is_on = True
                        else:
                            if curr_v >= sp_th: is_on = False
                        prod = hp_flow_opt if is_on else 0
                        if is_on: kwh_run += (p_hp_tot_input / 60.0)
                        curr_v = curr_v - cons + prod
                        if curr_v > tank_cap_opt: curr_v = tank_cap_opt
                        if curr_v < 0: curr_v = 0
                        if curr_v < min_soc: min_soc = curr_v
                    
                    # Evaluate
                    if min_soc > 0: # Valid strategy (no cold water)
                        if kwh_run < best_kwh:
                            best_kwh = kwh_run
                            best_start = start_test
                            best_stop = stop_test
            
            st.session_state.std_start = best_start
            st.session_state.std_stop = best_stop
            st.rerun()

        start_soc_std = st.slider("Avvio Reintegro se carica < (%)", 0, 100, key='std_start')
        stop_soc = st.slider("Stop Reintegro se carica > (%)", 0, 100, key='std_stop')

cost_tooltip = "DETTAGLIO COSTI:\n"
if total_cost > 0:
    for qty, size in config:
        if qty > 0:
            sub = qty * prices[size]
            cost_tooltip += f"- {qty}x i-{size}: € {sub:,.0f}\n"
    cost_tooltip += f"\nTOTALE: € {total_cost:,.0f}"
else:
    cost_tooltip = "Nessuna batteria selezionata"

sugg_air, sugg_water = get_suggested_hp(p_hp_tot_input)
all_sugg = sugg_air + sugg_water
hp_cost_disp = "N/A"
hp_tooltip = "Nessuna pompa di calore trovata nel range specificato."
if all_sugg:
    best_hp = min(all_sugg, key=lambda x: x.get('price', float('inf')))
    hp_cost_disp = f"€ {best_hp['price']:,.0f}"
    hp_tooltip = f"MIGLIOR PREZZO:\n{best_hp['brand']} {best_hp['model']}\nPotenza: {best_hp['kw']} kW\nGas: {best_hp['gas']}"

# --- VISUALIZZAZIONE CRUSCOTTO ---
st.subheader("📊 Analisi Prestazioni")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portata Target", f"{qp_lmin_target:.1f} L/min")
c2.metric("Potenza Batt. Req.", f"{p_req_batt:.1f} kW", help="Quota coperta dalla batteria")
c3.metric("Autonomia", autonomy_str, help="Durata alla portata di picco")
c4.metric("Costo Batterie", f"€ {total_cost:,.0f}", help=cost_tooltip)
c4.caption("⚠️ Contattare fornitore per conferma")

c1b, c2b, c3b, c4b = st.columns(4)
c1b.metric(f"Salto Termico", f"{dt_target:.1f} °C")
c2b.metric("Potenza PdC Calc.", f"{p_hp_tot_input:.1f} kW", help=f"Necessaria per ricarica in {recharge_min} min")
val_v40_disp = f"{total_v40_liters:.0f} L" if total_v40_liters < 99999 else "∞"
c3b.metric("Volume V40 Totale", val_v40_disp)
c4b.metric("Costo PdC (Min)", hp_cost_disp, help=hp_tooltip)
c4b.caption("⚠️ Contattare fornitore per conferma")

if p_hp_tot_input > 0 or p_hp_tot_input == 0: 
    st.markdown("### 🔍 Suggerimento Pompe di Calore (Multi-Brand)")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**🌬️ Aria / Acqua**")
        if sugg_air:
            for hp in sugg_air:
                price_disp = f"€ {hp['price']:,.0f}" if 'price' in hp else "N/A"
                st.info(f"**{hp['brand']} {hp['model']}** - {hp['kw']} kW [Gas: {hp['gas']}] | Est.: {price_disp}")
        else: st.warning("Nessun modello trovato in questo range.")
    with sc2:
        st.markdown("**💧 Acqua / Acqua (Geotermiche / Loop)**")
        if sugg_water:
            for hp in sugg_water:
                price_disp = f"€ {hp['price']:,.0f}" if 'price' in hp else "N/A"
                st.success(f"**{hp['brand']} {hp['model']}** - {hp['kw']} kW [Gas: {hp['gas']}] | Est.: {price_disp}")
        else: st.warning("Nessun modello trovato in questo range.")

st.divider()

# --- GRAFICI INTERATTIVI ---
sys_flows, sys_powers, sys_temps, sys_v40_volumes = get_system_curves(config, t_pcm, dt_target, p_hp_tot_input, total_energy_e0)

if len(sys_flows) > 0:
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.subheader("⚡ Potenza")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=sys_flows, y=sys_powers, fill='tozeroy', mode='lines', line=dict(color='#2a9d8f', width=2), name='Max Potenza', hovertemplate='Portata: %{x:.1f} L/min<br>Max Power: %{y:.1f} kW<extra></extra>'))
        limit_p = np.interp(qp_lmin_target, sys_flows, sys_powers)
        col_pt = 'green' if p_req_batt <= limit_p else 'red'
        fig1.add_trace(go.Scatter(x=[qp_lmin_target], y=[p_req_batt], mode='markers', marker=dict(color=col_pt, size=14, line=dict(color='black', width=2)), name='Punto Lavoro', hovertemplate='Richiesta: %{x:.1f} L/min<br>Serve: %{y:.1f} kW<extra></extra>'))
        fig1.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Potenza (kW)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig1, use_container_width=True)
    with col_g2:
        st.subheader("🌡️ Temperatura")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sys_flows, y=sys_temps, mode='lines', line=dict(color='#e76f51', width=2), name='Temp. Uscita', hovertemplate='Portata: %{x:.1f} L/min<br>Temp: %{y:.1f} °C<extra></extra>'))
        t_pt = np.interp(qp_lmin_target, sys_flows, sys_temps)
        fig2.add_trace(go.Scatter(x=[qp_lmin_target], y=[t_pt], mode='markers+text', marker=dict(color='orange', size=14, line=dict(color='black', width=2)), text=[f"{t_pt:.1f}°C"], textposition="top center", name='Punto Lavoro'))
        fig2.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Temp (°C)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig2, use_container_width=True)
    with col_g3:
        st.subheader("💧 Volume V40 Totale")
        fig3 = go.Figure()
        display_max_y = 10000 
        plot_v40_vol = [min(v, display_max_y) for v in sys_v40_volumes]
        fig3.add_trace(go.Scatter(x=sys_flows, y=plot_v40_vol, mode='lines', line=dict(color='#457b9d', width=2), name='Volume Disponibile', hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'))
        pt_v40_vis = min(total_v40_liters, display_max_y)
        label_v40 = f"{total_v40_liters:.0f} L" if total_v40_liters < 9000 else "> 9000 L"
        fig3.add_trace(go.Scatter(x=[qp_lmin_target], y=[pt_v40_vis], mode='markers', marker=dict(color='purple', size=14, line=dict(color='black', width=2)), name='Punto Lavoro', hovertemplate=f'<b>Punto Lavoro</b><br>Portata: %{{x:.1f}} L/min<br>Volume: {label_v40}<extra></extra>'))
        fig3.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Volume V40 (Litri)", margin=dict(l=20,r=20,t=30,b=20), height=350, yaxis=dict(range=[0, min(max(plot_v40_vol)*1.1, display_max_y)]))
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Aggiungi batterie per vedere i grafici.")

st.divider()

# --- GRAFICO SIMULAZIONE 24H (CON SOC E SMART REINTEGRO PV) ---
with st.expander("📈 Simulazione Profilo Giornaliero & Strategia Reintegro", expanded=True):
    hours_day, hourly_flow_lmin, total_daily_L = get_daily_profile_curve(sim_people, sim_type)
    
    # PVGIS Data Retrieval (per scalare la curva)
    # Prendiamo il mese corrente (1-12 -> 0-11 index)
    curr_month_idx = datetime.now().month - 1
    daily_kwh_kwp = get_pvgis_data(st.session_state.lat, st.session_state.lon, curr_month_idx)
    
    # 1. Setup Simulazione
    sim_minutes = 1440 
    x_time = np.arange(sim_minutes)
    
    # Interpolazione del consumo
    x_hours = np.linspace(0, 1440, 25) 
    y_flow = hourly_flow_lmin + [hourly_flow_lmin[0]]
    f_interp = interpolate.interp1d(x_hours, y_flow, kind='linear')
    consumption_curve_min = f_interp(x_time)
    
    # 2. Parametri
    tank_capacity_L = total_v40_liters if total_v40_liters < 99999 else 9999 
    hp_recharge_flow_lmin = (p_hp_tot_input * 60) / (4.186 * dt_target) if dt_target > 0 else 0
    
    # Curva Solare (Site Specific)
    solar_profile_norm = create_solar_curve_site_specific(st.session_state.lat, st.session_state.lon, daily_kwh_kwp)
    s_start, s_end = get_solar_window(st.session_state.get('solar_zone', 'C')) # Default C if not set
    start_min_solar = s_start * 60
    end_min_solar = s_end * 60
    
    # 3. Funzione di Simulazione (per Warm-up e Run Finale)
    def run_simulation_step(start_vol, start_hp_state):
        curr_v = start_vol
        is_hp_on = start_hp_state
        soc_hist = []
        hp_status_hist = []
        tot_kwh = 0
        sol_kwh = 0
        grd_kwh = 0
        cycle_count = 0
        kwh_per_m = p_hp_tot_input / 60.0
        
        for i in range(sim_minutes):
            solar_intensity = solar_profile_norm[i]
            is_sunny = solar_intensity > 0.05 # Threshold di "Sole"
            
            if recharge_strategy == "Autoconsumo Fotovoltaico":
                st_thr = tank_capacity_L * (start_soc_sun/100.0) if is_sunny else tank_capacity_L * (start_soc_night/100.0)
                sp_thr = tank_capacity_L * (stop_soc/100.0)
            else:
                st_thr = tank_capacity_L * (start_soc_std/100.0)
                sp_thr = tank_capacity_L * (stop_soc/100.0)
            
            cons_L = consumption_curve_min[i]
            
            if not is_hp_on:
                if curr_v < st_thr: 
                    is_hp_on = True
                    cycle_count += 1
            else:
                if curr_v >= sp_thr: is_hp_on = False
            
            prod_L = 0
            if is_hp_on:
                prod_L = hp_recharge_flow_lmin
                tot_kwh += kwh_per_m
                if is_sunny and recharge_strategy == "Autoconsumo Fotovoltaico":
                    # Calcola quota solare istantanea
                    s_part = kwh_per_m * (pv_coverage / 100.0) * solar_intensity 
                    if s_part > kwh_per_m: s_part = kwh_per_m
                    sol_kwh += s_part
                    grd_kwh += (kwh_per_m - s_part)
                else:
                    grd_kwh += kwh_per_m
            
            curr_v = curr_v - cons_L + prod_L
            if curr_v > tank_capacity_L: curr_v = tank_capacity_L
            if curr_v < 0: curr_v = 0
            
            soc_hist.append(curr_v)
            hp_status_hist.append(prod_L)
            
        return soc_hist, hp_status_hist, curr_v, is_hp_on, tot_kwh, sol_kwh, grd_kwh, cycle_count

    # Warm-up Day
    _, _, end_vol_0, end_state_0, _, _, _, _ = run_simulation_step(tank_capacity_L * 0.5, False)
    
    # Run Ufficiale
    soc_history, hp_status_history, _, _, total_kwh_used, solar_kwh_used, grid_kwh_used, cycle_total = run_simulation_step(end_vol_0, end_state_0)
    
    # Cumulative Consumption & Production (NEW LOGIC)
    cumulative_consumption = np.cumsum(consumption_curve_min)
    cumulative_production = np.cumsum(hp_status_history)
    
    # Average User Flow (L/min) over 24h
    avg_user_flow_lmin = cumulative_consumption[-1] / 1440.0
    
    # Calculate fills for cumulative graph
    y_fill_green = np.where(cumulative_production > cumulative_consumption, cumulative_production, cumulative_consumption)
    y_fill_red = np.where(cumulative_consumption > cumulative_production, cumulative_consumption, cumulative_production)

    # 4. Grafico
    fig_smart = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # --- ROW 1: MAIN SIMULATION (INSTANTANEOUS) ---
    # Area Solar Curve
    if recharge_strategy == "Autoconsumo Fotovoltaico":
        fig_smart.add_trace(go.Scatter(
            x=x_time/60, y=solar_profile_norm * tank_capacity_L, # Scalato visivamente
            mode='lines', fill='tozeroy', name='Produzione PV (Profilo)',
            line=dict(color='yellow', width=0),
            fillcolor='rgba(255, 215, 0, 0.4)',
            hoverinfo='skip'
        ), row=1, col=1, secondary_y=True)
    
    # Consumo Istantaneo (Rosso)
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=consumption_curve_min,
        mode='lines', fill='tozeroy', name='Prelievo Istantaneo',
        line=dict(color='#ff0000', width=1),
        fillcolor='rgba(255, 0, 0, 0.2)'
    ), row=1, col=1, secondary_y=False)
    
    # Produzione PdC Istantanea
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=hp_status_history,
        mode='lines', name='Reintegro PdC',
        line=dict(color='#2a9d8f', width=2),
        fill='tozeroy', fillcolor='rgba(42, 157, 143, 0.2)'
    ), row=1, col=1, secondary_y=False)
    
    # Target Flow Line (QP Target)
    fig_smart.add_trace(go.Scatter(
        x=[0, 24], y=[qp_lmin_target, qp_lmin_target],
        mode='lines', name='Target di Picco (EN 806)',
        line=dict(color='black', width=2, dash='dashdot')
    ), row=1, col=1, secondary_y=False)
    
    # Average User Flow Line (NEW)
    fig_smart.add_trace(go.Scatter(
        x=[0, 24], y=[avg_user_flow_lmin, avg_user_flow_lmin],
        mode='lines', name='Portata Media Utenza',
        line=dict(color='darkred', width=2, dash='dot')
    ), row=1, col=1, secondary_y=False)
    
    # Reference Lines (Max Capacity)
    max_sys_flow = 0
    nom_sys_flow = 0
    for qty, size in config:
        if qty > 0:
            last_pt = CURVES_DB[size][t_pcm][-1] 
            max_sys_flow += (last_pt[0] * qty)
            nom_sys_flow += (NOMINAL_FLOWS[size] * qty)
            
    fig_smart.add_trace(go.Scatter(x=[0, 24], y=[max_sys_flow, max_sys_flow], mode='lines', name='Capacità Max Batterie', line=dict(color='#2a9d8f', width=2, dash='dash')), row=1, col=1, secondary_y=False)
    
    # --- ROW 2: CUMULATIVE & SOC ---
    
    # 1. SoC Battery (Moved Here)
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=soc_history,
        mode='lines', name='Carica Batteria (SoC)',
        line=dict(color='#007acc', width=3)
    ), row=2, col=1)
    
    # 2. Cumulative Consumption (Red Line)
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=cumulative_consumption,
        mode='lines', name='Cumulato Prelievo',
        line=dict(color='#d62728', width=2)
    ), row=2, col=1)
    
    # Green Fill (Surplus)
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=y_fill_green,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(44, 160, 44, 0.2)', # Green light
        showlegend=False, hoverinfo='skip'
    ), row=2, col=1)
    
    # Cumulative Production (Green Line) - Plotted AFTER fill so it's on top
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=cumulative_production,
        mode='lines', name='Cumulato Produzione',
        line=dict(color='#2ca02c', width=2)
    ), row=2, col=1)
    
    # Red Fill (Deficit)
    fig_smart.add_trace(go.Scatter(x=x_time/60, y=y_fill_red, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 39, 40, 0.2)', showlegend=False), row=2, col=1) 

    fig_smart.update_layout(
        title=f"Strategia: {recharge_strategy} | PVGIS: {daily_kwh_kwp:.1f} kWh/kWp (Media Mensile)",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Axis Titles
    fig_smart.update_xaxes(title_text="Ora del Giorno (0-24h)", row=2, col=1)
    fig_smart.update_yaxes(title_text="Portata (L/min)", row=1, col=1, secondary_y=False)
    fig_smart.update_yaxes(title_text="Volume (L)", row=1, col=1, secondary_y=True, range=[0, tank_capacity_L*1.1])
    fig_smart.update_yaxes(title_text="Volume (L)", row=2, col=1)
    
    st.plotly_chart(fig_smart, use_container_width=True)
    st.caption(f"Nota: La portata media dell'utenza calcolata è di **{avg_user_flow_lmin:.1f} L/min** (Linea rossa tratteggiata), mentre il picco normativo è **{qp_lmin_target:.1f} L/min**.")
    
    # KPI Energetici
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Energia Totale PdC", f"{total_kwh_used:.1f} kWh/giorno")
    if recharge_strategy == "Autoconsumo Fotovoltaico":
        kpi2.metric("Da Fotovoltaico", f"{solar_kwh_used:.1f} kWh", delta=f"{(solar_kwh_used/total_kwh_used)*100:.0f}%" if total_kwh_used>0 else "0%")
        kpi3.metric("Da Rete", f"{grid_kwh_used:.1f} kWh", delta=f"-{(grid_kwh_used/total_kwh_used)*100:.0f}%" if total_kwh_used>0 else "0%", delta_color="inverse")
    else:
        kpi2.metric("Da Rete", f"{total_kwh_used:.1f} kWh")
        kpi3.metric("Autoconsumo", "N/A")
    kpi4.metric("Cicli ON/OFF", f"{cycle_total}", help="Numero di accensioni giornaliere della Pompa di Calore")

# --- TABELLA MIX BATTERIE ---
table_rows = ""
for qty, size in config:
    if qty > 0:
        nom_flow = NOMINAL_FLOWS[size]
        nom_power = (nom_flow * dt_target * 4.186) / 60.0
        last_pt = CURVES_DB[size][t_pcm][-1]
        f_max = last_pt[0] * qty
        p_max = last_pt[1] * qty
        u_price = prices[size]
        t_price = u_price * qty
        table_rows += f"""<tr style='border-bottom: 1px solid #eee;'>
            <td style='padding: 10px; font-weight:bold;'>i-{size}</td>
            <td style='padding: 10px; text-align:center;'>{qty}</td>
            <td style='padding: 10px; text-align:right;'>{nom_flow:.1f} L/min</td>
            <td style='padding: 10px; text-align:right;'>{nom_power:.1f} kW</td>
            <td style='padding: 10px; text-align:right; color:#666;'>{f_max:.1f} L/min</td>
            <td style='padding: 10px; text-align:right; color:#666;'>{p_max:.1f} kW</td>
            <td style='padding: 10px; text-align:right;'>€ {u_price:,.0f}</td>
            <td style='padding: 10px; text-align:right; font-weight:bold;'>€ {t_price:,.0f}</td>
        </tr>"""

if total_cost > 0:
    table_rows += f"""<tr style='background-color:#e9ecef; font-weight:bold; border-top: 2px solid #2a9d8f;'>
        <td colspan="7" style='padding: 12px; text-align:right;'>TOTALE</td>
        <td style='padding: 12px; text-align:right; color:#2a9d8f;'>€ {total_cost:,.0f}</td>
    </tr>"""

if table_rows:
    full_table_html = f"""<div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border:1px solid #ddd; margin-bottom: 20px;">
    <h3 style="margin-top:0; color:#333; border-bottom: 2px solid #2a9d8f; padding-bottom:10px;">🧩 Dettaglio Contributo Batterie</h3>
    <table style="width:100%; text-align:left; border-collapse: collapse;">
    <thead><tr style="background-color:#e9ecef; color:#444; font-size:14px;">
    <th style="padding: 10px;">Modello</th><th style="padding: 10px; text-align:center;">Q.tà</th>
    <th style="padding: 10px; text-align:right;">Portata Nom.</th><th style="padding: 10px; text-align:right;">Potenza Nom.</th>
    <th style="padding: 10px; text-align:right;">Portata Max</th><th style="padding: 10px; text-align:right;">Potenza Max</th>
    <th style="padding: 10px; text-align:right;">Prezzo Unit.</th><th style="padding: 10px; text-align:right;">Subtotale</th>
    </tr></thead><tbody>{table_rows}</tbody></table></div>"""
    st.markdown(full_table_html, unsafe_allow_html=True)

# --- GRAFICO EN 806-3 ---
with st.expander("📉 Vedi Grafico Normativo EN 806-3", expanded=True):
    x_vals = np.logspace(0, 3, 100)
    y_vals = [calcola_qd_en806(x, max_lu_unit) for x in x_vals]
    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Curva Normativa', line=dict(color='#007acc', width=3), hovertemplate='LU: %{x:.1f}<br>Portata: %{y:.2f} l/s<extra></extra>'))
    fig_en.add_trace(go.Scatter(x=[lu_totali], y=[qd_ls_target], mode='markers', name='Punto Progetto', marker=dict(color='red', size=15, line=dict(color='black', width=2)), hovertemplate='<b>IL TUO PROGETTO</b><br>Totale LU: %{x:.0f}<br>Portata Target: %{y:.2f} l/s<extra></extra>'))
    fig_en.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Load Units Totali (LU)", yaxis_title="Portata di Progetto (l/s)", margin=dict(l=20, r=20, t=30, b=20), height=400, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_en, use_container_width=True)
