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
import time

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Simulatore i-TES Pro", layout="wide")
st.title("🚰 Simulatore i-TES: Site-Specific & Confronto Tecnologico")

# --- GESTIONE STATO INIZIALE ---
if 'qty_6' not in st.session_state: st.session_state.qty_6 = 0
if 'qty_12' not in st.session_state: st.session_state.qty_12 = 0
if 'qty_20' not in st.session_state: st.session_state.qty_20 = 0
if 'qty_40' not in st.session_state: st.session_state.qty_40 = 0
if 'lat' not in st.session_state: st.session_state.lat = 41.9028 # Default Roma
if 'lon' not in st.session_state: st.session_state.lon = 12.4964
if 'address_found' not in st.session_state: st.session_state.address_found = "Roma, Italia (Default)"

# Inizializzazione Slider Strategie
if 'pv_start_sun' not in st.session_state: st.session_state.pv_start_sun = 50
if 'pv_start_night' not in st.session_state: st.session_state.pv_start_night = 15
if 'pv_stop' not in st.session_state: st.session_state.pv_stop = 100
if 'std_start' not in st.session_state: st.session_state.std_start = 70
if 'std_stop' not in st.session_state: st.session_state.std_stop = 98
if 'recharge_min' not in st.session_state: st.session_state.recharge_min = 60
if 'flow_correction_pct' not in st.session_state: st.session_state.flow_correction_pct = 0

# --- APPLY PENDING UPDATES (FIX PER STREAMLIT API ERROR) ---
# Questo blocco applica i valori calcolati dall'ottimizzatore PRIMA che i widget vengano disegnati
if 'pending_opt' in st.session_state:
    opt = st.session_state.pop('pending_opt') # Recupera e rimuove
    st.session_state.qty_6 = opt['q6']
    st.session_state.qty_12 = opt['q12']
    st.session_state.qty_20 = opt['q20']
    st.session_state.qty_40 = opt['q40']
    st.session_state.recharge_min = opt['rt']
    st.session_state.std_start = opt['start']
    st.session_state.std_stop = opt['stop']
    st.session_state.show_success_msg = f"🏆 Ottimizzazione completata!\nConfigurazione: {opt['q6']}xi6, {opt['q12']}xi12, {opt['q20']}xi20, {opt['q40']}xi40 | Ricarica: {opt['rt']}min"

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
    windows = {"A": (9, 17), "B": (9, 17), "C": (10, 16), "D": (10, 16), "E": (11, 15), "F": (11, 15)}
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
    params = {'lat': lat, 'lon': lon, 'peakpower': 1, 'loss': 14, 'outputformat': 'json', 'angle': 35, 'aspect': 0}
    try:
        r = requests.get(url, params=params)
        if r.status_code == 200:
            data = r.json()
            return data['outputs']['monthly']['fixed'][month_idx]['E_d']
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
        "Hotel":        [1, 0.5, 0.5, 0.5, 1, 3, 10, 15, 12, 8, 5, 4, 3, 3, 4, 6, 8, 12, 8, 5, 3, 2, 1]
    }
    selected_profile = profiles.get(building_type, profiles["Residenziale"])
    factor = 100.0 / sum(selected_profile)
    hourly_flow = [(p * factor / 100.0) * total_daily_vol for p in selected_profile]
    hourly_flow_lmin = [val / 60.0 for val in hourly_flow]
    return list(range(24)), hourly_flow_lmin, total_daily_vol

def calculate_water_tank_config(target_v40, t_store, t_net, db_to_use):
    if t_store <= 40: return None, 0, 0, 0
    factor = (t_store - t_net) / (40 - t_net)
    if factor <= 0: return None, 0, 0, 0
    needed_real_vol = target_v40 / factor
    sorted_db = sorted(db_to_use, key=lambda k: k['vol'], reverse=True)
    max_tank = sorted_db[0]
    selected_tank = None
    qty = 1
    if needed_real_vol <= max_tank['vol']:
        valid_tanks = [t for t in sorted_db if t['vol'] >= needed_real_vol]
        if valid_tanks: selected_tank = min(valid_tanks, key=lambda x: x['vol'])
        else: selected_tank = max_tank 
    else:
        selected_tank = max_tank
        qty = math.ceil(needed_real_vol / max_tank['vol'])
    total_vol = selected_tank['vol'] * qty
    total_price = selected_tank['price'] * qty
    return selected_tank, qty, total_vol, total_price

def calc_dispersion_kwh(volume_l, current_temp, ambient_temp=20.0, use_sanicube_data=False):
    if volume_l <= 0: return 0.0
    delta_t = max(0, current_temp - ambient_temp)
    
    if use_sanicube_data:
        ref_loss_per_liter = 1.4 / 500.0 
        current_loss_per_liter = ref_loss_per_liter * (delta_t / 40.0)
        daily_loss = current_loss_per_liter * volume_l
        return daily_loss / 1440.0 
    else:
        vol_m3 = volume_l / 1000.0
        radius = (vol_m3 / (4 * math.pi))**(1/3)
        area_m2 = 10 * math.pi * (radius**2)
        u_value = 0.35 
        power_loss_w = u_value * area_m2 * delta_t
        energy_loss_kwh = (power_loss_w / 1000.0) / 60.0
        return energy_loss_kwh

def estimate_cop(t_target):
    base_cop = 3.2
    base_t = 50.0
    diff = t_target - base_t
    penalty_factor = 0.025 
    new_cop = base_cop * (1 - (diff * penalty_factor))
    return max(1.5, new_cop)

# --- FUNZIONI SIMULAZIONE ---
def run_pcm_plot_data(soc_history_data, pcm_temp_func, t_pcm_val, t_in_val, consumption_data):
    curr_v = soc_history_data[0]
    soc_list = []
    temp_list = []
    for i in range(len(soc_history_data)):
        cons_L = consumption_data[i]
        curr_v = soc_history_data[i] 
        soc_list.append(curr_v)
        if cons_L > 0.1 and curr_v > 0:
            try: real_temp = float(pcm_temp_func(cons_L))
            except: real_temp = t_pcm_val
            temp_list.append(min(real_temp, t_pcm_val))
        elif curr_v > 0:
            temp_list.append(t_pcm_val)
        else:
            temp_list.append(t_in_val + (t_pcm_val - t_in_val) * 0.1) 
    return soc_list, temp_list

def run_simulation_step_water(start_vol, start_hp_state, water_tank_vol_total, consumption_curve_min, hp_recharge_flow_lmin, kwh_per_m, t_water_set, t_in, is_sanicube=False):
    curr_v = start_vol
    is_hp_on = start_hp_state
    soc_hist = []
    temp_hist = []
    tot_kwh = 0
    cycle_count = 0
    sim_minutes = len(consumption_curve_min)

    for i in range(sim_minutes):
        curr_temp = t_in + (curr_v / water_tank_vol_total) * (t_water_set - t_in)
        loss_kwh = calc_dispersion_kwh(water_tank_vol_total, curr_temp, 20.0, use_sanicube_data=is_sanicube)
        delta_T_sys = t_water_set - t_in
        loss_liters = 0
        if delta_T_sys > 0:
            loss_liters = loss_kwh / (delta_T_sys * 0.00116)
        
        cons_L = consumption_curve_min[i] + loss_liters
        st_thr = water_tank_vol_total * 0.85
        sp_thr = water_tank_vol_total * 1.0
        
        if not is_hp_on:
            if curr_v < st_thr: 
                is_hp_on = True
                cycle_count += 1
        else:
            if curr_v >= sp_thr: is_hp_on = False
        
        prod_L = hp_recharge_flow_lmin if is_hp_on else 0
        if is_hp_on: tot_kwh += kwh_per_m
        
        curr_v = curr_v - cons_L + prod_L
        if curr_v > water_tank_vol_total: curr_v = water_tank_vol_total
        if curr_v < 0: curr_v = 0
        
        soc_hist.append(curr_v)
        temp_hist.append(curr_temp)
        
    return soc_hist, temp_hist, curr_v, is_hp_on, tot_kwh, cycle_count

def get_suggested_hp(target_kw, target_temp):
    if target_kw <= 0: return []
    valid_hps = []
    search_min = target_kw * 0.8
    limit_upper = target_kw * 2.5 
    for hp in HP_DATABASE:
        if search_min <= hp['kw'] < limit_upper:
             if hp.get('max_t', 55) >= target_temp:
                 valid_hps.append(hp)
    valid_hps.sort(key=lambda x: x.get('price', float('inf')))
    return valid_hps

# --- FUNZIONE CURVE REALI SANICUBE (V40) ---
def get_sanicube_v40_at_flow(flow_lmin, t_store):
    data_500 = {
        50: {11: 1550, 12: 990, 15: 530, 20: 290, 25: 180, 30: 100, 35: 50},
        55: {11: 1900, 12: 1280, 15: 730, 20: 450, 25: 320, 30: 220, 35: 150},
        60: {11: 2400, 12: 1510, 15: 860, 20: 580, 25: 440, 30: 340, 35: 270},
        65: {11: 2200, 12: 1550, 15: 970, 20: 680, 25: 530, 30: 430, 35: 360},
        70: {11: 2300, 12: 1640, 15: 1060, 20: 770, 25: 610, 30: 500, 35: 430},
        75: {11: 2400, 12: 1700, 15: 1140, 20: 850, 25: 680, 30: 570, 35: 490}
    }
    avail_temps = sorted(data_500.keys())
    closest_t = min(avail_temps, key=lambda x: abs(x - t_store))
    curve_data = data_500[closest_t]
    flows = np.array(sorted(curve_data.keys()))
    v40s = np.array([curve_data[f] for f in flows])
    if flow_lmin < flows[0]: return v40s[0]
    if flow_lmin > flows[-1]: return v40s[-1]
    return np.interp(flow_lmin, flows, v40s)

# ==========================================
#  DATABASE
# ==========================================
prices = { 6: 3300.0, 12: 5100.0, 20: 7400.0, 40: 13200.0 }
NOMINAL_FLOWS = { 6: 10.0, 12: 20.0, 20: 25.0, 40: 50.0 }
params_v40 = {
    6:  {'V0': 130.0,  'max_lmin': 24.0}, 
    12: {'V0': 260.0,  'max_lmin': 32.0},
    20: {'V0': 525.0,  'max_lmin': 40.0},
    40: {'V0': 1050.0, 'max_lmin': 80.0}
}

PCM_SPECS_DB = {
    6:  {'w': 337, 'd': 695, 'h': 1204},
    12: {'w': 500, 'd': 695, 'h': 1500},
    20: {'w': 790, 'd': 1200, 'h': 950},
    40: {'w': 790, 'd': 1200, 'h': 1500},
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
    {"brand": "Climer", "model": "EcoFlex 150", "type": "Aria/Acqua", "kw": 1.8, "gas": "R290", "max_t": 70, "price": 2800},
    {"brand": "Climer", "model": "EcoFlex 200", "type": "Aria/Acqua", "kw": 2.2, "gas": "R290", "max_t": 70, "price": 3100},
    {"brand": "Climer", "model": "EcoFlex 300", "type": "Aria/Acqua", "kw": 3.0, "gas": "R290", "max_t": 70, "price": 3500},
    {"brand": "Climer", "model": "EcoFlex Plus", "type": "Aria/Acqua", "kw": 4.5, "gas": "R290", "max_t": 70, "price": 4200},
    {"brand": "Climer", "model": "EcoFlex EF02", "type": "Aria/Acqua", "kw": 2.2, "gas": "R134a/R513A", "price": 2600},
    {"brand": "Climer", "model": "EcoFlex EF04", "type": "Aria/Acqua", "kw": 3.8, "gas": "R134a/R513A", "price": 3100},
    {"brand": "Panasonic", "model": "Aquarea (J Gen)", "type": "Aria/Acqua", "kw": 3.2, "gas": "R32", "max_t": 60, "price": 3900},
    {"brand": "Climer", "model": "EcoHeat", "type": "Aria/Acqua", "kw": 3.5, "gas": "R290", "max_t": 70, "price": 3500},
    {"brand": "Climer", "model": "EcoPlus", "type": "Aria/Acqua", "kw": 8.0, "gas": "R290", "max_t": 70, "price": 5200},
    {"brand": "Climer", "model": "EcoPlus", "type": "Aria/Acqua", "kw": 12.0, "gas": "R290", "max_t": 70, "price": 6800},
    {"brand": "Daikin", "model": "Altherma 3 R", "type": "Aria/Acqua", "kw": 4.0, "gas": "R32", "max_t": 60, "price": 4200},
    {"brand": "Daikin", "model": "Altherma 3 R", "type": "Aria/Acqua", "kw": 8.0, "gas": "R32", "max_t": 60, "price": 4500},
    {"brand": "Mitsubishi", "model": "Ecodan Split", "type": "Aria/Acqua", "kw": 8.0, "gas": "R32", "max_t": 60, "price": 4200},
    {"brand": "Panasonic", "model": "Aquarea T-Cap", "type": "Aria/Acqua", "kw": 9.0, "gas": "R32", "max_t": 60, "price": 5400},
    {"brand": "Vaillant", "model": "aroTHERM plus", "type": "Aria/Acqua", "kw": 12.0, "gas": "R290", "max_t": 75, "price": 7500},
    {"brand": "Samsung", "model": "EHS TDM Plus", "type": "Aria/Acqua", "kw": 14.0, "gas": "R32", "max_t": 60, "price": 6200},
    {"brand": "LG", "model": "Therma V Split", "type": "Aria/Acqua", "kw": 16.0, "gas": "R32", "max_t": 65, "price": 6500},
    {"brand": "Climer", "model": "CA-20", "type": "Aria/Acqua", "kw": 20.0, "gas": "R410A", "max_t": 55, "price": 9500},
    {"brand": "Daikin", "model": "EWYT-B", "type": "Aria/Acqua", "kw": 25.0, "gas": "R32", "max_t": 60, "price": 11000},
    {"brand": "Climer", "model": "CA-30", "type": "Aria/Acqua", "kw": 30.0, "gas": "R410A", "max_t": 55, "price": 13000},
    {"brand": "Aermec", "model": "NRK", "type": "Aria/Acqua", "kw": 35.0, "gas": "R410A", "max_t": 65, "price": 14500},
    {"brand": "Carrier", "model": "AquaSnap 30RB", "type": "Aria/Acqua", "kw": 40.0, "gas": "R32", "max_t": 60, "price": 15500},
    {"brand": "Mitsubishi", "model": "CAHV-R", "type": "Aria/Acqua", "kw": 45.0, "gas": "R454B", "max_t": 70, "price": 17000},
    {"brand": "Mitsubishi", "model": "QAHV", "type": "Aria/Acqua (CO2)", "kw": 40.0, "gas": "R744", "max_t": 90, "price": 22000},
    {"brand": "Clivet", "model": "Thunder", "type": "Aria/Acqua", "kw": 40.0, "gas": "R290", "max_t": 75, "price": 18000},
    {"brand": "Daikin", "model": "EWYT-B", "type": "Aria/Acqua", "kw": 50.0, "gas": "R32", "max_t": 60, "price": 18500},
    {"brand": "Clivet", "model": "ELFOEnergy Sheen", "type": "Aria/Acqua", "kw": 60.0, "gas": "R32", "max_t": 60, "price": 22000},
    {"brand": "Carrier", "model": "AquaSnap 30RB", "type": "Aria/Acqua", "kw": 70.0, "gas": "R32", "max_t": 60, "price": 25000},
    {"brand": "Daikin", "model": "EWAT-B", "type": "Aria/Acqua", "kw": 85.0, "gas": "R32", "max_t": 55, "price": 29000},
    {"brand": "Climer", "model": "H-Series", "type": "Aria/Acqua", "kw": 90.0, "gas": "R410A", "max_t": 55, "price": 31000},
    {"brand": "Aermec", "model": "NRB", "type": "Aria/Acqua", "kw": 100.0, "gas": "R410A", "max_t": 55, "price": 34000},
    {"brand": "Mitsubishi", "model": "NX2-G02", "type": "Aria/Acqua", "kw": 110.0, "gas": "R454B", "max_t": 60, "price": 38000},
    {"brand": "Clivet", "model": "SpinChiller4", "type": "Aria/Acqua", "kw": 120.0, "gas": "R32", "max_t": 55, "price": 42000},
    {"brand": "Daikin", "model": "EWAT-B", "type": "Aria/Acqua", "kw": 150.0, "gas": "R32", "max_t": 55, "price": 48000},
    {"brand": "Aermec", "model": "NRK (Big)", "type": "Aria/Acqua", "kw": 150.0, "gas": "R410A", "max_t": 65, "price": 55000},
    {"brand": "Aermec", "model": "NRG", "type": "Aria/Acqua", "kw": 200.0, "gas": "R32", "max_t": 55, "price": 62000},
    {"brand": "Swegon", "model": "BlueBox Zeta Rev", "type": "Aria/Acqua", "kw": 150.0, "gas": "R290", "max_t": 70, "price": 65000},
]

BUFFER_TANK_DB_GENERIC = [
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 300, "h": 1600, "d": 650, "price": 900},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 500, "h": 1700, "d": 750, "price": 1200},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 800, "h": 1800, "d": 900, "price": 1600},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 1000, "h": 2100, "d": 950, "price": 1900},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 1500, "h": 2200, "d": 1100, "price": 2500},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 2000, "h": 2436, "d": 1300, "price": 2800},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 2500, "h": 2500, "d": 1350, "price": 3200},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 3000, "h": 2900, "d": 1350, "price": 3800},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 4000, "h": 2900, "d": 1500, "price": 4800},
    {"brand": "Cordivari", "model": "Volano Termico", "vol": 5000, "h": 2950, "d": 1700, "price": 5800},
]

DAIKIN_SANICUBE_DB = [
    {"brand": "Daikin", "model": "Sanicube EKHWP300B", "vol": 294, "h": 1646, "d": 615, "price": 1800},
    {"brand": "Daikin", "model": "Sanicube EKHWP500B", "vol": 477, "h": 1658, "d": 790, "price": 2400}, 
]

# --- SELETTORE CONFRONTO (Sidebar) ---
st.sidebar.markdown("---")
comp_target = st.sidebar.radio("Scegli Tecnologia di Confronto", ["Volano Termico Generico", "Daikin Sanicube (Drain-Back)"])
is_sanicube = comp_target == "Daikin Sanicube (Drain-Back)"

if is_sanicube:
    t_sanicube_set = st.sidebar.select_slider("Temp. Accumulo Sanicube (°C)", options=[50, 55, 60, 65, 70, 75], value=60)

# --- SEZIONE INPUT UTENTE (Sidebar) ---
st.sidebar.header("Parametri Progetto")
# Define sim inputs FIRST to be available for optimizer
sim_people = st.sidebar.number_input("Numero Utenti", min_value=1, value=4, step=1, key="sim_people_input")
sim_type = st.sidebar.selectbox("Tipo Edificio", ["Residenziale", "Ufficio", "Hotel"], key="sim_type_input")

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
recharge_min = st.sidebar.number_input("Tempo Reintegro Target (min)", value=60, step=10, min_value=1, key="recharge_min")

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

    # --- OTTIMIZZATORE GLOBALE ---
    st.markdown("### 🛠️ Ottimizzatore")
    flow_correction_pct = st.number_input("Coeff. Correzione Portata (%)", value=0, step=1, key="flow_correction_pct")
    
    if st.button("🚀 AUTO-PILOT: Trova Configurazione Ottimale"):
        with st.status("Analisi in corso...", expanded=True) as status:
            st.write("🔍 Fase 1: Esplorazione configurazioni hardware...")
            
            # --- PHASE 1: Find Hardware Combinations ---
            req_flow = qp_lmin_target
            valid_hardware = []
            
            for q40, q20, q12, q6 in itertools.product(range(5), range(5), range(5), range(5)):
                if q40==0 and q20==0 and q12==0 and q6==0: continue
                
                nom_flow_tot = (q6 * NOMINAL_FLOWS[6]) + (q12 * NOMINAL_FLOWS[12]) + (q20 * NOMINAL_FLOWS[20]) + (q40 * NOMINAL_FLOWS[40])
                effective_flow = nom_flow_tot * (1 + st.session_state.flow_correction_pct / 100.0)
                
                if effective_flow >= req_flow:
                    hw_cost = (q6*prices[6]) + (q12*prices[12]) + (q20*prices[20]) + (q40*prices[40])
                    valid_hardware.append({
                        'cfg': (q6, q12, q20, q40),
                        'cost': hw_cost,
                        'flow': effective_flow
                    })
            
            valid_hardware.sort(key=lambda x: x['cost'])
            candidates = valid_hardware[:3]
            
            st.write(f"✅ Trovate {len(valid_hardware)} configurazioni valide. Testo le top {len(candidates)}...")
            
            # --- PHASE 2: Simulate Strategies ---
            best_roi_years = float('inf')
            best_solution = None
            
            recharge_times = [60, 90, 120] 
            start_socs = [30, 50, 70] 
            stop_socs = [90, 100] 
            
            # Pre-calc consumption curve for simulation
            _, hourly_flow_opt, _ = get_daily_profile_curve(sim_people, sim_type)
            x_time_opt = np.arange(1440)
            flow_list_opt = list(hourly_flow_opt) + [hourly_flow_opt[0]]
            x_h_opt = np.linspace(0, 1440, len(flow_list_opt))
            f_int_opt = interpolate.interp1d(x_h_opt, flow_list_opt, kind='linear')
            cons_curve_opt = f_int_opt(x_time_opt)

            total_iterations = len(candidates) * len(recharge_times) * len(start_socs) * len(stop_socs)
            iter_count = 0
            
            for hw in candidates:
                q6, q12, q20, q40 = hw['cfg']
                
                e_tot = 0
                for q, s in zip([q6, q12, q20, q40], [6, 12, 20, 40]):
                    v0 = params_v40[s]['V0']
                    if t_pcm >= 58: v0 /= 0.76
                    e_tot += (v0 * 4.186 * dt_target) / 3600.0 * q
                
                for rt in recharge_times:
                    p_hp = e_tot / (rt/60.0)
                    hp_cost = 0
                    hp_sel = get_suggested_hp(p_hp, t_pcm)
                    if hp_sel: hp_cost = hp_sel[0]['price']
                    
                    ites_capex = hw['cost'] + hp_cost
                    
                    for start_s in start_socs:
                        for stop_s in stop_socs:
                            iter_count += 1
                            if iter_count % 50 == 0:
                                status.update(label=f"Simulazione scenario {iter_count}/{total_iterations}...", state="running")
                            
                            kwh_to_v40 = (3600) / (4.186 * dt_target)
                            tank_cap_v40 = e_tot * kwh_to_v40
                            hp_flow_v40 = (p_hp * 60) / (4.186 * dt_target)
                            
                            curr_v = tank_cap_v40 * 0.5
                            is_on = False
                            min_soc_reached = tank_cap_v40
                            kwh_consumed = 0
                            
                            st_th = tank_cap_v40 * (start_s/100.0)
                            sp_th = tank_cap_v40 * (stop_s/100.0)
                            
                            for _ in range(1440):
                                cons = cons_curve_opt[_]
                                if not is_on:
                                    if curr_v < st_th: is_on = True
                                else:
                                    if curr_v >= sp_th: is_on = False
                                
                                prod = hp_flow_v40 if is_on else 0
                                if is_on: kwh_consumed += (p_hp / 60.0)
                                
                                curr_v = curr_v - cons + prod
                                if curr_v > tank_cap_v40: curr_v = tank_cap_v40
                                if curr_v < 0: curr_v = 0
                                if curr_v < min_soc_reached: min_soc_reached = curr_v
                            
                            if min_soc_reached > 0:
                                cop = estimate_cop(t_pcm)
                                elec_kwh_yr = (kwh_consumed / cop) * 365
                                opex_yr = elec_kwh_yr * 0.31 
                                
                                tco = ites_capex + (opex_yr * 15)
                                
                                if best_solution is None or tco < best_solution['tco']:
                                    best_solution = {
                                        'tco': tco,
                                        'capex': ites_capex,
                                        'opex': opex_yr,
                                        'cfg': hw['cfg'],
                                        'rt': rt,
                                        'start': start_s,
                                        'stop': stop_s
                                    }
            
            status.update(label="Ottimizzazione completata!", state="complete", expanded=False)
            
            if best_solution:
                # Save to PENDING state to apply on next rerun
                st.session_state['pending_opt'] = {
                    'q6': best_solution['cfg'][0],
                    'q12': best_solution['cfg'][1],
                    'q20': best_solution['cfg'][2],
                    'q40': best_solution['cfg'][3],
                    'rt': best_solution['rt'],
                    'start': best_solution['start'],
                    'stop': best_solution['stop']
                }
                time.sleep(0.5) 
                st.rerun()
            else:
                st.error("Nessuna soluzione trovata che soddisfi i requisiti (prova ad aumentare la % Correzione Portata).")

# --- CHECK FOR PENDING UPDATES (MOVED HERE TO CATCH RERUN) ---
if 'pending_opt' in st.session_state:
    opt = st.session_state.pop('pending_opt')
    st.session_state.qty_6 = opt['q6']
    st.session_state.qty_12 = opt['q12']
    st.session_state.qty_20 = opt['q20']
    st.session_state.qty_40 = opt['q40']
    st.session_state.recharge_min = opt['rt']
    st.session_state.std_start = opt['start']
    st.session_state.std_stop = opt['stop']
    st.success(f"🏆 Ottimizzazione Applicata!\nConfig: {opt['q6']}xi6, {opt['q12']}xi12, {opt['q20']}xi20, {opt['q40']}xi40 | Ricarica: {opt['rt']}min")


# --- CALCOLI PRINCIPALI ---
# Ora che t_pcm è stato definito nella sidebar, possiamo usarlo
config = [(st.session_state.qty_6, 6), (st.session_state.qty_12, 12), 
          (st.session_state.qty_20, 20), (st.session_state.qty_40, 40)]
total_cost = sum(q*prices[s] for q,s in config)

total_energy_e0 = 0
for qty, size in config:
    v0_nominal = params_v40[size]['V0']
    if t_pcm >= 58: v0_nominal /= 0.76 
    e0_single = (v0_nominal * 4.186 * dt_target) / 3600.0
    total_energy_e0 += (e0_single * qty)

recharge_hours = st.session_state.recharge_min / 60.0
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

if is_sanicube:
    t_water_set = t_sanicube_set
else:
    t_water_set = 60.0 if t_pcm < 60 else t_pcm 

db_to_use = DAIKIN_SANICUBE_DB if is_sanicube else BUFFER_TANK_DB_GENERIC
suggested_tank, tank_qty, tank_total_vol, tank_total_price = calculate_water_tank_config(total_v40_liters, t_water_set, t_in, db_to_use)

cost_tooltip = "DETTAGLIO COSTI:\n"
if total_cost > 0:
    for qty, size in config:
        if qty > 0:
            sub = qty * prices[size]
            cost_tooltip += f"- {qty}x i-{size}: € {sub:,.0f}\n"
    cost_tooltip += f"\nTOTALE: € {total_cost:,.0f}"
else:
    cost_tooltip = "Nessuna batteria selezionata"

# --- SIMULAZIONE 24H INPUTS ---
with st.sidebar.expander("📈 Simulazione 24h & PV", expanded=True):
    # INPUTS MOVED UP BEFORE BUTTON
    # Just displaying them again if needed or rely on the ones above.
    # But since they are inputs, they must be unique.
    # The previous inputs were 'sim_people_input'. 
    # We can just use the values from session state here or display text.
    st.write(f"Utenti: {sim_people} | Tipo: {sim_type}")
    
    st.markdown("---")
    is_pv_mode = st.toggle("☀️ Attiva Autoconsumo Fotovoltaico", value=False)
    
    if is_pv_mode:
        recharge_strategy = "Autoconsumo Fotovoltaico"
        st.session_state.pv_start_sun = 50 
        st.session_state.pv_start_night = 15
        st.session_state.pv_stop = 100
        st.info("ℹ️ Modalità **Salvaguardia PdC** attivata automaticamente.")
    else:
        recharge_strategy = "Smart Hysteresis (Standard)"

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
    
    if is_pv_mode:
        pv_coverage = st.number_input("Copertura PV della PdC (%)", 0, 100, 50, step=10)
        st.markdown("**Soglie Intervento (Automatiche)**")
        st.text(f"Avvio con Sole: < {st.session_state.pv_start_sun}%")
        st.text(f"Avvio Notturno: < {st.session_state.pv_start_night}%")
    else:
        st.markdown("**Soglie Intervento (%)**")
        start_soc_std = st.slider("Avvio Reintegro se carica < (%)", 0, 100, key='std_start')
        stop_soc = st.slider("Stop Reintegro se carica > (%)", 0, 100, key='std_stop')

# ==========================================
#  VISUALIZZAZIONE DASHBOARD
# ==========================================

st.subheader("📊 Analisi Prestazioni")
c1, c2, c3 = st.columns(3)
c1.metric("Portata Target", f"{qp_lmin_target:.1f} L/min")
c2.metric("Potenza Batt. Req.", f"{p_req_batt:.1f} kW", help="Quota coperta dalla batteria")
c3.metric("Autonomia", autonomy_str, help="Durata alla portata di picco")

c1b, c2b, c3b = st.columns(3)
c1b.metric(f"Salto Termico", f"{dt_target:.1f} °C")
c2b.metric("Potenza PdC Calc.", f"{p_hp_tot_input:.1f} kW", help=f"Necessaria per ricarica in {recharge_min} min")
val_v40_disp = f"{total_v40_liters:.0f} L" if total_v40_liters < 99999 else "∞"
c3b.metric("Volume V40 Totale", val_v40_disp)

st.divider()

# --- CONFRONTO VOLANO TERMICO ---
if suggested_tank:
    st.subheader(f"🆚 Confronto Tecnologico: i-TES vs {comp_target}")
    
    ites_dims_str = ""
    batt_detail_str = ""
    ites_physical_vol_L = 0 
    for qty, size in config:
        if qty > 0 and size in PCM_SPECS_DB:
            spec = PCM_SPECS_DB[size]
            ites_dims_str += f"- {qty}x [L {spec['w']} x P {spec['d']} x H {spec['h']} mm]\n"
            batt_detail_str += f"- {qty}x i-{size}: € {qty * prices[size]:,.0f}\n"
            # Calc Volume Liters
            vol_single = (spec['w'] * spec['d'] * spec['h']) / 1000000.0
            ites_physical_vol_L += (vol_single * qty)
    
    tank_physical_vol_L = 0
    if suggested_tank:
        if is_sanicube:
            # Sanicube (Prisma): W x D x H. Assumiamo base quadrata con lato 'd'
            # Convert mm^3 to Liters
            vol_single_geo = (suggested_tank['d'] * suggested_tank['d'] * suggested_tank['h']) / 1000000.0
            tank_physical_vol_L = vol_single_geo * tank_qty
        else:
            # Generic (Cilindro): pi * r^2 * h
            radius_dm = (suggested_tank['d'] / 100.0) / 2.0
            height_dm = suggested_tank['h'] / 100.0
            vol_single_tank = math.pi * (radius_dm**2) * height_dm # Liters (dm^3)
            tank_physical_vol_L = vol_single_tank * tank_qty

    valid_hps_ites = get_suggested_hp(p_hp_tot_input, t_pcm)
    hp_ites_sel = valid_hps_ites[0] if valid_hps_ites else None
    
    valid_hps_water = get_suggested_hp(p_hp_tot_input, t_water_set + 5)
    hp_water_sel = valid_hps_water[0] if valid_hps_water else None

    hp_ites_price = hp_ites_sel['price'] if hp_ites_sel else 0
    total_system_ites_cost = total_cost + hp_ites_price

    hp_water_price = hp_water_sel['price'] if hp_water_sel else 0
    total_system_water_cost = tank_total_price + hp_water_price

    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.info(f"### 🔋 Soluzione i-TES (PCM)\n"
                f"**Temp. Stoccaggio:** {t_pcm}°C\n\n"
                f"**Ingombro Moduli:**\n{ites_dims_str}\n"
                f"**Volume Occupato:** {ites_physical_vol_L:.0f} Litri\n\n"
                f"**Dettaglio Costo Batterie:**\n{batt_detail_str}\n"
                f"**Costo Batterie Totale:** € {total_cost:,.0f}\n\n"
                f"---\n"
                f"**Pompa di Calore Suggerita:**\n"
                f"{hp_ites_sel['brand'] if hp_ites_sel else 'N/A'} {hp_ites_sel['model'] if hp_ites_sel else ''}\n"
                f"- Potenza: {hp_ites_sel['kw'] if hp_ites_sel else '-'} kW\n"
                f"- Gas: {hp_ites_sel['gas'] if hp_ites_sel else '-'}\n"
                f"- Max T: {hp_ites_sel.get('max_t','-') if hp_ites_sel else '-'}°C\n"
                f"- Costo Est.: € {hp_ites_price:,.0f}\n"
                f"---\n"
                f"### 💰 TOTALE SISTEMA: € {total_system_ites_cost:,.0f}")
        st.caption("⚠️ Contattare fornitore per conferma")
                
    with col_comp2:
        tech_icon = "🛢️" if not is_sanicube else "🧴"
        tech_name = "Volano Termico Equivalente" if not is_sanicube else "Daikin Sanicube Equivalente"
        
        st.success(f"### {tech_icon} {tech_name}\n"
                   f"**Temp. Stoccaggio:** {t_water_set}°C (Anti-Legionella)\n\n"
                   f"**Modello Suggerito:** {suggested_tank['brand']} {suggested_tank['model']}\n\n"
                   f"**Quantità:** {tank_qty}x ({tank_total_vol} Litri totali)\n\n"
                   f"**Dimensioni Unit:** Ø {suggested_tank['d']} mm x H {suggested_tank['h']} mm\n\n"
                   f"**Volume Occupato:** {tank_physical_vol_L:.0f} Litri\n\n"
                   f"**Costo Volano:** € {tank_total_price:,.0f}\n\n"
                   f"---\n"
                   f"**Pompa di Calore Suggerita (T_max > {t_water_set}°C):**\n"
                   f"{hp_water_sel['brand'] if hp_water_sel else 'N/A'} {hp_water_sel['model'] if hp_water_sel else ''}\n"
                   f"- Potenza: {hp_water_sel['kw'] if hp_water_sel else '-'} kW\n"
                   f"- Gas: {hp_water_sel['gas'] if hp_water_sel else '-'}\n"
                   f"- Max T: {hp_water_sel.get('max_t','-') if hp_water_sel else '-'}°C\n"
                   f"- Costo Est.: € {hp_water_price:,.0f}\n"
                   f"---\n"
                   f"### 💰 TOTALE SISTEMA: € {total_system_water_cost:,.0f}")
        st.caption("⚠️ Contattare fornitore per conferma")
        
        with st.expander("ℹ️ Nota Tecnica: Perché una PdC ad Alta Temperatura?"):
            st.markdown(f"""
            Per mantenere il volano a **{t_water_set}°C**, è necessario un generatore capace di raggiungere questa temperatura con un margine operativo adeguato (es. +5°C), per evitare di far lavorare il compressore sempre al limite.
            """)
        
        if is_sanicube:
             with st.expander("ℹ️ Info Daikin Sanicube"):
                 st.markdown("""
                 Il sistema **Daikin Sanicube** utilizza un accumulo di acqua tecnica (non potabile) con produzione istantanea di ACS.
                 * **Igiene:** Grazie al principio semi-istantaneo, non vi è ristagno di acqua potabile, riducendo drasticamente il rischio Legionella.
                 * **Materiali:** Costruito in materiale plastico, è esente da corrosione.
                 * **Drain-Back:** Può essere integrato con sistemi solari a svuotamento.
                 """)

    delta_capex_hp = 0
    if hp_ites_sel and hp_water_sel:
        delta_capex_hp = hp_water_sel['price'] - hp_ites_sel['price']

    # --- TABELLA CONFRONTO INGOMBRI ---
    st.markdown("#### 📏 Confronto Spazi Occupati")
    col_space1, col_space2 = st.columns(2)
    saving_vol_L = tank_physical_vol_L - ites_physical_vol_L
    saving_pct = (saving_vol_L / tank_physical_vol_L * 100) if tank_physical_vol_L > 0 else 0
    
    col_space1.metric("Volume Fisico i-TES", f"{ites_physical_vol_L:.0f} L")
    col_space2.metric(f"Volume Fisico {comp_target}", f"{tank_physical_vol_L:.0f} L", delta=f"-{saving_pct:.0f}% vs i-TES" if saving_pct > 0 else None, delta_color="inverse")

    st.caption(f"*Nota: Il volano termico deve contenere {tank_total_vol:.0f} litri d'acqua a {t_water_set}°C per eguagliare il V40 delle batterie i-TES.*")

st.divider()

# --- GRAFICI INTERATTIVI ---
sys_flows, sys_powers, sys_temps, sys_v40_volumes = get_system_curves(config, t_pcm, dt_target, p_hp_tot_input, total_energy_e0)

# FUNZIONE PER LE CURVE REALI SANICUBE (V40)
def get_sanicube_v40_at_flow(flow_lmin, t_store):
    data_500 = {
        50: {11: 1550, 12: 990, 15: 530, 20: 290, 25: 180, 30: 100, 35: 50},
        55: {11: 1900, 12: 1280, 15: 730, 20: 450, 25: 320, 30: 220, 35: 150},
        60: {11: 2400, 12: 1510, 15: 860, 20: 580, 25: 440, 30: 340, 35: 270},
        65: {11: 2200, 12: 1550, 15: 970, 20: 680, 25: 530, 30: 430, 35: 360},
        70: {11: 2300, 12: 1640, 15: 1060, 20: 770, 25: 610, 30: 500, 35: 430},
        75: {11: 2400, 12: 1700, 15: 1140, 20: 850, 25: 680, 30: 570, 35: 490}
    }
    avail_temps = sorted(data_500.keys())
    closest_t = min(avail_temps, key=lambda x: abs(x - t_store))
    curve_data = data_500[closest_t]
    flows = np.array(sorted(curve_data.keys()))
    v40s = np.array([curve_data[f] for f in flows])
    if flow_lmin < flows[0]: return v40s[0]
    if flow_lmin > flows[-1]: return v40s[-1]
    return np.interp(flow_lmin, flows, v40s)

# --- LAYOUT GRAFICI ---
# Riga 1: Potenza e Temperatura (i-TES)
col_pt1, col_pt2 = st.columns(2)

if len(sys_flows) > 0:
    with col_pt1:
        st.subheader("⚡ Potenza i-TES")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=sys_flows, y=sys_powers, fill='tozeroy', mode='lines', line=dict(color='#2a9d8f', width=2), name='Max Potenza'))
        limit_p = np.interp(qp_lmin_target, sys_flows, sys_powers)
        col_pt = 'green' if p_req_batt <= limit_p else 'red'
        fig1.add_trace(go.Scatter(x=[qp_lmin_target], y=[p_req_batt], mode='markers', marker=dict(color=col_pt, size=14, line=dict(color='black', width=2)), name='Punto Lavoro'))
        fig1.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Potenza (kW)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_pt2:
        st.subheader("🌡️ Temperatura i-TES")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sys_flows, y=sys_temps, mode='lines', line=dict(color='#e76f51', width=2), name='Temp. Uscita'))
        t_pt = np.interp(qp_lmin_target, sys_flows, sys_temps)
        fig2.add_trace(go.Scatter(x=[qp_lmin_target], y=[t_pt], mode='markers+text', marker=dict(color='orange', size=14, line=dict(color='black', width=2)), text=[f"{t_pt:.1f}°C"], textposition="top center", name='Punto Lavoro'))
        fig2.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Temp (°C)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig2, use_container_width=True)

# Riga 2: Volume V40 e Efficienza Volumetrica
col_g1, col_g2 = st.columns(2)

if len(sys_flows) > 0:
    with col_g1:
        st.subheader("💧 Volume V40 Totale")
        fig3 = go.Figure()
        display_max_y = 10000 
        plot_v40_vol = [min(v, display_max_y) for v in sys_v40_volumes]
        fig3.add_trace(go.Scatter(x=sys_flows, y=plot_v40_vol, mode='lines', line=dict(color='#457b9d', width=2), name='i-TES V40', hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'))
        
        sanicube_v40_curve = [] # Defined here to be used in next chart
        if is_sanicube:
            for f in sys_flows:
                 single_v40 = get_sanicube_v40_at_flow(f / tank_qty, t_water_set) 
                 sanicube_v40_curve.append(single_v40 * tank_qty)
            fig3.add_trace(go.Scatter(x=sys_flows, y=sanicube_v40_curve, mode='lines', line=dict(color='#e67e22', width=2, dash='dash'), name='Sanicube V40 (Reale)', hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'))

        pt_v40_vis = min(total_v40_liters, display_max_y)
        fig3.add_trace(go.Scatter(x=[qp_lmin_target], y=[pt_v40_vis], mode='markers', marker=dict(color='purple', size=14, line=dict(color='black', width=2)), name='Punto Lavoro'))
        fig3.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Volume V40 (Litri)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # --- NUOVO GRAFICO EFFICIENZA VOLUMETRICA ---
    with col_g2:
        st.subheader("📦 Efficienza Volumetrica (V40/Litro)")
        fig_ratio = go.Figure()
        
        # i-TES Ratio
        if ites_physical_vol_L > 0:
            ratio_ites = [v / ites_physical_vol_L for v in plot_v40_vol]
            fig_ratio.add_trace(go.Scatter(x=sys_flows, y=ratio_ites, mode='lines', line=dict(color='#457b9d', width=2), name='i-TES'))
            
            # Point
            curr_v40_ites = np.interp(qp_lmin_target, sys_flows, plot_v40_vol)
            curr_ratio_ites = curr_v40_ites / ites_physical_vol_L
            fig_ratio.add_trace(go.Scatter(x=[qp_lmin_target], y=[curr_ratio_ites], mode='markers', marker=dict(color='#457b9d', size=10), name='Punto Lavoro i-TES'))
        
        # Tank Ratio
        if tank_physical_vol_L > 0:
            if is_sanicube:
                ratio_tank = [v / tank_physical_vol_L for v in sanicube_v40_curve]
                fig_ratio.add_trace(go.Scatter(x=sys_flows, y=ratio_tank, mode='lines', line=dict(color='#e67e22', width=2, dash='dash'), name='Sanicube'))
                
                # Point
                curr_v40_sani = np.interp(qp_lmin_target, sys_flows, sanicube_v40_curve)
                curr_ratio_sani = curr_v40_sani / tank_physical_vol_L
                fig_ratio.add_trace(go.Scatter(x=[qp_lmin_target], y=[curr_ratio_sani], mode='markers', marker=dict(color='#e67e22', size=10), name='Punto Lavoro Sanicube'))

            else:
                # Generic Tank (Approximation)
                ratio_val = total_v40_liters / tank_physical_vol_L if tank_physical_vol_L > 0 else 0
                fig_ratio.add_trace(go.Scatter(x=[sys_flows[0], sys_flows[-1]], y=[ratio_val, ratio_val], mode='lines', line=dict(color='green', width=2, dash='dot'), name='Volano Gen.'))
                fig_ratio.add_trace(go.Scatter(x=[qp_lmin_target], y=[ratio_val], mode='markers', marker=dict(color='green', size=10), name='Punto Lavoro Volano'))
        
        fig_ratio.update_layout(
            xaxis_title="Portata (L/min)", 
            yaxis_title="Ratio V40 / Vol. Fisico", 
            title="Litri V40 erogati per ogni Litro di spazio occupato",
            margin=dict(l=20,r=20,t=40,b=20), 
            height=350
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

else:
    st.warning("Aggiungi batterie per vedere i grafici.")

st.divider()

# --- GRAFICO SIMULAZIONE 24H (CON SOC E SMART REINTEGRO PV) ---
with st.expander("📈 1. Analisi Dettagliata (i-TES)", expanded=True):
    hours_day, hourly_flow_lmin, total_daily_L = get_daily_profile_curve(sim_people, sim_type)
    
    curr_month_idx = datetime.now().month - 1
    daily_kwh_kwp = get_pvgis_data(st.session_state.lat, st.session_state.lon, curr_month_idx)
    
    sim_minutes = 1440 
    x_time = np.arange(sim_minutes)
    
    x_hours = np.linspace(0, 1440, 25) 
    flow_list_sim = list(hourly_flow_lmin)
    y_f_sim = flow_list_sim + [flow_list_sim[0]]
    x_h_sim = np.linspace(0, 1440, len(y_f_sim))
    f_interp = interpolate.interp1d(x_h_sim, y_f_sim, kind='linear')
    consumption_curve_min = f_interp(x_time)
    
    tank_capacity_L = total_v40_liters if total_v40_liters < 99999 else 9999 
    hp_recharge_flow_lmin = (p_hp_tot_input * 60) / (4.186 * dt_target) if dt_target > 0 else 0
    
    solar_profile_norm = create_solar_curve_site_specific(st.session_state.lat, st.session_state.lon, daily_kwh_kwp)
    
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
            is_sunny = solar_intensity > 0.05 
            
            if is_pv_mode:
                st_thr = tank_capacity_L * (st.session_state.pv_start_sun/100.0) if is_sunny else tank_capacity_L * (st.session_state.pv_start_night/100.0)
                sp_thr = tank_capacity_L * (st.session_state.pv_stop/100.0)
            else:
                st_thr = tank_capacity_L * (st.session_state.std_start/100.0)
                sp_thr = tank_capacity_L * (st.session_state.std_stop/100.0)
            
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
                kwh_curr = (p_hp_tot_input / 60.0)
                tot_kwh += kwh_curr
                if is_pv_mode and is_sunny:
                    s_part = kwh_curr * (pv_coverage / 100.0) * solar_intensity 
                    if s_part > kwh_curr: s_part = kwh_curr
                    sol_kwh += s_part
                    grd_kwh += (kwh_curr - s_part)
                else:
                    grd_kwh += kwh_curr
            
            curr_v = curr_v - cons_L + prod_L
            if curr_v > tank_capacity_L: curr_v = tank_capacity_L
            if curr_v < 0: curr_v = 0
            
            soc_hist.append(curr_v)
            hp_status_hist.append(prod_L)
            
        return soc_hist, hp_status_hist, curr_v, is_hp_on, tot_kwh, sol_kwh, grd_kwh, cycle_count

    start_vol_guess = tank_capacity_L * 0.5
    start_state_guess = False
    
    for _ in range(10):
        soc_history, hp_status_history, end_vol, end_state, total_kwh_used, solar_kwh_used, grid_kwh_used, cycle_total = run_simulation_step(start_vol_guess, start_state_guess)
        if abs(end_vol - start_vol_guess) < (tank_capacity_L * 0.01):
            break
        start_vol_guess = end_vol
        start_state_guess = end_state
    
    cumulative_consumption = np.cumsum(consumption_curve_min)
    cumulative_production = np.cumsum(hp_status_history)
    initial_soc_val = soc_history[0]
    total_availability = np.array([initial_soc_val + cp for cp in cumulative_production])
    y_fill_green = total_availability 
    y_fill_red = cumulative_consumption 
    avg_user_flow_lmin = cumulative_consumption[-1] / 1440.0
    
    fig_smart = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    if is_pv_mode:
        fig_smart.add_trace(go.Scatter(
            x=x_time/60, y=solar_profile_norm * tank_capacity_L,
            mode='lines', fill='tozeroy', name='Produzione PV (Profilo)',
            line=dict(color='yellow', width=0),
            fillcolor='rgba(255, 215, 0, 0.4)',
            hoverinfo='skip'
        ), row=1, col=1, secondary_y=True)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=consumption_curve_min,
        mode='lines', fill='tozeroy', name='Prelievo Istantaneo',
        line=dict(color='#ff0000', width=1),
        fillcolor='rgba(255, 0, 0, 0.2)'
    ), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time/60, y=hp_status_history,
        mode='lines', name='Reintegro PdC',
        line=dict(color='#2a9d8f', width=2),
        fill='tozeroy', fillcolor='rgba(42, 157, 143, 0.2)'
    ), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=[0, 24], y=[qp_lmin_target, qp_lmin_target],
        mode='lines', name='Target di Picco (EN 806)',
        line=dict(color='black', width=2, dash='dashdot')
    ), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=[0, 24], y=[avg_user_flow_lmin, avg_user_flow_lmin],
        mode='lines', name='Portata Media Utenza',
        line=dict(color='darkred', width=2, dash='dot')
    ), row=1, col=1, secondary_y=False)
    
    max_sys_flow = 0
    for qty, size in config:
        if qty > 0:
            last_pt = CURVES_DB[size][t_pcm][-1] 
            max_sys_flow += (last_pt[0] * qty)
            
    fig_smart.add_trace(go.Scatter(x=[0, 24], y=[max_sys_flow, max_sys_flow], mode='lines', name='Capacità Max Batterie', line=dict(color='#2a9d8f', width=2, dash='dash')), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(x=x_time/60, y=soc_history, mode='lines', name='Carica Batteria (SoC)', line=dict(color='#007acc', width=3)), row=2, col=1)
    fig_smart.add_trace(go.Scatter(x=x_time/60, y=cumulative_consumption, mode='lines', name='Cumulato Prelievo', line=dict(color='#d62728', width=2)), row=2, col=1)
    fig_smart.add_trace(go.Scatter(x=x_time/60, y=y_fill_green, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(44, 160, 44, 0.2)', showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig_smart.add_trace(go.Scatter(x=x_time/60, y=cumulative_production, mode='lines', name='Cumulato Produzione', line=dict(color='#2ca02c', width=2)), row=2, col=1)
    fig_smart.add_trace(go.Scatter(x=x_time/60, y=y_fill_red, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 39, 40, 0.2)', showlegend=False), row=2, col=1) 

    fig_smart.update_layout(
        title=f"Strategia: {recharge_strategy} | PVGIS: {daily_kwh_kwp:.1f} kWh/kWp (Media Mensile)",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_smart.update_xaxes(title_text="Ora del Giorno (0-24h)", row=2, col=1)
    fig_smart.update_yaxes(title_text="Portata (L/min)", row=1, col=1, secondary_y=False)
    fig_smart.update_yaxes(title_text="Volume (L)", row=1, col=1, secondary_y=True, range=[0, tank_capacity_L*1.1])
    fig_smart.update_yaxes(title_text="Volume Cumulato (L)", row=2, col=1)
    
    st.plotly_chart(fig_smart, use_container_width=True)
    st.caption(f"Nota: La portata media dell'utenza calcolata è di **{avg_user_flow_lmin:.1f} L/min** (Linea rossa tratteggiata), mentre il picco normativo è **{qp_lmin_target:.1f} L/min**.")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Energia Totale PdC", f"{total_kwh_used:.1f} kWh/giorno")
    if is_pv_mode:
        kpi2.metric("Da Fotovoltaico", f"{solar_kwh_used:.1f} kWh", delta=f"{(solar_kwh_used/total_kwh_used)*100:.0f}%" if total_kwh_used>0 else "0%")
        kpi3.metric("Da Rete", f"{grid_kwh_used:.1f} kWh", delta=f"-{(grid_kwh_used/total_kwh_used)*100:.0f}%" if total_kwh_used>0 else "0%", delta_color="inverse")
    else:
        kpi2.metric("Da Rete", f"{total_kwh_used:.1f} kWh")
        kpi3.metric("Autoconsumo", "N/A")
    kpi4.metric("Cicli ON/OFF", f"{cycle_total}", help="Numero di accensioni giornaliere della Pompa di Calore")

# --- CONFRONTO TECNOLOGICO E TABELLE ---
with st.expander(f"🆚 2. Confronto Tecnologico (i-TES vs {comp_target})", expanded=True):
    # Use global function run_simulation_step_water
    start_v_w = tank_total_vol * 0.8
    start_s_w = False
    
    cost_extra_temp = 0 # Initialize here to be safe

    # Convergence loop
    for _ in range(5):
        soc_water, temp_water, end_v_w, end_s_w, kwh_water_therm, cycles_water = run_simulation_step_water(
            start_v_w, start_s_w, tank_total_vol, consumption_curve_min, 
            hp_recharge_flow_lmin, (p_hp_tot_input / 60.0), t_water_set, t_in, is_sanicube
        )
        start_v_w = end_v_w
        start_s_w = end_s_w

    kwh_pcm = total_kwh_used
    cycles_pcm = cycle_total
    
    s_temp_water = pd.Series(temp_water)
    temp_water_ma = s_temp_water.rolling(window=60, min_periods=1).mean()
    
    if len(sys_flows) > 0:
        func_temp_pcm = interpolate.interp1d(sys_flows, sys_temps, kind='linear', fill_value="extrapolate")
    else:
        func_temp_pcm = lambda x: t_in 

    soc_pcm, temp_pcm = run_pcm_plot_data(soc_history, func_temp_pcm, t_pcm, t_in, consumption_curve_min)

    fig_comp = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5],
        subplot_titles=("Stato di Carica (Litri Equivalenti)", "Temperatura Erogazione (°C)")
    )
    
    fig_comp.add_trace(go.Scatter(x=x_time/60, y=soc_pcm, mode='lines', name='Accumulo i-TES (PCM)', line=dict(color='#007acc', width=3)), row=1, col=1)
    fig_comp.add_trace(go.Scatter(x=x_time/60, y=soc_water, mode='lines', name=f'Accumulo {comp_target}', line=dict(color='#2ca02c', width=2, dash='dot')), row=1, col=1) 
    fig_comp.add_trace(go.Scatter(x=x_time/60, y=temp_pcm, mode='lines', name='Temp. Acqua i-TES (PCM)', line=dict(color='#007acc', width=3)), row=2, col=1)
    fig_comp.add_trace(go.Scatter(x=x_time/60, y=temp_water, mode='lines', name=f'Temp. Istantanea {comp_target}', line=dict(color='#ff7f0e', width=1, dash='dot')), row=2, col=1) 
    fig_comp.add_trace(go.Scatter(x=x_time/60, y=temp_water_ma, mode='lines', name=f'Temp. Media Mobile (60m) {comp_target}', line=dict(color='#d62728', width=2.5)), row=2, col=1)
    
    fig_comp.update_layout(title="Confronto Dinamico Completo", height=700)
    fig_comp.update_yaxes(title_text="Volume (L)", row=1, col=1)
    fig_comp.update_yaxes(title_text="Temperatura (°C)", row=2, col=1)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Calculations for Table
    cop_pcm = estimate_cop(t_pcm)
    cop_water = estimate_cop(t_water_set)
    elec_price = 0.31
    cost_per_cycle = 0.15
    startup_penalty_kwh = 0.05
    
    kwh_elec_pcm_year = (kwh_pcm * 365) / cop_pcm
    kwh_elec_water_year = (kwh_water_therm * 365) / cop_water
    
    # Cost components
    pcm_cost_energy = kwh_elec_pcm_year * elec_price
    water_cost_energy = kwh_elec_water_year * elec_price
    
    # Delta calculations
    delta_cycles_year = (cycles_water - cycles_pcm) * 365
    saving_energy_year = water_cost_energy - pcm_cost_energy
    saving_maint_year = delta_cycles_year * cost_per_cycle
    saving_startup_elec_year = delta_cycles_year * startup_penalty_kwh * elec_price
    
    total_saving_year = saving_energy_year + saving_maint_year + saving_startup_elec_year
    
    if t_water_set > t_pcm:
        energy_pcm_at_water_cop = (kwh_pcm * 365) / cop_water
        energy_pcm_at_pcm_cop = (kwh_pcm * 365) / cop_pcm
        cost_extra_temp = (energy_pcm_at_water_cop - energy_pcm_at_pcm_cop) * elec_price

    # TABLE DATA
    pcm_total_annual_cost = pcm_cost_energy + (cycles_pcm * 365 * startup_penalty_kwh * elec_price)
    water_total_annual_cost = water_cost_energy + (cycles_water * 365 * startup_penalty_kwh * elec_price)
    
    st.markdown("### 📊 Dettaglio Consumi Assoluti (Stima Annuale)")
    consumption_data = {
        "Metrica": ["Energia Termica (kWh_t)", "Energia Elettrica (kWh_e)", "Costo Elettrico (€)", "Costo Start-up (€)", "Costo Manutenzione (€)"],
        "i-TES (PCM)": [
            f"{kwh_pcm*365:,.0f}", 
            f"{kwh_elec_pcm_year:,.0f}", 
            f"€ {pcm_cost_energy:,.0f}",
            f"€ {cycles_pcm * 365 * startup_penalty_kwh * elec_price:,.0f}",
            "Included"
        ],
        f"{comp_target}": [
            f"{kwh_water_therm*365:,.0f}", 
            f"{kwh_elec_water_year:,.0f}", 
            f"€ {water_cost_energy:,.0f}",
            f"€ {cycles_water * 365 * startup_penalty_kwh * elec_price:,.0f}",
            f"€ {delta_cycles_year * cost_per_cycle:,.0f} (Extra)"
        ],
        "Differenza (Risparmio)": [
            f"{(kwh_water_therm-kwh_pcm)*365:,.0f}", 
            f"{(kwh_elec_water_year-kwh_elec_pcm_year):,.0f}", 
            f"€ {saving_energy_year:,.0f}",
            f"€ {saving_startup_elec_year:,.0f}",
            f"€ {saving_maint_year:,.0f}"
        ]
    }
    st.table(pd.DataFrame(consumption_data))
    
    # Saving Metrics Display
    st.markdown("### 💰 Riepilogo Risparmio Economico")
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    col_res1.metric("Risp. Elettrico (COP & Disp.)", f"€ {saving_energy_year:,.0f}", help="Risparmio dovuto alla miglior efficienza (COP) e minori dispersioni")
    col_res2.metric("Risp. Elettrico (Start-up)", f"€ {saving_startup_elec_year:,.0f}", help="Energia risparmiata evitando i continui avvii del compressore")
    col_res3.metric("Risp. Manutenzione (Usura)", f"€ {saving_maint_year:,.0f}", help="Minor usura meccanica dovuta alla riduzione dei cicli")
    col_res4.metric("TOTALE RISPARMIO ANNUO", f"€ {total_saving_year:,.0f}", delta="Totale", delta_color="normal")
    
    with st.expander("ℹ️ Dettaglio Risparmio Elettrico (COP & Disp.)"):
        st.markdown(f"""
        Questo risparmio combina due fattori fondamentali:
        1. **Miglior Efficienza (COP):** La PdC per i-TES lavora a {t_pcm}°C (COP {cop_pcm:.2f}), mentre per il {comp_target} lavora a {t_water_set}°C (COP {cop_water:.2f}).
        2. **Minori Dispersioni:** Mantenere un volume d'acqua sempre ad alta temperatura comporta maggiori perdite termiche rispetto al PCM.

        **Calcolo:**
        * **Consumo Elettrico {comp_target}:** {kwh_elec_water_year:,.0f} kWh/anno
        * **Consumo Elettrico i-TES:** {kwh_elec_pcm_year:,.0f} kWh/anno
        * **Delta Energia:** {kwh_elec_water_year - kwh_elec_pcm_year:,.0f} kWh
        * **Risparmio:** {kwh_elec_water_year - kwh_elec_pcm_year:,.0f} kWh × {elec_price} €/kWh = **€ {saving_energy_year:,.0f}**
        """)

    if cost_extra_temp > 0:
        with st.expander("ℹ️ Dettaglio Calcolo Penalità COP (Legionella)"):
            st.markdown(f"""
            Per prevenire la legionella, il volano ad acqua deve essere mantenuto a **{t_water_set:.1f}°C**, mentre la batteria PCM lavora a **{t_pcm}°C**.
            L'aumento della temperatura di mandata riduce l'efficienza (COP) della Pompa di Calore:
            
            1.  **COP a {t_pcm}°C (PCM):** {cop_pcm:.2f}
            2.  **COP a {t_water_set:.1f}°C (Volano):** {cop_water:.2f} (Penalità: {((cop_pcm-cop_water)/cop_pcm)*100:.1f}%)
            """)
            
    with st.expander("ℹ️ Dettaglio Risparmio Manutenzione (Usura)"):
        st.markdown(f"""
        Il risparmio è calcolato sulla riduzione dello stress meccanico del compressore.
        * **Cicli Annui i-TES:** {cycles_pcm * 365:,.0f}
        * **Cicli Annui {comp_target}:** {cycles_water * 365:,.0f}
        * **Differenza:** {delta_cycles_year:,.0f} cicli in meno.
        """)

    with st.expander("ℹ️ Dettaglio Risparmio Elettrico (Transitori Start-up)"):
        st.markdown(f"""
        Ogni volta che la Pompa di Calore si avvia, necessita di una fase di "ramp-up" a bassa efficienza.
        * **Penalità Energetica per Avvio:** {startup_penalty_kwh} kWh
        * **Calcolo:** {delta_cycles_year:,.0f} cicli evitati × {startup_penalty_kwh} kWh/ciclo × {elec_price} €/kWh = **€ {saving_startup_elec_year:,.0f}**
        """)

    st.caption(f"**Fonte Prezzo Energia:** Stima basata su dati ARERA (~0.31 €/kWh).")

    # --- ANALISI BREAK-EVEN (ROI) ---
    st.markdown("### 💸 Analisi di Break-even (ROI)")
    
    delta_investment = total_system_ites_cost - total_system_water_cost
    
    if delta_investment <= 0:
        st.success(f"✅ **L'investimento i-TES è più economico o uguale.**\n\nRisparmio immediato all'acquisto: **€ {-delta_investment:,.0f}**\nOltre a un risparmio operativo annuo di **€ {total_saving_year:,.0f}**.")
    elif total_saving_year <= 0:
        st.error(f"❌ **Il sistema i-TES costa di più (€ {delta_investment:,.0f}) e non genera risparmio operativo in queste condizioni.**\nVerificare i parametri (es. costo energia, temperature).")
    else:
        years = 15
        x_years = list(range(years + 1))
        
        # Cumulative Cash Flow (Net Benefit Approach)
        # Year 0: -Delta Investment
        # Year N: Previous + Annual Saving
        cash_flow = []
        curr = -delta_investment
        for y in x_years:
            cash_flow.append(curr)
            curr += total_saving_year
        
        # Create Plot
        fig_roi = go.Figure()
        
        # Line 1: Net Cumulative Savings
        fig_roi.add_trace(go.Scatter(
            x=x_years, 
            y=cash_flow, 
            mode='lines+markers', 
            name='Flusso di Cassa Cumulato',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)' # Light green area below
        ))
        
        # Zero Line (Break-even)
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Point")
        
        fig_roi.update_layout(
            title="Rientro dell'Investimento (Differenziale)",
            xaxis_title="Anni",
            yaxis_title="Bilancio Economico (€)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
        
        roi_years = delta_investment / total_saving_year
        st.metric("Tempo di Ritorno (Payback)", f"{roi_years:.1f} Anni", help="Tempo necessario affinché i risparmi operativi coprano il maggior costo iniziale.")
        
        if roi_years < 5:
            st.success(f"🚀 **Ottimo investimento!** Il sistema si ripaga in soli {roi_years:.1f} anni.")
        elif roi_years < 10:
            st.warning(f"⚠️ **Investimento a medio termine.** Rientro in {roi_years:.1f} anni.")
        else:
            st.info(f"ℹ️ **Investimento a lungo termine.** Rientro in {roi_years:.1f} anni.")

# --- GRAFICO EN 806 ---
with st.expander("📉 Vedi Grafico Normativo EN 806-3", expanded=False):
    x_vals = np.logspace(0, 3, 100)
    y_vals = [calcola_qd_en806(x, max_lu_unit) for x in x_vals]
    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Curva Normativa'))
    fig_en.add_trace(go.Scatter(x=[lu_totali], y=[qd_ls_target], mode='markers', name='Punto Progetto'))
    fig_en.update_layout(xaxis_type="log", yaxis_type="log")
    st.plotly_chart(fig_en, use_container_width=True)
