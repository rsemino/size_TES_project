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

# Placeholder per il titolo dinamico con logo
title_placeholder = st.empty() 

# --- GESTIONE STATO INIZIALE ---
if 'qty_6' not in st.session_state: st.session_state.qty_6 = 0
if 'qty_12' not in st.session_state: st.session_state.qty_12 = 0
if 'qty_20' not in st.session_state: st.session_state.qty_20 = 0
if 'qty_40' not in st.session_state: st.session_state.qty_40 = 0
if 'ind_p_hp' not in st.session_state: st.session_state.ind_p_hp = 0.0
if 'lat' not in st.session_state: st.session_state.lat = 41.9028 
if 'lon' not in st.session_state: st.session_state.lon = 12.4964
if 'address_found' not in st.session_state: st.session_state.address_found = "Roma, Italia (Default)"

if 'pv_start_sun' not in st.session_state: st.session_state.pv_start_sun = 50
if 'pv_start_night' not in st.session_state: st.session_state.pv_start_night = 15
if 'pv_stop' not in st.session_state: st.session_state.pv_stop = 100
if 'std_start' not in st.session_state: st.session_state.std_start = 70
if 'std_stop' not in st.session_state: st.session_state.std_stop = 98
if 'recharge_min' not in st.session_state: st.session_state.recharge_min = 60
if 'flow_correction_pct' not in st.session_state: st.session_state.flow_correction_pct = 0
if 'shower_window_min' not in st.session_state: st.session_state.shower_window_min = 60

# --- APPLY PENDING UPDATES ---
if 'pending_opt' in st.session_state:
    opt = st.session_state.pop('pending_opt')
    st.session_state.qty_6 = opt['q6']
    st.session_state.qty_12 = opt['q12']
    st.session_state.qty_20 = opt['q20']
    st.session_state.qty_40 = opt['q40']
    st.session_state.recharge_min = opt['rt']
    st.session_state.std_start = opt['start']
    st.session_state.std_stop = opt['stop']
    st.session_state.flow_correction_pct = opt['fc_pct']
    st.toast(f"🏆 Ottimizzazione Applicata! (Correzione Portata: +{opt['fc_pct']}%)", icon="✅")

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

def get_system_curves(config, t_pcm, dt_calc, p_hp_tot, e_tot_e0, flow_correction_pct=0):
    interpolators_p = {}
    interpolators_t = {}
    limits = {}
    
    correction_factor = 1 + (flow_correction_pct / 100.0)
    
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
    
    alpha_values = np.linspace(0.01, 1.5, 50) 
    sys_flows, sys_powers, sys_temps, sys_v40_volumes = [], [], [], []
    
    for alpha in alpha_values:
        f_tot, p_tot_batt, weighted_temp_sum = 0, 0, 0
        for qty, size in config:
            if qty > 0:
                f_single_nominal = limits[size] * alpha
                f_single_corrected = f_single_nominal * correction_factor
                
                p_single = float(interpolators_p[size](f_single_nominal))
                t_single = float(interpolators_t[size](f_single_nominal))
                
                f_block = f_single_corrected * qty
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
    elif building_type == "Industria (2 Turni)": liters_per_person = 40
    elif building_type == "Industria (3 Turni)": liters_per_person = 40
    
    total_daily_vol = n_people * liters_per_person
    
    profiles = {
        "Residenziale": [0.5, 0.2, 0.1, 0.1, 0.5, 2.0, 8.0, 12.0, 9.0, 6.0, 5.0, 4.0, 5.0, 4.0, 3.0, 3.0, 4.0, 6.0, 10.0, 11.0, 5.0, 2.0, 1.0, 0.6],
        "Ufficio":      [0, 0, 0, 0, 0, 0, 2, 8, 15, 12, 10, 15, 12, 10, 8, 5, 3, 0, 0, 0, 0, 0, 0, 0],
        "Hotel":        [1, 0.5, 0.5, 0.5, 1, 3, 10, 15, 12, 8, 5, 4, 3, 3, 4, 6, 8, 12, 8, 5, 3, 2, 1],
        "Industria (2 Turni)": [1, 1, 1, 1, 1, 15, 20, 15, 1, 1, 1, 1, 1, 15, 20, 15, 1, 1, 1, 1, 1, 1, 1, 1],
        "Industria (3 Turni)": [1, 1, 1, 1, 1, 10, 15, 10, 1, 1, 1, 1, 10, 15, 10, 1, 1, 1, 10, 15, 10, 1, 1, 1]
    }
    selected_profile = profiles.get(building_type, profiles["Residenziale"])
    factor = 100.0 / sum(selected_profile)
    hourly_flow = [(p * factor / 100.0) * total_daily_vol for p in selected_profile]
    hourly_flow_lmin = [val / 60.0 for val in hourly_flow]
    return list(range(24)), hourly_flow_lmin, total_daily_vol

def calculate_water_tank_config(target_v40, t_store, t_net, db_to_use, is_sanicube=False):
    if t_store <= 40: return None, 0, 0, 0
    if is_sanicube: t_return = t_net
    else: t_return = max(t_net + 5, 20.0) 
    if t_store <= t_return: return None, 0, 0, 0
    factor = (t_store - t_return) / (40 - t_net)
    if factor <= 0: return None, 0, 0, 0
    needed_real_vol = target_v40 / factor
    sorted_db = sorted(db_to_use, key=lambda k: k['vol'])
    if is_sanicube:
        max_tank = sorted_db[-1]
        qty = math.ceil(needed_real_vol / max_tank['vol'])
        return max_tank, qty, max_tank['vol'] * qty, max_tank['price'] * qty
    else:
        preferred_max_vol = 2500 
        best_qty = 1
        best_tank = sorted_db[-1]
        min_score = float('inf')
        for tank in sorted_db:
            qty = math.ceil(needed_real_vol / tank['vol'])
            excess = (qty * tank['vol']) - needed_real_vol
            penalty = 999999 if needed_real_vol > preferred_max_vol and tank['vol'] > preferred_max_vol else 0
            score = excess + penalty
            if score < min_score:
                min_score = score
                best_tank = tank
                best_qty = qty
            elif score == min_score:
                if abs(tank['vol'] - 2000) < abs(best_tank['vol'] - 2000):
                    best_tank = tank
                    best_qty = qty
        total_vol = best_tank['vol'] * best_qty
        total_price = best_tank['price'] * best_qty
        return best_tank, best_qty, total_vol, total_price

def get_mac_config(target_flow_lmin):
    if target_flow_lmin <= 20: return {"desc": "1x Preparatore Rapido ACS 50 kW", "price": 1500, "qty": 1}
    elif target_flow_lmin <= 40: return {"desc": "1x Preparatore Rapido ACS 100 kW", "price": 2200, "qty": 1}
    else:
        qty = math.ceil(target_flow_lmin / 40.0)
        return {"desc": f"{qty}x Preparatore Rapido ACS 100 kW in Cascata", "price": (2200 * qty) + 800, "qty": qty}

def calc_dispersion_kwh(volume_l, current_temp, ambient_temp=20.0, use_sanicube_data=False):
    if volume_l <= 0: return 0.0
    delta_t = max(0, current_temp - ambient_temp)
    if use_sanicube_data:
        return (1.4 / 500.0) * (delta_t / 40.0) * volume_l / 1440.0 
    else:
        vol_m3 = volume_l / 1000.0
        radius = (vol_m3 / (4 * math.pi))**(1/3)
        area_m2 = 10 * math.pi * (radius**2)
        return ((0.35 * area_m2 * delta_t) / 1000.0) / 60.0

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

def run_simulation_step_water(start_vol, start_hp_state, water_tank_vol_total, consumption_curve_min, hp_recharge_flow_tank, kwh_per_m, t_water_set, t_in, factor_w, is_sanicube=False):
    curr_v = start_vol
    is_hp_on = start_hp_state
    soc_hist = []
    temp_erogazione_hist = []
    temp_accumulo_hist = []
    tot_kwh = 0
    tot_loss_kwh = 0
    cycle_count = 0
    sim_minutes = len(consumption_curve_min)
    t_return = max(t_in + 5, 20.0) if not is_sanicube else t_in

    for i in range(sim_minutes):
        curr_temp_accumulo = t_return + (curr_v / water_tank_vol_total) * (t_water_set - t_return) if water_tank_vol_total > 0 else t_water_set
        loss_kwh = calc_dispersion_kwh(water_tank_vol_total, curr_temp_accumulo, 20.0, use_sanicube_data=is_sanicube)
        tot_loss_kwh += loss_kwh
        delta_T_sys = t_water_set - t_return
        loss_liters = loss_kwh / (delta_T_sys * 0.00116) if delta_T_sys > 0 else 0
        
        cons_L_tank = (consumption_curve_min[i] / factor_w) + loss_liters
        st_thr = water_tank_vol_total * 0.85
        sp_thr = water_tank_vol_total * 1.0
        
        if not is_hp_on:
            if curr_v < st_thr: 
                is_hp_on = True
                cycle_count += 1
        else:
            if curr_v >= sp_thr: is_hp_on = False
        
        prod_L = hp_recharge_flow_tank if is_hp_on else 0
        if is_hp_on: tot_kwh += kwh_per_m
        
        curr_v = curr_v - cons_L_tank + prod_L
        if curr_v > water_tank_vol_total: curr_v = water_tank_vol_total
        if curr_v < 0: curr_v = 0
        
        soc_hist.append(curr_v * factor_w) 
        temp_erogazione_hist.append(40.0 if curr_v > 0 and consumption_curve_min[i] > 0 else t_water_set) 
        temp_accumulo_hist.append(curr_temp_accumulo)
        
    return soc_hist, temp_erogazione_hist, temp_accumulo_hist, curr_v, is_hp_on, tot_kwh, cycle_count, tot_loss_kwh

def get_suggested_generator(target_kw, target_temp, gen_type="Pompa di Calore"):
    if target_kw <= 0: return None
    db = HP_DATABASE if gen_type == "Pompa di Calore" else BOILER_DATABASE
    valid_gens = [g for g in db if g.get('max_t', 55) >= target_temp]
    if not valid_gens: return None
    valid_gens.sort(key=lambda x: x['kw'])
    for g in valid_gens:
        if g['kw'] >= target_kw * 0.9: 
            return {"qty": 1, "gen": g}
    largest = valid_gens[-1]
    qty = math.ceil(target_kw / largest['kw'])
    return {"qty": qty, "gen": largest}

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

BOILER_DATABASE = [
    {"brand": "Elco", "model": "Thision L Plus 60", "type": "Caldaia a Condensazione", "kw": 56.9, "gas": "Metano/H2", "max_t": 85, "price": 4500},
    {"brand": "Elco", "model": "Thision L Plus 70", "type": "Caldaia a Condensazione", "kw": 65.4, "gas": "Metano/H2", "max_t": 85, "price": 5000},
    {"brand": "Elco", "model": "Thision L Plus 100", "type": "Caldaia a Condensazione", "kw": 90.2, "gas": "Metano/H2", "max_t": 85, "price": 6500},
    {"brand": "Elco", "model": "Thision L Plus 120", "type": "Caldaia a Condensazione", "kw": 110.8, "gas": "Metano/H2", "max_t": 85, "price": 7500},
    {"brand": "Elco", "model": "Thision L Plus 140", "type": "Caldaia a Condensazione", "kw": 130.5, "gas": "Metano/H2", "max_t": 85, "price": 8500},
    {"brand": "Elco", "model": "Thision L Plus 170", "type": "Caldaia a Condensazione", "kw": 155.5, "gas": "Metano/H2", "max_t": 85, "price": 9500},
    {"brand": "Elco", "model": "Thision L Plus 200", "type": "Caldaia a Condensazione", "kw": 180.3, "gas": "Metano/H2", "max_t": 85, "price": 10500},
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
show_comparison = st.sidebar.toggle("🆚 Attiva Confronto Tecnologico", value=True)

if show_comparison:
    comp_target = st.sidebar.radio("Scegli Tecnologia di Confronto", ["Volano Inerziale + MAC", "Daikin Sanicube (Drain-Back)"])
    is_sanicube = comp_target == "Daikin Sanicube (Drain-Back)"
    if is_sanicube:
        t_water_set = st.sidebar.select_slider("Temp. Accumulo Sanicube (°C)", options=[50, 55, 60, 65, 70, 75], value=60)
    else:
        t_water_set = st.sidebar.slider("Temp. Accumulo Volano (°C)", min_value=50, max_value=85, value=60, step=5)
else:
    is_sanicube = False
    comp_target = "Nessuno"
    t_water_set = 60.0

# --- SEZIONE INPUT UTENTE (Sidebar) ---
st.sidebar.header("Parametri Progetto")
with st.sidebar.expander("1. Profilo Utenza", expanded=True):
    sim_people = st.number_input("Numero Utenti", min_value=1, value=4, step=1)
    sim_type = st.selectbox("Tipo Edificio", ["Residenziale", "Ufficio", "Hotel", "Industria (2 Turni)", "Industria (3 Turni)"])
    is_industrial = sim_type in ["Industria (2 Turni)", "Industria (3 Turni)"]
    
    if is_industrial:
        st.session_state.shower_window_min = st.selectbox("Finestra Lavaggio Fine Turno (min)", [30, 60, 90, 120, 150, 180, 210, 240], index=1)

t_in = st.sidebar.number_input("Temp. Acqua Rete (°C)", value=12.5, step=0.5, format="%.1f")
dt_target = 40.0 - t_in
if dt_target <= 0: dt_target = 27.5

with st.sidebar.expander("🏗️ 2. Dettaglio Utenze (LU)", expanded=False):
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

# AUTO-CORREZIONE DEL PICCO PER GRANDI UTENZE
_, hourly_flow_opt_pre, total_daily_L_calc = get_daily_profile_curve(sim_people, sim_type)

x_time_opt = np.arange(1440)

if is_industrial:
    num_shifts = 2 if "2 Turni" in sim_type else 3
    vol_per_shift = total_daily_L_calc / num_shifts
    sim_peak_flow_lmin = vol_per_shift / st.session_state.shower_window_min
    
    consumption_curve_min = np.zeros(1440)
    shift_times = [840, 1320] if num_shifts == 2 else [360, 840, 1320]
    
    for start_t in shift_times:
        for m in range(st.session_state.shower_window_min):
            idx = (start_t + m) % 1440
            consumption_curve_min[idx] = sim_peak_flow_lmin
else:
    flow_list_opt = list(hourly_flow_opt_pre) + [hourly_flow_opt_pre[0]]
    x_h_opt = np.linspace(0, 1440, len(flow_list_opt))
    f_int_opt = interpolate.interp1d(x_h_opt, flow_list_opt, kind='linear')
    consumption_curve_min = f_int_opt(x_time_opt)
    c_sum_val = np.sum(consumption_curve_min)
    if c_sum_val > 0:
        consumption_curve_min = consumption_curve_min * (total_daily_L_calc / c_sum_val)
    sim_peak_flow_lmin = max(consumption_curve_min)

if sim_peak_flow_lmin > qp_lmin_target:
    qp_lmin_target = sim_peak_flow_lmin
    st.sidebar.info(f"⚠️ **Autocorrezione:** Il calcolo richiede un picco statistico/onda quadra di **{sim_peak_flow_lmin:.1f} L/min**, superiore alle Utenze LU inserite. Dimensionamento basato sul picco reale.")
else:
    st.sidebar.info(f"Target (EN 806): **{qp_lmin_target:.1f} L/min**")

st.sidebar.subheader("🔥 3. Generazione")
gen_type = st.sidebar.radio("Tecnologia Generatore", ["Pompa di Calore", "Caldaia a Condensazione"])

if is_industrial:
    contributo_contestuale = st.sidebar.toggle("⚡ Integrazione Contestuale", value=False, help="Se ON: Il generatore eroga potenza istantanea in aiuto alle batterie durante il picco. Se OFF: le batterie coprono il picco totalmente da sole.")
else:
    contributo_contestuale = True

# --- SIMULAZIONE 24H INPUTS ---
with st.sidebar.expander("📈 4. Strategia 24h & PV", expanded=True):
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
    
    if is_pv_mode:
        pv_coverage = st.number_input("Copertura PV del Generatore (%)", 0, 100, 50, step=10)
        st.markdown("**Soglie Intervento (Automatiche)**")
        st.text(f"Avvio con Sole: < {st.session_state.pv_start_sun}%")
        st.text(f"Avvio Notturno: < {st.session_state.pv_start_night}%")
    else:
        st.markdown("**Soglie Intervento (%)**")
        start_soc_std = st.slider("Avvio Reintegro se carica < (%)", 0, 100, key='std_start')
        stop_soc = st.slider("Stop Reintegro se carica > (%)", 0, 100, key='std_stop')

with st.sidebar.expander("🔋 5. Batterie i-TES", expanded=True):
    
    if is_industrial:
        st.markdown("### ⚙️ Dimensionamento Industriale Avanzato")
        
        t_pcm = st.radio("Temp. PCM (°C)", [48, 58, 74], horizontal=True)
        batt_size = st.selectbox("Taglia Moduli i-TES", [40, 20, 12, 6], index=0)
        
        v0_nom = params_v40[batt_size]['V0']
        if t_pcm >= 58: v0_nom /= 0.76

        # Calcolo Deficit Integrale Esatto (Balancing Core)
        p_gen_ideal = (total_daily_L_calc * dt_target * 0.00116) / 24.0
        f_gen_ideal = (p_gen_ideal * 60) / (4.186 * dt_target) if dt_target > 0 else 0
        
        cv = 0
        min_v = 0
        max_v = 0
        for cons in consumption_curve_min:
            cv += (f_gen_ideal - cons)
            if cv < min_v: min_v = cv
            if cv > max_v: max_v = cv
            
        req_vol_net = max_v - min_v
        qty_ideal = math.ceil(req_vol_net / v0_nom) if v0_nom > 0 else 1
        qty_ideal = max(1, int(qty_ideal * 1.1)) 
        
        max_batt_slider = max(10, qty_ideal * 3)
        
        st.markdown(f"Volume netto di accumulo richiesto per bilanciare i turni: **{req_vol_net:.0f} L**")
        
        qty_calc = st.slider("Numero di Batterie i-TES", min_value=1, max_value=int(max_batt_slider), value=int(qty_ideal), help="L'impostazione di default garantisce il bilancio perfetto sulle 24h coprendo il deficit matematico del turno.")
        
        st.session_state.qty_40 = qty_calc if batt_size == 40 else 0
        st.session_state.qty_20 = qty_calc if batt_size == 20 else 0
        st.session_state.qty_12 = qty_calc if batt_size == 12 else 0
        st.session_state.qty_6  = qty_calc if batt_size == 6 else 0
        
        st.session_state.flow_correction_pct = st.number_input("Coeff. Correzione Portata (%)", step=1, value=st.session_state.flow_correction_pct)
        
        # Calcolo Baseload consequenziale esatto
        p_loss = 28.1 * qty_calc / 1000.0
        p_min_s = p_gen_ideal + p_loss
        p_max_s = max(p_min_s * 10, 10)
        found_p = p_max_s
        
        v_cap = qty_calc * v0_nom
        loss_f = (p_loss * 60) / (4.186 * dt_target)
        
        st_th = v_cap * (st.session_state.std_start/100.0) if not is_pv_mode else v_cap * (st.session_state.pv_start_sun/100.0)
        sp_th = v_cap * (st.session_state.std_stop/100.0) if not is_pv_mode else v_cap * (st.session_state.pv_stop/100.0)
        
        for p_test in np.linspace(p_min_s, p_max_s, 50):
            f_test = (p_test * 60) / (4.186 * dt_target)
            v = v_cap * 0.5
            is_on = False
            failed = False
            for _ in range(2): 
                for cons in consumption_curve_min:
                    if not is_on:
                        if v < st_th: is_on = True
                    else:
                        if v >= sp_th: is_on = False
                    
                    prod = f_test if is_on else 0
                    v += (prod - cons - loss_f)
                    if v > v_cap: v = v_cap
                    if v < 0: 
                        failed = True
                        break
                if failed: break
            if not failed:
                found_p = p_test
                break
                
        st.session_state.ind_p_hp = found_p * 1.02 
        st.info(f"Generatore Baseload Minimo Calcolato per compensare setup e dispersioni: **{st.session_state.ind_p_hp:.1f} kW**")
        
    else:
        enable_autopilot = st.toggle("🤖 Attiva Modalità Autopilota", value=False)
        
        if enable_autopilot:
            if sim_people < 10:
                r_q6, r_q12, r_q20, r_q40 = range(6), range(6), [0], [0]
                hw_rule_str = "i-6 e i-12 (Residenziale < 10 utenti)"
            else:
                max_40 = max(10, min(60, math.ceil(sim_people / 40)))
                r_q6, r_q12, r_q20, r_q40 = [0], [0], range(6), range(max_40 + 1)
                hw_rule_str = f"i-20 e i-40 (fino a {max_40} unità i-40)"

            st.info(f"💡 Ottimizzazione Automatica (Standard).\n\n"
                    f"**Regole impostate:**\n"
                    f"- Edificio **{sim_type}** ({sim_people} utenti): L'algoritmo testerà moduli **{hw_rule_str}**.\n"
                    f"- **Portata:** Coefficiente spinto fino al +50% per cercare il miglior compromesso.\n"
                    f"- **Ottimizzazione:** L'algoritmo valuterà diversi tempi di ricarica standard (1-2h) per trovare il miglior Break-even.")           
            
            t_pcm = st.radio("Temp. PCM (°C)", [48, 58, 74], horizontal=True)
            
            if st.button("🚀 TROVA CONFIGURAZIONE OTTIMALE"):
                with st.status("Analisi in corso...", expanded=True) as status:
                    st.write("🔍 Fase 1: Esplorazione hardware per gestione picchi...")
                    
                    req_flow = qp_lmin_target
                    valid_hardware = []
                    
                    for fc_test in range(0, 51, 10): 
                        for q40, q20, q12, q6 in itertools.product(r_q40, r_q20, r_q12, r_q6):
                            if q40==0 and q20==0 and q12==0 and q6==0: continue
                            nom_flow_tot = (q6 * NOMINAL_FLOWS[6]) + (q12 * NOMINAL_FLOWS[12]) + (q20 * NOMINAL_FLOWS[20]) + (q40 * NOMINAL_FLOWS[40])
                            effective_flow = nom_flow_tot * (1 + fc_test / 100.0)
                            
                            if effective_flow >= req_flow:
                                hw_cost = (q6*prices[6]) + (q12*prices[12]) + (q20*prices[20]) + (q40*prices[40])
                                valid_hardware.append({'cfg': (q6, q12, q20, q40), 'cost': hw_cost, 'fc_test': fc_test})
                    
                    valid_hardware.sort(key=lambda x: (x['cost'], x['fc_test']))
                    candidates = valid_hardware[:20] 
                    
                    st.write(f"✅ Trovate {len(valid_hardware)} opzioni. Simulo integrazione generatore e cicli termici...")
                    
                    best_solution = None
                    E_daily_kwh_opt = total_daily_L_calc * dt_target * 0.00116
                    P_avg_opt = E_daily_kwh_opt / 24.0
                    P_peak_opt = qp_lmin_target * dt_target * 0.0697
                    
                    start_socs = [30, 50, 70]
                    stop_socs = [90, 100]

                    for hw in candidates:
                        q6, q12, q20, q40 = hw['cfg']
                        fc_test = hw['fc_test']
                        e_tot = 0
                        
                        tot_batt_test = sum([q6, q12, q20, q40])
                        P_loss_kW_test = 28.1 * tot_batt_test / 1000.0
                        loss_flow_v40 = (P_loss_kW_test * 60) / (4.186 * dt_target) if dt_target > 0 else 0
                        
                        for q, s in zip([q6, q12, q20, q40], [6, 12, 20, 40]):
                            v0 = params_v40[s]['V0']
                            t_pcm_val = t_pcm
                            if t_pcm_val >= 58: v0 /= 0.76
                            e_tot += (v0 * 4.186 * dt_target) / 3600.0 * q
                        
                        p_gen_tests_with_rt = [(e_tot / (rt/60.0), rt) for rt in [60, 90, 120]]

                        for p_gen, rt_val in p_gen_tests_with_rt:
                            
                            p_hp_for_v40_opt = p_gen
                            p_req_batt_opt = max(0, P_peak_opt - p_hp_for_v40_opt)
                            
                            test_cfg = [(q6, 6), (q12, 12), (q20, 20), (q40, 40)]
                            s_flows, s_powers, _, _ = get_system_curves(test_cfg, t_pcm_val, dt_target, p_hp_for_v40_opt, e_tot, fc_test)
                            if not s_flows: continue
                            
                            limit_p = np.interp(qp_lmin_target, s_flows, s_powers)
                            if p_req_batt_opt >= limit_p:
                                continue 
                            
                            hp_cost = 0
                            hp_sel = get_suggested_generator(p_gen, t_pcm_val, gen_type)
                            if hp_sel: hp_cost = hp_sel['qty'] * hp_sel['gen']['price']
                            else: continue
                            
                            ites_capex = hw['cost'] + hp_cost
                            
                            for start_s in start_socs:
                                for stop_s in stop_socs:
                                    kwh_to_v40 = (3600) / (4.186 * dt_target)
                                    tank_cap_v40 = e_tot * kwh_to_v40
                                    hp_flow_v40 = (p_gen * 60) / (4.186 * dt_target)
                                    curr_v = tank_cap_v40 * 0.5
                                    is_on = False
                                    min_soc = tank_cap_v40
                                    kwh_cons = 0
                                    st_th = tank_cap_v40 * (start_s/100.0)
                                    sp_th = tank_cap_v40 * (stop_s/100.0)
                                    
                                    for _ in range(1440):
                                        cons = consumption_curve_min[_]
                                        if not is_on:
                                            if curr_v < st_th: is_on = True
                                        else:
                                            if curr_v >= sp_th: is_on = False
                                        
                                        prod = hp_flow_v40 if is_on else 0
                                        if is_on: kwh_cons += (p_gen / 60.0)
                                        curr_v = curr_v - cons - loss_flow_v40 + prod
                                        if curr_v > tank_cap_v40: curr_v = tank_cap_v40
                                        if curr_v < 0: curr_v = 0
                                        if curr_v < min_soc: min_soc = curr_v
                                    
                                    if min_soc > 0:
                                        if gen_type == "Pompa di Calore":
                                            eff = estimate_cop(t_pcm_val)
                                            opex = (kwh_cons / eff) * 365 * 0.31
                                        else:
                                            eff = 1.05 if t_pcm_val < 60 else 0.98
                                            opex = (kwh_cons / eff) * 365 * 0.11
                                        
                                        tco = ites_capex + (opex * 15)
                                        if best_solution is None or tco < best_solution['tco']:
                                            rt_eq = (e_tot / p_gen) * 60 if p_gen > 0 else 999
                                            best_solution = {
                                                'tco':tco, 'cfg':hw['cfg'], 'rt': int(rt_val), 
                                                'start':start_s, 'stop':stop_s, 'fc_pct': fc_test
                                            }
                    
                    status.update(label="Ottimizzazione completata!", state="complete", expanded=False)
                    
                    if best_solution:
                        st.session_state['pending_opt'] = {
                            'q6': best_solution['cfg'][0], 'q12': best_solution['cfg'][1],
                            'q20': best_solution['cfg'][2], 'q40': best_solution['cfg'][3],
                            'rt': best_solution['rt'], 'start': best_solution['start'], 
                            'stop': best_solution['stop'], 'fc_pct': best_solution['fc_pct']
                        }
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Nessuna soluzione trovata che rispetti tutti i vincoli termodinamici. Riprova con parametri diversi.")
        else:
            manage_qty('qty_6', "i-6")
            st.divider()
            manage_qty('qty_12', "i-12")
            st.divider()
            manage_qty('qty_20', "i-20")
            st.divider()
            manage_qty('qty_40', "i-40")
            st.markdown("---")
            t_pcm = st.radio("Temp. PCM (°C)", [48, 58, 74], horizontal=True)
            st.markdown("---")
            recharge_min = st.number_input("Tempo Reintegro Target (min)", step=10, min_value=1, value=st.session_state.recharge_min)
            st.session_state.recharge_min = recharge_min
            st.session_state.flow_correction_pct = st.number_input("Coeff. Correzione Portata (%)", step=1, value=st.session_state.flow_correction_pct)

# --- CALCOLI PRINCIPALI ---
config = [(st.session_state.qty_6, 6), (st.session_state.qty_12, 12), 
          (st.session_state.qty_20, 20), (st.session_state.qty_40, 40)]
total_cost = sum(q*prices[s] for q,s in config)
tot_batteries = sum(q for q, s in config)

total_energy_e0 = 0
installed_v40 = 0
v40_breakdown = []
for qty, size in config:
    v0_nominal = params_v40[size]['V0']
    if t_pcm >= 58: v0_nominal /= 0.76 
    e0_single = (v0_nominal * 4.186 * dt_target) / 3600.0
    total_energy_e0 += (e0_single * qty)
    if qty > 0:
        installed_v40 += (v0_nominal * qty)
        v40_breakdown.append(f"{qty} x {v0_nominal:.0f} L")

breakdown_str = " + ".join(v40_breakdown) if v40_breakdown else "0 L"

P_loss_kW = 28.1 * tot_batteries / 1000.0

if is_industrial:
    p_hp_tot_input = st.session_state.ind_p_hp
    p_hp_for_v40 = p_hp_tot_input if contributo_contestuale else 0.0
else:
    p_hp_tot_input = total_energy_e0 / (st.session_state.recharge_min / 60.0) if st.session_state.recharge_min > 0 and total_energy_e0 > 0 else 0.0
    p_hp_for_v40 = p_hp_tot_input

p_load = qp_lmin_target * dt_target * 0.0697
p_req_batt = max(0, p_load - p_hp_for_v40)

net_power_deficit = p_load - p_hp_for_v40
if net_power_deficit > 0.1:
    total_v40_liters = (total_energy_e0 / net_power_deficit) * 60 * qp_lmin_target
    autonomy_min = total_v40_liters / qp_lmin_target if qp_lmin_target > 0 else 0
    autonomy_str = f"{autonomy_min:.1f} min"
else:
    total_v40_liters = 99999
    autonomy_str = "∞ (Illimitata)"

P_avg_daily = (total_daily_L_calc * dt_target * 0.00116) / 24.0 + P_loss_kW
if not is_industrial and p_hp_tot_input < P_avg_daily and p_hp_tot_input > 0:
    st.warning(f"⚠️ **Attenzione: Generatore Sottodimensionato per il Baseload!** La potenza calcolata del generatore ({p_hp_tot_input:.1f} kW) è inferiore al fabbisogno medio continuo dell'utenza incluse le dispersioni ({P_avg_daily:.1f} kW). Le batterie si scaricheranno. Diminuisci il tempo di reintegro o attivare l'Autopilota.")

db_to_use = DAIKIN_SANICUBE_DB if is_sanicube else BUFFER_TANK_DB_GENERIC
tank_target_v40 = total_v40_liters if total_v40_liters < 99999 else total_daily_L_calc * 0.8
suggested_tank, tank_qty, tank_total_vol, tank_total_price = calculate_water_tank_config(tank_target_v40, t_water_set, t_in, db_to_use, is_sanicube)

tank_upsized_for_temp = False

if not is_sanicube and suggested_tank and show_comparison:
    factor_w = (t_water_set - max(t_in + 5, 20.0)) / (40 - t_in) if (40 - t_in) > 0 else 1
    if factor_w > 0:
        hp_recharge_flow_tank = ((p_hp_tot_input * 60) / (4.186 * dt_target)) / factor_w
        for _loop in range(20):
            start_v_w = tank_total_vol * 0.8
            start_s_w = False
            for _w in range(2): 
                _, _, temp_accumulo_water, start_v_w, start_s_w, _, _, _ = run_simulation_step_water(
                    start_v_w, start_s_w, tank_total_vol, consumption_curve_min, 
                    hp_recharge_flow_tank, (p_hp_tot_input / 60.0), t_water_set, t_in, factor_w, is_sanicube
                )
            
            if min(temp_accumulo_water) >= 45.0:
                break
            else:
                tank_upsized_for_temp = True
                tank_target_v40 *= 1.2
                suggested_tank, tank_qty, tank_total_vol, tank_total_price = calculate_water_tank_config(tank_target_v40, t_water_set, t_in, db_to_use, is_sanicube)

mac_config = {"desc": "N/A", "price": 0, "qty": 0}
if not is_sanicube and show_comparison:
    mac_config = get_mac_config(qp_lmin_target)

cost_tooltip = "DETTAGLIO COSTI:\n"
if total_cost > 0:
    for qty, size in config:
        if qty > 0:
            sub = qty * prices[size]
            cost_tooltip += f"- {qty}x i-{size}: € {sub:,.0f}\n"
    cost_tooltip += f"\nTOTALE: € {total_cost:,.0f}"
else:
    cost_tooltip = "Nessuna batteria selezionata"

# ==========================================
#  VISUALIZZAZIONE DASHBOARD HEADER E TITOLO
# ==========================================

with title_placeholder.container():
    col_logo, col_titolo = st.columns([1, 15])
    with col_logo:
        try:
            st.image("logo.png", width=80) 
        except:
            st.title("🔋") 
            
    with col_titolo:
        if show_comparison:
            st.title("Simulatore i-TES: Site-Specific & Confronto Tecnologico")
        else:
            addr_clean = st.session_state.address_found.replace(" (Default)", "")
            st.title(f"Simulatore i-TES: Site-Specific - {addr_clean}")

st.markdown("### 📋 Riepilogo Parametri di Progetto (Setup)")
utenze_list = []
if inputs['LU1']['qty'] > 0: utenze_list.append(f"{inputs['LU1']['qty']}x Lavabo")
if inputs['LU2']['qty'] > 0: utenze_list.append(f"{inputs['LU2']['qty']}x Doccia")
if inputs['LU3']['qty'] > 0: utenze_list.append(f"{inputs['LU3']['qty']}x Orinatoio")
if inputs['LU4']['qty'] > 0: utenze_list.append(f"{inputs['LU4']['qty']}x Vasca")
if inputs['LU5']['qty'] > 0: utenze_list.append(f"{inputs['LU5']['qty']}x Giardino")
if inputs['LU8']['qty'] > 0: utenze_list.append(f"{inputs['LU8']['qty']}x Comm.")
if inputs['LU15']['qty'] > 0: utenze_list.append(f"{inputs['LU15']['qty']}x Valvola")
utenze_str = ", ".join(utenze_list) if utenze_list else "Nessuna utenza"

num_shifts = 2 if "2 Turni" in sim_type else 3
num_docce = inputs['LU2']['qty']
y_utenti = int(sim_people / num_shifts) if is_industrial else 0
z_scaglioni = math.ceil(y_utenti / num_docce) if (is_industrial and num_docce > 0) else 0

if is_pv_mode:
    strat_str = f"Fotovoltaico (Avvio Sole < {st.session_state.pv_start_sun}%, Notte < {st.session_state.pv_start_night}%, Stop {st.session_state.pv_stop}%) | 📍 **Località:** {st.session_state.address_found}"
else:
    strat_str = f"Standard (Avvio < {st.session_state.std_start}%, Stop {st.session_state.std_stop}%)"

if is_industrial:
    st.info(f"""
    - **Profilo Utenza:** {sim_people} Utenti ({sim_type}) | **Modello Onda Quadra:** {y_utenti} utenti/turno in {z_scaglioni} scaglioni su {num_docce} docce (Finestra {st.session_state.shower_window_min} min).
    - **Utenze Sanitarie:** {utenze_str} | **Temperatura Rete:** {t_in}°C
    - **Impostazioni Batteria i-TES:** Temp. PCM **{t_pcm}°C** | Correzione Portata: **{st.session_state.flow_correction_pct:+}%**
    - **Dimensionamento (Modo Industria):** Bilancio Integrale Automatico | Strategia: {strat_str}
    """)
else:
    st.info(f"""
    - **Profilo Utenza:** {sim_people} Utenti ({sim_type}) | **Temperatura Rete:** {t_in}°C
    - **Utenze Sanitarie:** {utenze_str}
    - **Impostazioni Batteria i-TES:** Temp. PCM **{t_pcm}°C** | Correzione Portata: **{st.session_state.flow_correction_pct:+}%**
    - **Reintegro Gen.:** Tempo Target **{st.session_state.recharge_min} min** | Strategia: {strat_str}
    """)
st.divider()

st.subheader("📊 Analisi Prestazioni")
c1, c2, c3 = st.columns(3)
c1.metric("Portata Target Effettiva", f"{qp_lmin_target:.1f} L/min", help="Picco massimo tra normativa EN 806 e stima del Profilo Utenza")
c2.metric("Potenza Batt. Req.", f"{p_req_batt:.1f} kW", help="Quota di potenza istantanea coperta interamente dalla batteria")
c3.metric("Autonomia", autonomy_str, help="Durata alla portata di picco")

c1b, c2b, c3b = st.columns(3)
c1b.metric(f"Salto Termico", f"{dt_target:.1f} °C")

help_testo_gen = f"Potenza necessaria calcolata in automatico per garantire il baseload continuo" if is_industrial else f"Necessaria per ricaricare le batterie in {st.session_state.recharge_min} min"
c2b.metric("Potenza Gen. Calc.", f"{p_hp_tot_input:.1f} kW", help=help_testo_gen)

if total_v40_liters < 99999:
    integrazione_v40 = max(0, total_v40_liters - installed_v40)
    c3b.metric("Volume V40 Totale Erogabile", f"{total_v40_liters:.0f} L")
    c3b.caption(f"🔋 **Stand-alone:** {installed_v40:.0f} L {breakdown_str} \n⚡ **Integrazione PdC:** +{integrazione_v40:.0f} L")
else:
    c3b.metric("Volume V40 Totale Erogabile", "∞ L")
    c3b.caption(f"🔋 **Stand-alone:** {installed_v40:.0f} L {breakdown_str} \n⚡ **Integrazione PdC:** Potenza sufficiente a ciclo continuo")

st.divider()

# --- CONFRONTO VOLANO TERMICO O STANDALONE ---
st.subheader(f"🆚 Dettaglio Soluzione: i-TES{f' vs {comp_target}' if show_comparison else ''}")

ites_dims_str = ""
batt_detail_str = ""
ites_physical_vol_L = 0 
for qty, size in config:
    if qty > 0 and size in PCM_SPECS_DB:
        spec = PCM_SPECS_DB[size]
        ites_dims_str += f"- {qty}x [L {spec['w']} x P {spec['d']} x H {spec['h']} mm]\n"
        batt_detail_str += f"- {qty}x i-{size}: € {qty * prices[size]:,.0f}\n"
        vol_single = (spec['w'] * spec['d'] * spec['h']) / 1000000.0
        ites_physical_vol_L += (vol_single * qty)

tank_physical_vol_L = 0
if suggested_tank:
    if is_sanicube:
        vol_single_geo = (suggested_tank['d'] * suggested_tank['d'] * suggested_tank['h']) / 1000000.0
        tank_physical_vol_L = vol_single_geo * tank_qty
    else:
        radius_dm = (suggested_tank['d'] / 100.0) / 2.0
        height_dm = suggested_tank['h'] / 100.0
        vol_single_tank = math.pi * (radius_dm**2) * height_dm 
        tank_physical_vol_L = vol_single_tank * tank_qty

gen_ites = get_suggested_generator(p_hp_tot_input, t_pcm, gen_type)
hp_ites_price = gen_ites['qty'] * gen_ites['gen']['price'] if gen_ites else 0
total_system_ites_cost = total_cost + hp_ites_price

gen_water = get_suggested_generator(p_hp_tot_input, t_water_set + 5, gen_type)
hp_water_price = gen_water['qty'] * gen_water['gen']['price'] if gen_water else 0
total_system_water_cost = tank_total_price + hp_water_price + mac_config['price']

if show_comparison:
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.info(f"### 🔋 Soluzione i-TES (PCM)\n"
                f"**Temp. Stoccaggio:** {t_pcm}°C\n\n"
                f"**Ingombro Moduli:**\n{ites_dims_str}\n"
                f"**Volume Occupato:** {ites_physical_vol_L:.0f} Litri\n\n"
                f"**Dettaglio Costo Batterie:**\n{batt_detail_str}\n"
                f"**Costo Batterie Totale:** € {total_cost:,.0f}\n\n"
                f"---\n"
                f"**Generatore Suggerito ({gen_type}):**\n"
                f"{gen_ites['qty']}x {gen_ites['gen']['brand']} {gen_ites['gen']['model']} " if gen_ites else "N/A"
                f"\n- Potenza Totale: {gen_ites['qty'] * gen_ites['gen']['kw']:.1f} kW " if gen_ites else ""
                f"\n- Alimentazione: {gen_ites['gen']['gas']} " if gen_ites else ""
                f"\n- Max T: {gen_ites['gen'].get('max_t','-')}°C " if gen_ites else ""
                f"\n- Costo Est.: € {hp_ites_price:,.0f}\n"
                f"---\n"
                f"### 💰 TOTALE SISTEMA: € {total_system_ites_cost:,.0f}")
        st.caption("⚠️ Contattare fornitore per conferma")
                
    with col_comp2:
        if is_sanicube:
            tech_icon = "🧴"
            st.success(f"### {tech_icon} {comp_target}\n"
                       f"**Temp. Stoccaggio:** {t_water_set}°C (Acqua Tecnica)\n\n"
                       f"**Modello Suggerito:** {suggested_tank['brand']} {suggested_tank['model']}\n\n"
                       f"**Quantità:** {tank_qty}x ({tank_total_vol} Litri totali)\n\n"
                       f"**Dimensioni Unit:** Ø {suggested_tank['d']} mm x H {suggested_tank['h']} mm\n\n"
                       f"**Volume Occupato:** {tank_physical_vol_L:.0f} Litri\n\n"
                       f"**Costo Sanicube:** € {tank_total_price:,.0f}\n\n"
                       f"---\n"
                       f"**Generatore Suggerito ({gen_type}):**\n"
                       f"{gen_water['qty']}x {gen_water['gen']['brand']} {gen_water['gen']['model']} " if gen_water else "N/A"
                       f"\n- Potenza Totale: {gen_water['qty'] * gen_water['gen']['kw']:.1f} kW " if gen_water else ""
                       f"\n- Alimentazione: {gen_water['gen']['gas']} " if gen_water else ""
                       f"\n- Max T: {gen_water['gen'].get('max_t','-')}°C " if gen_water else ""
                       f"\n- Costo Est.: € {hp_water_price:,.0f}\n"
                       f"---\n"
                       f"### 💰 TOTALE SISTEMA: € {total_system_water_cost:,.0f}")
            st.caption("⚠️ Contattare fornitore per conferma")
        else:
            tech_icon = "🛢️"
            st.success(f"### {tech_icon} {comp_target}\n"
                       f"**Temp. Stoccaggio:** {t_water_set}°C (Acqua Tecnica)\n\n"
                       f"**Modelli Suggeriti:** {suggested_tank['brand']} {suggested_tank['model']}\n\n"
                       f"**Quantità Puffer:** {tank_qty}x ({tank_total_vol} Litri totali)\n\n"
                       f"**Dimensioni Unit:** Ø {suggested_tank['d']} mm x H {suggested_tank['h']} mm\n\n"
                       f"**Volume Occupato:** {tank_physical_vol_L:.0f} Litri\n\n"
                       f"**Costo Volani:** € {tank_total_price:,.0f}\n\n"
                       f"---\n"
                       f"**Modulo Produzione ACS Istantanea:**\n"
                       f"- {mac_config['desc']}\n"
                       f"- Costo Est. MAC: € {mac_config['price']:,.0f}\n\n"
                       f"---\n"
                       f"**Generatore Suggerito ({gen_type}):**\n"
                       f"{gen_water['qty']}x {gen_water['gen']['brand']} {gen_water['gen']['model']} " if gen_water else "N/A"
                       f"\n- Potenza Totale: {gen_water['qty'] * gen_water['gen']['kw']:.1f} kW " if gen_water else ""
                       f"\n- Alimentazione: {gen_water['gen']['gas']} " if gen_water else ""
                       f"\n- Max T: {gen_water['gen'].get('max_t','-')}°C " if gen_water else ""
                       f"\n- Costo Est. Gen.: € {hp_water_price:,.0f}\n"
                       f"---\n"
                       f"### 💰 TOTALE SISTEMA: € {total_system_water_cost:,.0f}")
            st.caption("⚠️ Contattare fornitore per conferma")
            
            if tank_upsized_for_temp:
                st.warning("⚠️ **Vincolo Modulo MAC (T > 45°C):** Per garantire che la temperatura dell'accumulo non scenda mai sotto i 45°C (limite per scambiare calore e produrre ACS in istantanea), il simulatore ha dovuto moltiplicare automaticamente il volume dei volani rispetto al fabbisogno teorico minimo.")
            
else:
    # STAND-ALONE MODE
    st.info(f"### 🔋 Soluzione i-TES (PCM)\n"
            f"**Temp. Stoccaggio:** {t_pcm}°C\n\n"
            f"**Ingombro Moduli:**\n{ites_dims_str}\n"
            f"**Volume Occupato:** {ites_physical_vol_L:.0f} Litri\n\n"
            f"**Dettaglio Costo Batterie:**\n{batt_detail_str}\n"
            f"**Costo Batterie Totale:** € {total_cost:,.0f}\n\n"
            f"---\n"
            f"**Dimensionamento Generatore ({gen_type}):**\n"
            f"- Potenza Minima Baseload: **{p_hp_tot_input:.1f} kW**\n"
            f"- Unità Consigliate: **{gen_ites['qty']} in parallelo** " if gen_ites else "N/A"
            f"\n- Temp. Max Erogazione: **$\ge$ {t_pcm}°C**\n"
            f"- Alimentazione (es.): **{gen_ites['gen']['gas']}** " if gen_ites else "")

# --- TABELLA CONFRONTO INGOMBRI ---
if show_comparison:
    st.markdown("#### 📏 Confronto Spazi Occupati")
    col_space1, col_space2 = st.columns(2)
    saving_vol_L = tank_physical_vol_L - ites_physical_vol_L
    saving_pct = (saving_vol_L / tank_physical_vol_L * 100) if tank_physical_vol_L > 0 else 0
    
    col_space1.metric("Volume Fisico i-TES", f"{ites_physical_vol_L:.0f} L")
    col_space2.metric(f"Volume Fisico {comp_target}", f"{tank_physical_vol_L:.0f} L", delta=f"-{saving_pct:.0f}% vs i-TES" if saving_pct > 0 else None, delta_color="inverse")
    st.caption(f"*Nota: Il sistema alternativo deve contenere {tank_total_vol:.0f} litri d'acqua a {t_water_set}°C per eguagliare la quantità di V40 prodotta dalle batterie i-TES alla portata di progetto.*")

st.divider()

# --- GRAFICI INTERATTIVI ---
sys_flows, sys_powers, sys_temps, sys_v40_volumes = get_system_curves(config, t_pcm, dt_target, p_hp_tot_input, total_energy_e0, st.session_state.flow_correction_pct)

# --- LAYOUT GRAFICI ---
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

col_g1, col_g2 = st.columns(2)

if len(sys_flows) > 0:
    with col_g1:
        st.subheader("💧 Volume V40 Costante")
        fig3 = go.Figure()
        
        display_max_y = installed_v40 * 3 if installed_v40 > 0 else 10000
        
        if not contributo_contestuale:
            plot_v40_vol = [installed_v40] * len(sys_flows)
        else:
            plot_v40_vol = [min(v, display_max_y) for v in sys_v40_volumes]
            
        fig3.add_trace(go.Scatter(x=sys_flows, y=plot_v40_vol, mode='lines', line=dict(color='#457b9d', width=2), name='i-TES V40', hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'))
        
        if show_comparison:
            if is_sanicube:
                sanicube_v40_curve = []
                for f in sys_flows:
                     single_v40 = get_sanicube_v40_at_flow(f / tank_qty, t_water_set) 
                     sanicube_v40_curve.append(single_v40 * tank_qty)
                fig3.add_trace(go.Scatter(x=sys_flows, y=sanicube_v40_curve, mode='lines', line=dict(color='#e67e22', width=2, dash='dash'), name='Sanicube V40 (Reale)', hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'))
            else:
                factor_w = (t_water_set - max(t_in + 5, 20.0)) / (40 - t_in) if (40 - t_in) > 0 else 1
                actual_v40_tank = tank_total_vol * factor_w
                v40_tank_flat = [actual_v40_tank] * len(sys_flows)
                fig3.add_trace(go.Scatter(x=sys_flows, y=v40_tank_flat, mode='lines', line=dict(color='green', width=2, dash='dot'), name='Volano Inerziale V40', hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'))

        pt_v40_vis = min(total_v40_liters, display_max_y) if contributo_contestuale else installed_v40
        fig3.add_trace(go.Scatter(x=[qp_lmin_target], y=[pt_v40_vis], mode='markers', marker=dict(color='purple', size=14, line=dict(color='black', width=2)), name='Punto Lavoro'))
        fig3.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Volume V40 (Litri)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col_g2:
        st.subheader("📦 Efficienza Volumetrica (V40/Litro)")
        fig_ratio = go.Figure()
        
        if ites_physical_vol_L > 0:
            ratio_ites = [v / ites_physical_vol_L for v in plot_v40_vol]
            fig_ratio.add_trace(go.Scatter(x=sys_flows, y=ratio_ites, mode='lines', line=dict(color='#457b9d', width=2), name='i-TES'))
            curr_v40_ites = np.interp(qp_lmin_target, sys_flows, plot_v40_vol)
            curr_ratio_ites = curr_v40_ites / ites_physical_vol_L
            fig_ratio.add_trace(go.Scatter(x=[qp_lmin_target], y=[curr_ratio_ites], mode='markers', marker=dict(color='#457b9d', size=10), name='Punto Lavoro i-TES'))
        
        if show_comparison and tank_physical_vol_L > 0:
            if is_sanicube:
                ratio_tank = [v / tank_physical_vol_L for v in sanicube_v40_curve]
                fig_ratio.add_trace(go.Scatter(x=sys_flows, y=ratio_tank, mode='lines', line=dict(color='#e67e22', width=2, dash='dash'), name='Sanicube'))
                curr_v40_sani = np.interp(qp_lmin_target, sys_flows, sanicube_v40_curve)
                curr_ratio_sani = curr_v40_sani / tank_physical_vol_L
                fig_ratio.add_trace(go.Scatter(x=[qp_lmin_target], y=[curr_ratio_sani], mode='markers', marker=dict(color='#e67e22', size=10), name='Punto Lavoro Sanicube'))
            else:
                factor_w = (t_water_set - max(t_in + 5, 20.0)) / (40 - t_in) if (40 - t_in) > 0 else 1
                ratio_val = (tank_total_vol * factor_w) / tank_physical_vol_L
                ratio_array = [ratio_val] * len(sys_flows)
                fig_ratio.add_trace(go.Scatter(x=sys_flows, y=ratio_array, mode='lines', line=dict(color='green', width=2, dash='dot'), name='Volano Gen.'))
                fig_ratio.add_trace(go.Scatter(x=[qp_lmin_target], y=[ratio_val], mode='markers', marker=dict(color='green', size=10), name='Punto Lavoro Volano'))
        
        fig_ratio.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Ratio V40 / Vol. Fisico", margin=dict(l=20,r=20,t=40,b=20), height=350)
        st.plotly_chart(fig_ratio, use_container_width=True)

else:
    st.warning("Aggiungi batterie per vedere i grafici.")

st.divider()

# --- GRAFICO SIMULAZIONE 24H ---
with st.expander("📈 Analisi Dettagliata", expanded=True):
    tank_capacity_L = total_v40_liters if total_v40_liters < 99999 else 9999 
    hp_recharge_flow_lmin = (p_hp_tot_input * 60) / (4.186 * dt_target) if dt_target > 0 else 0
    
    # Run Full Simulation Logic Definition
    def run_full_simulation(start_vol, start_hp_state, tank_capacity, hp_recharge_flow, hp_power_kW):
        curr_v = start_vol
        is_hp_on = start_hp_state
        soc_hist = []
        hp_status_hist = []
        
        gen_to_user_hist = []
        gen_to_batt_hist = []
        batt_to_user_hist = [] 
        batt_loss_hist = []
        
        tot_kwh = 0
        sol_kwh = 0
        grd_kwh = 0
        cycle_count = 0
        kwh_per_m = hp_power_kW / 60.0
        
        loss_L_per_m = (P_loss_kW * 60) / (4.186 * dt_target) if dt_target > 0 else 0
        
        for i in range(1440):
            solar_intensity = solar_profile_norm[i]
            is_sunny = solar_intensity > 0.05 
            
            if is_pv_mode:
                st_thr = tank_capacity * (st.session_state.pv_start_sun/100.0) if is_sunny else tank_capacity * (st.session_state.pv_start_night/100.0)
                sp_thr = tank_capacity * (st.session_state.pv_stop/100.0)
            else:
                st_thr = tank_capacity * (st.session_state.std_start/100.0)
                sp_thr = tank_capacity * (st.session_state.std_stop/100.0)
            
            cons_L = consumption_curve_min[i]
            
            if not is_hp_on:
                if curr_v < st_thr: 
                    is_hp_on = True
                    cycle_count += 1
            else:
                if curr_v >= sp_thr: is_hp_on = False
            
            prod_L = 0
            if is_hp_on:
                prod_L = hp_recharge_flow
                kwh_curr = kwh_per_m
                tot_kwh += kwh_curr
                if is_pv_mode and is_sunny:
                    s_part = kwh_curr * (pv_coverage / 100.0) * solar_intensity 
                    if s_part > kwh_curr: s_part = kwh_curr
                    sol_kwh += s_part
                    grd_kwh += (kwh_curr - s_part)
                else:
                    grd_kwh += kwh_curr
            
            gen_to_u_gross = min(prod_L, cons_L)
            gen_to_b_gross = max(0, prod_L - cons_L)
            batt_to_u_gross = max(0, cons_L - prod_L)
            loss_gross = loss_L_per_m
            
            new_v = curr_v + gen_to_b_gross - batt_to_u_gross - loss_gross
            
            gen_to_b_actual = gen_to_b_gross
            batt_to_u_actual = batt_to_u_gross
            loss_actual = loss_gross
            
            if new_v > tank_capacity:
                spill = new_v - tank_capacity
                gen_to_b_actual = max(0, gen_to_b_gross - spill)
                new_v = tank_capacity
            elif new_v < 0:
                shortfall = 0 - new_v
                if loss_actual >= shortfall:
                    loss_actual -= shortfall
                else:
                    shortfall -= loss_actual
                    loss_actual = 0
                    batt_to_u_actual = max(0, batt_to_u_actual - shortfall)
                new_v = 0
                
            gen_to_user_hist.append(gen_to_u_gross)
            gen_to_batt_hist.append(gen_to_b_actual)
            batt_to_user_hist.append(batt_to_u_actual)
            batt_loss_hist.append(loss_actual)
            
            curr_v = new_v
            soc_hist.append(curr_v)
            hp_status_hist.append(prod_L)
            
        return soc_hist, hp_status_hist, batt_to_user_hist, gen_to_user_hist, gen_to_batt_hist, batt_loss_hist, curr_v, is_hp_on, tot_kwh, sol_kwh, grd_kwh, cycle_count

    daily_kwh_kwp = get_pvgis_data(st.session_state.lat, st.session_state.lon, datetime.now().month - 1)
    solar_profile_norm = create_solar_curve_site_specific(st.session_state.lat, st.session_state.lon, daily_kwh_kwp)
    
    start_vol_guess = tank_capacity_L * 0.5
    start_state_guess = False
    
    for _ in range(10):
        soc_history, hp_status_history, batt_contrib_history, gen_to_user_history, gen_to_batt_history, batt_loss_history, end_vol, end_state, total_kwh_used, solar_kwh_used, grid_kwh_used, cycle_total = run_full_simulation(start_vol_guess, start_state_guess, tank_capacity_L, hp_recharge_flow_lmin, p_hp_tot_input)
        if abs(end_vol - start_vol_guess) < (tank_capacity_L * 0.01):
            break
        start_vol_guess = end_vol
        start_state_guess = end_state
    
    cumulative_consumption = np.cumsum(consumption_curve_min)
    cumulative_production = np.cumsum(hp_status_history)
    cum_gen_to_user = np.cumsum(gen_to_user_history)
    cum_batt_to_user = np.cumsum(batt_contrib_history)
    cum_gen_to_batt = np.cumsum(gen_to_batt_history)
    cum_batt_loss = np.cumsum(batt_loss_history)
    
    fig_smart = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=False, 
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # ------------------
    # ROW 1 (Top Chart)
    # ------------------
    if is_pv_mode:
        fig_smart.add_trace(go.Scatter(
            x=x_time_opt/60, y=solar_profile_norm * tank_capacity_L,
            mode='lines', fill='tozeroy', name='Produzione PV (Profilo)',
            line=dict(color='yellow', width=0),
            fillcolor='rgba(255, 215, 0, 0.4)',
            hoverinfo='skip',
            legend='legend'
        ), row=1, col=1, secondary_y=True)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=gen_to_user_history,
        mode='lines', name='Gen. ➔ Utenza (Diretto)',
        line=dict(color='#2ca02c', width=0),
        stackgroup='flow', fillcolor='rgba(44, 160, 44, 0.7)',
        hovertemplate='Gen ➔ Utenza: %{y:.1f} L/min<extra></extra>',
        legendgroup='gen_user',
        legend='legend'
    ), row=1, col=1, secondary_y=False)

    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=batt_contrib_history,
        mode='lines', name='Batt. ➔ Utenza (Scarica)',
        line=dict(color='#e67e22', width=0),
        stackgroup='flow', fillcolor='rgba(230, 126, 34, 0.7)',
        hovertemplate='Batt ➔ Utenza: %{y:.1f} L/min<extra></extra>',
        legendgroup='batt_user',
        legend='legend'
    ), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=gen_to_batt_history,
        mode='lines', name='Gen. ➔ Batterie (Ricarica)',
        line=dict(color='#1f77b4', width=0),
        stackgroup='flow', fillcolor='rgba(31, 119, 180, 0.5)',
        hovertemplate='Gen ➔ Batterie: %{y:.1f} L/min (Equiv.)<extra></extra>',
        legendgroup='gen_batt',
        legend='legend'
    ), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=batt_loss_history,
        mode='lines', name='Batt. ➔ Dispersioni',
        line=dict(color='#8c564b', width=0),
        stackgroup='flow', fillcolor='rgba(140, 86, 75, 0.7)',
        hovertemplate='Dispersioni: %{y:.1f} L/min<extra></extra>',
        legendgroup='batt_disp',
        legend='legend'
    ), row=1, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=consumption_curve_min,
        mode='lines', name='Prelievo Richiesto (Tetto)',
        line=dict(color='#ff0000', width=2),
        hovertemplate='Richiesta Utenza: %{y:.1f} L/min<extra></extra>',
        legendgroup='user_req',
        legend='legend'
    ), row=1, col=1, secondary_y=False)

    # ------------------
    # ROW 2 (Bottom Chart)
    # ------------------
    
    # Asse Sinistro (Utenza & SoC) - Batt SOTTO, Gen SOPRA
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=cum_batt_to_user,
        mode='lines', name='Cumul. Batt. ➔ Utenza',
        line=dict(color='#e67e22', width=0),
        stackgroup='cum_u', fillcolor='rgba(230, 126, 34, 0.5)',
        hovertemplate='Cum. Batt➔Utenza: %{y:.0f} L<extra></extra>',
        legendgroup='batt_user',
        legend='legend2',
        showlegend=False
    ), row=2, col=1, secondary_y=False)

    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=cum_gen_to_user,
        mode='lines', name='Cumul. Gen. ➔ Utenza',
        line=dict(color='#2ca02c', width=0),
        stackgroup='cum_u', fillcolor='rgba(44, 160, 44, 0.5)',
        hovertemplate='Cum. Gen➔Utenza: %{y:.0f} L<extra></extra>',
        legendgroup='gen_user',
        legend='legend2',
        showlegend=False
    ), row=2, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=cumulative_consumption,
        mode='lines', name='Volume Prelievo Totale',
        line=dict(color='#ff0000', width=2),
        hovertemplate='Prelievo Tot: %{y:.0f} L<extra></extra>',
        legendgroup='user_req',
        legend='legend2',
        showlegend=False
    ), row=2, col=1, secondary_y=False)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=soc_history,
        mode='lines', name='Carica Batteria (SoC)',
        line=dict(color='#000000', width=2, dash='dot'),
        hovertemplate='Carica (SoC): %{y:.0f} L<extra></extra>',
        legend='legend2'
    ), row=2, col=1, secondary_y=False)

    # Asse Destro (Generatore & Produzione)
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=cum_gen_to_batt,
        mode='lines', name='Cumul. Gen. ➔ Batterie',
        line=dict(color='#1f77b4', width=0),
        stackgroup='cum_gen', fillcolor='rgba(31, 119, 180, 0.4)',
        hovertemplate='Cum. Gen➔Batterie: %{y:.0f} L<extra></extra>',
        legendgroup='gen_batt',
        legend='legend2',
        showlegend=False
    ), row=2, col=1, secondary_y=True)

    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=cum_batt_loss,
        mode='lines', name='Cumul. Batt. ➔ Dispersioni',
        line=dict(color='#8c564b', width=0),
        stackgroup='cum_gen', fillcolor='rgba(140, 86, 75, 0.5)',
        hovertemplate='Cum. Dispersioni: %{y:.0f} L<extra></extra>',
        legendgroup='batt_disp',
        legend='legend2',
        showlegend=False
    ), row=2, col=1, secondary_y=True)
    
    fig_smart.add_trace(go.Scatter(
        x=x_time_opt/60, y=cumulative_production,
        mode='lines', name='Produzione Totale Gen.',
        line=dict(color='#1b5e20', width=3, dash='dash'),
        hovertemplate='Prod Tot Gen: %{y:.0f} L<extra></extra>',
        legend='legend2'
    ), row=2, col=1, secondary_y=True)

    # Generazione Ticks per Asse X
    x_ticks_vals = np.arange(0, 24.5, 0.5)
    x_ticks_labels = [str(int(x)) if x % 1 == 0 else " " for x in x_ticks_vals]

    fig_smart.update_layout(
        title=f"Strategia: {recharge_strategy}",
        height=800,
        legend=dict(y=1.0, yanchor="top", x=1.08, xanchor="left", title_text="<b>Clicca per nascondere</b>"),
        legend2=dict(y=0.4, yanchor="top", x=1.08, xanchor="left", title_text="<b>Linee Specifiche</b>"),
        margin=dict(r=200) # Spazio per le legende separate a destra
    )
    
    # Asse X Top
    fig_smart.update_xaxes(
        title_text="",
        tickmode='array', tickvals=x_ticks_vals, ticktext=x_ticks_labels, 
        ticks="outside", showticklabels=True,
        range=[0, 24], row=1, col=1
    )
    
    # Asse X Bottom
    fig_smart.update_xaxes(
        title_text="Ora del Giorno (0-24h)",
        tickmode='array', tickvals=x_ticks_vals, ticktext=x_ticks_labels, 
        ticks="outside", showticklabels=True,
        range=[0, 24], row=2, col=1
    )
    
    # Calcolo tetto massimo UNICO per allineare perfettamente i due assi
    max_vol_tot = max(np.max(cumulative_consumption), np.max(cumulative_production), np.max(soc_history), 1) * 1.05

    # Assi Y
    fig_smart.update_yaxes(title_text="Portata (L/min)", row=1, col=1, secondary_y=False, rangemode="tozero")
    if is_pv_mode:
        fig_smart.update_yaxes(title_text="Volume (L)", row=1, col=1, secondary_y=True, range=[0, tank_capacity_L*1.1], rangemode="tozero")
        
    fig_smart.update_yaxes(title_text="Volumi Utenza / Batt. (L)", row=2, col=1, secondary_y=False, range=[0, max_vol_tot], rangemode="tozero")
    fig_smart.update_yaxes(title_text="Volumi Generatore (L)", row=2, col=1, secondary_y=True, range=[0, max_vol_tot], rangemode="tozero", showgrid=False)
    
    st.plotly_chart(fig_smart, use_container_width=True)
    
    # --- DIAGRAMMA DI SANKEY ---
    st.markdown("### 🔄 Bilancio di Flusso Giornaliero")
    
    total_gen_to_user = np.sum(gen_to_user_history)
    total_gen_to_batt = np.sum(gen_to_batt_history)
    total_batt_to_user = np.sum(batt_contrib_history)
    total_batt_loss = np.sum(batt_loss_history)
    
    total_gen_out = total_gen_to_user + total_gen_to_batt
    total_batt_in = total_gen_to_batt
    total_batt_out = total_batt_to_user + total_batt_loss
    total_received = total_gen_to_user + total_batt_to_user
    
    fig_sankey = go.Figure(data=[go.Sankey(
        textfont=dict(size=25, color="black"),
        node = dict(
          pad = 25, thickness = 30, line = dict(color = "black", width = 1),
          label = [
              f"GENERATORE ({total_gen_out:,.0f} L)", 
              f"BATTERIE i-TES ({total_batt_in:,.0f} L IN / {total_batt_out:,.0f} L OUT)", 
              f"UTENZA FINALE ({total_received:,.0f} L)",
              f"DISPERSIONI TERMICHE ({total_batt_loss:,.0f} L)"
          ],
          color = ["#2ca02c", "#1f77b4", "#ff0000", "#8c564b"]
        ),
        link = dict(
          source = [0, 0, 1, 1], 
          target = [2, 1, 2, 3],
          value = [total_gen_to_user, total_gen_to_batt, total_batt_to_user, total_batt_loss],
          color = ["rgba(44, 160, 44, 0.4)", "rgba(31, 119, 180, 0.4)", "rgba(230, 126, 34, 0.4)", "rgba(140, 86, 75, 0.4)"],
          label = ["Gen ➔ Utenza", "Gen ➔ Carica Batt.", "Batt ➔ Utenza", "Batt ➔ Dispersioni"]
        )
    )])
    
    fig_sankey.update_layout(
        title_text="<b>Distribuzione dei Volumi V40 Equivalenti [Litri/Giorno]</b>", 
        height=400
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # Costruzione stringhe descrittive box finali
    tot_batt_report = sum(q for q, s in config)
    flow_parts_report = [f"{q}x i-{s} ({NOMINAL_FLOWS[s]} L/min cad.)" for q, s in config if q > 0]
    nom_flow_text_str = ", ".join(flow_parts_report) if flow_parts_report else "0 L/min"
    corr_pct_val = st.session_state.flow_correction_pct
    
    if is_industrial:
        st.markdown(f"""
        **ℹ️ Dettagli Profilo di Prelievo & Carico di Base (Industria):**
        * **Normativa di Riferimento (Volume Giornaliero):** Il fabbisogno totale stimato in **{total_daily_L_calc:.0f} Litri** si basa sulle indicazioni della **UNI 9182** per utenze di tipo '{sim_type}'. Il generatore è dimensionato sul *Base-Load* necessario a coprire questo volume durante l'intera giornata.
        * **Profilo di Erogazione Istantaneo:** L'algoritmo genera un'**Onda Quadra** che simula l'ingresso scaglionato degli utenti alle docce durante il cambio turno.
        * **Dinamica dei Flussi (Le Aree Colorate):** L'Area Verde (Gen ➔ Utenza) è impilata sopra l'Area Arancione (Batt ➔ Utenza) sull'asse di sinistra, mostrando esattamente come si riempie il fabbisogno. Sull'asse di destra, la linea tratteggiata verde scuro mostra la Produzione Cumulata Totale del Generatore.
        * **Gestione del Picco di Progetto:** L'algoritmo **"Onda Quadra Integrale"** ha calcolato il volume geometrico esatto mancante all'interno della finestra di prelievo. La prestazione è garantita dall'utilizzo di **{tot_batt_report} batterie** con portate nominali di **{nom_flow_text_str}** e correzione portata del **{corr_pct_val}%**.
        * **Efficienza Termica:** È stata calcolata una dispersione termica costante di **28,1 W per ogni modulo i-TES**, il cui volume equivalente perso (Area Marrone) viene automaticamente compensato dal generatore per mantenere l'impianto in perfetto equilibrio 24h.
        """)
    else:
        st.markdown(f"""
        **ℹ️ Dettagli Profilo di Prelievo:**
        * **Normativa di Riferimento (Volume Giornaliero):** Il fabbisogno totale stimato in **{total_daily_L_calc:.0f} Litri** si basa sulle indicazioni della **UNI 9182** per utenze di tipo '{sim_type}'.
        * **Dinamica dei Flussi (Le Aree Colorate):** La distribuzione dei prelievi (Linea Rossa) fa riferimento ai **Profili di Carico Standard (UNI TS 11300-2)**. L'Area Verde è il contributo diretto del generatore all'utenza. Il picco rimanente è soddisfatto dalle **Batterie i-TES (Area Arancione)**. Quando l'**Area Blu** oltrepassa la curva rossa, il generatore sta ricaricando le batterie. (Sincronizzazione Legende Attiva).
        * **Gestione del Picco di Progetto:** Il picco normativo richiesto di **{qp_lmin_target:.1f} L/min** (UNI EN 806-3 / Picco Profilo) è interamente coperto. Tale prestazione è garantita dall'utilizzo di **{tot_batt_report} batterie** con portate nominali base di **{nom_flow_text_str}**, alle quali viene applicato un coefficiente di correzione della portata del **{corr_pct_val}%** (un incremento che rientra pienamente nei limiti di sicurezza gestibili dalla batteria).
        * **Efficienza Termica:** È stata calcolata una dispersione termica costante di **28,1 W per ogni modulo i-TES**, il cui volume equivalente perso (Area Marrone) viene automaticamente compensato dal generatore per mantenere l'impianto in perfetto equilibrio 24h.
        """)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(f"Energia Totale Generata", f"{total_kwh_used:.1f} kWh/g")
    if is_pv_mode:
        kpi2.metric("Da Fotovoltaico", f"{solar_kwh_used:.1f} kWh", delta=f"{(solar_kwh_used/total_kwh_used)*100:.0f}%" if total_kwh_used>0 else "0%")
        kpi3.metric("Da Rete", f"{grid_kwh_used:.1f} kWh", delta=f"-{(grid_kwh_used/total_kwh_used)*100:.0f}%" if total_kwh_used>0 else "0%", delta_color="inverse")
    else:
        kpi2.metric("Da Rete (Gas/Elec)", f"{total_kwh_used:.1f} kWh")
        kpi3.metric("Autoconsumo PV", "N/A")
    kpi4.metric("Cicli ON/OFF Generatore", f"{cycle_total}")

# --- CONFRONTO TECNOLOGICO E TABELLE ---
if show_comparison:
    with st.expander(f"🆚 Analisi Economica & Tecnologica", expanded=True):
        
        if is_sanicube:
            factor_w = (t_water_set - t_in) / (40 - t_in) if (40 - t_in) > 0 else 1
        else:
            factor_w = (t_water_set - max(t_in + 5, 20.0)) / (40 - t_in) if (40 - t_in) > 0 else 1
            
        hp_recharge_flow_tank = ((p_hp_tot_input * 60) / (4.186 * dt_target)) / factor_w

        start_v_w = tank_total_vol * 0.8
        start_s_w = False
        cost_extra_temp = 0

        for _ in range(5):
            soc_water, temp_erogazione_water, temp_accumulo_water, end_v_w, end_s_w, kwh_water_therm, cycles_water, tot_water_loss_kwh = run_simulation_step_water(
                start_v_w, start_s_w, tank_total_vol, consumption_curve_min, 
                hp_recharge_flow_tank, (p_hp_tot_input / 60.0), t_water_set, t_in, factor_w, is_sanicube
            )
            start_v_w = end_v_w
            start_s_w = end_s_w

        kwh_pcm = total_kwh_used
        cycles_pcm = cycle_total
        
        if len(sys_flows) > 0:
            func_temp_pcm = interpolate.interp1d(sys_flows, sys_temps, kind='linear', fill_value="extrapolate")
        else:
            func_temp_pcm = lambda x: t_in 

        soc_pcm, temp_pcm = run_pcm_plot_data(soc_history, func_temp_pcm, t_pcm, t_in, consumption_curve_min)

        fig_comp = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=False,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5],
            subplot_titles=("Stato di Carica (Litri Equivalenti V40)", "Dinamica delle Temperature (°C)")
        )
        
        fig_comp.add_trace(go.Scatter(x=x_time_opt/60, y=soc_pcm, mode='lines', name='Accumulo i-TES (PCM)', line=dict(color='#007acc', width=3)), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=x_time_opt/60, y=soc_water, mode='lines', name=f'Accumulo {comp_target}', line=dict(color='#2ca02c', width=2, dash='dot')), row=1, col=1) 
        
        fig_comp.add_trace(go.Scatter(x=x_time_opt/60, y=temp_pcm, mode='lines', name='Temp. Erogazione i-TES', line=dict(color='#007acc', width=3)), row=2, col=1)
        fig_comp.add_trace(go.Scatter(x=x_time_opt/60, y=temp_erogazione_water, mode='lines', name=f'Temp. Erogazione {comp_target} (Miscelata)', line=dict(color='#ff7f0e', width=2, dash='dot')), row=2, col=1) 
        
        if not is_sanicube:
            fig_comp.add_trace(go.Scatter(x=x_time_opt/60, y=temp_accumulo_water, mode='lines', name=f'Temp. Interna Puffer', line=dict(color='#2ca02c', width=1, dash='dashdot')), row=2, col=1)
        
        fig_comp.update_xaxes(
            title_text="",
            tickmode='array', tickvals=x_ticks_vals, ticktext=x_ticks_labels, 
            ticks="outside", showticklabels=True,
            range=[0, 24], row=1, col=1
        )
        
        fig_comp.update_xaxes(
            title_text="Ora del Giorno (0-24h)",
            tickmode='array', tickvals=x_ticks_vals, ticktext=x_ticks_labels, 
            ticks="outside", showticklabels=True,
            range=[0, 24], row=2, col=1
        )

        fig_comp.update_layout(title="Confronto Dinamico Completo", height=700)
        fig_comp.update_yaxes(title_text="Volume (L)", row=1, col=1)
        fig_comp.update_yaxes(title_text="Temperatura (°C)", row=2, col=1)
        st.plotly_chart(fig_comp, use_container_width=True)
            
        if not is_sanicube:
            testo_dinamica = (
                "💡 **Dinamica Termica del Volano:** La linea verde punteggiata mostra la temperatura reale all'interno del puffer. "
                "Ad ogni prelievo, l'acqua fredda in ingresso abbassa la temperatura media del serbatoio, costringendo il Generatore "
                "a frequenti riaccensioni per ripristinare il setpoint. L'i-TES, invece, cede calore a temperatura costante sfruttando il "
                "cambiamento di fase, azzerando questi sbalzi e riducendo drasticamente le accensioni (cicli) del generatore."
            )
            st.info(testo_dinamica)
        
        if gen_type == "Pompa di Calore":
            eff_pcm = estimate_cop(t_pcm)
            eff_water = estimate_cop(t_water_set)
            energy_price = 0.31
            startup_penalty = 0.05
            cost_per_cycle = 0.15
            energy_name = "Energia Elettrica (kWh_e)"
        else:
            eff_pcm = 1.05 if t_pcm < 60 else 0.98
            eff_water = 1.05 if t_water_set < 60 else 0.95
            energy_price = 0.11
            startup_penalty = 0.01
            cost_per_cycle = 0.02
            energy_name = "Energia Termica Primaria (kWh)"

        kwh_energy_pcm_year = (kwh_pcm * 365) / eff_pcm
        kwh_energy_water_year = (kwh_water_therm * 365) / eff_water
        
        pcm_cost_energy = kwh_energy_pcm_year * energy_price
        water_cost_energy = kwh_energy_water_year * energy_price
        
        delta_cycles_year = (cycles_water - cycles_pcm) * 365
        saving_energy_year = water_cost_energy - pcm_cost_energy
        saving_maint_year = delta_cycles_year * cost_per_cycle
        saving_startup_elec_year = delta_cycles_year * startup_penalty * energy_price
        total_saving_year = saving_energy_year + saving_maint_year + saving_startup_elec_year
        
        if t_water_set > t_pcm:
            cost_extra_temp = (((kwh_pcm * 365) / eff_water) - ((kwh_pcm * 365) / eff_pcm)) * energy_price

        st.markdown("### 📊 Dettaglio Consumi Assoluti (Stima Annuale)")
        
        consumption_data = {
            "Metrica": ["Energia Richiesta (kWh_t)", "Dispersioni Termiche (kWh_t)", energy_name, "Costo Energia (€)", "Costo Start-up (€)", "Costo Manutenzione (€)"],
            "i-TES (PCM)": [f"{kwh_pcm*365:,.0f}", f"{P_loss_kW * 24 * 365:,.0f}", f"{kwh_energy_pcm_year:,.0f}", f"€ {pcm_cost_energy:,.0f}", f"€ {cycles_pcm * 365 * startup_penalty * energy_price:,.0f}", "Incluso"],
            f"{comp_target}": [f"{kwh_water_therm*365:,.0f}", f"{tot_water_loss_kwh * 365:,.0f}", f"{kwh_energy_water_year:,.0f}", f"€ {water_cost_energy:,.0f}", f"€ {cycles_water * 365 * startup_penalty * energy_price:,.0f}", f"€ {delta_cycles_year * cost_per_cycle:,.0f} (Extra)"],
            "Differenza (Risparmio)": [f"{(kwh_water_therm-kwh_pcm)*365:,.0f}", f"{(tot_water_loss_kwh * 365) - (P_loss_kW * 24 * 365):,.0f}", f"{(kwh_energy_water_year-kwh_energy_pcm_year):,.0f}", f"€ {saving_energy_year:,.0f}", f"€ {saving_startup_elec_year:,.0f}", f"€ {saving_maint_year:,.0f}"]
        }
            
        st.table(pd.DataFrame(consumption_data))
        
        st.markdown("### 💰 Riepilogo Risparmio Economico")
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Risp. Energia (Efficienza & Disp.)", f"€ {saving_energy_year:,.0f}")
        col_res2.metric("Risp. Energia (Start-up)", f"€ {saving_startup_elec_year:,.0f}")
        col_res3.metric("Risp. Manutenzione (Usura)", f"€ {saving_maint_year:,.0f}")
        col_res4.metric("TOTALE RISPARMIO ANNUO", f"€ {total_saving_year:,.0f}", delta="Totale", delta_color="normal")
        
        with st.expander("ℹ️ Dettaglio Risparmio Energetico"):
            st.markdown(f"""
            Questo risparmio combina due fattori fondamentali:
            1. **Miglior Efficienza (Resa):** Il generatore per i-TES lavora a {t_pcm}°C (Rendimento {eff_pcm*100:.0f}%), mentre per il sistema alternativo lavora a {t_water_set}°C (Rendimento {eff_water*100:.0f}%).
            2. **Minori Dispersioni:** Mantenere un volume d'acqua enorme sempre ad alta temperatura comporta maggiori perdite termiche rispetto al PCM.
            """)

        if cost_extra_temp > 0:
            with st.expander("ℹ️ Dettaglio Calcolo Penalità di Generazione"):
                st.markdown(f"La batteria PCM lavora a **{t_pcm}°C**. Il sistema alternativo è mantenuto a **{t_water_set:.1f}°C** peggiorando le performance del generatore.")
                
        with st.expander("ℹ️ Dettaglio Risparmio Manutenzione (Usura)"):
            st.markdown(f"Il risparmio è calcolato sulla riduzione dello stress termomeccanico. Differenza: {delta_cycles_year:,.0f} cicli in meno.")

        with st.expander("ℹ️ Dettaglio Risparmio Energia (Transitori Start-up)"):
            st.markdown(f"Ogni avvio necessita di una fase a bassa efficienza. Calcolo: {delta_cycles_year:,.0f} cicli evitati × {startup_penalty} kWh/ciclo × {energy_price} € = **€ {saving_startup_elec_year:,.0f}**")

        st.markdown("### 💸 Analisi di Break-even (ROI)")
        delta_investment = total_system_ites_cost - total_system_water_cost
        
        if delta_investment <= 0:
            st.success(f"✅ **L'investimento i-TES è più economico o uguale.**\n\nRisparmio immediato all'acquisto: **€ {-delta_investment:,.0f}**\nOltre a un risparmio operativo annuo di **€ {total_saving_year:,.0f}**.")
        elif total_saving_year <= 0:
            st.error(f"❌ **Il sistema i-TES costa di più (€ {delta_investment:,.0f}) e non genera risparmio operativo.**")
        else:
            x_years = list(range(16))
            cash_flow = [(-delta_investment) + (total_saving_year * y) for y in x_years]
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(x=x_years, y=cash_flow, mode='lines+markers', name='Flusso di Cassa Cumulato', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)'))
            fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Point")
            fig_roi.update_layout(title="Rientro dell'Investimento (Differenziale)", xaxis_title="Anni", yaxis_title="Bilancio Economico (€)", height=400)
            
            st.plotly_chart(fig_roi, use_container_width=True)
            
            roi_years = delta_investment / total_saving_year
            st.metric("Tempo di Ritorno (Payback)", f"{roi_years:.1f} Anni")
            
            if roi_years < 5: st.success(f"🚀 **Ottimo investimento!** Il sistema si ripaga in soli {roi_years:.1f} anni.")
            elif roi_years < 10: st.warning(f"⚠️ **Investimento a medio termine.** Rientro in {roi_years:.1f} anni.")
            else: st.info(f"ℹ️ **Investimento a lungo termine.** Rientro in {roi_years:.1f} anni.")

# --- GRAFICO EN 806 ---
with st.expander("📉 Vedi Grafico Normativo EN 806-3", expanded=False):
    x_vals = np.logspace(0, 3, 100)
    y_vals = [calcola_qd_en806(x, max_lu_unit) for x in x_vals]
    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Curva Normativa', line=dict(color='#1f77b4', width=2)))
    
    fig_en.add_trace(go.Scatter(
        x=[lu_totali], y=[qd_ls_target], 
        mode='markers+text', 
        marker=dict(color='red', size=16, line=dict(color='black', width=2)),
        text=["Punto Progetto"], textposition="top center",
        name='Punto Progetto'
    ))
    
    fig_en.update_layout(
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Unità di Carico Totali (LU)",
        yaxis_title="Portata di Progetto (L/s)",
        title="Curva di Contemporaneità (UNI EN 806-3)"
    )
    st.plotly_chart(fig_en, use_container_width=True)
