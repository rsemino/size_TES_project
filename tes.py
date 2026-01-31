import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools
from scipy import interpolate

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Simulatore i-TES Pro", layout="wide")
st.title("🚰 Simulatore i-TES: Curve Reali & Ottimizzazione")

# --- GESTIONE STATO ---
if 'qty_6' not in st.session_state: st.session_state.qty_6 = 0
if 'qty_12' not in st.session_state: st.session_state.qty_12 = 0
if 'qty_20' not in st.session_state: st.session_state.qty_20 = 1
if 'qty_40' not in st.session_state: st.session_state.qty_40 = 0

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

# ==========================================
#  DATABASE
# ==========================================
prices = { 6: 3300.0, 12: 5100.0, 20: 7400.0, 40: 13200.0, 'pdc_per_kw': 0.0 }

# V0: Litri V40 nominali della sola batteria
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

# --- FUNZIONI DI CALCOLO ---
def calcola_qd_en806(total_lu, max_single_lu):
    max_single_qa = max_single_lu * 0.1 
    conv_lu = 300.0; conv_qd = 1.7
    if total_lu <= 0: return 0.0
    if total_lu >= conv_lu:
        return 0.1 * math.sqrt(total_lu)
    else:
        if total_lu < max_single_lu: total_lu = max_single_lu
        x1, y1 = math.log(max_single_lu), math.log(max_single_qa)
        x2, y2 = math.log(conv_lu), math.log(conv_qd)
        m = 0 if x2 == x1 else (y2 - y1) / (x2 - x1)
        return math.exp(y1 + m * (math.log(total_lu) - x1))

def get_system_curves(config, t_pcm, dt_calc, p_hp_tot, e_tot_e0):
    """
    Calcola le curve del sistema.
    """
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

    # Range portate da visualizzare
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
        
        # --- CALCOLO CURVA V40 ---
        p_req_at_flow = f_tot * dt_calc * 0.0697
        p_deficit = p_req_at_flow - p_hp_tot
        
        if p_deficit > 0:
            v40_vol = (e_tot_e0 / p_deficit) * 60 * f_tot
        else:
            v40_vol = 99999 
            
        sys_v40_volumes.append(v40_vol)
        
    return sys_flows, sys_powers, sys_temps, sys_v40_volumes

# --- SIDEBAR ---
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
    st.info(f"Target: **{qp_lmin_target:.1f} L/min**")

st.sidebar.subheader("🔥 2. Generazione")
p_hp = st.sidebar.number_input("Potenza PdC (kW)", value=10.0, step=1.0, format="%.0f")
n_hp = st.sidebar.number_input("Numero PdC", value=1, min_value=1)
p_hp_tot_input = p_hp * n_hp

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

    if st.button("🚀 OTTIMIZZA COSTI"):
        st.toast("Calcolo...", icon="⏳")
        req_flow = qp_lmin_target
        best_cost = float('inf'); best_cfg = None
        max_search = int(req_flow / 10) + 2
        for q40, q20, q12, q6 in itertools.product(range(max_search if max_search < 5 else 6), range(6), range(10), range(6)):
            if q40==0 and q20==0 and q12==0 and q6==0: continue
            f_sys = (q6 * 24) + (q12 * 32) + (q20 * 40) + (q40 * 80)
            if f_sys >= req_flow:
                curr_cost = (q6*prices[6]) + (q12*prices[12]) + (q20*prices[20]) + (q40*prices[40])
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_cfg = (q6, q12, q20, q40)
        if best_cfg:
            st.session_state.qty_6, st.session_state.qty_12 = best_cfg[0], best_cfg[1]
            st.session_state.qty_20, st.session_state.qty_40 = best_cfg[2], best_cfg[3]
            st.success(f"Ottimizzato: €{best_cost:,.0f}")
            st.rerun()

# --- CALCOLI PRINCIPALI ---
config = [(st.session_state.qty_6, 6), (st.session_state.qty_12, 12), 
          (st.session_state.qty_20, 20), (st.session_state.qty_40, 40)]
total_cost = sum(q*prices[s] for q,s in config)

# 1. Calcolo Energia Totale (E0)
total_energy_e0 = 0
for qty, size in config:
    v0_nominal = params_v40[size]['V0']
    if t_pcm >= 58: v0_nominal /= 0.76 
    # E0 = V0 * 4.186 * dt / 3600
    e0_single = (v0_nominal * 4.186 * dt_target) / 3600.0
    total_energy_e0 += (e0_single * qty)

# 2. Potenza Carico Richiesta
p_load = qp_lmin_target * dt_target * 0.0697
p_req_batt = max(0, p_load - p_hp_tot_input)

# 3. Calcolo Volume V40 Totale (Puntuale)
net_power_deficit = p_load - p_hp_tot_input
if net_power_deficit > 0.1:
    total_v40_liters = (total_energy_e0 / net_power_deficit) * 60 * qp_lmin_target
else:
    total_v40_liters = 99999 

# --- VISUALIZZAZIONE ---
st.subheader("📊 Analisi Prestazioni")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Portata Target", f"{qp_lmin_target:.1f} L/min")
k2.metric(f"Salto Termico", f"{dt_target:.1f} °C")
k3.metric("Potenza Batt. Richiesta", f"{p_req_batt:.1f} kW")
k4.metric("Costo Batterie", f"€ {total_cost:,.0f}")

st.divider()

# --- GRAFICI INTERATTIVI (PLOTLY) ---
sys_flows, sys_powers, sys_temps, sys_v40_volumes = get_system_curves(config, t_pcm, dt_target, p_hp_tot_input, total_energy_e0)

if len(sys_flows) > 0:
    col_g1, col_g2, col_g3 = st.columns(3)
    
    # 1. POTENZA
    with col_g1:
        st.subheader("⚡ Potenza")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=sys_flows, y=sys_powers, fill='tozeroy', mode='lines', line=dict(color='#2a9d8f', width=2), name='Max Potenza', hovertemplate='Portata: %{x:.1f} L/min<br>Max Power: %{y:.1f} kW<extra></extra>'))
        limit_p = np.interp(qp_lmin_target, sys_flows, sys_powers)
        col_pt = 'green' if p_req_batt <= limit_p else 'red'
        fig1.add_trace(go.Scatter(x=[qp_lmin_target], y=[p_req_batt], mode='markers', marker=dict(color=col_pt, size=14, line=dict(color='black', width=2)), name='Punto Lavoro', hovertemplate='Richiesta: %{x:.1f} L/min<br>Serve: %{y:.1f} kW<extra></extra>'))
        fig1.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Potenza (kW)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig1, use_container_width=True)

    # 2. TEMPERATURA
    with col_g2:
        st.subheader("🌡️ Temperatura")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sys_flows, y=sys_temps, mode='lines', line=dict(color='#e76f51', width=2), name='Temp. Uscita', hovertemplate='Portata: %{x:.1f} L/min<br>Temp: %{y:.1f} °C<extra></extra>'))
        t_pt = np.interp(qp_lmin_target, sys_flows, sys_temps)
        fig2.add_trace(go.Scatter(x=[qp_lmin_target], y=[t_pt], mode='markers+text', marker=dict(color='orange', size=14, line=dict(color='black', width=2)), text=[f"{t_pt:.1f}°C"], textposition="top center", name='Punto Lavoro'))
        fig2.update_layout(xaxis_title="Portata (L/min)", yaxis_title="Temp (°C)", margin=dict(l=20,r=20,t=30,b=20), height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # 3. VOLUME V40
    with col_g3:
        st.subheader("💧 Volume V40 Totale")
        fig3 = go.Figure()
        
        display_max_y = 10000 
        plot_v40_vol = [min(v, display_max_y) for v in sys_v40_volumes]
        
        fig3.add_trace(go.Scatter(
            x=sys_flows, y=plot_v40_vol,
            mode='lines',
            line=dict(color='#457b9d', width=2),
            name='Volume Disponibile',
            hovertemplate='Portata: %{x:.1f} L/min<br>Volume: %{y:.0f} L<extra></extra>'
        ))
        
        pt_v40_vis = min(total_v40_liters, display_max_y)
        label_v40 = f"{total_v40_liters:.0f} L" if total_v40_liters < 9000 else "> 9000 L"
        
        fig3.add_trace(go.Scatter(
            x=[qp_lmin_target], y=[pt_v40_vis],
            mode='markers',
            marker=dict(color='purple', size=14, line=dict(color='black', width=2)),
            name='Punto Lavoro',
            hovertemplate=f'<b>Punto Lavoro</b><br>Portata: %{{x:.1f}} L/min<br>Volume: {label_v40}<extra></extra>'
        ))
        
        fig3.update_layout(
            xaxis_title="Portata (L/min)", yaxis_title="Volume V40 (Litri)",
            margin=dict(l=20,r=20,t=30,b=20), height=350,
            yaxis=dict(range=[0, min(max(plot_v40_vol)*1.1, display_max_y)])
        )
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning("Aggiungi batterie per vedere i grafici.")

st.divider()

# --- TABELLA MIX BATTERIE ---
table_rows = ""
for qty, size in config:
    if qty > 0:
        last_pt = CURVES_DB[size][t_pcm][-1]
        f_max = last_pt[0] * qty
        p_max = last_pt[1] * qty
        table_rows += f"<tr style='border-bottom: 1px solid #ddd;'><td style='padding: 12px; font-weight:bold; font-size:18px;'>i-{size}</td><td style='padding: 12px; font-size:18px; color:#2a9d8f; font-weight:bold;'>{qty} pz</td><td style='padding: 12px; font-size:18px;'>{p_max:.1f} kW</td><td style='padding: 12px; font-size:18px;'>{f_max:.1f} L/min</td></tr>"

if table_rows:
    full_table_html = f"""
<div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border:1px solid #ddd; margin-bottom: 20px;">
<h3 style="margin-top:0; color:#333; border-bottom: 2px solid #2a9d8f; padding-bottom:10px;">🧩 Dettaglio Contributo Batterie (Max)</h3>
<table style="width:100%; text-align:left; border-collapse: collapse;">
<thead>
<tr style="background-color:#e9ecef; color:#444;">
<th style="padding: 12px; font-size:16px;">Modello</th>
<th style="padding: 12px; font-size:16px;">Quantità</th>
<th style="padding: 12px; font-size:16px;">Potenza Max</th>
<th style="padding: 12px; font-size:16px;">Portata Max</th>
</tr>
</thead>
<tbody>{table_rows}</tbody>
</table></div>"""
    st.markdown(full_table_html, unsafe_allow_html=True)

# --- GRAFICO EN 806-3 INTERATTIVO (PLOTLY) ---
with st.expander("📉 Vedi Grafico Normativo EN 806-3", expanded=True):
    # Generazione dati
    x_vals = np.logspace(0, 3, 100)
    y_vals = [calcola_qd_en806(x, max_lu_unit) for x in x_vals]
    
    fig_en = go.Figure()

    # Curva Blu
    fig_en.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        name='Curva Normativa',
        line=dict(color='#007acc', width=3),
        hovertemplate='LU: %{x:.1f}<br>Portata: %{y:.2f} l/s<extra></extra>'
    ))

    # Punto Rosso
    fig_en.add_trace(go.Scatter(
        x=[lu_totali], y=[qd_ls_target],
        mode='markers',
        name='Punto Progetto',
        marker=dict(color='red', size=15, line=dict(color='black', width=2)),
        hovertemplate='<b>IL TUO PROGETTO</b><br>Totale LU: %{x:.0f}<br>Portata Target: %{y:.2f} l/s<extra></extra>'
    ))

    # Layout Logaritmico
    fig_en.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Load Units Totali (LU)",
        yaxis_title="Portata di Progetto (l/s)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig_en, use_container_width=True)
ax.set_xlabel("LU Totali")
ax.set_ylabel("Portata [l/s]")
ax.legend()
st.pyplot(fig)
