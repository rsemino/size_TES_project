import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Simulatore i-TES Pro", layout="wide")
st.title("🚰 Simulatore i-TES: Tecnico & Economico")

# --- GESTIONE STATO (Per aggiornare i valori dopo l'ottimizzazione) ---
if 'qty_6' not in st.session_state: st.session_state.qty_6 = 0
if 'qty_12' not in st.session_state: st.session_state.qty_12 = 0
if 'qty_20' not in st.session_state: st.session_state.qty_20 = 1
if 'qty_40' not in st.session_state: st.session_state.qty_40 = 0

# ==========================================
#  DATABASE PREZZI
# ==========================================
prices = {
    6:  3300.0,
    12: 5100.0,
    20: 7400.0,
    40: 13200.0,
    'pdc_per_kw': 0.0 
}

# --- 2. LOGICA EN 806-3 ---
def calcola_qd_en806(total_lu, max_single_lu):
    max_single_qa = max_single_lu * 0.1 
    conv_lu = 300.0
    conv_qd = 1.7
    if total_lu <= 0: return 0.0
    if total_lu >= conv_lu:
        return 0.1 * math.sqrt(total_lu)
    else:
        if total_lu < max_single_lu: total_lu = max_single_lu
        x1, y1 = math.log(max_single_lu), math.log(max_single_qa)
        x2, y2 = math.log(conv_lu), math.log(conv_qd)
        m = 0 if x2 == x1 else (y2 - y1) / (x2 - x1)
        return math.exp(y1 + m * (math.log(total_lu) - x1))

# --- 3. DATABASE CARATTERISTICHE ---
params_v40 = {
    6:  {'E0': 4.1569,  'label': 'i-6',  'nom_lmin': 10.0, 'max_lmin': 12.0}, 
    12: {'E0': 8.3138,  'label': 'i-12', 'nom_lmin': 20.0, 'max_lmin': 24.0},
    20: {'E0': 16.7876, 'label': 'i-20', 'nom_lmin': 25.0, 'max_lmin': 30.0},
    40: {'E0': 33.5752, 'label': 'i-40', 'nom_lmin': 50.0, 'max_lmin': 60.0}
}

# --- 4. SIDEBAR: INPUT ---
st.sidebar.header("Parametri Progetto")

# A. Utenze (LU)
with st.sidebar.expander("🏗️ 1. Utenze e Carichi (LU)", expanded=False):
    inputs = {}
    inputs['LU1'] = {'qty': st.number_input("Lavabo, Bidet (1 LU)", 0, value=0), 'val': 1}
    inputs['LU2'] = {'qty': st.number_input("Doccia, Lavello (2 LU)", 0, value=2), 'val': 2}
    inputs['LU3'] = {'qty': st.number_input("Orinatoio (3 LU)", 0, value=0), 'val': 3}
    inputs['LU4'] = {'qty': st.number_input("Vasca (4 LU)", 0, value=0), 'val': 4}
    inputs['LU5'] = {'qty': st.number_input("Giardino (5 LU)", 0, value=0), 'val': 5}
    inputs['LU8'] = {'qty': st.number_input("Commerciale (8 LU)", 0, value=0), 'val': 8}
    inputs['LU15'] = {'qty': st.number_input("Valvola DN20 (15 LU)", 0, value=0), 'val': 15}
    
    lu_totali = sum(item['qty'] * item['val'] for item in inputs.values())
    max_lu_unit = max([item['val'] for item in inputs.values() if item['qty'] > 0] or [1])
    if lu_totali == 0: lu_totali = 1
    
    # Calcolo target per ottimizzatore
    qd_ls_target = calcola_qd_en806(lu_totali, max_lu_unit)
    qp_lmin_target = qd_ls_target * 60 
    
    st.info(f"Totale: **{lu_totali} LU**\nTarget: **{qp_lmin_target:.1f} L/min**")

# B. Generazione
st.sidebar.subheader("🔥 2. Generazione")
p_hp = st.sidebar.number_input("Potenza Singola PdC (kW)", value=10.0, step=1.0, format="%.0f")
n_hp = st.sidebar.number_input("Numero PdC", value=1, min_value=1)
p_hp_tot_input = p_hp * n_hp

# C. Batterie (Selezione)
with st.sidebar.expander("🔋 3. Parco Batterie", expanded=True):
    # Callback per aggiornare lo stato quando l'utente cambia manualmente
    def update_qty():
        st.session_state.qty_6 = st.session_state._qty_6
        st.session_state.qty_12 = st.session_state._qty_12
        st.session_state.qty_20 = st.session_state._qty_20
        st.session_state.qty_40 = st.session_state._qty_40

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.number_input("i-6", key='_qty_6', value=st.session_state.qty_6, min_value=0, on_change=update_qty)
        st.number_input("i-12", key='_qty_12', value=st.session_state.qty_12, min_value=0, on_change=update_qty)
    with col_b2:
        st.number_input("i-20", key='_qty_20', value=st.session_state.qty_20, min_value=0, on_change=update_qty)
        st.number_input("i-40", key='_qty_40', value=st.session_state.qty_40, min_value=0, on_change=update_qty)
    
    st.markdown("---")
    t_pcm = st.radio("Temp. PCM (°C)", [48, 58, 74], horizontal=True)

    # --- ALGORITMO DI OTTIMIZZAZIONE ---
    if st.button("🚀 TROVA CONFIGURAZIONE OTTIMALE"):
        st.toast("Calcolo ottimizzazione in corso...", icon="⏳")
        
        # Parametri target
        req_flow = qp_lmin_target
        # Nota: La potenza è legata alla portata, quindi se soddisfiamo la portata, 
        # soddisfiamo la potenza (a meno di PdC negative, impossibile).
        # Flow constraint is the hardest constraint.
        
        best_cost = float('inf')
        best_cfg = None
        
        # Definizione range di ricerca (Limitiamo per velocità)
        # Un i-40 porta 60 l/min. Se servono 120 l/min, max 2-3 i-40.
        # i-6 porta 12 l/min. Se servono 120 l/min, max 10 i-6.
        # Mettiamo bounds ragionevoli per l'app web.
        max_search = int(req_flow / 10) + 2 # Euristic limit
        r_40 = range(max_search if max_search < 5 else 6) # Max 5 i-40
        r_20 = range(6)
        r_12 = range(10) # i-12 è efficiente, controlliamone di più
        r_6  = range(6)
        
        # Iterazione Brute Force (Veloce per questi numeri)
        for q40, q20, q12, q6 in itertools.product(r_40, r_20, r_12, r_6):
            if q40==0 and q20==0 and q12==0 and q6==0: continue
            
            # Calcolo Capacità Configurazione
            curr_flow = (q6 * params_v40[6]['max_lmin']) + \
                        (q12 * params_v40[12]['max_lmin']) + \
                        (q20 * params_v40[20]['max_lmin']) + \
                        (q40 * params_v40[40]['max_lmin'])
            
            # Verifica Vincoli (Deve coprire la portata richiesta)
            if curr_flow >= req_flow:
                # Calcolo Costo
                curr_cost = (q6 * prices[6]) + (q12 * prices[12]) + \
                            (q20 * prices[20]) + (q40 * prices[40])
                
                # Se è più economico (o uguale ma con meno batterie totali), salva
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_cfg = (q6, q12, q20, q40)
        
        if best_cfg:
            st.session_state.qty_6 = best_cfg[0]
            st.session_state.qty_12 = best_cfg[1]
            st.session_state.qty_20 = best_cfg[2]
            st.session_state.qty_40 = best_cfg[3]
            st.success(f"Trovato! Costo: €{best_cost:,.0f}")
            st.rerun() # Ricarica la pagina coi nuovi valori
        else:
            st.error("Nessuna configurazione valida trovata nei limiti di ricerca.")

# --- 5. CALCOLI ---
# Recupero valori (aggiornati o manuali)
qty_6, qty_12, qty_20, qty_40 = st.session_state.qty_6, st.session_state.qty_12, st.session_state.qty_20, st.session_state.qty_40

# Calcoli Tecnici
dt_progetto = 27.5
coeff_termico = 0.0697
p_load = qp_lmin_target * dt_progetto * coeff_termico

total_energy = 0
total_max_flow = 0
total_cost_batteries = 0 

batteries_config = [(qty_6, 6), (qty_12, 12), (qty_20, 20), (qty_40, 40)]

for qty, size in batteries_config:
    if qty > 0:
        e_single = params_v40[size]['E0']
        if t_pcm >= 58: e_single /= 0.76 # Approx per 58 e 74
        
        total_energy += (e_single * qty)
        total_max_flow += (params_v40[size]['max_lmin'] * qty)
        total_cost_batteries += (prices[size] * qty)

p_batteries_max = total_max_flow * dt_progetto * coeff_termico
net_pwr_deficit = p_load - p_hp_tot_input 
grand_total_cost = total_cost_batteries + (p_hp_tot_input * prices['pdc_per_kw'])

# --- 6. VISUALIZZAZIONE ---
st.subheader("📊 Analisi Progetto")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Portata Target", f"{qp_lmin_target:.1f} L/min", delta=f"{total_max_flow - qp_lmin_target:.1f} marg.", delta_color="normal")
kpi2.metric("Potenza Picco", f"{p_load:.1f} kW")
kpi3.metric("💰 Costo Configurazione", f"€ {total_cost_batteries:,.0f}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ Verifiche Sistema")
    
    delta_flow = total_max_flow - qp_lmin_target
    if delta_flow < -0.1: # Tolleranza float
        st.error(f"❌ **PORTATA INSUFFICIENTE**: Mancano {abs(delta_flow):.1f} L/min")
    else:
        st.success(f"✅ **Portata OK**: Coperta al {total_max_flow/qp_lmin_target*100:.0f}%")

    p_guaranteed_total = p_batteries_max + p_hp_tot_input
    delta_pwr = p_guaranteed_total - p_load
    if delta_pwr < -0.1:
        st.error(f"❌ **POTENZA INSUFFICIENTE**")
    else:
        st.success(f"✅ **Potenza OK**")
        
    st.caption(f"Mix Batterie: {qty_6}x i-6 | {qty_12}x i-12 | {qty_20}x i-20 | {qty_40}x i-40")

with col2:
    st.markdown("#### 🔋 Autonomia")
    
    if net_pwr_deficit <= 0:
        st.success("♾️ **ILLIMITATA** (PdC autosufficiente)")
    else:
        power_gap = p_load - p_hp_tot_input
        # Se la batteria non riesce a scaricare abbastanza potenza (Flow Limit), l'autonomia reale è "compromessa"
        # ma qui calcoliamo la durata teorica alla massima scarica possibile o richiesta.
        
        discharge_power = min(power_gap, p_batteries_max)
        if discharge_power > 0:
            autonomia_min = (total_energy / discharge_power) * 60
            v40_tot = qp_lmin_target * autonomia_min
            
            st.metric("Tempo Scarica", f"{autonomia_min:.1f} min")
            st.metric("Volume Equivalente", f"{v40_tot:.0f} Litri")
        else:
             st.error("Errore Calcolo")

st.divider()
st.subheader("📉 Grafico EN 806-3")
fig, ax = plt.subplots(figsize=(8, 3))
x_vals = np.logspace(0, 3, 100)
y_vals = [calcola_qd_en806(x, max_lu_unit) for x in x_vals]
ax.plot(x_vals, y_vals, color='#007acc')
ax.scatter([lu_totali], [qd_ls_target], color='red', s=100, zorder=5)
ax.axhline(total_max_flow/60, color='green', linestyle='--', label='Limite Batterie')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.set_xlabel("LU Totali")
ax.set_ylabel("Portata [l/s]")
ax.legend()
st.pyplot(fig)