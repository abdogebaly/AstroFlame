import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from PIL import Image

st.set_page_config(
    page_title="AstroFlame | NASA Microgravity Combustion Simulator",
    page_icon="🔥",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔥 AstroFlame: NASA Microgravity Combustion Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified AI & Physics Simulation Suite for Spacecraft Fire Safety and Combustion in Free Fall</div>', unsafe_allow_html=True)

# Cache Models
@st.cache_resource
def load_all_models():
    models = {}
    paths = {
        'bass_ext': "models/bass_extinction_model.joblib",
        'bass_co': "models/bass_co_regressor.joblib",
        'flex_k': "models/flex_burning_rate_model.joblib",
        'flex_end': "models/flex_end_mode_model.joblib",
        'cfig_time': "models/cfig_burn_time_model.joblib",
        'cfig_ext': "models/cfig_self_ext_model.joblib",
        'spice': "models/spice_model.joblib"
    }
    for k, p in paths.items():
        if os.path.exists(p):
            models[k] = joblib.load(p)
    return models

models = load_all_models()

# Navigation Tabs
tabs = st.tabs([
    "🛡️ Solid Fire Safety (BASS-II)",
    "💧 Droplet Physics & d² Law (FLEX)",
    "🧪 Gaseous & Cool Flames (CFI-G)",
    "💨 Soot & Smoke Point (SPICE)",
    "📊 Empirical Model Analytics",
    "🔬 Space Physics Reference"
])

# ==============================================================================
# TAB 1: BASS-II SPACECRAFT CABIN FIRE SAFETY
# ==============================================================================
with tabs[0]:
    st.subheader("🛡️ Spacecraft Cabin Material Flammability & Extinction Assessment")
    st.caption("Predict whether solid materials in the spacecraft cabin will self-extinguish or sustain combustion under ventilation airflow.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🛠️ Cabin & Material Parameters")
        mat_type = st.selectbox("Material Type", ["PMMA", "SIBAL", "Nomex", "Wax"], index=0)
        geom = st.selectbox("Sample Geometry", ["Film", "Slab", "Rod", "Fabric"], index=0)
        color = st.selectbox("Material Optical Tint", ["standard", "clear", "black"], index=0)
        thickness = st.slider("Material Thickness / Diameter (mm)", 0.1, 15.0, 1.0, step=0.1)
        width = st.slider("Sample Width (cm)", 0.5, 5.0, 2.0, step=0.5)
        fan_speed = st.slider("Cabin Fan Flow Display (Ventilation Velocity)", 10.0, 80.0, 50.0, step=1.0)
        o2_init = st.slider("Cabin Initial O₂ (% by Volume)", 15.0, 30.0, 21.0, step=0.5)
        
        input_bass = pd.DataFrame([{
            'material_type': mat_type, 'geometry': geom, 'color': color,
            'thickness_mm': thickness, 'width_cm': width, 'flow_restrictor': 1,
            'fan_display': fan_speed, 'air_display': fan_speed / 10.0, 'o2_initial': o2_init
        }])
        
    with col2:
        st.markdown("### 🎯 Safety Prediction Outcome")
        if 'bass_ext' in models:
            ext_pred = models['bass_ext'].predict(input_bass)[0]
            ext_prob = models['bass_ext'].predict_proba(input_bass)[0]
            pred_co = max(0.0, float(models['bass_co'].predict(input_bass)[0])) if 'bass_co' in models else 20.0
            
            if ext_pred == 1:
                st.success("✅ **SAFE: High Probability of Self-Extinction**\n\nThe flame is projected to extinguish naturally under this ventilation flow and material geometry without spreading.")
            else:
                st.error("🚨 **HAZARD: Sustained Fire Risk Detected!**\n\nMaterial sustains combustion! Active fire suppression intervention is required.")
                
            c_m1, c_m2 = st.columns(2)
            c_m1.metric("Extinction Probability", f"{ext_prob[1]*100:.1f}%")
            c_m2.metric("Projected Toxic CO Output", f"{pred_co:.1f} ppm")
            
            hazard_score = (1.0 - ext_prob[1]) * 100.0
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=hazard_score,
                title={'text': "Combustion Hazard Index (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b" if hazard_score > 50 else "#00cc96"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 204, 150, 0.2)"},
                        {'range': [30, 70], 'color': "rgba(255, 170, 0, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if pred_co > 50:
                st.warning("⚠️ **Toxicology Alert**: Estimated CO exceeds spacecraft maximum allowable 7-day concentration limit (50 ppm).")
            else:
                st.info("ℹ️ Projected CO generation is within manageable cabin scrubber (ECLSS) limits.")

# ==============================================================================
# TAB 2: FLEX DROPLET COMBUSTION (d² LAW)
# ==============================================================================
with tabs[1]:
    st.subheader("💧 Microgravity Droplet Combustion Physics & d²-Law Simulation")
    st.caption("Simulate spherically symmetric liquid fuel droplet combustion under varying cabin atmospheres and pressures.")
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.markdown("### 🧪 Droplet & Atmosphere Inputs")
        fuel = st.selectbox("Liquid Fuel", ["Heptane", "Methanol"], index=0)
        p_bar = st.slider("Ambient Pressure (bar)", 0.2, 3.0, 1.0, step=0.1)
        d0 = st.slider("Initial Droplet Diameter d₀ (mm)", 1.0, 5.0, 3.0, step=0.1)
        x_o2 = st.slider("Oxygen (O₂)", 0.10, 0.40, 0.21, step=0.01)
        x_co2 = st.slider("Carbon Dioxide (CO₂ - Thermal Quencher)", 0.0, 0.50, 0.0, step=0.05)
        x_he = st.slider("Helium (He - High Thermal Conductor)", 0.0, 0.50, 0.0, step=0.05)
        x_n2 = max(0.0, 1.0 - (x_o2 + x_co2 + x_he))
        st.caption(f"Nitrogen (N₂) balance: {x_n2:.2f}")
        
        input_flex = pd.DataFrame([{
            'fuel': fuel, 'pressure_bar': p_bar, 'x_o2': x_o2, 'x_n2': x_n2,
            'x_co2': x_co2, 'x_he': x_he, 'd0_mm': d0
        }])
        
    with col_f2:
        st.markdown("### 📈 Physics Simulation Results")
        if 'flex_k' in models:
            k_val = max(0.05, float(models['flex_k'].predict(input_flex)[0]))
            t_total = (d0**2) / k_val
            
            c_k1, c_k2 = st.columns(2)
            c_k1.metric("Burning Rate Constant (K)", f"{k_val:.4f} mm²/s")
            c_k2.metric("Total Burning Lifetime", f"{t_total:.2f} s")
            
            t_steps = np.linspace(0, t_total, 100)
            d_vals = np.sqrt(np.maximum(0, d0**2 - k_val * t_steps))
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=t_steps, y=d_vals, mode='lines', name='d(t)', line=dict(color='#38bdf8', width=3)))
            fig_curve.update_layout(
                title=f"Droplet Diameter Decay vs Time (d²-Law)",
                xaxis_title="Time (seconds)", yaxis_title="Droplet Diameter (mm)",
                height=280, margin=dict(l=10, r=10, t=35, b=10)
            )
            st.plotly_chart(fig_curve, use_container_width=True)

# ==============================================================================
# TAB 3: CFI-G GASEOUS & COOL FLAMES
# ==============================================================================
with tabs[2]:
    st.subheader("🧪 Gaseous Alkanes & Cool Flame Physics (CFI-G)")
    st.caption("Investigate laminar co-flow gas flames and self-extinction limits across varying pressures and fuel mass flow rates.")
    
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.markdown("### ⚙️ Burner Conditions")
        c_fuel = st.selectbox("Hydrocarbon Fuel", ["ethane", "propane", "50% butane", "30% butane", "30% propane"], index=0)
        c_p = st.slider("Chamber Pressure (bar)", 0.5, 3.0, 1.2, step=0.1)
        c_xf = st.slider("Fuel Mole Fraction (X_f)", 0.1, 1.0, 0.4, step=0.05)
        c_xo2 = st.slider("Initial Oxygen Fraction (X_O₂)", 0.15, 0.50, 0.25, step=0.02)
        c_mhc = st.slider("Hydrocarbon Mass Flow Rate (mg/s)", 0.2, 3.5, 1.2, step=0.1)
        c_ign = st.selectbox("Hot Ignition Assist", ["Yes", "No"], index=0)
        
        input_cfig = pd.DataFrame([{
            'fuel': c_fuel, 'pressure_bar': c_p, 'xf': c_xf,
            'xo2_initial': c_xo2, 'mhc_mg_s': c_mhc, 'hot_ignition': c_ign
        }])
        
    with col_c2:
        st.markdown("### 📊 Extinction & Duration Projection")
        if 'cfig_ext' in models and 'cfig_time' in models:
            c_ext = models['cfig_ext'].predict(input_cfig)[0]
            c_ext_prob = models['cfig_ext'].predict_proba(input_cfig)[0][1]
            c_time = max(0.0, float(models['cfig_time'].predict(input_cfig)[0]))
            
            if c_ext == 1:
                st.info(f"🔵 **Self-Extinction Projected** (Probability: {c_ext_prob*100:.1f}%)\nFlame terminates naturally due to low-strain diffusion quenching.")
            else:
                st.warning(f"🟡 **Sustained Gas Flame Projected** (Manual flow termination required).")
                
            c_g1, c_g2 = st.columns(2)
            c_g1.metric("Predicted Burn Duration", f"{c_time:.1f} sec")
            c_g2.metric("Extinction Probability", f"{c_ext_prob*100:.1f}%")

# ==============================================================================
# TAB 4: SPICE SOOT & SMOKE POINT
# ==============================================================================
with tabs[3]:
    st.subheader("💨 Soot Emission & Smoke Point Prediction (SPICE)")
    st.caption("Determine critical fuel flow rate at which soot escapes the flame tip without oxidizing.")
    
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        st.markdown("### ⛽ Fuel Jet & Co-flow Parameters")
        s_fuel = st.selectbox("Jet Fuel", ["Propylene", "75% Propylene", "50% Propylene", "Propane", "Ethylene", "Methane"], index=0)
        s_nozzle = st.selectbox("Burner Nozzle Diameter (mm)", [0.4064, 0.7620, 1.6002], index=1)
        s_air = st.slider("Co-flow Air Velocity (cm/s)", 5.0, 80.0, 30.0, step=1.0)
        
        input_spice = pd.DataFrame([{
            'fuel': s_fuel, 'nozzle_mm': s_nozzle, 'air_rate_cm_s': s_air
        }])
        
    with col_s2:
        st.markdown("### 🎯 Critical Smoke Point")
        if 'spice' in models:
            crit_flow = max(0.1, float(models['spice'].predict(input_spice)[0]))
            st.metric("Critical Fuel Rate at Smoke Point", f"{crit_flow:.3f} mg/s")
            st.info("""
            **Soot Dynamics Insight**:
            In microgravity, longer molecular residence times promote soot agglomeration. Exceeding this critical fuel rate causes soot release beyond the flame envelope.
            """)

# ==============================================================================
# TAB 5: EMPIRICAL MODEL ANALYTICS & VISUALIZATIONS
# ==============================================================================
with tabs[4]:
    st.subheader("📊 Empirical Model Analytics & Validation Plots")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        if os.path.exists("results/bass_feature_importance.png"):
            st.image(Image.open("results/bass_feature_importance.png"), caption="Top Factors Influencing Flame Extinction in Microgravity")
            
    with col_v2:
        if os.path.exists("results/bass_confusion_matrix.png"):
            st.image(Image.open("results/bass_confusion_matrix.png"), caption="BASS-II Confusion Matrix (81.8% Accuracy)")
            
    if os.path.exists("results/flex_pred_vs_actual.png"):
        st.image(Image.open("results/flex_pred_vs_actual.png"), caption="FLEX Droplet Burning Rate: AI vs Empirical Experiment (R² = 0.70)")

# ==============================================================================
# TAB 6: SPACE PHYSICS REFERENCE
# ==============================================================================
with tabs[5]:
    st.markdown("""
    ### 🚀 The Physics of Free-Fall Combustion
    
    1. **Buoyancy Elimination**: Gravity-induced natural convection ($g \cdot \Delta \rho$) vanishes.
    2. **Diffusion Dominated**: Oxygen transport is governed purely by Fickian diffusion:
       $$J = -D \\nabla C$$
    3. **Quasi-steady Cool Flames**: Discovered during FLEX droplet experiments; n-alkanes undergo low-temperature chain-branching chemistry (~600–1000 K) with near-zero visible emissions.
    4. **Critical Spacecraft Airspeeds**: Cabin airflow between 20–50 arbitrary fan units supplies oxygen to sustain solid fires without providing sufficient convective cooling to achieve blow-off extinction.
    
    *Data Source: NASA Physical Sciences Informatics (PSI) System.*
    """)
