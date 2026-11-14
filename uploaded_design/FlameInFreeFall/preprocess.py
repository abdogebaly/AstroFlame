import os
import re
import pandas as pd
import numpy as np

os.makedirs("cleaned_data", exist_ok=True)

def parse_num_range(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("none", "").replace("None", "").replace("?", "")
    if not s:
        return np.nan
    # Split on space, slash, dash
    parts = re.findall(r"[-+]?\d*\.?\d+", s)
    if not parts:
        return np.nan
    nums = [float(p) for p in parts]
    return float(np.mean(nums))

def clean_bass_ii():
    print("Processing BASS-II...")
    df = pd.read_csv("PSI-25_Experimental table_BASS-II.csv")
    
    # Standardize column names
    col_map = {
        'PI': 'pi',
        'Test #': 'test_id',
        'Date': 'date',
        'GMT': 'gmt',
        'Sample #': 'sample_id',
        'Fuel Sample Material': 'raw_material',
        'Flow restrictor': 'flow_restrictor',
        'Fan display': 'fan_display_raw',
        'Air display': 'air_display_raw',
        'Radiometer gain   (1000   250   50   10)': 'radiometer_gain',
        'Total Frames Shot': 'total_frames_raw',
        'Calibrated  initial O2 % by vol ': 'o2_initial_raw',
        'Calibrated final O2 % by vol': 'o2_final_raw',
        'Initial CO2 % by vol': 'co2_initial_raw',
        'Final CO2 % by vol': 'co2_final_raw',
        'Initial CO (ppm)': 'co_initial_raw',
        'Final CO (ppm)': 'co_final_raw'
    }
    df = df.rename(columns=col_map)
    
    # Parse material characteristics
    def parse_material(mat):
        s = str(mat).lower()
        # Material type
        if "pmma" in s or "film" in s or "rod" in s or "slab" in s:
            mat_type = "PMMA"
        elif "sibal" in s:
            mat_type = "SIBAL"
        elif "nomex" in s:
            mat_type = "Nomex"
        elif "wax" in s or "candle" in s:
            mat_type = "Wax"
        else:
            mat_type = "Other"
            
        # Geometry
        if "film" in s:
            geom = "Film"
        elif "rod" in s:
            geom = "Rod"
        elif "slab" in s:
            geom = "Slab"
        elif "sibal" in s or "nomex" in s:
            geom = "Fabric"
        elif "candle" in s or "wax" in s:
            geom = "Candle"
        elif "spherical" in s:
            geom = "Spherical"
        else:
            geom = "Other"
            
        # Thickness / Diameter in mm
        thickness = np.nan
        if "micron" in s:
            m = re.search(r"(\d+)\s*micron", s)
            if m:
                thickness = float(m.group(1)) / 1000.0  # micron to mm
        elif "mm" in s:
            m = re.search(r"(\d*\.?\d+)\s*mm", s)
            if m:
                thickness = float(m.group(1))
        elif '1/4"' in s or "1/4" in s:
            thickness = 6.35  # 0.25 inch to mm
        elif '3/8"' in s or "3/8" in s:
            thickness = 9.525 # 0.375 inch to mm
        elif '1/2"' in s or "1/2" in s or "0.5\"" in s:
            thickness = 12.7  # 0.5 inch to mm
            
        # Width in cm
        width = np.nan
        m_w = re.search(r"(\d*\.?\d+)\s*cm", s)
        if m_w:
            width = float(m_w.group(1))
            
        # Color (rods)
        color = "clear" if "clear" in s else ("black" if "black" in s or "blac" in s else "standard")
        
        return pd.Series([mat_type, geom, thickness, width, color], 
                         index=['material_type', 'geometry', 'thickness_mm', 'width_cm', 'color'])
                         
    parsed_mat = df['raw_material'].apply(parse_material)
    df = pd.concat([df, parsed_mat], axis=1)
    
    # Fill defaults for thickness and width
    df['thickness_mm'] = df['thickness_mm'].fillna(df['thickness_mm'].median())
    df['width_cm'] = df['width_cm'].fillna(2.0)
    
    # Clean numeric fields
    df['fan_display'] = df['fan_display_raw'].apply(parse_num_range)
    df['air_display'] = df['air_display_raw'].apply(parse_num_range)
    df['total_frames'] = df['total_frames_raw'].apply(parse_num_range).fillna(0)
    df['o2_initial'] = df['o2_initial_raw'].apply(parse_num_range)
    df['o2_final'] = df['o2_final_raw'].apply(parse_num_range)
    df['co2_initial'] = df['co2_initial_raw'].apply(parse_num_range)
    df['co2_final'] = df['co2_final_raw'].apply(parse_num_range)
    df['co_initial'] = df['co_initial_raw'].apply(parse_num_range).clip(lower=0)
    df['co_final'] = df['co_final_raw'].apply(parse_num_range).clip(lower=0)
    
    # Compute combustion deltas
    df['delta_o2'] = (df['o2_initial'] - df['o2_final']).clip(lower=0)
    df['delta_co2'] = (df['co2_final'] - df['co2_initial']).clip(lower=0)
    df['delta_co'] = (df['co_final'] - df['co_initial']).clip(lower=0)
    
    # Target 1: Fire Extinction / Suppression Event
    # A fire is classified as extinguished/suppressed if delta_o2 < 0.15% or total_frames < 60
    df['extinguished'] = ((df['delta_o2'] < 0.15) | (df['total_frames'] < 60)).astype(int)
    
    # Cleaned output
    cols_to_keep = [
        'test_id', 'pi', 'material_type', 'geometry', 'color', 
        'thickness_mm', 'width_cm', 'flow_restrictor', 'fan_display', 'air_display',
        'radiometer_gain', 'total_frames', 'o2_initial', 'o2_final', 'delta_o2',
        'co2_initial', 'co2_final', 'delta_co2', 'co_initial', 'co_final', 'delta_co',
        'extinguished'
    ]
    df_clean = df[cols_to_keep].copy()
    df_clean.to_csv("cleaned_data/bass_ii_clean.csv", index=False)
    print(f"BASS-II saved with {len(df_clean)} rows.")
    return df_clean

def clean_flex():
    print("Processing FLEX...")
    df = pd.read_csv("PSI-69_Experimental table_FLEX.csv", encoding="cp1252")
    col_map = {
        'FLEX Test #': 'test_id',
        'FLEX Identifier': 'flex_id',
        'Test Date': 'test_date',
        'Test GMT': 'test_gmt',
        'Fuel': 'fuel',
        'Ambient pressure; mmHg': 'pressure_mmHg',
        'O initial ambient composition; mole fraction': 'x_o2',
        'N initial ambient composition; mole fraction': 'x_n2',
        'CO initial ambient composition; mole fraction': 'x_co2',
        'He initial ambient composition; mole fraction': 'x_he',
        'Droplet initial diameter; mm': 'd0_mm_raw',
        'Visible flame extinction diameter; mm': 'd_ext_mm_raw',
        'Burning rate; mm': 'burning_rate_raw',
        'Burn time; s': 'burn_time_s_raw',
        'Test end': 'test_end'
    }
    df = df.rename(columns=col_map)
    
    df['d0_mm'] = df['d0_mm_raw'].apply(parse_num_range)
    df['d_ext_mm'] = df['d_ext_mm_raw'].apply(parse_num_range)
    df['burning_rate'] = df['burning_rate_raw'].apply(parse_num_range)
    df['burn_time_s'] = df['burn_time_s_raw'].apply(parse_num_range)
    
    # Impute missing values with medians by fuel
    df['d0_mm'] = df.groupby('fuel')['d0_mm'].transform(lambda x: x.fillna(x.median()))
    df['burning_rate'] = df.groupby('fuel')['burning_rate'].transform(lambda x: x.fillna(x.median()))
    df['burn_time_s'] = df.groupby('fuel')['burn_time_s'].transform(lambda x: x.fillna(x.median()))
    
    # Convert pressure from mmHg to bar (1 mmHg = 0.00133322 bar)
    df['pressure_bar'] = df['pressure_mmHg'] * 0.00133322
    
    cols = ['test_id', 'fuel', 'pressure_mmHg', 'pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he',
            'd0_mm', 'd_ext_mm', 'burning_rate', 'burn_time_s', 'test_end']
    df_clean = df[cols].copy()
    df_clean.to_csv("cleaned_data/flex_clean.csv", index=False)
    print(f"FLEX saved with {len(df_clean)} rows.")
    return df_clean

def clean_cfig():
    print("Processing CFI-G...")
    df_norm = pd.read_csv("PSI-159_Experimental table_CFI-G_Normal Flames.csv")
    df_inv = pd.read_csv("PSI-159_Experimental table_CFI-G_Inverse Flames.csv").dropna(how='all')
    
    # Normal flames
    df_norm = df_norm.rename(columns={
        'Flame Type': 'flame_type', 'Day': 'day', 'Fuel': 'fuel', 'Test': 'test_id',
        'Cool flame?': 'cool_flame_raw', ' XO2 before test': 'xo2_initial',
        'XO2  after test': 'xo2_final', 'approx p (bar)': 'pressure_bar',
        'Xf': 'xf', 'mHC (mg/s)': 'mhc_mg_s', 'TFP': 'tfp', 'Tad (K)': 'tad_k',
        'Zst': 'zst', 'Hot ignition': 'hot_ignition', 'burn time (sec)': 'burn_time_s',
        'Self-ext or flow term': 'termination_mode', 'Peak burner T (C )': 'peak_burner_t_c',
        'Comments': 'comments'
    })
    df_norm['has_cool_flame'] = df_norm['cool_flame_raw'].apply(lambda x: 1 if pd.notna(x) and 'yes' in str(x).lower() else 0)
    
    # Inverse flames
    df_inv = df_inv.rename(columns={
        'Flame Type': 'flame_type', 'Day': 'day', 'Fuel': 'fuel', 'Test': 'test_id',
        'Xf  before test': 'xf_initial', 'Xf  after test': 'xf_final',
        'approx p (bar)': 'pressure_bar', 'XO2': 'xo2', 'HC burnrate (mg/s)': 'mhc_mg_s',
        'O2 flow rate (mg/s)': 'o2_flow_rate_mg_s', 'TFP': 'tfp', 'Tad (K)': 'tad_k',
        'Zst': 'zst', 'Hot ignition': 'hot_ignition', 'burn time (sec)': 'burn_time_s',
        'Self -ext or flow term': 'termination_mode', 'Peak burner Temp (C)': 'peak_burner_t_c',
        'Peak burner temp (K)': 'peak_burner_t_k', 'Color': 'flame_color', 'Comments': 'comments'
    })
    
    df_norm.to_csv("cleaned_data/cfig_normal_clean.csv", index=False)
    df_inv.to_csv("cleaned_data/cfig_inverse_clean.csv", index=False)
    print(f"CFI-G Normal ({len(df_norm)} rows) & Inverse ({len(df_inv)} rows) saved.")

def clean_spice():
    print("Processing SPICE...")
    df = pd.read_csv("PSI-107_Experimental table_Spice.csv")
    df = df.rename(columns={
        'GMT date': 'gmt_date', 'GMT ignition': 'ignition_time', 'GMT  Flame Off ': 'flame_off_time',
        'Fuel': 'fuel', 'Nozzle (mm)': 'nozzle_mm', 'Fuel Rate (sccm)': 'fuel_rate_sccm',
        'Fuel Rate (mg/s)': 'fuel_rate_mg_s', 'HC Fuel Rate (mg/s)': 'hc_fuel_rate_mg_s',
        'Air Rate (cm/s)': 'air_rate_cm_s'
    })
    df.to_csv("cleaned_data/spice_clean.csv", index=False)
    print(f"SPICE saved with {len(df)} rows.")

if __name__ == "__main__":
    clean_bass_ii()
    clean_flex()
    clean_cfig()
    clean_spice()
    print("All datasets cleaned successfully!")

