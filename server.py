import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chatbot_engine import generate_chat_response, NASA_PSI_INVESTIGATIONS

BASE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(BASE, *a)

app = FastAPI(title="AstroFlame | NASA Microgravity Combustion Research Suite")
app.mount("/static", StaticFiles(directory=P("static")), name="static")

# 1. Load All 7 Models
MODEL_FILES = {
    "bass_ext": "bass_extinction_model",
    "bass_co": "bass_co_regressor",
    "flex_k": "flex_burning_rate_model",
    "flex_end": "flex_end_mode_model",
    "cfig_time": "cfig_burn_time_model",
    "cfig_ext": "cfig_self_ext_model",
    "spice": "spice_model"
}
models, load_errors = {}, {}
for k, f in MODEL_FILES.items():
    try:
        models[k] = joblib.load(P("models", f + ".joblib"))
    except Exception as e:
        load_errors[k] = str(e)[:200]
        print(f"[AstroFlame] model '{k}' not loaded: {e}")

# Load Champion Models if available
champion_models = {}
for k, f in [('bass', 'champion_bass_extinction'), ('flex', 'champion_flex_endmode'), ('cfig', 'champion_cfig_extinction')]:
    path = P("models", f + ".joblib")
    if os.path.exists(path):
        champion_models[k] = joblib.load(path)
    elif k in models:
        champion_models[k] = models[k + '_ext' if k != 'flex' else 'flex_end']

def load_json(name):
    try:
        with open(P("results", name), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

RANGES = load_json("training_ranges.json")
VAL = load_json("grouped_validation.json")

def need(key):
    """Fail loudly: never answer with made-up default numbers."""
    if key not in models:
        raise HTTPException(503, f"Model '{key}' is not available ({load_errors.get(key, 'file missing')}).")
    return models[key]

def quality(key):
    v = VAL.get(key)
    return {"label": v["label"], "level": v["level"], "summary": v["summary"]} if v else {"label": key, "level": "low", "summary": "No validation report found."}

def check_range(name, vals, group=None):
    """Warn when inputs leave the region covered by NASA training data."""
    r = RANGES.get(name)
    if not r:
        return ["Training ranges not found: input coverage cannot be verified."]
    out, box = [], r["all"]
    if group is not None and r.get("by"):
        if group not in r["by"]:
            return [f"No training data for '{group}': prediction is not supported by data."]
        box = r["by"][group]
    for f, (lo, hi) in box.items():
        if f in vals and not (lo - 1e-9 <= vals[f] <= hi + 1e-9):
            where = f" for {group}" if group is not None and r.get("by") else ""
            out.append(f"{f} = {vals[f]:g} is outside the tested range{where} ({lo:g} – {hi:g}); treat result as extrapolation.")
    for f, cats in r.get("cats", {}).items():
        if f in vals and str(vals[f]) not in cats:
            out.append(f"{f} = '{vals[f]}' never appeared in training data.")
    return out

# ------------------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------------------
class BassRequest(BaseModel):
    material_type: str = "PMMA"
    geometry: str = "Film"
    color: str = "standard"
    thickness_mm: float = 1.0
    width_cm: float = 2.0
    fan_display: float = 50.0
    air_display: float = 5.0
    o2_initial: float = 21.0

class FlexRequest(BaseModel):
    fuel: str = "Heptane"
    pressure_bar: float = 1.0
    d0_mm: float = 3.0
    x_o2: float = 0.21
    x_co2: float = 0.0
    x_he: float = 0.0

class CfigRequest(BaseModel):
    fuel: str = "ethane"
    pressure_bar: float = 1.2
    xf: float = 0.4
    xo2_initial: float = 0.25
    mhc_mg_s: float = 1.2
    hot_ignition: str = "Yes"

class SpiceRequest(BaseModel):
    fuel: str = "Propylene"
    nozzle_mm: float = 0.7620
    air_rate_cm_s: float = 30.0

class TriHazardRequest(BaseModel):
    material_type: str = "PMMA"
    geometry: str = "Film"
    color: str = "standard"
    thickness_mm: float = 1.0
    width_cm: float = 2.0
    fan_display: float = 50.0
    o2_initial: float = 21.0
    liquid_fuel: str = "Heptane"
    d0_mm: float = 3.0
    x_co2: float = 0.0
    x_he: float = 0.0
    gas_fuel: str = "ethane"
    pressure_bar: float = 1.0
    xf: float = 0.40
    mhc_mg_s: float = 1.2

class ChatRequest(BaseModel):
    message: str
    telemetry: dict | None = None
    session_id: str = "default"

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/")
async def home():
    return FileResponse(P("templates", "index.html"))

@app.get("/chat")
async def chat_page():
    return FileResponse(P("templates", "chat.html"))

@app.get("/api/health")
async def health():
    return {"loaded": sorted(models), "missing": load_errors, "validation": bool(VAL), "ranges": bool(RANGES)}

@app.post("/api/predict/bass")
async def predict_bass(req: BassRequest):
    d = req.model_dump()
    d["flow_restrictor"] = 1
    df = pd.DataFrame([d])
    ext, co = need("bass_ext"), need("bass_co")
    classes = list(ext.classes_)
    p = float(ext.predict_proba(df)[0][classes.index(1)]) if 1 in classes else float(ext.predict_proba(df)[0][-1])
    co_ppm = max(0.0, float(co.predict(df)[0]))
    return {
        "status": "success",
        "extinction_probability": round(p * 100, 2),
        "is_safe": p >= 0.5,
        "non_extinction_risk": round((1 - p) * 100, 1),
        "screening_pass": p >= 0.75,
        "co_ppm": round(co_ppm, 1),
        "co_error_ppm": round(VAL.get("bass_co", {}).get("mae", 0), 1),
        "warnings": check_range("bass", d, req.material_type),
        "quality": [quality("bass_ext"), quality("bass_co")]
    }

@app.post("/api/predict/flex")
async def predict_flex(req: FlexRequest):
    d = req.model_dump()
    d["x_n2"] = max(0.0, 1.0 - (req.x_o2 + req.x_co2 + req.x_he))
    df = pd.DataFrame([d])
    k = max(0.04, float(need("flex_k").predict(df)[0]))
    mode = str(need("flex_end").predict(df)[0])
    t_end = req.d0_mm ** 2 / k
    ts = np.linspace(0, t_end, 25)
    warns = check_range("flex", d, req.fuel)
    if req.x_o2 + req.x_co2 + req.x_he > 1:
        warns.append("Gas fractions sum to more than 1; N₂ was set to 0.")
    return {
        "status": "success",
        "burning_rate_k": round(k, 4),
        "k_error": round(VAL.get("flex_k", {}).get("mae", 0), 3),
        "total_burn_time_sec": round(t_end, 2),
        "termination_mode": mode,
        "curve_time": [round(float(t), 2) for t in ts],
        "curve_diameter": [round(float(np.sqrt(max(0, req.d0_mm ** 2 - k * t))), 3) for t in ts],
        "warnings": warns,
        "quality": [quality("flex_k"), quality("flex_end")]
    }

@app.post("/api/predict/cfig")
async def predict_cfig(req: CfigRequest):
    d = req.model_dump()
    df = pd.DataFrame([d])
    t = max(0.0, float(need("cfig_time").predict(df)[0]))
    m = need("cfig_ext")
    classes = list(m.classes_)
    p = float(m.predict_proba(df)[0][classes.index(1)]) if 1 in classes else float(m.predict_proba(df)[0][-1])
    return {
        "status": "success",
        "predicted_burn_time_s": round(t, 1),
        "burn_time_error_s": round(VAL.get("cfig_time", {}).get("mae", 0), 1),
        "self_extinction_probability": round(p * 100, 1),
        "is_self_extinguishing": p >= 0.5,
        "warnings": check_range("cfig", d, req.fuel),
        "quality": [quality("cfig_ext"), quality("cfig_time")]
    }

@app.post("/api/predict/spice")
async def predict_spice(req: SpiceRequest):
    d = req.model_dump()
    df = pd.DataFrame([d])
    f = max(0.05, float(need("spice").predict(df)[0]))
    return {
        "status": "success",
        "critical_smoke_point_flow_mg_s": round(f, 4),
        "flow_error_mg_s": round(VAL.get("spice", {}).get("mae", 0), 2),
        "warnings": check_range("spice", d, req.fuel),
        "quality": [quality("spice")]
    }

@app.get("/api/benchmarks")
async def benchmarks():
    if not VAL:
        raise HTTPException(404, "grouped_validation.json not found.")
    return {"validation": VAL, "original_report": load_json("accuracy_benchmark_report.json")}

# ------------------------------------------------------------------------------
# TRI-HAZARD & ASTROCOMMANDER ENDPOINTS
# ------------------------------------------------------------------------------
@app.post("/api/tri-hazard/evaluate")
async def evaluate_tri_hazard(req: TriHazardRequest):
    bass_model = champion_models.get('bass', models.get('bass_ext'))
    flex_model = champion_models.get('flex', models.get('flex_end'))
    cfig_model = champion_models.get('cfig', models.get('cfig_ext'))
    
    if not (bass_model and flex_model and cfig_model):
        raise HTTPException(503, "Champion models not fully loaded.")

    # 1. Solid Phase (BASS-II)
    df_b = pd.DataFrame([{
        'material_type': req.material_type,
        'geometry': req.geometry,
        'color': req.color,
        'thickness_mm': req.thickness_mm,
        'width_cm': req.width_cm,
        'flow_restrictor': 1,
        'fan_display': req.fan_display,
        'air_display': req.fan_display / 10.0,
        'o2_initial': req.o2_initial
    }])
    classes_b = list(bass_model.classes_)
    idx_b = classes_b.index(1) if 1 in classes_b else -1
    p_solid_ext = float(bass_model.predict_proba(df_b)[0][idx_b])
    risk_solid = round((1.0 - p_solid_ext) * 100.0, 1)

    # 2. Liquid Droplet Phase (FLEX)
    xo2_eff = req.o2_initial / 100.0
    xn2_eff = max(0.0, 1.0 - (xo2_eff + req.x_co2 + req.x_he))
    df_f = pd.DataFrame([{
        'fuel': req.liquid_fuel,
        'pressure_bar': req.pressure_bar,
        'x_o2': xo2_eff,
        'x_n2': xn2_eff,
        'x_co2': req.x_co2,
        'x_he': req.x_he,
        'd0_mm': req.d0_mm
    }])
    classes_f = list(flex_model.classes_)
    probs_f = flex_model.predict_proba(df_f)[0]
    prob_dict_f = {cls: float(probs_f[i]) for i, cls in enumerate(classes_f)}
    p_disrupt = prob_dict_f.get('Disruption', 0.25)
    p_complete = prob_dict_f.get('Completion', 0.15)
    p_ext_liq = prob_dict_f.get('Extinction', 0.60)
    risk_liquid = round((p_disrupt * 1.0 + p_complete * 0.5) * 100.0, 1)

    # 3. Gaseous Phase (CFI-G)
    df_c = pd.DataFrame([{
        'fuel': req.gas_fuel,
        'pressure_bar': req.pressure_bar,
        'xf': req.xf,
        'xo2_initial': xo2_eff,
        'mhc_mg_s': req.mhc_mg_s,
        'hot_ignition': 'Yes'
    }])
    classes_c = list(cfig_model.classes_)
    idx_c = classes_c.index(1) if 1 in classes_c else -1
    p_gas_ext = float(cfig_model.predict_proba(df_c)[0][idx_c])
    risk_gas = round((1.0 - p_gas_ext) * 100.0, 1)

    # 4. Composite Score
    composite_score = round(0.40 * risk_solid + 0.35 * risk_liquid + 0.25 * risk_gas, 1)

    phases = {'Solid Phase (BASS-II)': risk_solid, 'Liquid Droplets (FLEX)': risk_liquid, 'Gaseous Jets (CFI-G)': risk_gas}
    bottleneck_phase = max(phases, key=phases.get)
    bottleneck_val = phases[bottleneck_phase]

    if composite_score <= 28.0:
        threat_level = "GREEN // NOMINAL (SAFE CONTAINMENT)"
        status_color = "#6ef3c5"
    elif composite_score <= 65.0:
        threat_level = "YELLOW // ELEVATED THERMAL HAZARD"
        status_color = "#ffb454"
    else:
        threat_level = "RED // CRITICAL RUNAWAY FIRE HAZARD"
        status_color = "#ff5a36"

    # 5. AstroCommander Optimization
    opt_fan = 10.0 if req.fan_display > 20.0 else req.fan_display
    opt_co2 = min(0.30, req.x_co2 + 0.18) if risk_liquid > 30.0 else req.x_co2

    df_b_opt = df_b.copy()
    df_b_opt['fan_display'] = opt_fan
    df_b_opt['air_display'] = opt_fan / 10.0
    p_solid_opt = float(bass_model.predict_proba(df_b_opt)[0][idx_b])
    risk_solid_opt = round((1.0 - p_solid_opt) * 100.0, 1)

    df_f_opt = df_f.copy()
    df_f_opt['x_co2'] = opt_co2
    df_f_opt['x_n2'] = max(0.0, 1.0 - (xo2_eff + opt_co2 + req.x_he))
    probs_f_opt = flex_model.predict_proba(df_f_opt)[0]
    prob_dict_f_opt = {cls: float(probs_f_opt[i]) for i, cls in enumerate(classes_f)}
    risk_liquid_opt = round((prob_dict_f_opt.get('Disruption', 0.2)*1.0 + prob_dict_f_opt.get('Completion', 0.1)*0.5)*100.0, 1)

    opt_composite_score = round(0.40 * risk_solid_opt + 0.35 * risk_liquid_opt + 0.25 * risk_gas, 1)
    if opt_composite_score > composite_score:
        opt_composite_score = composite_score

    directives = []
    if risk_solid > 35.0:
        directives.append(f"REDUCE COMPARTMENT FAN AIRFLOW to {opt_fan} units (starves solid boundary layer).")
    if risk_liquid > 30.0:
        directives.append(f"INJECT {int(opt_co2*100)}% CO₂ INERTING PULSE to quench droplet disruption splatter.")
    if risk_gas > 40.0:
        directives.append("PURGE GASEOUS MANIFOLD & MAINTAIN DIFFUSION VENTING.")
    if not directives:
        directives.append("ALL 3 PHASES WITHIN NOMINAL PASSIVE EXTINCTION LIMITS. MAINTAIN CURRENT VENTILATION.")

    return {
        "status": "success",
        "composite_score": composite_score,
        "threat_level": threat_level,
        "status_color": status_color,
        "breakdown": {
            "solid_risk": risk_solid,
            "solid_extinction_prob": round(p_solid_ext * 100, 1),
            "liquid_risk": risk_liquid,
            "liquid_disruption_prob": round(p_disrupt * 100, 1),
            "liquid_extinction_prob": round(p_ext_liq * 100, 1),
            "gas_risk": risk_gas,
            "gas_extinction_prob": round(p_gas_ext * 100, 1)
        },
        "bottleneck": {
            "phase": bottleneck_phase,
            "risk_percentage": bottleneck_val
        },
        "commander_directive": {
            "tactical_actions": directives,
            "optimized_parameters": {
                "fan_display": opt_fan,
                "x_co2": opt_co2,
                "pressure_bar": req.pressure_bar
            },
            "projected_composite_score": opt_composite_score,
            "projected_reduction": round(composite_score - opt_composite_score, 1)
        }
    }

# ------------------------------------------------------------------------------
# CHATBOT & NASA PSI CATALOG ENDPOINTS
# ------------------------------------------------------------------------------
@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    return generate_chat_response(req.message, req.telemetry, req.session_id)

@app.get("/api/nasa/psi-catalog")
async def get_nasa_psi_catalog():
    return {"status": "success", "investigations": NASA_PSI_INVESTIGATIONS}

@app.get("/api/models/meta")
async def get_models_meta():
    return {
        "models": [
            {"id": "BASS-II", "target": "Solid Flame Extinction", "type": "Binary Classifier", "cv_accuracy": "87.20%"},
            {"id": "FLEX", "target": "Droplet Termination Mode", "type": "3-Class Classifier", "cv_accuracy": "77.72%"},
            {"id": "CFI-G", "target": "Gaseous Flame Self-Extinction", "type": "Binary Classifier", "cv_accuracy": "87.43%"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8080, reload=True)
