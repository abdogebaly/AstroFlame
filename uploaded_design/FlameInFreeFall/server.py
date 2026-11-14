import os, json, joblib
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(BASE, *a)
app = FastAPI(title="AstroFlame | NASA Microgravity Combustion Research Suite")
app.mount("/static", StaticFiles(directory=P("static")), name="static")

MODEL_FILES = {"bass_ext": "bass_extinction_model", "bass_co": "bass_co_regressor", "flex_k": "flex_burning_rate_model",
               "flex_end": "flex_end_mode_model", "cfig_time": "cfig_burn_time_model", "cfig_ext": "cfig_self_ext_model", "spice": "spice_model"}
models, load_errors = {}, {}
for k, f in MODEL_FILES.items():
    try: models[k] = joblib.load(P("models", f + ".joblib"))
    except Exception as e: load_errors[k] = str(e)[:200]; print(f"[AstroFlame] model '{k}' not loaded: {e}")

def load_json(name):
    try:
        with open(P("results", name), encoding="utf-8") as fh: return json.load(fh)
    except Exception: return {}
RANGES, VAL = load_json("training_ranges.json"), load_json("grouped_validation.json")  # made by evaluate_grouped.py

def need(key):
    """Fail loudly: never answer with made-up default numbers."""
    if key not in models:
        raise HTTPException(503, f"Model '{key}' is not available ({load_errors.get(key, 'file missing')}). Run train_advanced_models.py.")
    return models[key]

def quality(key):
    v = VAL.get(key)
    return {"label": v["label"], "level": v["level"], "summary": v["summary"]} if v else {"label": key, "level": "low", "summary": "No validation report found (run evaluate_grouped.py)."}

def check_range(name, vals, group=None):
    """Warn when inputs leave the region covered by the NASA training data (tree models cannot extrapolate)."""
    r = RANGES.get(name)
    if not r: return ["Training ranges not found: input coverage cannot be verified."]
    out, box = [], r["all"]
    if group is not None and r.get("by"):
        if group not in r["by"]: return [f"No training data for '{group}': prediction is not supported by data."]
        box = r["by"][group]
    for f, (lo, hi) in box.items():
        if f in vals and not (lo - 1e-9 <= vals[f] <= hi + 1e-9):
            where = f" for {group}" if group is not None and r.get("by") else ""
            out.append(f"{f} = {vals[f]:g} is outside the tested range{where} ({lo:g} – {hi:g}); treat the result as an extrapolation.")
    for f, cats in r.get("cats", {}).items():
        if f in vals and str(vals[f]) not in cats: out.append(f"{f} = '{vals[f]}' never appeared in the training data.")
    return out

class BassRequest(BaseModel):
    material_type: str = "PMMA"; geometry: str = "Film"; color: str = "standard"; thickness_mm: float = 1.0
    width_cm: float = 2.0; fan_display: float = 50.0; air_display: float = 5.0; o2_initial: float = 21.0
class FlexRequest(BaseModel):
    fuel: str = "Heptane"; pressure_bar: float = 1.0; d0_mm: float = 3.0; x_o2: float = 0.21; x_co2: float = 0.0; x_he: float = 0.0
class CfigRequest(BaseModel):
    fuel: str = "ethane"; pressure_bar: float = 1.2; xf: float = 0.4; xo2_initial: float = 0.25; mhc_mg_s: float = 1.2; hot_ignition: str = "Yes"
class SpiceRequest(BaseModel):
    fuel: str = "Propylene"; nozzle_mm: float = 0.7620; air_rate_cm_s: float = 30.0

@app.get("/")
async def home(): return FileResponse(P("templates", "index.html"))

@app.get("/api/health")
async def health(): return {"loaded": sorted(models), "missing": load_errors, "validation": bool(VAL), "ranges": bool(RANGES)}

@app.post("/api/predict/bass")
async def predict_bass(req: BassRequest):
    d = req.model_dump(); d["flow_restrictor"] = 1
    df = pd.DataFrame([d])
    ext, co = need("bass_ext"), need("bass_co")
    classes = list(ext.classes_)
    p = float(ext.predict_proba(df)[0][classes.index(1)]) if 1 in classes else float(ext.predict_proba(df)[0][-1])
    co_ppm = max(0.0, float(co.predict(df)[0]))
    return {"status": "success", "extinction_probability": round(p * 100, 2), "is_safe": p >= 0.5,
            "non_extinction_risk": round((1 - p) * 100, 1),            # = 100 - P(extinction); not an independent physical index
            "screening_pass": p >= 0.75,                                # research screening rule; CO is NOT part of it (low reliability)
            "co_ppm": round(co_ppm, 1), "co_error_ppm": round(VAL.get("bass_co", {}).get("mae", 0), 1),
            "warnings": check_range("bass", d, req.material_type),
            "quality": [quality("bass_ext"), quality("bass_co")]}

@app.post("/api/predict/flex")
async def predict_flex(req: FlexRequest):
    d = req.model_dump(); d["x_n2"] = max(0.0, 1.0 - (req.x_o2 + req.x_co2 + req.x_he))
    df = pd.DataFrame([d])
    k = max(0.04, float(need("flex_k").predict(df)[0]))
    mode = str(need("flex_end").predict(df)[0])
    t_end = req.d0_mm ** 2 / k
    ts = np.linspace(0, t_end, 25)
    warns = check_range("flex", d, req.fuel)
    if req.x_o2 + req.x_co2 + req.x_he > 1: warns.append("Gas fractions sum to more than 1; N₂ was set to 0.")
    return {"status": "success", "burning_rate_k": round(k, 4), "k_error": round(VAL.get("flex_k", {}).get("mae", 0), 3),
            "total_burn_time_sec": round(t_end, 2), "termination_mode": mode,
            "curve_time": [round(float(t), 2) for t in ts],
            "curve_diameter": [round(float(np.sqrt(max(0, req.d0_mm ** 2 - k * t))), 3) for t in ts],  # idealised d²-law
            "warnings": warns, "quality": [quality("flex_k"), quality("flex_end")]}

@app.post("/api/predict/cfig")
async def predict_cfig(req: CfigRequest):
    d = req.model_dump(); df = pd.DataFrame([d])
    t = max(0.0, float(need("cfig_time").predict(df)[0]))
    m = need("cfig_ext"); classes = list(m.classes_)
    p = float(m.predict_proba(df)[0][classes.index(1)]) if 1 in classes else float(m.predict_proba(df)[0][-1])
    return {"status": "success", "predicted_burn_time_s": round(t, 1), "burn_time_error_s": round(VAL.get("cfig_time", {}).get("mae", 0), 1),
            "self_extinction_probability": round(p * 100, 1), "is_self_extinguishing": p >= 0.5,
            "warnings": check_range("cfig", d, req.fuel), "quality": [quality("cfig_ext"), quality("cfig_time")]}

@app.post("/api/predict/spice")
async def predict_spice(req: SpiceRequest):
    d = req.model_dump(); df = pd.DataFrame([d])
    f = max(0.05, float(need("spice").predict(df)[0]))
    return {"status": "success", "critical_smoke_point_flow_mg_s": round(f, 4), "flow_error_mg_s": round(VAL.get("spice", {}).get("mae", 0), 2),
            "warnings": check_range("spice", d, req.fuel), "quality": [quality("spice")]}

@app.get("/api/benchmarks")
async def benchmarks():
    if not VAL: raise HTTPException(404, "grouped_validation.json not found. Run evaluate_grouped.py.")
    return {"validation": VAL, "original_report": load_json("accuracy_benchmark_report.json")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
