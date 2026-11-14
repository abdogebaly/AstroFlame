"""Honest validation + training-range extraction.  Run:  python evaluate_grouped.py
Random-split CV lets repeated runs of the SAME experimental setting land in both train and test (leakage).
Grouped CV holds out whole settings, which measures generalisation to settings the model never saw."""
import json, warnings, joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
warnings.filterwarnings("ignore")
D = lambda n: pd.read_csv(f"cleaned_data/{n}_clean.csv")
bass, flex, spice = D("bass_ii"), D("flex"), D("spice").dropna(subset=["hc_fuel_rate_mg_s", "air_rate_cm_s", "nozzle_mm"])
cfig = D("cfig_normal").dropna(subset=["burn_time_s", "peak_burner_t_c"])
BF = ["material_type", "geometry", "color", "thickness_mm", "width_cm", "flow_restrictor", "fan_display", "air_display", "o2_initial"]
FF = ["fuel", "pressure_bar", "x_o2", "x_n2", "x_co2", "x_he", "d0_mm"]
CF = ["fuel", "pressure_bar", "xf", "xo2_initial", "mhc_mg_s", "hot_ignition"]
SF = ["fuel", "nozzle_mm", "air_rate_cm_s"]
SPECS = [
 ("bass_ext", "BASS-II extinction", "bass_extinction_model", bass, BF, bass["extinguished"], "clf", ""),
 ("bass_co", "BASS-II CO emission", "bass_co_regressor", bass, BF, bass["delta_co"].fillna(0), "reg", "ppm"),
 ("flex_k", "FLEX burning rate", "flex_burning_rate_model", flex, FF, flex["burning_rate"], "reg", "mm²/s"),
 ("flex_end", "FLEX end mode", "flex_end_mode_model", flex, FF, flex["test_end"], "clf", ""),
 ("cfig_ext", "CFI-G self-extinction", "cfig_self_ext_model", cfig, CF, (cfig["termination_mode"] == "SE").astype(int), "clf", ""),
 ("cfig_time", "CFI-G burn time", "cfig_burn_time_model", cfig, CF, cfig["burn_time_s"], "reg", "s"),
 ("spice", "SPICE smoke point", "spice_model", spice, SF, spice["hc_fuel_rate_mg_s"], "reg", "mg/s")]

def level(task, g, base, r2=None):
    if task == "clf": return "high" if g >= .85 and g - base >= .15 else "medium" if g >= .75 and g - base >= .05 else "low"
    return "high" if r2 >= .6 else "medium" if r2 >= .3 else "low"

out = {}
for key, label, mf, df, feats, y, task, unit in SPECS:
    m = y.notna().values; X, y = df.loc[m, feats], y[m]
    if task == "clf" and (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)): y = y.astype(int)
    model = joblib.load(f"models/{mf}.joblib")
    groups = pd.factorize(X.apply(lambda r: "|".join(map(str, r.values)), axis=1))[0]
    res = {"label": label, "task": task, "unit": unit, "n": int(len(y)), "settings": int(groups.max() + 1)}
    preds = {"random": cross_val_predict(clone(model), X, y, cv=KFold(5, shuffle=True, random_state=42)),
             "grouped": cross_val_predict(clone(model), X, y, cv=GroupKFold(5), groups=groups)}
    if task == "clf":
        res["baseline"] = float(y.value_counts(normalize=True).iloc[0])
        for k, p in preds.items(): res[k] = float(accuracy_score(y, p))
        res["metric"], res["level"] = "Accuracy", level(task, res["grouped"], res["baseline"])
        res["summary"] = f"Accuracy on unseen settings {res['grouped']:.0%} (always guessing the majority class: {res['baseline']:.0%})."
    else:
        res["baseline"] = float(mean_absolute_error(y, np.full(len(y), np.median(y))))
        for k, p in preds.items(): res[k] = float(r2_score(y, p))
        res["mae"] = float(mean_absolute_error(y, preds["grouped"]))
        res["metric"], res["level"] = "R²", level(task, 0, 0, res["grouped"])
        res["summary"] = f"R² on unseen settings {res['grouped']:.2f}; typical error ±{res['mae']:.3g} {unit}."
    out[key] = res
    print(f"{key:9s} random={res['random']:.3f} grouped={res['grouped']:.3f} -> {res['level']}")
json.dump(out, open("results/grouped_validation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

def ranges(df, feats, by=None):
    num = [f for f in feats if pd.api.types.is_numeric_dtype(df[f]) and f != "flow_restrictor"]
    mm = lambda d: {f: [float(d[f].min()), float(d[f].max())] for f in num if d[f].notna().any()}
    r = {"all": mm(df), "cats": {f: sorted(map(str, df[f].dropna().unique())) for f in feats if f not in num and f != "flow_restrictor"}}
    if by: r["by_col"], r["by"] = by, {str(k): mm(g) for k, g in df.groupby(by)}
    return r
json.dump({"bass": ranges(bass, BF, "material_type"), "flex": ranges(flex, FF, "fuel"), "cfig": ranges(cfig, CF, "fuel"), "spice": ranges(spice, SF, "fuel")},
          open("results/training_ranges.json", "w", encoding="utf-8"), indent=1)
print("saved results/grouped_validation.json and results/training_ranges.json")
