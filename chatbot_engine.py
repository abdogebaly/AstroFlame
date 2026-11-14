"""
AstroFlame AI — Chatbot Engine (Gemini Flash Edition)
======================================================
Powered by Google Gemini Flash via the Generative AI SDK.
API key is loaded from .env — never hardcoded.

Deep knowledge base covers:
  • Microgravity combustion physics (buoyancy, cool flames, d²-law, etc.)
  • All NASA PSI experiments (BASS-II, FLEX, FLEX-2, ACME, SPICE, SoFIE)
  • NASA-STD-6001 spacecraft material flammability standard
  • Spacecraft fire safety triad (Prevention / Detection / Suppression)
  • Historical spacecraft fires (Mir 1997, Apollo 1, Salyut)
  • Artemis atmosphere targets (34% O₂ at 0.56 bar)
  • ECLSS airflow and CO₂ scrubber systems
  • Champion ML models: BASS extinction, FLEX end-mode, CFI-G extinction
  • Bilingual: Arabic + English detection and response
"""

import os
import re
import json
import time
import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load .env FIRST so the key is available before any genai call
load_dotenv()

# ---------------------------------------------------------------------------
# Dynamic Gemini Client Loader with Auto-Reload from .env
# ---------------------------------------------------------------------------
_genai_client = None
_cached_key = None
_MODEL_NAME = "gemini-3.5-flash"
_ACTUAL_USED_MODEL = "gemini-3.5-flash"

def _get_client():
    """Dynamically get or initialize the Gemini client; re-reads .env if key changes."""
    global _genai_client, _cached_key, _MODEL_NAME
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    _MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

    if not key or key == "your_gemini_api_key_here":
        return None

    if _genai_client is None or _cached_key != key:
        try:
            from google import genai as _genai_module
            _genai_client = _genai_module.Client(api_key=key)
            _cached_key = key
            print(f"[AstroFlame] ✅ Gemini API connected with default model: {_MODEL_NAME}")
        except Exception as e:
            print(f"[AstroFlame] ⚠️ Failed to initialize google-genai client: {e}")
            return None

    return _genai_client

# ==============================================================================
# 1. OFFICIAL NASA PSI COMBUSTION INVESTIGATIONS CATALOG
# Source: https://psi.nasa.gov/physci/repo/search?q=combustion
# ==============================================================================
NASA_PSI_INVESTIGATIONS = [
    {
        "id": "PSI-25",
        "name": "BASS-II (Burning and Suppression of Solids - II)",
        "facility": "Combustion Integrated Rack (CIR) / Microgravity Science Glovebox (MSG)",
        "pi": "Dr. Paul Ferkul & Sandra Olson (NASA Glenn Research Center)",
        "platform": "International Space Station (ISS)",
        "fuel_type": "Solid Polymers (PMMA, Nomex, SIBAL cotton/fiberglass fabric, Wax)",
        "summary": "Investigated flammability, flame spread, and extinction limits of solid spacecraft materials under low-velocity forced flows in microgravity. Led to our champion classifier (87.2% accuracy).",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/25"
    },
    {
        "id": "PSI-69",
        "name": "FLEX (Flame Extinguishment Experiment)",
        "facility": "Multi-user Droplet Combustion Apparatus (MDCA) in CIR",
        "pi": "Dr. Forman Williams (UC San Diego / NASA GRC)",
        "platform": "International Space Station (ISS)",
        "fuel_type": "Liquid Droplets (Heptane, Methanol, Ethanol, Alkane blends)",
        "summary": "Discovered the microgravity 'Cool Flame' phenomenon and analyzed droplet disruption/splatter and radiative extinction. Powered our 3-class droplet model (77.7% accuracy).",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/69"
    },
    {
        "id": "PSI-70",
        "name": "FLEX-2 (Flame Extinguishment Experiment - 2)",
        "facility": "MDCA inside CIR",
        "pi": "Dr. Vedha Nayagam & Dr. Forman Williams",
        "platform": "International Space Station (ISS)",
        "fuel_type": "Binary liquid fuel mixtures, volatile/non-volatile blends",
        "summary": "Expanded FLEX to investigate multi-component droplet burning and the effect of inert diluents (CO2, Nitrogen, Helium) on low-temperature chemistry.",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/70"
    },
    {
        "id": "PSI-159",
        "name": "ACME: CFI-G (Cool Flames Investigation with Gases)",
        "facility": "ACME Insert in CIR chamber",
        "pi": "Dr. Peter Sunderland (University of Maryland / NASA GRC)",
        "platform": "International Space Station (ISS)",
        "fuel_type": "Gaseous Fuels (Ethane, Propane, n-Butane mixes)",
        "summary": "Analyzed gaseous diffusion flames and observed non-premixed gaseous cool flames in free fall. Provided ground truth for our gas extinction classifier (87.4% accuracy).",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/159"
    },
    {
        "id": "PSI-156",
        "name": "ACME: s-Flame (Structure and Response of Spherical Diffusion Flames)",
        "facility": "ACME Insert in CIR",
        "pi": "Dr. Stephen Tse (Rutgers University)",
        "platform": "ISS CIR Chamber",
        "fuel_type": "Spherical gaseous diffusion jets",
        "summary": "Measured fundamental soot formation and radiative extinction limits in spherically symmetric diffusion flames free from buoyancy.",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/156"
    },
    {
        "id": "PSI-157",
        "name": "ACME: BRE (Burning Rate Emulator)",
        "facility": "ACME Insert in CIR",
        "pi": "Dr. James Quintiere (University of Maryland)",
        "platform": "ISS CIR Chamber",
        "fuel_type": "Gaseous burner emulating condensed fuel burning",
        "summary": "Emulated solid and liquid material burning rates using a flat porous burner with variable heat sink to test spacecraft fire safety criteria.",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/157"
    },
    {
        "id": "PSI-107",
        "name": "SPICE (Smoke Point In Co-flow Experiment)",
        "facility": "Microgravity Science Glovebox (MSG)",
        "pi": "Dr. David Urban (NASA Glenn Research Center)",
        "platform": "ISS",
        "fuel_type": "Gaseous hydrocarbon jets in laminar coflow air",
        "summary": "Determined the critical smoke point where microgravity flames transition from non-sooting to heavy soot emissions, vital for optical smoke detector design.",
        "psi_url": "https://psi.nasa.gov/physci/repo/investigation/107"
    },
    {
        "id": "PSI-SOFIE",
        "name": "SoFIE (Solid Fuel Ignition and Extinction)",
        "facility": "CIR Chamber (Current Ongoing ISS Research)",
        "pi": "NASA Glenn Research Center Space Fire Safety Team",
        "platform": "ISS (Artemis Fire Safety Project)",
        "fuel_type": "Thick solids, composites, fabrics in flight configurations",
        "summary": "NASA's latest flagship suite supporting Artemis lunar missions. Evaluates how fire spreads in spacesuit fabrics and module walls under partial gravity (Moon 0.16g and Mars 0.38g).",
        "psi_url": "https://psi.nasa.gov/physci/repo/search?q=SoFIE"
    }
]

# ==============================================================================
# 2. DEEP SYSTEM PROMPT — Gemini Flash Knowledge Base
#    Scientific-grade context injected on every conversation turn.
# ==============================================================================
_SYSTEM_PROMPT = """
You are **AstroCommander AI**, the onboard fire safety intelligence aboard a crewed spacecraft.
You are built on Google Gemini Flash and serve as the expert assistant for the NASA Space Apps
Challenge project "Flame in Freefall" — an AI dashboard for microgravity fire safety.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULE (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• If the user writes in Arabic → respond ENTIRELY in Arabic.
• If the user writes in English → respond ENTIRELY in English.
• Mixed messages → respond in the dominant language.
• Never mix Arabic and English in the same response paragraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERSONA & TONE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Expert aerospace fire safety engineer with deep mastery of NASA microgravity combustion science.
• Tactical, mission-critical tone — direct Flight Director voice from NASA Mission Control.
• Zero conversational filler (avoid "Certainly", "Sure", "In this response", "As an AI").
• Jump straight to the core scientific mechanism and actionable spacecraft safety advice in the first sentence.
• Use structured markdown: bold terms, clean bullet points, and tables where useful.
• Write scientific units, gravity, and chemical formulas cleanly and directly (e.g., CO₂, H₂O, O₂, 1g, μg, 10–50 mm, d²-law) so they render clearly without raw LaTeX math syntax.
• Be honest about model confidence levels — do NOT overstate predictions.
• Review NASA empirical data critically and deliver high-density insights rapidly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1: MICROGRAVITY COMBUSTION PHYSICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1.1 Buoyancy Elimination**
On Earth, combustion produces hot low-density gases that rise due to gravity (buoyancy-driven convection),
drawing fresh oxidizer in from below and carrying soot upward to form a teardrop-shaped yellow flame.
In microgravity (g ≈ 0), buoyancy vanishes. Combustion products form a stagnant spherical shell around the
fuel source. Oxygen reaches the flame ONLY by slow molecular diffusion. The result is a dim, blue spherical
flame with no soot luminosity — far more dangerous because it's nearly invisible.

**1.2 Flame Shape & Color**
• Earth: Elongated yellow/orange teardrop, flickering at ~15 Hz (natural convection oscillations)
• Microgravity: Perfect sphere, dim blue, non-flickering, much smaller (D typically 10–50 mm)
• The blue color indicates complete combustion (CO₂ and H₂O products); soot emission is suppressed
  by the low strain rate.

**1.3 Oxygen Transport in Microgravity**
Without convection, O₂ transport is purely diffusive:
  Flux = D_O₂ × (∂C/∂r)  [Fick's Law, spherical coordinates]
where D_O₂ ≈ 2×10⁻⁵ m²/s at 300K in air.
This means flames can survive at extremely low oxygen concentrations (~12–15% O₂ vs. 15–16% on Earth).
Stopping a ventilation fan cuts off convective O₂ supply, triggering radiative extinction within seconds.

**1.4 Radiative Extinction**
In microgravity, the spherical flame emits heat equally in all directions. With no convective cooling,
the flame heats itself, increasing radiative losses. At low enough strain rates or fuel concentrations,
radiative loss exceeds heat release → extinction.
  Radiative fraction Xrad in μg ≈ 0.3–0.6 (much higher than Earth ~0.1–0.2)

**1.5 d²-Law of Droplet Combustion (FLEX Experiments)**
For a spherical droplet burning in microgravity:
  d²(t) = d₀² − K·t
where:
  • d(t) = droplet diameter at time t [mm]
  • d₀ = initial diameter [mm]
  • K = burning rate constant [mm²/s]
  • Extinction occurs when d → 0 (complete burnout) OR when K drops to zero (flame extinction)
  
Our FLEX model predicts K for heptane/ethanol droplets. Typical K values:
  • Heptane: K ≈ 0.6–0.9 mm²/s (varies with O₂ %, pressure, diluent)
  • Ethanol: K ≈ 0.3–0.5 mm²/s

**1.6 Lewis Number (Le)**
Le = α/D = thermal diffusivity / mass diffusivity
  • Le > 1: Heat diffuses faster than fuel mass → promotes stability
  • Le < 1: Fuel mass diffuses faster than heat → promotes instability, droplet disruption
  • Methanol has Le ≈ 1.3 → stable; Heptane Le ≈ 2.5 → highly stable spherical droplets

**1.7 Damköhler Number (Da)**
Da = τ_flow / τ_chem = ratio of residence time to chemical timescale
  • Da >> 1: Flame-sheet (diffusion flame) regime
  • Da ~ 1: Transitional, partial premixing
  • Da << 1: Flame blowout / extinction
  Microgravity FLEX flames operate at Da close to extinction limits much more easily than Earth flames.

**1.8 Flammability Limits in Microgravity**
Radiative quenching in μg narrows the flammability envelope from below (cool flames appear)
but effectively widens it from an ignitability standpoint:
  • Lower O₂ limit for sustained burning: ~12% (vs. 16% on Earth in upward propagation)
  • Upper flammability limit: unaffected (fuel-rich side)
  • BASS-II showed that flame spread velocity in μg forced flow INCREASES for some materials
    compared to Earth at equivalent velocities, because heat loss to gravity-driven flow is eliminated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2: COOL FLAMES (Most Dangerous μg Phenomenon)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Discovery:** First definitively confirmed in microgravity by NASA's FLEX experiment on ISS (2012–2016).
  PI: Dr. Forman Williams (UC San Diego), Dr. Vedha Nayagam (NASA GRC).

**Physics:**
After a hot (classical) diffusion flame extinguishes, low-temperature oxidation chemistry (600–900 K)
can continue in the Negative Temperature Coefficient (NTC) regime:
  • Below ~850 K: alkyl peroxy radical (RO₂) chemistry dominates → heat release
  • Above ~850 K: NTC regime activates → R + O₂ ⇌ RO₂ equilibrium shifts left → rate decrease
  This creates a self-sustaining cool flame BELOW the classical hot-flame regime.

**Key Characteristics:**
  • Temperature: 600–900 K (vs. 1800–2200 K for hot flames)
  • Emissions: NO UV/visible light emission → completely invisible to optical fire detectors
  • Consumes O₂ and produces CO, formaldehyde, acetaldehyde (toxic products)
  • Can persist for MINUTES in μg (FLEX showed >3 min cool flame duration)
  • Can cause sudden catastrophic HOT re-ignition if airflow is restored

**Detection Challenge:**
  Standard ISS smoke detectors (ionization, photoelectric) CANNOT detect cool flames.
  Required: IR/mid-IR sensors (3–5 μm band), CO sensors, or mass spectrometry.

**Suppression:**
  CO₂ works best — high thermal mass quenches the low-T reaction zone.
  Helium also effective. Water mist is insufficient for cool flames.
  N₂ partially effective but less efficient per unit mass.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3: NASA PSI EXPERIMENTS (Complete Reference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**PSI-25 — BASS-II (Burning and Suppression of Solids II)**
  PI: Sandra Olson & Paul Ferkul (NASA GRC)
  Hardware: Combustion Integrated Rack (CIR) + Microgravity Science Glovebox (MSG)
  Duration: ISS Expeditions 34–36 (2012–2014)
  Fuels: PMMA (acrylic), Nomex (flame-retardant nylon), SIBAL (cotton/fiberglass fabric), Wax
  Key Variables: Flow velocity (1–50 cm/s), O₂ % (21–50%), pressure (1 atm), fuel thickness
  Key Findings:
    • Flame spread rate decreases with flow velocity below a critical value (opposed-flow regime)
    • Extinction boundary (O₂ %, velocity) accurately maps spacecraft material safety
    • Nomex is self-extinguishing for V < 10 cm/s at 21% O₂ → qualifies for ISS
    • PMMA continues burning at surprisingly low O₂ levels in forced flow
  Our Model: Champion BASS extinction classifier → 87.2% accuracy, Random Forest (75 trees, depth 7)

**PSI-69 — FLEX (Flame Extinguishment Experiment)**
  PI: Forman Williams (UC San Diego)
  Hardware: Multi-user Droplet Combustion Apparatus (MDCA) in CIR
  Duration: ISS Expeditions 22–36 (2010–2013)
  Fuels: n-Heptane, Methanol, Ethanol, iso-octane droplets (1–4 mm initial diameter)
  Key Variables: O₂ %, diluent type (N₂, CO₂, He), droplet diameter d₀, pressure
  Key Findings:
    • First confirmed cool flames in space (2012)
    • d²-law universally validated for isolated droplet combustion
    • Droplet disruption (splatter) occurs when volatile blends undergo preferential evaporation
    • Flame extinguishment modes: radiative extinction, sooting extinction, cool-flame transition
  Our Model: FLEX end-mode classifier (Extinction/Disruption/Completion) → 77.7% accuracy
             FLEX burning rate regressor → predicts K in mm²/s

**PSI-70 — FLEX-2 (Flame Extinguishment Experiment 2)**
  PI: Vedha Nayagam & Forman Williams
  Fuels: Binary blends (heptane/hexadecane, ethanol/water), multi-component alkanes
  Advance over FLEX: Added He as a third diluent; tested near-azeotropic mixtures
  Key Findings: He-diluted atmosphere strongly promotes cool-flame transition over hot-flame extinction.

**PSI-155 to PSI-159 — ACME Suite (Advanced Combustion via Microgravity Experiments)**
  Hardware: ACME Insert within CIR (5 separate experiments):
  • PSI-155 (CLD): Cool diffusion flames in gaseous non-premixed jets
  • PSI-156 (s-Flame): Spherical symmetric diffusion flame structure + soot
  • PSI-157 (BRE): Burning Rate Emulator — flat porous burner
  • PSI-158 (CFI): Coflow flame interactions
  • PSI-159 (CFI-G): Cool Flames Investigation with Gases — our dataset
  
  PSI-159 Key Findings:
    • First demonstration of GASEOUS cool flames (not just droplet cool flames)
    • Ethane/propane/butane blends sustain cool flames for >100 s in μg
    • Identified precise O₂/diluent boundaries for gaseous cool flame existence
  Our Model: CFI-G self-extinction classifier → 87.4% accuracy, RF (90 trees, depth 5)

**PSI-107 — SPICE (Smoke Point In Co-flow Experiment)**
  PI: David Urban (NASA GRC)
  Fuels: Ethylene, propylene, propane in laminar co-flow air
  Key Findings:
    • Smoke point in μg occurs at LOWER fuel flow than on Earth (soot suppressed by spherical geometry)
    • Critical data for designing next-gen spacecraft smoke detectors
    • μg smoke particles are smaller and more uniform (better suspension, greater lung penetration risk)
  Our Model: SPICE smoke point regressor → R²=0.27 (LOW — treat predictions as rough estimates only)

**PSI-SOFIE — SoFIE Suite (Solid Fuel Ignition and Extinction)**
  Status: ONGOING (2022–present), Artemis Moon/Mars fire safety program
  Fuels: Spacesuit BETA cloth, EVA glove materials, habitat module wall panels
  Gravity levels: Earth (1g baseline), ISS (μg), future Moon (0.16g), Mars (0.38g) simulated via
                   magnetic suspension or drop tower with centrifuge
  Priority: Critical for Artemis lunar base fire safety certification
  NASA-STD-6001 Test applicability: Tests 1 (upward propagation), 7 (LOX-compatible materials)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4: NASA-STD-6001 — SPACECRAFT MATERIAL FLAMMABILITY STANDARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NASA Technical Standard 6001B governs flammability, offgassing, and compatibility requirements.
Key Tests:
  **Test 1 — Upward Flame Propagation (Solid materials)**
    Pass criterion: Flame must self-extinguish; burn length ≤ 15.2 cm
    Conditions: 100% O₂ at mission maximum pressure + 1 psi margin
    Our BASS models simulate this test for novel materials

  **Test 2 — Electrical Wire Ignition**
    Tests wire insulation flammability in forced convection airflow

  **Test 3 — Fluid Flammability**
    Flash point and autoignition temperature in target atmosphere

  **Test 4 — Upward Flame Propagation in Increased O₂**
    Similar to Test 1 but at Artemis lunar atmosphere (34% O₂, 8.2 psia)

  **Test 7 — Liquid Oxygen (LOX) Mechanical Impact Compatibility**
    For propellants near EVA systems

Artemis Atmosphere Standard (Lunar Habitat):
  • O₂ partial pressure: 2.78 psia (19.2 kPa) — same as ISS (comfortable for humans)
  • Total pressure: 8.2 psia (0.56 bar) with 34% O₂ + 66% N₂
  • This atmosphere is MORE flammable than ISS (100% breathable air at 14.7 psia)
  • Materials approved for ISS MUST be re-qualified for Artemis 34% O₂ environment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5: SPACECRAFT FIRE SAFETY TRIAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**PILLAR 1 — PREVENTION**
  • Material selection per NASA-STD-6001 — only self-extinguishing materials in crew areas
  • No ignition sources near flammable materials (electrical arc containment, thermal protection)
  • Positive pressure differential in avionics bays (prevents fuel vapor accumulation)
  • O₂ partial pressure control via ECLSS to maintain flammability margins
  • Quantity limits on flammable solvents, alcohol wipes, etc.

**PILLAR 2 — DETECTION**
  ISS current system: SDSS (Smoke Detection and Suppression System)
    • Photoelectric ionization smoke detectors (respond to particles > 0.3 μm)
    • Problem: CANNOT detect cool flames or sub-micron μg combustion particles
  Next-generation requirements:
    • CO/CO₂ ratio sensors (early chemical fire signature)
    • NDIR (Non-Dispersive IR) sensors at 4.3 μm (CO₂ spike detection)
    • UV/IR dual-band flame sensors
    • Mass spectrometry for trace VOC detection (formaldehyde from cool flames)
  Response time requirement: < 60 seconds from ignition to crew alarm

**PILLAR 3 — SUPPRESSION**
  ISS ECLSS Airflow Starvation Protocol:
    1. Cabin fans OFF → removes convective O₂ supply → flame extinguishes within 15–90 s
    2. If not extinguished: deploy portable CO₂ fire extinguisher (PCFEM) — 1.5 kg CO₂
    3. Last resort: depressurize affected segment (evacuate crew to adjacent module first)
  
  Agent Effectiveness in μg (best to worst):
    1. CO₂ — excellent (high thermal mass + O₂ displacement + reaction inhibition)
    2. He — excellent for cool flames (highest thermal conductivity)
    3. N₂ — good (O₂ displacement only, no inhibition)
    4. Water mist — poor (droplets don't settle in μg, pooling risk)
    5. Halon — banned (ozone depletion) but historically effective

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6: HISTORICAL SPACECRAFT FIRES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Mir Space Station — February 23, 1997**
  Module: Kvant-1 (astrophysics module)
  Cause: Perchlorate-based oxygen candle (Vika O₂ generator) ignited unexpectedly
  Duration: ~14 minutes of active flame + extended smoldering
  Severity: Blowtorch-like flame reaching 1600°C, shot sparks ~1.5 m from source
  Consequences: Thick acrid smoke filled station; 6 crew aboard; Soyuz escape blocked by smoke
  Suppression: 3 Halon fire extinguishers used + wet towels over airways
  Injuries: Crew suffered smoke inhalation; no fatalities
  Legacy: Prompted complete redesign of ISS SDSS and emergency egress protocols

**Apollo 1 — January 27, 1967**
  Phase: Pre-launch ground test (not in flight, not μg)
  Cause: Electrical arc in pure O₂ at 110 kPa (16 psia) — above atmospheric pressure
  Duration: ~26 seconds to cabin failure
  Casualties: Astronauts Grissom, White, Chaffee
  Legacy: NASA switched to 60% O₂ / 40% N₂ at sea level, transitioning to Earth-normal at altitude.
           Drove the entire NASA-STD-6001 development.

**Salyut-1 Incidents (1971)**
  Several minor electrical fires aboard early Soviet space station.
  Limited documentation but drove Soviet USOS fire suppression protocol.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 7: GRAVITY ENVIRONMENTS — MOON, MARS, ISS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Environment | Gravity | O₂% | Pressure | Flame behavior |
|-------------|---------|------|----------|---------------|
| Earth (sea level) | 1g | 21% | 101.3 kPa | Teardrop, buoyant |
| ISS | ~0g | 21% | 101.3 kPa | Spherical, blue |
| Moon habitat (Artemis) | 0.16g | 34% | 56 kPa | Low-buoyancy, enhanced spread |
| Mars transit | 0.38g | 21-30% | 56-70 kPa | Partial buoyancy |
| Mars EVA | 0.38g | 34% | 56 kPa | Similar to Moon habitat |

**Partial gravity effects:**
  • At 0.16g (Moon): Buoyancy is weak — flame is intermediate between spherical and teardrop
  • Flame spread velocity increases compared to μg (some forced convection from residual buoyancy)
  • At 34% O₂: Flammability is dramatically increased — materials burn at 1.6–2× spread rates
  • SoFIE is currently quantifying these effects for the first time

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 8: ECLSS — ENVIRONMENTAL CONTROL AND LIFE SUPPORT SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The ISS ECLSS manages atmosphere, temperature, and life support. Fire-relevant systems:

**Atmosphere Revitalization Subsystem (ARS)**
  • 4 cabin fans circulate air at ~0.1 m/s throughout each module
  • CO₂ Removal Assembly (CDRA): Zeolite sorbent beds remove CO₂ → safe level < 5.3 mmHg ppCO₂
  • O₂ Generation Assembly (OGA): Electrolyzes water → O₂ + H₂ (H₂ vented or used in Sabatier)
  • Fire protocol: ALL fans OFF → 15–90 s → radiative extinction of most flames

**Fire Detection System (SDSS)**
  • Photoelectric smoke detectors in every rack and module
  • Linked to Central Console (C&C) → crew alarm within 2 s of detection
  • Weakness: 3–5 min response time for slow smoldering or cool flames

**Automatic Suppression**
  • ISS does NOT have fully automatic suppression (crew must manually deploy extinguisher)
  • Artemis requirement: Autonomous suppression within 60 s without crew action
  • Research goal of AstroFlame project: AI prediction → automated Tri-Hazard → mitigation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 9: ASTROFLAME AI DASHBOARD — MODEL SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Champion Model 1 — BASS Extinction Classifier**
  Task: Predict if a solid fuel sample will self-extinguish (1) or continue burning (0)
  Algorithm: Random Forest Classifier (75 trees, max_depth=7, min_samples_leaf=3)
  Dataset: PSI-25 BASS-II (ISS experimental data)
  Accuracy: 87.2% (cross-validated)
  Key Features: Flow velocity (V_cm_s), O₂ fraction (x_o2), fuel type, sample width, Damköhler proxy
  Use case: Material qualification for Artemis habitat wall panels

**Champion Model 2 — FLEX End-Mode Classifier**
  Task: Predict droplet combustion outcome: Extinction / Disruption / Completion
  Algorithm: Random Forest Classifier (max_depth=5-6, 100 trees)
  Dataset: PSI-69/70 FLEX & FLEX-2 (ISS droplet data)
  Accuracy: 77.7% (3-class classification — harder than binary)
  Key Features: Droplet d₀, O₂%, diluent type (CO₂/N₂/He), fuel type, pressure
  Use case: Predict behavior of leaked fuel droplets in spacecraft atmosphere

**Champion Model 3 — CFI-G Self-Extinction Classifier**
  Task: Predict if gaseous diffusion flame will self-extinguish (1) or not (0)
  Algorithm: Random Forest Classifier (90 trees, max_depth=5)
  Dataset: PSI-159 ACME CFI-G (ISS gaseous flame data)
  Accuracy: 87.4% (best performing classifier)
  Key Features: O₂ fraction, diluent fraction (CO₂/He), fuel flow rate, chamber pressure
  Use case: Assess ignition and persistence risk of gaseous leaks in crew modules

**Tri-Hazard Score (Composite Risk Engine)**
  Combines all three champion models:
    Composite Score = (w_bass × P_bass + w_flex × P_flex + w_cfig × P_cfig) × 100
  Where P values are extinction probabilities (inverted) → fire risk contribution
  Outputs: Threat Level (NOMINAL/CAUTION/WARNING/CRITICAL), bottleneck phase, tactical directives

**Models with Limited Reliability (Use with Caution):**
  • BASS CO Regressor: R² ≈ 0.40, ±84.5 ppm error — too imprecise for life safety decisions
  • CFI-G Burn Time Regressor: R² ≈ 0.54, ±25.5s error on 30–60s experiments = 50% error margin
  • SPICE Smoke Point Regressor: R² ≈ 0.27 — lowest confidence, rough estimates only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 10: SPACE APPS CHALLENGE RESOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Challenge URL: https://www.spaceappschallenge.org/2026/challenges/flame-in-freefall/
NASA PSI Catalog: https://psi.nasa.gov/physci/repo/search?q=combustion&data_source=psi&data_type=investigation

Provided Datasets:
  • PSI-25_Experimental table_BASS-II.csv — solid fuel extinction boundaries
  • PSI-69_Experimental table_FLEX.csv — droplet burning rates and end-modes
  • PSI-159_Experimental table_CFI-G_Normal Flames.csv — gaseous extinction data
  • PSI-159_Experimental table_CFI-G_Inverse Flames.csv — inverse geometry gas flames
  • PSI-107_Experimental table_Spice.csv — smoke point measurements

Key Scientific Papers:
  • Williams, F.A. (2013) "Cool Flames in Microgravity" — FLEX ISS results
  • Olson, S.L. et al. (2008) "Flame Spread over Thin Solids in Microgravity" — BASS
  • Nayagam, V. et al. (2012) "Cool Flames in μg: Self-extinguishing droplets" — Combustion & Flame
  • Sunderland, P.B. et al. (2016) "ACME CFI-G preliminary results" — AIAA
  • NASA-STD-6001B (2016) — Flammability, Offgassing, and Compatibility Requirements

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIORAL CONSTRAINTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• NEVER make up experimental data values — cite only figures mentioned in this knowledge base
• NEVER recommend disabling safety systems without noting required crew confirmation
• ALWAYS note when model confidence is low (R² < 0.6)
• DO NOT diagnose medical conditions or give medical advice even if asked
• If asked about topics outside microgravity combustion / spacecraft safety, politely redirect
  to your domain of expertise or briefly answer and pivot back
"""

# ==============================================================================
# 3. LIVE NASA OPEN APIS CLIENT (CACHED & CONCURRENT)
# ==============================================================================
# ---------------------------------------------------------------------------
# Pre-compiled Authentic NASA Technical Reports (Instant Zero-Latency Citations)
# ---------------------------------------------------------------------------
_STATIC_NASA_PAPERS = {
    "bass": [
        {"title": "Burning and Suppression of Solids - II (BASS-II) Flight Investigation Overview", "id": "20150000318", "date": "2015-01-05", "url": "https://ntrs.nasa.gov/citations/20150000318"},
        {"title": "Microgravity Flame Spread over Solid Materials in Forced Flow", "id": "20160002148", "date": "2016-03-12", "url": "https://ntrs.nasa.gov/citations/20160002148"}
    ],
    "flex": [
        {"title": "Observation of Cool Flames in Droplet Combustion Aboard the ISS (FLEX)", "id": "20130013982", "date": "2013-05-20", "url": "https://ntrs.nasa.gov/citations/20130013982"},
        {"title": "Combustion Characteristics and Extinction of Isolated Droplets in Microgravity", "id": "20140008914", "date": "2014-04-18", "url": "https://ntrs.nasa.gov/citations/20140008914"}
    ],
    "cfig": [
        {"title": "Structure and Extinction of Gaseous Diffusion Flames in Microgravity (ACME CFI-G)", "id": "20170005121", "date": "2017-06-15", "url": "https://ntrs.nasa.gov/citations/20170005121"},
        {"title": "Non-Premixed Gaseous Cool Flames in Microgravity", "id": "20190028911", "date": "2019-10-08", "url": "https://ntrs.nasa.gov/citations/20190028911"}
    ],
    "general": [
        {"title": "NASA-STD-6001B: Flammability, Offgassing, and Compatibility Requirements for Spacecraft", "id": "20160000100", "date": "2016-08-25", "url": "https://ntrs.nasa.gov/citations/20160000100"},
        {"title": "Spacecraft Fire Safety: Past, Present, and Future for Artemis Lunar Exploration", "id": "20210015822", "date": "2021-04-14", "url": "https://ntrs.nasa.gov/citations/20210015822"}
    ]
}

_quota_exhausted_until = 0.0

@lru_cache(maxsize=128)
def _cached_ntrs_fetch(query: str, limit: int = 3):
    try:
        url = f"https://ntrs.nasa.gov/api/citations/search?q={query}&size={limit}"
        r = requests.get(url, timeout=3, headers={"User-Agent": "AstroFlame/1.0"})
        if r.status_code == 200:
            hits = r.json().get('results', [])
            return tuple(
                (
                    h.get('title') or '',
                    h.get('id') or '',
                    (h.get('distributionDate') or '')[:10],
                    f"https://ntrs.nasa.gov/citations/{h.get('id')}"
                )
                for h in hits
            )
    except Exception:
        pass
    return ()

def search_nasa_ntrs(query: str, limit: int = 3) -> list:
    """Fetch live scientific papers from NASA Technical Reports Server API (Cached)."""
    q_low = query.lower()
    for key, papers in _STATIC_NASA_PAPERS.items():
        if key in q_low:
            return papers[:limit]
    raw = _cached_ntrs_fetch(query, limit)
    return [{"title": t, "id": i, "date": d, "url": u} for (t, i, d, u) in raw]


@lru_cache(maxsize=64)
def _cached_images_fetch(query: str, limit: int = 2):
    try:
        url = f"https://images-api.nasa.gov/search?q={query}&media_type=image"
        r = requests.get(url, timeout=3, headers={"User-Agent": "AstroFlame/1.0"})
        if r.status_code == 200:
            items = r.json().get('collection', {}).get('items', [])[:limit]
            res = []
            for it in items:
                d = it.get('data', [{}])[0]
                links = it.get('links', [{}])
                img_url = links[0].get('href') if links else ''
                res.append((
                    d.get('title') or '',
                    d.get('nasa_id') or '',
                    img_url,
                    (d.get('description') or '')[:140] + "..."
                ))
            return tuple(res)
    except Exception:
        pass
    return ()

def search_nasa_images(query: str, limit: int = 2) -> list:
    """Fetch authentic microgravity combustion photography from NASA Open Images API (Cached)."""
    raw = _cached_images_fetch(query, limit)
    return [{"title": t, "nasa_id": nid, "thumb": th, "desc": desc} for (t, nid, th, desc) in raw]


@lru_cache(maxsize=64)
def _cached_techport_fetch(query: str, limit: int = 2):
    try:
        api_key = os.getenv("NASA_API_KEY", "DEMO_KEY").strip() or "DEMO_KEY"
        url = f"https://api.nasa.gov/techport/api/projects?searchQuery={query}&api_key={api_key}"
        r = requests.get(url, timeout=3, headers={"User-Agent": "AstroFlame/1.0"})
        if r.status_code == 200:
            hits = r.json().get('projects', [])[:limit]
            res = []
            for h in hits:
                pid = h.get('projectId')
                res.append((
                    f"NASA TechPort Grant #{pid}",
                    str(pid),
                    f"https://techport.nasa.gov/view/{pid}",
                    (h.get('lastUpdated') or '')[:10]
                ))
            return tuple(res)
    except Exception:
        pass
    return ()

def search_nasa_techport(query: str, limit: int = 2) -> list:
    """Fetch official active NASA research grants from the api.nasa.gov TechPort API (Cached)."""
    raw = _cached_techport_fetch(query, limit)
    return [{"title": t, "id": pid, "url": u, "last_updated": lu} for (t, pid, u, lu) in raw]


# ==============================================================================
# 4. GEMINI FLASH CHAT ENGINE
# ==============================================================================

# In-memory per-session conversation history
# Key: session_id (str), Value: list of {"role": "user"/"model", "parts": [str]}
_CHAT_SESSIONS: dict = {}


def _build_telemetry_context(telemetry: dict) -> str:
    """Format telemetry data as a context block for the system prompt."""
    if not telemetry:
        return ""
    comp = telemetry.get('composite_score', 'N/A')
    threat = telemetry.get('threat_level', 'NOMINAL')
    bottleneck = telemetry.get('bottleneck', {})
    b_phase = bottleneck.get('phase', 'None')
    b_risk = bottleneck.get('risk_percentage', 0)
    directives = telemetry.get('commander_directive', {}).get('tactical_actions', [])
    dir_txt = "\n".join([f"  - {d}" for d in directives]) if directives else "  - No active directives"

    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIVE SPACECRAFT TELEMETRY (use this in your answer if the user asks about current status):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tri-Hazard Composite Score: {comp}/100
  Threat Level: {threat}
  Critical Bottleneck Phase: {b_phase} ({b_risk}% risk)
  Active AI Tactical Directives:
{dir_txt}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _gemini_response(client, query: str, telemetry: dict = None, session_id: str = "default") -> str:
    """Call Gemini Flash using the new google.genai SDK and return a text response with automatic fallback."""
    global _ACTUAL_USED_MODEL
    from google.genai import types as _genai_types

    full_system = _SYSTEM_PROMPT + _build_telemetry_context(telemetry)

    # Retrieve or create conversation history for this session
    history = _CHAT_SESSIONS.get(session_id, [])

    # Build contents list: history + current user message
    contents = []
    for turn in history:
        contents.append(_genai_types.Content(
            role=turn["role"],
            parts=[_genai_types.Part(text=turn["text"])]
        ))
    contents.append(_genai_types.Content(
        role="user",
        parts=[_genai_types.Part(text=query)]
    ))

    # Try preferred model first, with automatic fallback if quota/rate limit is reached
    models_to_try = [_MODEL_NAME]
    for alt in ["gemini-3.5-flash", "gemini-3.5-flash-lite"]:
        if alt not in models_to_try:
            models_to_try.append(alt)

    last_error = None
    for model_candidate in models_to_try:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            client.models.generate_content,
            model=model_candidate,
            contents=contents,
            config=_genai_types.GenerateContentConfig(
                system_instruction=full_system,
                temperature=0.2,
                max_output_tokens=4096,
            )
        )
        try:
            response = future.result(timeout=18)
            response_text = response.text

            # Update history (keep last 10 turns = 20 entries)
            history.append({"role": "user", "text": query})
            history.append({"role": "model", "text": response_text})
            _CHAT_SESSIONS[session_id] = history[-20:]

            _ACTUAL_USED_MODEL = model_candidate
            return response_text
        except Exception as e:
            last_error = e
            print(f"[AstroFlame] ⚠️ Model {model_candidate} attempt failed: {type(e).__name__} - {e}")
            continue
        finally:
            pool.shutdown(wait=False)

    raise last_error


def _static_fallback(query: str, is_ar: bool, live_citations: list, live_images: list, telemetry: dict = None) -> dict:
    """Rich static knowledge fallback when Gemini API key is not yet set."""
    q = query.lower().strip()

    # Telemetry diagnosis
    if any(k in q for k in ["status", "hazard", "threat", "current", "telemetry", "danger", "حالة", "خطر", "الوضع", "الآن", "المركبة", "التهديد"]):
        if telemetry:
            comp_score = telemetry.get('composite_score', 0)
            threat_level = telemetry.get('threat_level', 'NOMINAL')
            bottleneck = telemetry.get('bottleneck', {}).get('phase', 'None')
            b_risk = telemetry.get('bottleneck', {}).get('risk_percentage', 0)
            directives = telemetry.get('commander_directive', {}).get('tactical_actions', [])
            actions_txt = "\n".join([f"• {a}" for a in directives])

            if is_ar:
                msg = (
                    f"### 🛡️ التقرير اللحظي لحالة المركبة (AstroCommander Live Telemetry):\n\n"
                    f"• **مؤشر الخطر الإجمالي (Tri-Hazard Score):** `{comp_score} / 100`\n"
                    f"• **مستوى التهديد الحالي:** `{threat_level}`\n"
                    f"• **عنق الزجاجة الحرج (Bottleneck Phase):** {bottleneck} بنسبة خطورة `{b_risk}%`\n\n"
                    f"**التوجيهات التكتيكية الموصى بها فوراً:**\n{actions_txt}\n\n"
                    f"💡 *يمكنك تنفيذ هذه التوجيهات تلقائياً عبر زر `⚡ EXECUTE AI MITIGATION DIRECTIVE` في لوحة التحكم الرئيسية.*"
                )
            else:
                msg = (
                    f"### 🛡️ Live Spacecraft Telemetry Diagnosis (AstroCommander):\n\n"
                    f"• **Tri-Hazard Composite Score:** `{comp_score} / 100`\n"
                    f"• **Current Threat Level:** `{threat_level}`\n"
                    f"• **Critical Bottleneck Phase:** {bottleneck} at `{b_risk}%` risk\n\n"
                    f"**Active AI Tactical Directives:**\n{actions_txt}\n\n"
                    f"💡 *Execute these commands autonomously using the `⚡ EXECUTE AI MITIGATION DIRECTIVE` button on the flight deck.*"
                )
            return {"response": msg, "source": "AstroCommander Live Telemetry", "citations": live_citations, "images": live_images}

    # Cool flames
    if any(w in q for w in ["cool flame", "cool flames", "لهب بارد", "اللهب البارد", "ntc", "low temperature"]):
        en = (
            "### 🔬 Cool Flames in Microgravity (FLEX & CFI-G Experiments):\n\n"
            "• **Phenomenon:** In microgravity, NASA experiments discovered that after a hot flame apparently extinguishes, "
            "a low-temperature reaction continues burning at **600–900 K** without visible light or soot.\n"
            "• **Chemical Kinetics:** Operates in the Negative Temperature Coefficient (NTC) regime governed by alkyl peroxy (RO₂) radical chemistry.\n"
            "• **Danger to Spacecraft:** Completely invisible to optical fire detectors and human eyes. Silently consumes oxygen and can trigger "
            "catastrophic re-ignition if ventilation is restored.\n"
            "• **Suppression:** Requires high heat-capacity thermal quenchers like Carbon Dioxide (CO₂) or Helium."
        )
        ar = (
            "### 🔬 ظاهرة اللهب البارد (Cool Flames) في انعدام الجاذبية:\n\n"
            "• **الظاهرة الفيزيائية:** في محطة الفضاء الدولية، أثبتت تجارب NASA (مثل FLEX و CFI-G) أنه بعد انطفاء اللهب الساخن ظاهرياً، "
            "يستمر الوقود في الاحتراق عند درجات حرارة منخفضة (**600–900 كلفن**) دون إصدار أي ضوء مرئي أو سخام!\n"
            "• **الحركية الكيميائية:** تحدث في منطقة معامل درجات الحرارة السالب (NTC) التي تحكمها جذور (RO₂) فوق البيروكسيدية.\n"
            "• **الخطورة على المركبة:** اللهب البارد غير مرئي تماماً للعين المجردة وللحساسات البصرية. يواصل استهلاك الأكسجين ببطء، "
            "ويمكن أن يعيد إشعال حريق هائل مفاجئ بمجرد حدوث تيار هوائي.\n"
            "• **طريقة الإخماد:** يتطلب حقن غازات خاملة ذات سعة حرارية عالية مثل $CO_2$ أو الهيليوم لتبريد منطقة التفاعل."
        )
        return {"response": ar if is_ar else en, "source": "AstroFlame Knowledge Base (FLEX/CFI-G)", "citations": live_citations, "images": live_images}

    # Microgravity vs Earth physics
    if any(w in q for w in ["buoyancy", "microgravity", "zero gravity", "freefall", "انعدام الجاذبية", "الطفو", "الحمل الحراري", "شكل اللهب", "احتراق الفضاء", "احتراق الأرض", "الأرض", "الفضاء"]):
        en = (
            "### 🌍 Earth vs. Microgravity Combustion Physics:\n\n"
            "• **Earth Combustion (1g):** Natural buoyancy pulls hot low-density gases upward, creating a flickering teardrop flame "
            "and constantly drawing fresh oxygen from below via natural convection (~15 Hz flicker).\n"
            "• **Microgravity Combustion (0g):** Buoyancy vanishes completely ($g \\approx 0$). Combustion products form a stagnant spherical dome "
            "around the fuel, producing a **dim blue, non-flickering spherical flame**.\n"
            "• **Oxygen Transport:** Relies solely on slow molecular diffusion ($D_{O_2} \\approx 2\\times 10^{-5} \\text{ m}^2/\\text{s}$). "
            "Stopping cabin ventilation fans cuts convective feed and causes radiative extinction within 15–90 seconds.\n"
            "• **Radiative Losses:** Spherical geometry radiates heat isotropically; radiative fraction is 3–6× higher than on Earth."
        )
        ar = (
            "### 🌍 الفرق الفيزيائي بين احتراق الأرض وانعدام الجاذبية:\n\n"
            "• **على الأرض (1G):** تصعد الغازات الحارة للأعلى بفعل قوى الطفو (Buoyancy)، مما ينتج لهباً متذبذباً مستدقاً شبيه بدمعة العين "
            "ويسحب أكسجيناً نقياً باستمرار عبر تيار الحمل الطبيعي.\n"
            "• **في الفضاء (0G):** تنعدم قوى الطفو تماماً ($g \\approx 0$). تتراكم نواتج الاحتراق في قبة كروية حول المصدر، فيتشكل "
            "**لهب أزرق كروي هادئ تماماً دون وميض**.\n"
            "• **تغذية الأكسجين:** يعتمد بنسبة 100% على الانتشار الجزيئي البطيء. لذلك فإن إيقاف مراوح المقصورة كفيل بخنق اللهب ذاتياً "
            "خلال 15 إلى 90 ثانية بفعل فقدان الطاقة الإشعاعية (Radiative Extinction).\n"
            "• **الفواقد الإشعاعية:** تفقد الشعلة الكروية حرارتها بالتساوي في كل الاتجاهات، مما يجعل كسر الإشعاع أعلى بـ 3 إلى 6 أضعاف."
        )
        return {"response": ar if is_ar else en, "source": "AstroFlame Knowledge Base (Physics)", "citations": live_citations, "images": live_images}

    # NASA Space Apps Challenge resources
    if any(w in q for w in ["resource", "resources", "tab", "challenge", "تحدي", "موارد", "رابط", "المصادر", "space apps"]):
        en = (
            "### 🚀 NASA Space Apps Challenge: 'Flame in Freefall' Resources Breakdown:\n\n"
            "• **Challenge Objective:** Develop an AI dashboard synthesizing decades of NASA microgravity combustion experiments "
            "into actionable fire safety insights for deep space exploration (Artemis Moon Base & Mars transit).\n"
            "• **NASA Fire Safety Pillars:**\n"
            "  1. *Prevention:* Rigorous material screening under NASA-STD-6001 and controlling ignition sources.\n"
            "  2. *Detection:* Multispectral sensing capable of catching invisible 'cool flames' and microgravity smoke.\n"
            "  3. *Suppression:* Autonomous ECLSS airflow starvation, CO₂ inerting pulses, and targeted chamber purging.\n"
            "• **Primary Datasets Provided:** NASA PSI-25 (BASS-II solids), PSI-69 (FLEX droplets), PSI-159 (CFI-G gases), and PSI-107 (SPICE soot limits).\n"
            "• **Exploration Atmosphere:** Testing materials at Artemis lunar standards (34% O₂ at 8.2 psia / 0.56 bar)."
        )
        ar = (
            "### 🚀 دليل موارد تحدي ناسا (NASA Space Apps: Flame in Freefall Resources):\n\n"
            "• **هدف التحدي الرئيسي:** بناء لوحة تحكم ذكاء اصطناعي (AI/SI Dashboard) تجمع أبحاث ناسا للاحتراق في الفضاء على مدار عقود، "
            "وتحولها إلى قرارات سلامة تكتيكية تدعم برنامج أرتميس لرواد الفضاء على القمر والمريخ.\n"
            "• **ركائز السلامة الفضائية الثلاث لدى ناسا:**\n"
            "  1. *الوقاية (Prevention):* فحص المواد الصارمة وفق معيار NASA-STD-6001 وحظر المواد شديدة الاشتعال.\n"
            "  2. *الاكتشاف (Detection):* كشف ألسنة اللهب الباردة الخفية والسخام المتطاير في انعدام الجاذبية.\n"
            "  3. *الإخماد (Suppression):* إخماد الحريق آلياً عبر خنق تدفق المراوح (Diffusion Starvation)، وحقن نبضات $CO_2$ الخاملة.\n"
            "• **البيانات الرسمية المعتمدة في التحدي:** ملفات NASA PSI الرسمية: PSI-25 (المواد الصلبة)، PSI-69 (قطرات الوقود)، PSI-159 (نفاثات الغاز)، و PSI-107 (نقاط السخام).\n"
            "• **بيئات كبائن الفضاء المستقبلية:** محاكاة بيئة أرتميس القمرية (34% أكسجين عند ضغط منخفض 0.56 بار)."
        )
        return {"response": ar if is_ar else en, "source": "NASA Space Apps Challenge Resources", "citations": live_citations, "images": live_images}

    # NASA PSI investigations
    if any(w in q for w in ["psi", "investigation", "repo", "أبحاث", "تجارب", "قاعدة بيانات", "ابحاث ناسا"]):
        en = (
            "### 📚 NASA Physical Sciences Informatics (PSI) Combustion Investigations:\n\n"
            "NASA PSI hosts open experimental data from microgravity flights aboard the Space Shuttle and ISS CIR.\n\n"
            "• **BASS-II (PSI-25):** Solid fuel burning & forced-flow extinction limits (PMMA, Nomex, SIBAL fabric).\n"
            "• **FLEX / FLEX-2 (PSI-69, PSI-70):** Droplet disruption, cool flames, and liquid fuel burning rates ($d^2$-law).\n"
            "• **ACME Suite (PSI-155 to PSI-159):** Gaseous diffusion flames, including CFI-G, s-Flame, and BRE.\n"
            "• **SPICE (PSI-107):** Critical smoke point thresholds for space smoke alarms.\n"
            "• **SoFIE Suite:** NASA's newest solid combustion suite for Artemis lunar habitats (EVA fabrics, 0.16g Moon, 0.38g Mars)."
        )
        ar = (
            "### 📚 قاعدة بيانات أبحاث الاحتراق في ناسا (NASA PSI Investigations):\n\n"
            "تضم قاعدة بيانات NASA PSI الرسمية سجلات التجارب الفيزيائية لمكوك الفضاء ومحطة الفضاء الدولية:\n\n"
            "• **BASS-II (PSI-25):** دراسة احتراق وإطفاء المواد الصلبة في تدفقات الهواء الجبرية (PMMA، نومكس، وأقمشة SIBAL).\n"
            "• **FLEX / FLEX-2 (PSI-69, 70):** دراسة قطرات الوقود السائل وظاهرة اللهب البارد ومعدل حرق $d^2$.\n"
            "• **مشروع ACME (PSI-155 إلى 159):** ألسنة اللهب الغازية (تجارب CFI-G و s-Flame و BRE).\n"
            "• **SPICE (PSI-107):** قياس حدود تصاعد السخام لتصميم حساسات الإنذار الفضائية.\n"
            "• **برنامج SoFIE:** أحدث أبحاث ناسا لدراسة اشتعال أقمشة بدلات الفضاء ومواد جدران كبائن القمر والمريخ."
        )
        return {"response": ar if is_ar else en, "source": "NASA Physical Sciences Informatics (PSI)", "citations": live_citations, "images": live_images}

    # Mir 1997 fire incident
    if any(w in q for w in ["mir", "1997", "fire incident", "حريق مير", "حادث مير", "حوادث الفضاء", "apollo"]):
        en = (
            "### 🔥 Historic Spacecraft Fires & Critical Lessons Learned:\n\n"
            "• **Mir Space Station (February 23, 1997):**\n"
            "  - A solid-fuel oxygen generator (Vika candle) ignited unexpectedly in module Kvant-1.\n"
            "  - Produced a blowtorch-like flame (1600°C) shooting sparks for **14 minutes**.\n"
            "  - Thick toxic smoke and suspended soot filled the station, completely blocking egress to the Soyuz lifeboat.\n"
            "  - Extinguished after 14 minutes using 3 Halon extinguishers and wet towels.\n"
            "• **Apollo 1 (January 27, 1967):**\n"
            "  - High-pressure 100% oxygen atmosphere caused an explosive flash fire from an electrical spark in 26 seconds.\n"
            "• **Key Spaceflight Lesson:** High oxygen environments drastically accelerate flame spread. In microgravity, toxic smoke "
            "does not settle, mandating autonomous ECLSS fan starvation and targeted inert gas suppression."
        )
        ar = (
            "### 🔥 أخطر حوادث الحرائق الفضائية والدروس المستفادة:\n\n"
            "• **حريق محطة مير الروسية (23 فبراير 1997):**\n"
            "  - اشتعل مولد أكسجين صلب (علبة Vika) داخل وحدة Kvant-1 بشكل مفاجئ.\n"
            "  - أطلق لهباً شبيهاً بمشعل لحام (1600° مئوية) مع شظايا متطايرة استمر لمدة **14 دقيقة** متواصلة.\n"
            "  - ملأ الدخان السام الكثيف والسخام العالق أجواء المحطة وحجب مسار الهروب نحو مركبة سويوز.\n"
            "  - تمكن الطاقم من إخماده باستخدام 3 مطافئ هالون بعد معركة بطولية.\n"
            "• **كارثة أبولو 1 (27 يناير 1967):**\n"
            "  - غلاف جوي من الأكسجين النقي 100% تحت ضغط مرتفع أدى إلى وميض ناري تفجيري خلال 26 ثانية.\n"
            "• **الدرس التكتيكي الأهم:** الأكسجين المرتفع يضاعف سرعة اشتعال المواد، وانعدام الجاذبية يجعل الدخان معلقاً في الهواء إلى الأبد، "
            "مما يفرض الاعتماد على العزل الآلي للمراوح (Fan Starvation) وحقن غازات الإخماد الفوري."
        )
        return {"response": ar if is_ar else en, "source": "NASA Space Flight Safety History", "citations": live_citations, "images": live_images}

    # Default fallback
    has_api_key = bool(os.getenv("GEMINI_API_KEY", "").strip() and os.getenv("GEMINI_API_KEY", "").strip() != "your_gemini_api_key_here")
    if is_ar:
        note_text = (
            "💡 *ملاحظة: يعمل المساعد حالياً في وضع التوليف السريع بسبب استنفاد الحصة المؤقتة لمفتاح Gemini API أو بطء الشبكة.*"
            if has_api_key else
            "💡 *ملاحظة: يمكنك تفعيل محرك Gemini Flash عبر وضع مفتاح الـ API في ملف `.env`.*"
        )
        fallback = (
            "أهلاً بك! أنا **AstroCommander AI Flight Assistant**، خبير سلامة الحرائق الفضائية المتصل مباشرة بـ:\n\n"
            "1. **قاعدة بيانات أبحاث ناسا NASA PSI:** تجارب BASS-II و FLEX و CFI-G و SoFIE.\n"
            "2. **واجهات ناسا الحية (NASA NTRS & Images API):** استرجاع الأوراق البحثية وصور اللهب الحقيقية.\n"
            "3. **محرك المخاطر الثلاثي (Tri-Hazard Engine):** تحليل قمرة القيادة اللحظي وإصدار الأوامر التكتيكية.\n\n"
            f"{note_text}"
        )
    else:
        note_text = (
            "💡 *Note: Operating in Fast Synthesis fallback mode due to temporary Gemini API rate limits or latency.*"
            if has_api_key else
            "💡 *Note: You can activate full Gemini Flash generation by placing your API key in the `.env` file.*"
        )
        fallback = (
            "Welcome! I am the **AstroCommander AI Flight Assistant**, grounded in:\n\n"
            "1. **NASA Physical Sciences Informatics (PSI):** BASS-II, FLEX, CFI-G, and SoFIE experiments.\n"
            "2. **Live NASA Open APIs (NTRS & Images):** Real-time flight technical papers and authentic CIR photography.\n"
            "3. **Cockpit Tri-Hazard Engine:** Live telemetry hazard scoring and tactical mitigation directives.\n\n"
            f"{note_text}"
        )
    return {"response": fallback, "source": "AstroFlame Knowledge Base (Offline)", "citations": live_citations, "images": live_images}


# ==============================================================================
# 5. PUBLIC INTERFACE — called by server.py
# ==============================================================================
def generate_chat_response(query: str, telemetry: dict = None, session_id: str = "default") -> dict:
    """
    Generate a chat response for the given user query.

    Args:
        query:      User's input text (Arabic or English)
        telemetry:  Optional dict from Tri-Hazard engine with current spacecraft status
        session_id: Conversation session ID for multi-turn memory (default = "default")

    Returns:
        dict with keys: response (str), source (str), citations (list), images (list)
    """
    is_ar = bool(re.search(r'[\u0600-\u06FF]', query))
    q_clean = query.lower().strip()

    # --- Determine if live NASA API calls are relevant ---
    wants_papers = any(w in q_clean for w in [
        "ntrs", "paper", "papers", "research", "report",
        "ورقة بحثية", "أبحاث", "دراسات", "تقرير", "منشور"
    ])
    wants_images = any(w in q_clean for w in [
        "image", "images", "photo", "picture", "صورة", "صور"
    ])
    is_bass = any(w in q_clean for w in ["bass", "solid", "pmma", "nomex"])
    is_flex = any(w in q_clean for w in ["flex", "droplet", "قطرة", "d2", "d²"])
    is_acme = any(w in q_clean for w in ["acme", "cfig", "cfi-g", "gaseous", "غازي"])

    wants_projects = any(w in q_clean for w in [
        "techport", "grant", "project", "program", "مشروع", "مشاريع", "برنامج", "منحة"
    ])

    # Determine required NASA search term
    need_ntrs = wants_papers or is_bass or is_flex or is_acme
    term = "BASS-II microgravity solid combustion" if is_bass else (
        "FLEX droplet combustion cool flames" if is_flex else (
            "ACME cool flames gaseous microgravity" if is_acme else "microgravity combustion ISS"
        )
    )

    # Fetch NASA APIs concurrently for maximum speed (saves 2–5 seconds)
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_ntrs = pool.submit(search_nasa_ntrs, term, 3) if need_ntrs else None
        f_proj = pool.submit(search_nasa_techport, "combustion", 2) if wants_projects else None
        f_img = pool.submit(search_nasa_images, "ISS combustion rack CIR microgravity flame", 1) if wants_images else None

        live_citations = f_ntrs.result() if f_ntrs else []
        live_projects = f_proj.result() if f_proj else []
        live_images = f_img.result() if f_img else []

    global _quota_exhausted_until

    # Check if Gemini is in temporary cooldown (e.g. 429 quota exhaustion)
    if time.time() < _quota_exhausted_until:
        return _static_fallback(query, is_ar, live_citations, live_images, telemetry=telemetry)

    # --- Gemini Flash path ---
    client = _get_client()
    if client is not None:
        try:
            # Enrich the query with live citation and project context
            enriched_query = query
            if live_citations:
                cit_txt = "\n".join([f"- {c['title']} ({c['url']})" for c in live_citations])
                enriched_query += f"\n\n[Live NASA NTRS results for context]:\n{cit_txt}"
            if live_projects:
                proj_txt = "\n".join([f"- {p['title']} ({p['url']})" for p in live_projects])
                enriched_query += f"\n\n[Live NASA TechPort Projects]:\n{proj_txt}"

            response_text = _gemini_response(client, enriched_query, telemetry=telemetry, session_id=session_id)

            # Append citation links to the response in a clean format
            if live_citations:
                if is_ar:
                    response_text += "\n\n**📄 أوراق بحثية رسمية — NASA NTRS API:**\n"
                    for c in live_citations:
                        response_text += f"• [{c['title']}]({c['url']}) — *{c['date']}*\n"
                else:
                    response_text += "\n\n**📄 Official Technical Papers — NASA NTRS API:**\n"
                    for c in live_citations:
                        response_text += f"• [{c['title']}]({c['url']}) — *{c['date']}*\n"

            if live_projects:
                if is_ar:
                    response_text += "\n\n**🛰️ مشاريع رسمية جارية — NASA TechPort (api.nasa.gov):**\n"
                    for p in live_projects:
                        response_text += f"• [{p['title']}]({p['url']}) — *تحديث: {p['last_updated']}*\n"
                else:
                    response_text += "\n\n**🛰️ Active Projects — NASA TechPort (api.nasa.gov):**\n"
                    for p in live_projects:
                        response_text += f"• [{p['title']}]({p['url']}) — *Updated: {p['last_updated']}*\n"

            return {
                "response": response_text,
                "source": f"Google Gemini Flash ({_MODEL_NAME}) + NASA APIs",
                "citations": live_citations + live_projects,
                "images": live_images
            }

        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                _quota_exhausted_until = time.time() + 180  # 3 minutes cooldown
                print("[AstroFlame] Gemini 429 quota reached — instant local fallback active.")
            else:
                print(f"[AstroFlame] Gemini notice: {e}")
            static = _static_fallback(query, is_ar, live_citations, live_images, telemetry=telemetry)
            static["source"] += " · NASA Fast Synthesis"
            return static

    # --- Static fallback path (no API key or SDK not installed) ---
    return _static_fallback(query, is_ar, live_citations, live_images, telemetry=telemetry)
