// ASTROFLAME | deep-space mission console
const $ = id => document.getElementById(id);
const val = id => $(id).value, num = id => parseFloat(val(id));
const post = async (url, body) => {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
};
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

// 1. STARFIELD: 3 parallax layers, twinkling, constellations, meteors
const sc = $('starfield'), sx = sc.getContext('2d');
let W, H, stars = [], conn = [], meteors = [], mx = 0, my = 0;
function initSky() {
  W = sc.width = innerWidth; H = sc.height = innerHeight;
  stars = Array.from({length: Math.min(320, W * H / 5500 | 0)}, () => {
    const d = Math.random();
    return {x: Math.random() * W, y: Math.random() * H, d, r: .3 + d * 1.5, a: .25 + d * .7, p: Math.random() * 6.28, hue: Math.random() < .15 ? 30 : 220};
  });
  conn = stars.filter(s => s.d > .8).slice(0, 38);
}
function sky(t) {
  sx.clearRect(0, 0, W, H);
  for (const s of stars) {
    s.y -= .04 + s.d * .1; if (s.y < 0) s.y = H;
    const px = s.x + mx * s.d * 22, py = s.y + my * s.d * 22;
    sx.globalAlpha = s.a * (.65 + .35 * Math.sin(t / 900 + s.p));
    sx.fillStyle = `hsl(${s.hue},90%,88%)`;
    sx.beginPath(); sx.arc(px, py, s.r, 0, 6.283); sx.fill();
    if (s.d > .93) { sx.globalAlpha *= .5; sx.fillRect(px - 7, py - .4, 14, .8); sx.fillRect(px - .4, py - 7, .8, 14); }
  }
  sx.lineWidth = .6; sx.strokeStyle = '#8b9bff';
  for (let i = 0; i < conn.length; i++) for (let j = i + 1; j < conn.length; j++) {
    const a = conn[i], b = conn[j], d = Math.hypot(a.x - b.x, a.y - b.y);
    if (d < 190) { sx.globalAlpha = (1 - d / 190) * .3; sx.beginPath(); sx.moveTo(a.x + mx * a.d * 22, a.y + my * a.d * 22); sx.lineTo(b.x + mx * b.d * 22, b.y + my * b.d * 22); sx.stroke(); }
  }
  if (Math.random() < .006 && meteors.length < 2) meteors.push({x: Math.random() * W, y: Math.random() * H * .4, l: 90 + Math.random() * 80, v: 9 + Math.random() * 6, o: 1});
  meteors = meteors.filter(m => {
    const g = sx.createLinearGradient(m.x, m.y, m.x - m.l * .8, m.y - m.l * .5);
    g.addColorStop(0, `rgba(255,255,255,${m.o})`); g.addColorStop(1, 'rgba(124,200,255,0)');
    sx.globalAlpha = 1; sx.strokeStyle = g; sx.lineWidth = 1.8; sx.beginPath(); sx.moveTo(m.x, m.y); sx.lineTo(m.x - m.l * .8, m.y - m.l * .5); sx.stroke();
    m.x += m.v * .85; m.y += m.v * .5; m.o -= .014; return m.o > 0;
  });
  if (!still) requestAnimationFrame(sky);
}
addEventListener('resize', initSky);
addEventListener('mousemove', e => { mx = e.clientX / W - .5; my = e.clientY / H - .5; });
initSky(); requestAnimationFrame(sky);

// 2. SPHERICAL FLAME (microgravity): glow, diffusion sparks, orbit rings
const fc = $('flameCanvas'), fx = fc.getContext('2d');
const flame = {r: 55, mode: 'hot', t: 0};
const sparks = Array.from({length: 26}, (_, i) => ({a: i * 2.4, k: Math.random()}));
function sizeFlame() { fc.width = fc.clientWidth * devicePixelRatio; fc.height = fc.clientHeight * devicePixelRatio; }
function drawFlame() {
  if (fc.clientWidth) { if (fc.width !== fc.clientWidth * devicePixelRatio) sizeFlame();
    const w = fc.width, h = fc.height, cx = w / 2, cy = h / 2 - 10, u = devicePixelRatio;
    flame.t += .03; fx.clearRect(0, 0, w, h);
    const r = (flame.r + Math.sin(flame.t * 2.4) * 3) * u, safe = flame.mode === 'safe';
    fx.strokeStyle = 'rgba(160,170,255,.18)'; fx.lineWidth = 1;
    [1.9, 2.8].forEach(k => { fx.beginPath(); fx.arc(cx, cy, r * k, 0, 6.283); fx.stroke(); });
    const g = fx.createRadialGradient(cx, cy, r * .1, cx, cy, r * 2.2);
    if (safe) { g.addColorStop(0, 'rgba(110,243,197,.9)'); g.addColorStop(.4, 'rgba(124,200,255,.3)'); }
    else { g.addColorStop(0, 'rgba(255,250,215,.95)'); g.addColorStop(.25, 'rgba(255,180,84,.8)'); g.addColorStop(.65, 'rgba(255,70,30,.3)'); }
    g.addColorStop(1, 'rgba(0,0,0,0)');
    fx.fillStyle = g; fx.beginPath(); fx.arc(cx, cy, r * 2.2, 0, 6.283); fx.fill();
    for (const s of sparks) {
      s.k += .004; if (s.k > 1) s.k = 0;
      const d = r * (.9 + s.k * 1.6); fx.globalAlpha = (1 - s.k) * .8;
      fx.fillStyle = safe ? '#9ff7dc' : '#ffd28a';
      fx.beginPath(); fx.arc(cx + Math.cos(s.a + flame.t * .2) * d, cy + Math.sin(s.a + flame.t * .2) * d, 1.6 * u, 0, 6.283); fx.fill();
    }
    fx.globalAlpha = 1; fx.fillStyle = '#fff'; fx.shadowColor = '#fff'; fx.shadowBlur = 16;
    fx.beginPath(); fx.arc(cx, cy, 6 * u, 0, 6.283); fx.fill(); fx.shadowBlur = 0;
  }
  if (!still) requestAnimationFrame(drawFlame);
}
drawFlame();

// 3. TABS + AUDIO BLIPS
let ac;
function blip(f = 480, d = .06) {
  try { ac = ac || new (window.AudioContext || window.webkitAudioContext)();
    const o = ac.createOscillator(), g = ac.createGain();
    o.frequency.value = f; g.gain.setValueAtTime(.03, ac.currentTime); g.gain.exponentialRampToValueAtTime(.001, ac.currentTime + d);
    o.connect(g); g.connect(ac.destination); o.start(); o.stop(ac.currentTime + d); } catch (e) {}
}
document.querySelectorAll('.tab').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tab,.pane').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); $('pane-' + b.dataset.tab).classList.add('active'); blip();
  if (b.dataset.tab === 'benchmark') drawGauges();
});

// 4. CONTROLS (generated from specs; IDs match the FastAPI clients)
// ['s', id, label, options, default] | ['r', id, label, min, max, step, default, unit]
const SPECS = {
  bass: [['s', 'b_mat', 'Material', [['PMMA', 'PMMA · cast acrylic'], ['SIBAL', 'SIBAL · cotton/fiberglass fabric'], ['Nomex', 'Nomex · flame-resistant suit'], ['Wax', 'Wax candle']]],
    ['s', 'b_geom', 'Sample geometry', [['Film', 'Thin film'], ['Slab', 'Solid slab'], ['Rod', 'Cylindrical rod'], ['Fabric', 'Woven fabric']]],
    ['s', 'b_color', 'Optical tint', [['standard', 'Standard'], ['clear', 'Clear'], ['black', 'Black']]],
    ['r', 'b_thick', 'Thickness / diameter', .1, 15, .1, 1, ' mm'], ['r', 'b_width', 'Sample width', .5, 5, .5, 2, ' cm'],
    ['r', 'b_fan', 'Cabin ventilation airflow', 10, 80, 1, 50, ''], ['r', 'b_o2', 'Initial cabin O₂', 15, 34, .5, 21, ' %']],
  flex: [['s', 'f_fuel', 'Fuel', [['Heptane', 'Heptane · cool-flame transition'], ['Methanol', 'Methanol · clean burning']]],
    ['r', 'f_p', 'Ambient pressure', .2, 3, .1, 1, ' bar'], ['r', 'f_d0', 'Initial diameter d₀', 1, 5, .1, 3, ' mm'],
    ['r', 'f_o2', 'Oxygen fraction X_O₂', .1, .4, .01, .21, ''], ['r', 'f_co2', 'CO₂ diluent X_CO₂', 0, .5, .05, 0, ''], ['r', 'f_he', 'Helium X_He', 0, .5, .05, 0, '']],
  cfig: [['s', 'c_fuel', 'Hydrocarbon fuel', [['ethane', 'Ethane (C₂H₆)'], ['propane', 'Propane (C₃H₈)'], ['50% butane', '50% butane mix'], ['30% butane', '30% butane mix'], ['30% propane', '30% propane mix']]],
    ['r', 'c_p', 'Chamber pressure', .5, 3, .1, 1.2, ' bar'], ['r', 'c_xf', 'Fuel fraction X_f', .1, 1, .05, .4, ''],
    ['r', 'c_xo2', 'Initial oxygen X_O₂', .15, .5, .02, .25, ''], ['r', 'c_mhc', 'Mass flow rate', .2, 3.5, .1, 1.2, ' mg/s'],
    ['s', 'c_ign', 'Hot ignition pulse', [['Yes', 'Yes'], ['No', 'No']]]],
  spice: [['s', 's_fuel', 'Co-flow fuel jet', [['Propylene', 'Propylene · high soot'], ['75% Propylene', '75% propylene'], ['50% Propylene', '50% propylene'], ['Propane', 'Propane'], ['Ethylene', 'Ethylene'], ['Methane', 'Methane · low soot']]],
    ['s', 's_nozzle', 'Nozzle diameter', [['0.4064', '0.4064 mm'], ['0.7620', '0.7620 mm'], ['1.6002', '1.6002 mm']], '0.7620'],
    ['r', 's_air', 'Co-flow air velocity', 5, 80, 1, 30, ' cm/s']]
};
function build(key, fn) {
  const spec = SPECS[key], el = $('ctl-' + key);
  el.innerHTML = spec.map(c => c[0] === 's'
    ? `<label class="ctl"><span>${c[2]}</span><select id="${c[1]}">${c[3].map(o => `<option value="${o[0]}">${o[1]}</option>`).join('')}</select></label>`
    : `<label class="ctl"><span>${c[2]}<b id="v_${c[1]}">${c[6]}${c[7]}</b></span><input type="range" id="${c[1]}" min="${c[3]}" max="${c[4]}" step="${c[5]}" value="${c[6]}"></label>`).join('');
  spec.forEach(c => { if (c[4] && c[0] === 's') $(c[1]).value = c[4]; });
  el.querySelectorAll('select,input').forEach(i => i.addEventListener('input', () => {
    const c = spec.find(x => x[1] === i.id), l = $('v_' + i.id); if (l) l.textContent = i.value + c[7]; fn();
  }));
}

// 5. API CLIENTS: fail loudly, warn on extrapolation, show confidence
function notes(key, d) {
  const w = $('warn-' + key);
  w.style.display = d.warnings.length ? 'block' : 'none';
  w.innerHTML = d.warnings.map(x => `<div>⚠ ${x}</div>`).join('');
  $('q-' + key).innerHTML = d.quality.map(x => `<span class="chip ${x.level}">${x.label} · ${x.level} confidence</span>${x.summary}`).join('<br>');
}
function fail(key, e, ids) {
  const w = $('warn-' + key); w.style.display = 'block'; w.innerHTML = `<div>⛔ Prediction unavailable: ${e.message}</div>`;
  ids.forEach(i => $(i).textContent = '--'); $('q-' + key).textContent = '';
}
const pm = (v, u) => `<span class="pm">± ${v} ${u}</span>`;

async function fetchBassPrediction() {
  try {
    const d = await post('/api/predict/bass', {material_type: val('b_mat'), geometry: val('b_geom'), color: val('b_color'), thickness_mm: num('b_thick'),
      width_cm: num('b_width'), fan_display: num('b_fan'), air_display: num('b_fan') / 10, o2_initial: num('b_o2')});
    flame.mode = d.is_safe ? 'safe' : 'hot';
    flame.r = d.is_safe ? 28 : 55 + num('b_thick') * 1.5;
    $('flame_status_text').textContent = d.is_safe ? 'FLAME STARVING · SELF-EXTINCTION LIKELY' : 'SUSTAINED DIFFUSION FLAME LIKELY';
    $('bass_ext_prob').textContent = d.extinction_probability + '%';
    $('bass_co_ppm').innerHTML = '≈ ' + d.co_ppm + ' ppm' + pm(d.co_error_ppm, 'ppm');
    $('bass_hazard_idx').textContent = d.non_extinction_risk + '%';
    const b = $('bass_banner');
    if (d.screening_pass) { b.className = 'banner safe'; b.innerHTML = '✦ <strong>Screening: likely self-extinguishing.</strong> The model estimates at least a 75% chance the flame starves on its own. A research estimate, not a compliance result.'; }
    else if (d.is_safe) { b.className = 'banner mid'; b.innerHTML = '◐ <strong>Inconclusive.</strong> Extinction is more likely than not, but below the 75% screening threshold.'; }
    else { b.className = 'banner danger'; b.innerHTML = '⚠ <strong>Sustained-burn risk.</strong> The model expects this fire to keep burning. Treat it as a scenario that needs suppression planning.'; }
    notes('bass', d);
  } catch (e) { fail('bass', e, ['bass_ext_prob', 'bass_co_ppm', 'bass_hazard_idx']); $('bass_banner').className = 'banner mid'; $('bass_banner').textContent = 'No prediction available.'; }
}
let flexChart;
async function fetchFlexPrediction() {
  try {
    const d = await post('/api/predict/flex', {fuel: val('f_fuel'), pressure_bar: num('f_p'), d0_mm: num('f_d0'), x_o2: num('f_o2'), x_co2: num('f_co2'), x_he: num('f_he')});
    $('flex_k_val').innerHTML = d.burning_rate_k + ' mm²/s' + pm(d.k_error, '');
    $('flex_burn_time').textContent = '≈ ' + d.total_burn_time_sec + ' s';
    $('flex_term_mode').textContent = d.termination_mode;
    notes('flex', d);
    if (!flexChart) {
      const tick = {color: '#98a0c8'}, grid = {color: 'rgba(160,170,255,.08)'};
      flexChart = new Chart($('flexChart'), {type: 'line',
        data: {labels: d.curve_time, datasets: [{label: 'Idealised d²-law: d(t) [mm]', data: d.curve_diameter, borderColor: '#ffb454', backgroundColor: 'rgba(255,90,54,.14)', fill: true, tension: .25, borderWidth: 2.5, pointRadius: 0}]},
        options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {labels: {color: '#98a0c8'}}},
          scales: {x: {title: {display: true, text: 'Time (s)', color: '#98a0c8'}, ticks: tick, grid}, y: {min: 0, title: {display: true, text: 'Diameter (mm)', color: '#98a0c8'}, ticks: tick, grid}}}});
    } else { flexChart.data.labels = d.curve_time; flexChart.data.datasets[0].data = d.curve_diameter; flexChart.update(); }
  } catch (e) { fail('flex', e, ['flex_k_val', 'flex_burn_time', 'flex_term_mode']); }
}
async function fetchCfigPrediction() {
  try {
    const d = await post('/api/predict/cfig', {fuel: val('c_fuel'), pressure_bar: num('c_p'), xf: num('c_xf'), xo2_initial: num('c_xo2'), mhc_mg_s: num('c_mhc'), hot_ignition: val('c_ign')});
    $('cfig_time_val').innerHTML = d.predicted_burn_time_s + ' s' + pm(d.burn_time_error_s, 's');
    $('cfig_ext_prob').textContent = d.self_extinction_probability + '%';
    notes('cfig', d);
  } catch (e) { fail('cfig', e, ['cfig_time_val', 'cfig_ext_prob']); }
}
async function fetchSpicePrediction() {
  try {
    const d = await post('/api/predict/spice', {fuel: val('s_fuel'), nozzle_mm: num('s_nozzle'), air_rate_cm_s: num('s_air')});
    $('spice_flow_val').innerHTML = d.critical_smoke_point_flow_mg_s + ' mg/s' + pm(d.flow_error_mg_s, 'mg/s');
    notes('spice', d);
  } catch (e) { fail('spice', e, ['spice_flow_val']); }
}

// 6. VERIFICATION (live from /api/benchmarks) + HERO COUNTERS
const LVC = {high: '#6ef3c5', medium: '#ffb454', low: '#ff8a70'};
const pct = v => Math.round(v * 100) + '%';
async function drawGauges() {
  let V;
  try { V = (await (await fetch('/api/benchmarks')).json()).validation; } catch (e) {}
  if (!V) { $('gauges').innerHTML = '<p class="note">Validation report not found. Run <code>python evaluate_grouped.py</code>.</p>'; return; }
  const C = 2 * Math.PI * 70;
  $('gauges').innerHTML = ['bass_ext', 'cfig_ext', 'flex_end'].map(k => { const r = V[k], c = LVC[r.level];
    return `<div class="gauge"><svg viewBox="0 0 170 170"><circle class="bg" cx="85" cy="85" r="70"/><circle class="fg" cx="85" cy="85" r="70" stroke="${c}" data-d="${r.grouped * C}"/></svg><div class="pct" style="color:${c}">${pct(r.grouped)}</div><p>${r.label}<br>unseen settings</p></div>`; }).join('');
  $('valtbl').innerHTML = '<tr><th>Model</th><th class="n">Metric</th><th class="n">Random split</th><th class="n">Unseen settings</th><th class="n">Error vs naive guess</th><th>Confidence</th></tr>' +
    Object.values(V).map(r => { const f = v => r.task === 'clf' ? pct(v) : v.toFixed(2);
      const e = r.task === 'clf' ? `${pct(1 - r.grouped)} wrong vs ${pct(1 - r.baseline)}` : `±${r.mae.toPrecision(2)} vs ±${r.baseline.toPrecision(2)} ${r.unit}`;
      return `<tr><td>${r.label}</td><td class="n">${r.metric}</td><td class="n">${f(r.random)}</td><td class="n">${f(r.grouped)}</td><td class="n">${e}</td><td><span class="chip ${r.level}">${r.level}</span></td></tr>`; }).join('');
  requestAnimationFrame(() => requestAnimationFrame(() => document.querySelectorAll('.fg').forEach(c => c.style.strokeDasharray = c.dataset.d + ' 999')));
}
document.querySelectorAll('[data-count]').forEach(el => {
  const end = +el.dataset.count, t0 = performance.now();
  (function step(t) { const p = Math.min(1, (t - t0) / 1800); el.textContent = Math.round(end * (1 - Math.pow(1 - p, 3))).toLocaleString(); if (p < 1) requestAnimationFrame(step); })(t0);
});

// BOOT
build('bass', fetchBassPrediction); build('flex', fetchFlexPrediction); build('cfig', fetchCfigPrediction); build('spice', fetchSpicePrediction);
fetchBassPrediction(); fetchFlexPrediction(); fetchCfigPrediction(); fetchSpicePrediction();
