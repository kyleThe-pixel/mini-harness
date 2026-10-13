"""results.json -> leaderboard.html (self-contained, sortable, no deps)."""
import json
from pathlib import Path


def generate(results_file, out_path):
    results = []
    p = Path(results_file)
    if p.exists():
        try:
            results = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = []
    html = _TEMPLATE.replace("__RESULTS_JSON__", json.dumps(results))
    Path(out_path).write_text(html, encoding="utf-8")


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RTX 3050 Benchmark Leaderboard</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
h1{font-size:1.6rem} h2{font-size:1.2rem;margin-top:2rem}
table{border-collapse:collapse;width:100%} th,td{padding:8px 10px;border-bottom:1px solid #333;text-align:left}
th{cursor:pointer;user-select:none} th:hover{color:#ffcc00}
tr.best td{background:#1d331f} .err{color:#ff6b6b} .muted{color:#999}
a{color:#7cc4ff}
</style></head><body>
<h1>RTX 3050 Benchmark Leaderboard</h1>
<p class="muted">Click a column header to sort. Best per model is highlighted.</p>
<h2>Best per model</h2><div id="best"></div>
<h2>All runs</h2>
<table><thead><tr>
<th data-k="timestamp">Date</th><th data-k="model">Model</th><th data-k="adapter">Adapter</th>
<th data-k="primary">Result</th><th data-k="wall_seconds">Wall (s)</th>
<th data-k="vram">Peak VRAM (MB)</th><th data-k="temp">Peak Temp (C)</th>
<th data-k="power">Avg Power (W)</th><th data-k="ram">Peak RAM (MB)</th><th>Status</th>
</tr></thead><tbody id="rows"></tbody></table>
<script>
const RESULTS = __RESULTS_JSON__;
const tbody = document.getElementById('rows');
function val(r, k){
  const m = r.metrics || {};
  if (k==='primary') return r.metrics ? (r.metrics.primary_value ?? '') : '';
  if (k==='vram') return m.peak_vram_mb ?? '';
  if (k==='temp') return m.peak_temp_c ?? '';
  if (k==='power') return m.avg_power_w ?? '';
  if (k==='ram') return m.peak_ram_mb ?? '';
  return r[k] ?? '';
}
function render(rows){
  tbody.innerHTML = rows.map(r => {
    const m = r.metrics || {};
    const cls = r.status === 'error' ? 'err' : '';
    const res = r.status === 'error' ? ('ERROR: ' + (r.error || '')) :
      (m.primary_value ?? '') + ' ' + (m.primary_label || '');
    return `<tr class="${cls}"><td>${r.timestamp}</td><td>${r.model}</td><td>${r.adapter}</td>` +
      `<td>${res}</td><td>${r.wall_seconds ?? ''}</td><td>${m.peak_vram_mb ?? ''}</td>` +
      `<td>${m.peak_temp_c ?? ''}</td><td>${m.avg_power_w ?? ''}</td><td>${m.peak_ram_mb ?? ''}</td>` +
      `<td>${r.status}</td></tr>`;
  }).join('');
}
let rows = [...RESULTS].reverse();
render(rows);
// best per model: tok/s -> max wins, seconds -> min wins
const byModel = {};
for (const r of RESULTS) {
  if (r.status !== 'ok' || !r.metrics) continue;
  const cur = byModel[r.model];
  const v = r.metrics.primary_value;
  const higher = (r.metrics.primary_label || '').includes('tok/s');
  if (!cur || (higher ? v > cur.metrics.primary_value : v < cur.metrics.primary_value)) byModel[r.model] = r;
}
document.getElementById('best').innerHTML = Object.values(byModel).map(r =>
  `<div><b>${r.model}</b>: ${r.metrics.primary_value} ${r.metrics.primary_label} ` +
  `(wall ${r.wall_seconds}s, VRAM ${r.metrics.peak_vram_mb ?? '?'} MB, ${r.timestamp})</div>`).join('') || '<span class="muted">No runs yet.</span>';
// sort on header click
let asc = {};
document.querySelectorAll('th[data-k]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k; asc[k] = !asc[k];
  rows.sort((a, b) => {
    const x = val(a, k), y = val(b, k);
    const cmp = (typeof x === 'number' && typeof y === 'number') ? x - y : String(x).localeCompare(String(y));
    return asc[k] ? cmp : -cmp;
  });
  render(rows);
}));
</script></body></html>
"""
