/* global Chart */
// ------------------------------------------------------------
// main.js  — Pyodide front-end for CBB → NBA comparator
// ------------------------------------------------------------
let pyodide;          // will hold the Pyodide runtime
let compareFunc;      // Python function we call from JS

// -------------- 1.  Boot the Python runtime -----------------
async function bootPyodide () {
  // a) spin-up CPython in the browser
  document.getElementById('status').textContent = 'Loading Python…';
  pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.1/full/'
  });

  // b) load the binary packages we need
  await pyodide.loadPackage([
    'numpy', 'pandas', 'matplotlib', 'scikit-learn',
    'lxml', 'beautifulsoup4', 'pyodide-http', 'requests'
  ]);

  // c) copy the data-and-code files from GitHub pages → Pyodide FS
  const files = [
    'legacy_script.py',
    'pace.csv', 'cbb_stats.csv', 'nba_stats.csv'
  ];
  for (const f of files) {
    const resp = await fetch(f);
    pyodide.FS.writeFile(f, new Uint8Array(await resp.arrayBuffer()));
  }

  // d) ***MAKE legacy_script A REAL MODULE***
  //    (this is what fixes the “No module named 'legacy_script'” error)
  await pyodide.runPythonAsync(`
import sys, importlib.util, types
spec = importlib.util.spec_from_file_location("legacy_script", "legacy_script.py")
legacy_script = importlib.util.module_from_spec(spec)
sys.modules["legacy_script"] = legacy_script
spec.loader.exec_module(legacy_script)
  `);

  // e) pull out the compare function for quick reuse
  compareFunc = pyodide.globals.get('legacy_script').get('run_compare');
  document.getElementById('status').textContent = 'Python ready ✔';
}

// -------------- 2.  Run a comparison on button-click --------
async function runCompare () {
  const url = document.getElementById('url').value.trim();
  if (!url) { alert('Paste a URL first'); return; }

  document.getElementById('status').textContent = 'Running…';
  try {
    // call Python → get plain JS back
    const res = compareFunc(url).toJs({ dict_converter: Object });

    // show text report
    document.getElementById('report').textContent = res.text;

    // (optional) draw radar chart if res.chart_labels & res.chart_vals exist
    if (res.chart_labels && res.chart_vals) {
      new Chart(
        document.getElementById('chart'),
        {
          type: 'radar',
          data: {
            labels: res.chart_labels,
            datasets: [{ label: 'Similarity', data: res.chart_vals }]
          },
          options: { responsive: true, scales: { r: { beginAtZero: true } } }
        }
      );
    }
    document.getElementById('status').textContent = 'Done ✔';
  } catch (err) {
    console.error(err);
    document.getElementById('status').textContent = 'Error – see console';
    alert(err);
  }
}

// -------------- 3.  Wire things up --------------------------
document.getElementById('runBtn').onclick = runCompare;
bootPyodide();