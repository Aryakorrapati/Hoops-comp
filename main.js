/* global Chart */
let pyodide, compareFunc

async function bootPyodide () {
  document.getElementById('status').textContent = 'Loading Python…'
  pyodide = await loadPyodide({ indexURL:
      'https://cdn.jsdelivr.net/pyodide/v0.27.1/full/' })
  await pyodide.loadPackage([
    'micropip', 'numpy', 'pandas', 'matplotlib',      // …
    'scikit-learn', 'lxml', 'beautifulsoup4',
    'pyodide-http', 'requests'
  ])

  // mount data files exactly as before …
}

async function runCompare () {
  const url = document.getElementById('url').value.trim()
  if (!url) { alert('Paste a URL first'); return }

  document.getElementById('status').textContent = 'Running…'
  const { run_compare } = pyodide.pyimport('legacy_script')

  // call Python function and get a JS-friendly object back
  const result = run_compare(url).toJs({ dict_converter: Object })

  document.getElementById('report').textContent = result.text
  const ctx = document.getElementById('chart')
  /* draw the first radar chart with Chart.js (example) */
  new Chart(ctx, {
    type: 'radar',
    data: { /* …convert PNG bytes if you still want charts in JS… */ }
  })
  document.getElementById('status').textContent = 'Done'
}

document.getElementById('runBtn').onclick = runCompare
bootPyodide()
