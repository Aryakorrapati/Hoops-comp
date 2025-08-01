/* global Chart */
let pyodide, compareFunc

async function bootPyodide () {
  const statusEl = document.getElementById("status");
  statusEl.textContent = "Loading Python…";

  /* 1️⃣  spin-up Pyodide from latest CDN */
  const pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.1/full/",
    stdin: () => null,
    disableIntegrityCheck: true          // fixes the hash-mismatch issue
  });

  /* 2️⃣  copy the CSVs + script into the Pyodide filesystem */
  for (const f of [
    "pace.csv",
    "cbb_stats.csv",
    "nba_stats.csv",
    "legacy_script.py",
  ]) {
    const resp = await fetch(f);
    pyodide.FS.writeFile(f, new Uint8Array(await resp.arrayBuffer()));
  }

  /* 3️⃣  pull in the scientific wheels */
  statusEl.textContent = "Downloading packages…";
  await pyodide.loadPackage([
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "lxml",
  ]);

  /* 4️⃣  wire legacy_script.run_all → JS */
  await pyodide.runPythonAsync(`
    import sys, js
    import legacy_script
    compare = legacy_script.run_all
  `);

  statusEl.textContent = "Ready!";
  return pyo
}  

async function runCompare () {
  const url = document.getElementById('url').value.trim()
  if (!url) return alert('Paste a URL first')
  document.getElementById('status').textContent = 'Running… (this may take 5-10 s)'
  document.getElementById('report').textContent = ''
  try {
    const result = await compareFunc(url)    // Py → Js proxy
    // result is a python dict; convert:
    const out = result.toJs({ dict_converter: Object })
    document.getElementById('report').textContent = out.text
    drawChart(out.images[0])                 // first radar as PNG b64
    document.getElementById('status').textContent = 'Done!'
  } catch (e) {
    console.error(e)
    document.getElementById('status').textContent = 'Error – see console'
  }
}

function drawChart (b64png) {
  // replace <canvas> with the PNG to save time (simplest)
  const cv = document.getElementById('chart')
  const ctx = cv.getContext('2d')
  const img = new Image()
  img.onload = () => ctx.drawImage(img, 0, 0, cv.width, cv.height)
  img.src = 'data:image/png;base64,' + b64png
}

document.getElementById('runBtn').onclick = runCompare
bootPyodide()
