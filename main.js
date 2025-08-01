/* global Chart */
let pyodide, compareFunc

async function bootPyodide () {
  document.getElementById('status').textContent = 'Loading Python…'
  pyodide = await loadPyodide({ stdin: () => null })
  // Mount data files into Pyodide FS
  for (const f of ['pace.csv', 'cbb_stats.csv', 'nba_stats.csv', 'legacy_script.py']) {
    const resp = await fetch(f)
    pyodide.FS.writeFile(f, new Uint8Array(await resp.arrayBuffer()))
  }

  await pyodide.loadPackage([
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "lxml",
    "beautifulsoup4",
    "pillow"          // used by matplotlib for PNG
  ]);
  await pyodide.runPythonAsync(`
    import sys, js, importlib.util, types
    # pandas & numpy are already built-in to Pyodide
    import legacy_script  # executes your file once
    compare = legacy_script.run_all        # entry point
  `)
  compareFunc = pyodide.globals.get('compare')
  document.getElementById('status').textContent = 'Ready!'
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
