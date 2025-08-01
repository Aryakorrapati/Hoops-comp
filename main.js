/* global Chart */
let pyodide, compareFunc

// main.js
async function bootPyodide () {
  const log = txt => (document.getElementById("status").textContent = txt);

  log("Loading Python runtime…");
  const pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.1/full/",
    disableIntegrityCheck: true,   // avoids the hash mismatch fiasco
  });

  /* ① copy data files into Pyodide’s virtual FS */
  for (const f of [
    "pace.csv",
    "cbb_stats.csv",
    "nba_stats.csv",
    "legacy_script.py",
  ]) {
    const buf = await (await fetch(f)).arrayBuffer();
    pyodide.FS.writeFile(f, new Uint8Array(buf));
  }

  /* ② install the HTTP shim so `import requests` works */
  log("Patching HTTP…");
  await pyodide.loadPackage("pyodide-http");    // tiny wheel
  await pyodide.runPythonAsync(`
    import pyodide_http
    pyodide_http.patch_all()   # provides fetch-backed 'requests'
  `);

  /* ③ pull the heavy scientific stack */
  log("Downloading packages");
  await pyodide.loadPackage([
    "pandas", "numpy", "matplotlib", "scikit-learn", "scipy", "lxml", "requests", "beautifulsoup4",
  ]);

  /* ④ import your monolith and expose run_all */
  await pyodide.runPythonAsync("import legacy_script as ls");
  log("Ready!");

  return pyodide.globals.get("ls").get("run_all");
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
