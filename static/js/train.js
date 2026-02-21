/**
 * train.js
 *
 * Two responsibilities:
 * 1. uploadDataset() — XHR upload with progress events (streaming to server)
 * 2. startTraining() — POST config, open SSE, drive charts + layer highlight
 */

// ── Chart setup ───────────────────────────────────────────────────────
const chartBase = (label, color) => ({
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: `Train ${label}`, data: [], borderColor: color, backgroundColor: color + '22',
                tension: 0.4, pointRadius: 3, fill: true
            },
            {
                label: `Val ${label}`, data: [], borderColor: '#94a3b8', backgroundColor: '#94a3b822',
                tension: 0.4, pointRadius: 3, fill: false, borderDash: [4, 3]
            },
        ]
    },
    options: {
        animation: { duration: 350 },
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
        scales: {
            x: { ticks: { color: '#64748b' }, grid: { color: '#1e2d45' } },
            y: { ticks: { color: '#64748b' }, grid: { color: '#1e2d45' } },
        }
    }
});

const lossChart = new Chart(document.getElementById('loss-chart'), chartBase('Loss', '#6366f1'));
const accChart = new Chart(document.getElementById('acc-chart'), chartBase('Acc', '#22d3a0'));

function pushChartPoint(epoch, trainLoss, valLoss, trainAcc, valAcc) {
    lossChart.data.labels.push(`E${epoch}`);
    lossChart.data.datasets[0].data.push(trainLoss);
    lossChart.data.datasets[1].data.push(valLoss);
    lossChart.update();
    accChart.data.labels.push(`E${epoch}`);
    accChart.data.datasets[0].data.push(trainAcc);
    accChart.data.datasets[1].data.push(valAcc);
    accChart.update();
}

function resetCharts() {
    [lossChart, accChart].forEach(c => {
        c.data.labels = [];
        c.data.datasets.forEach(d => d.data = []);
        c.update();
    });
}

// ── Helpers ───────────────────────────────────────────────────────────
function log(msg, type = 'info') {
    const el = document.getElementById('log');
    const div = document.createElement('div');
    div.className = `log-line ${type}`;
    div.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
}

function setStat(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

// ── SSE handlers ──────────────────────────────────────────────────────
function onInfo(data) {
    log(data.msg, 'info');
    if (data.device) {
        const badge = document.getElementById('device-badge');
        badge.textContent = data.device.startsWith('cuda') ? '⚡ GPU' : '🖥 CPU';
        badge.style.color = data.device.startsWith('cuda') ? 'var(--green)' : '';
    }
}

function onModelReady(data) {
    log(`Model ready — ${data.total_params.toLocaleString()} trainable params`, 'success');
    setStat('stat-status', 'Training…');
}

function onEpochStart(data) {
    log(`── Epoch ${data.epoch} / ${data.total_epochs} ──`, 'warn');
    setStat('stat-epoch', `${data.epoch} / ${data.total_epochs}`);
    setStat('stat-status', `Epoch ${data.epoch}`);
    highlightLayer(data.epoch - 1);
}

function onBatch(data) {
    setStat('stat-loss', data.loss);
    setStat('stat-acc', data.acc + '%');
}

function onEpochEnd(data) {
    pushChartPoint(data.epoch, data.train_loss, data.val_loss, data.train_acc, data.val_acc);
    setStat('stat-loss', data.train_loss);
    setStat('stat-val-loss', data.val_loss);
    setStat('stat-acc', data.train_acc + '%');
    setStat('stat-val-acc', data.val_acc + '%');
    log(`Epoch ${data.epoch}: loss=${data.train_loss}  val=${data.val_loss}  acc=${data.train_acc}%  val_acc=${data.val_acc}%`, 'success');
}

function onDone(data) {
    log(data.msg, 'success');
    setStat('stat-status', '✅ Done');
    document.getElementById('btn-train').disabled = false;
    document.getElementById('btn-train').textContent = '▶ Start Training';
    highlightLayer(-1);
    if (data.session_id) showDownloadBtn(data.session_id);
}

function showDownloadBtn(sessionId) {
    document.getElementById('dl-btn')?.remove();

    const wrap = document.createElement('div');
    wrap.id = 'dl-btn';
    wrap.style.cssText = 'display:flex;gap:.5rem;margin-top:.75rem;flex-wrap:wrap;';

    const makeBtn = (href, label, colors) => {
        const a = document.createElement('a');
        a.href = href;
        a.textContent = label;
        a.style.cssText = `
      display:inline-block;padding:.5rem 1rem;
      background:linear-gradient(135deg,${colors});
      color:#fff;font-weight:600;font-size:.8rem;
      border-radius:8px;text-decoration:none;
      box-shadow:0 4px 12px rgba(0,0,0,.25);
      transition:all .2s;font-family:Inter,sans-serif;
    `;
        a.onmouseover = () => a.style.transform = 'translateY(-1px)';
        a.onmouseout = () => a.style.transform = '';
        a.onclick = () => setTimeout(() => {
            a.textContent = '\u2713 Downloaded';
            a.style.opacity = '0.5';
            a.style.pointerEvents = 'none';
        }, 500);
        return a;
    };

    wrap.appendChild(makeBtn(`/download/${sessionId}/`, '\u2b07 model.pt', '#059669,#0d9488'));
    wrap.appendChild(makeBtn(`/metrics/${sessionId}/`, '\u2b07 metrics.csv', '#7c3aed,#6366f1'));

    document.getElementById('log').appendChild(wrap);
    document.getElementById('log').scrollTop = 9999;
}

function onError(data) {
    log('ERROR: ' + data.msg, 'err');
    setStat('stat-status', '❌ Error');
    document.getElementById('btn-train').disabled = false;
    document.getElementById('btn-train').textContent = '▶ Start Training';
}

function highlightLayer(idx) {
    const layers = window.getLayers();
    const active = layers.length ? idx % layers.length : -1;
    renderDiagram('net-svg-train', active);
}

// ── Streaming upload via XHR ──────────────────────────────────────────
/**
 * Uses XMLHttpRequest (not fetch) because XHR exposes upload.onprogress,
 * giving us byte-level progress while the file streams to the server.
 * fetch() supports progress only via ReadableStream which is more complex.
 */
function uploadDataset(file, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const form = new FormData();
        form.append('file', file);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) onProgress(e.loaded / e.total * 100);
        };
        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                if (data.error) reject(new Error(data.error));
                else resolve(data.upload_id);
            } else {
                reject(new Error(`Upload failed (HTTP ${xhr.status})`));
            }
        };
        xhr.onerror = () => reject(new Error('Network error during upload'));
        xhr.open('POST', '/upload/');
        xhr.send(form);
    });
}

// ── Upload progress UI ────────────────────────────────────────────────
function showUploadProgress(show) {
    document.getElementById('upload-box').style.display = show ? 'flex' : 'none';
}
function setUploadPct(pct) {
    document.getElementById('upload-progress-bar').style.width = pct + '%';
    document.getElementById('upload-pct').textContent = pct.toFixed(0) + '%';
}

// ── Main: Start Training ──────────────────────────────────────────────
window.startTraining = async function () {
    const layers = window.getLayers();
    if (layers.length === 0) { alert('Add at least one layer first!'); return; }

    const config = {
        dataset: document.getElementById('dataset').value,
        optimizer: document.getElementById('optimizer').value,
        loss: document.getElementById('loss').value,
        lr: parseFloat(document.getElementById('lr').value),
        epochs: parseInt(document.getElementById('epochs').value),
        layers,
    };

    // Reset UI
    resetCharts();
    document.getElementById('log').innerHTML = '';
    ['stat-epoch', 'stat-loss', 'stat-val-loss', 'stat-acc', 'stat-val-acc'].forEach(id => setStat(id, '—'));
    setStat('stat-status', 'Starting…');
    showUploadProgress(false);

    const btn = document.getElementById('btn-train');
    btn.disabled = true;
    btn.textContent = '⏳ Working…';

    // ── Custom dataset: upload first ──────────────────────────────────
    if (config.dataset === 'custom') {
        const file = window._droppedFile || document.getElementById('custom-file').files[0];
        if (!file) {
            alert('Please select or drag-and-drop a dataset ZIP file first!');
            btn.disabled = false;
            btn.textContent = '▶ Start Training';
            return;
        }

        setStat('stat-status', 'Uploading…');
        showUploadProgress(true);
        setUploadPct(0);

        // Switch to training tab so user sees upload progress
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-train').classList.add('active');
        document.getElementById('tab-train').classList.add('active');

        log(`Uploading "${file.name}" (${(file.size / 1024 / 1024).toFixed(1)} MB)…`, 'info');

        try {
            const uploadId = await uploadDataset(file, (pct) => {
                setUploadPct(pct);
                if (pct % 10 < 1) log(`Upload ${pct.toFixed(0)}%`, 'info');
            });
            setUploadPct(100);
            log(`Upload complete! ID: ${uploadId}`, 'success');
            config.upload_id = uploadId;
        } catch (e) {
            log('Upload failed: ' + e.message, 'err');
            setStat('stat-status', '❌ Upload Error');
            btn.disabled = false;
            btn.textContent = '▶ Start Training';
            showUploadProgress(false);
            return;
        }

        showUploadProgress(false);
    } else {
        // For built-in datasets, switch panel immediately
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-train').classList.add('active');
        document.getElementById('tab-train').classList.add('active');
    }

    btn.textContent = '⏳ Training…';
    renderDiagram('net-svg-train', -1);

    log(`Config: dataset=${config.dataset}, opt=${config.optimizer}, loss=${config.loss}, lr=${config.lr}, epochs=${config.epochs}`, 'info');

    // POST config → session_id
    let sessionId;
    try {
        const resp = await fetch('/train/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const json = await resp.json();
        if (json.error) throw new Error(json.error);
        sessionId = json.session_id;
    } catch (e) {
        log('Failed to start training: ' + e.message, 'err');
        setStat('stat-status', '❌ Error');
        btn.disabled = false;
        btn.textContent = '▶ Start Training';
        return;
    }

    // Open SSE connection
    const sse = new EventSource(`/stream/${sessionId}/`);

    const handlers = {
        info: onInfo,
        model_ready: onModelReady,
        epoch_start: onEpochStart,
        batch: onBatch,
        epoch_end: onEpochEnd,
        done: (d) => { onDone(d); sse.close(); },
        error: (d) => { onError(d); sse.close(); },
    };

    Object.entries(handlers).forEach(([event, fn]) => {
        sse.addEventListener(event, (e) => fn(JSON.parse(e.data)));
    });

    sse.onerror = () => {
        log('SSE connection dropped', 'err');
        sse.close();
    };
};
