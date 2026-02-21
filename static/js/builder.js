/**
 * builder.js
 *
 * Node-based Neural Network Visualization:
 * - Each layer = vertical column of circles (neurons)
 * - Bezier curves connect every node in layer i → layer i+1
 * - Active layer during training pulses with a glow effect
 * - Layer count per column is capped at MAX_NODES; real count shown with "+N"
 */

// ── State ─────────────────────────────────────────────────────────────
let layers = [];

const DEFAULTS = {
    conv2d: { in_channels: 1, out_channels: 32, kernel_size: 3, padding: 1 },
    maxpool2d: { kernel_size: 2 },
    flatten: {},
    linear: { in_features: 128, out_features: 64 },
    relu: {},
    leakyrelu: {},
    sigmoid: {},
    tanh: {},
    dropout: { p: 0.5 },
    batchnorm2d: { num_features: 32 },
};

const COLORS = {
    conv2d: { fill: '#1a3055', stroke: '#2563eb', text: '#60a5fa', ring: '#3b82f6' },
    maxpool2d: { fill: '#1a3055', stroke: '#1d4ed8', text: '#93c5fd', ring: '#3b82f6' },
    flatten: { fill: '#2a1a55', stroke: '#7c3aed', text: '#a78bfa', ring: '#8b5cf6' },
    linear: { fill: '#1a3d2e', stroke: '#059669', text: '#34d399', ring: '#10b981' },
    relu: { fill: '#3a2a0e', stroke: '#d97706', text: '#fbbf24', ring: '#f59e0b' },
    leakyrelu: { fill: '#3a2a0e', stroke: '#b45309', text: '#fcd34d', ring: '#f59e0b' },
    sigmoid: { fill: '#3a2a0e', stroke: '#c2410c', text: '#fb923c', ring: '#f97316' },
    tanh: { fill: '#3a2a0e', stroke: '#a16207', text: '#facc15', ring: '#eab308' },
    dropout: { fill: '#3a1818', stroke: '#dc2626', text: '#f87171', ring: '#ef4444' },
    batchnorm2d: { fill: '#1a3838', stroke: '#0d9488', text: '#2dd4bf', ring: '#14b8a6' },
};

// ── Node count helpers ────────────────────────────────────────────────
const MAX_NODES = 6;

function getActualNodeCount(spec) {
    if (spec.type === 'conv2d') return spec.out_channels || 32;
    if (spec.type === 'linear') return spec.out_features || 64;
    if (spec.type === 'batchnorm2d') return spec.num_features || 32;
    return null; // "carry forward" types
}

function computeNodeCounts() {
    const counts = [];
    let prev = 4;
    for (const s of layers) {
        const actual = getActualNodeCount(s);
        let visible;
        if (actual !== null) {
            visible = Math.min(actual, MAX_NODES);
            prev = visible;
        } else if (s.type === 'flatten') {
            visible = Math.min(6, MAX_NODES);
            prev = visible;
        } else if (s.type === 'maxpool2d') {
            visible = Math.max(2, Math.ceil(prev * 0.67));
            prev = visible;
        } else {
            visible = prev; // activations, dropout
        }
        counts.push(visible);
    }
    return counts;
}

// ── Layer description (for right-panel list) ──────────────────────────
function describeLayer(spec) {
    const t = spec.type;
    if (t === 'conv2d') return `${spec.in_channels}→${spec.out_channels}, k=${spec.kernel_size}, p=${spec.padding}`;
    if (t === 'linear') return `${spec.in_features}→${spec.out_features}`;
    if (t === 'dropout') return `p=${spec.p}`;
    if (t === 'batchnorm2d') return `features=${spec.num_features}`;
    if (t === 'maxpool2d') return `k=${spec.kernel_size}`;
    return '';
}

function estimateParams() {
    let total = 0;
    for (const s of layers) {
        if (s.type === 'conv2d')
            total += s.out_channels * s.in_channels * s.kernel_size * s.kernel_size + s.out_channels;
        else if (s.type === 'linear')
            total += s.in_features * s.out_features + s.out_features;
        else if (s.type === 'batchnorm2d')
            total += s.num_features * 2;
    }
    return total;
}

// ── SVG Node Diagram ──────────────────────────────────────────────────
/**
 * Renders a horizontal neural network diagram as SVG.
 * Layout: left-to-right, one column of circles per layer.
 * Bezier curves connect consecutive columns.
 * activeIdx highlights that column with a glow.
 */
function renderDiagram(svgId, activeIdx = -1) {
    const svg = document.getElementById(svgId);
    if (!svg) return;

    if (layers.length === 0) {
        svg.setAttribute('width', 10);
        svg.setAttribute('height', 10);
        svg.innerHTML = '';
        return;
    }

    const NODE_R = 9;
    const NODE_GAP = 22;
    const COL_W = 84;
    const SVG_H = 280;
    const MARGIN = 36;
    const CY = SVG_H / 2;

    const nodeCounts = computeNodeCounts();
    const totalW = MARGIN * 2 + layers.length * COL_W;

    svg.setAttribute('width', totalW);
    svg.setAttribute('height', SVG_H);

    // ── Build SVG string ──────────────────────────────────────────────
    let out = `
  <defs>
    <filter id="glow-node" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-soft" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;

    // Draw connections (behind nodes)
    for (let i = 0; i < layers.length - 1; i++) {
        const n1 = nodeCounts[i];
        const n2 = nodeCounts[i + 1];
        const x1 = MARGIN + (i + 0.5) * COL_W + NODE_R;
        const x2 = MARGIN + (i + 1.5) * COL_W - NODE_R;
        const mx = (x1 + x2) / 2;
        const isActiveConn = (i === activeIdx || i + 1 === activeIdx);
        const opacity = isActiveConn ? 0.55 : 0.18;
        const connColor = isActiveConn
            ? (COLORS[layers[i + 1]?.type] || COLORS.relu).stroke
            : '#334155';

        for (let a = 0; a < n1; a++) {
            const y1 = CY + (a - (n1 - 1) / 2) * NODE_GAP;
            for (let b = 0; b < n2; b++) {
                const y2 = CY + (b - (n2 - 1) / 2) * NODE_GAP;
                out += `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"
          fill="none" stroke="${connColor}" stroke-width="0.8" opacity="${opacity}"/>`;
            }
        }
    }

    // Draw nodes per layer
    for (let i = 0; i < layers.length; i++) {
        const s = layers[i];
        const c = COLORS[s.type] || COLORS.relu;
        const isActive = i === activeIdx;
        const n = nodeCounts[i];
        const cx = MARGIN + (i + 0.5) * COL_W;
        const filter = isActive ? 'filter="url(#glow-node)"' : 'filter="url(#glow-soft)"';

        // Pulse ring for active layer
        if (isActive) {
            const ringH = (n - 1) * NODE_GAP + NODE_R * 2 + 12;
            const ringY = CY - (n - 1) / 2 * NODE_GAP - NODE_R - 6;
            out += `<rect x="${cx - 18}" y="${ringY}" width="36" height="${ringH}"
        rx="16" fill="${c.ring}22" stroke="${c.ring}" stroke-width="1"
        opacity="0.7"/>`;
        }

        // Nodes (circles)
        for (let j = 0; j < n; j++) {
            const cy = CY + (j - (n - 1) / 2) * NODE_GAP;
            out += `<circle cx="${cx}" cy="${cy}" r="${NODE_R}"
        fill="${c.fill}" stroke="${isActive ? c.text : c.stroke}"
        stroke-width="${isActive ? 2.5 : 1.2}" ${filter}/>`;
            // Inner dot
            out += `<circle cx="${cx}" cy="${cy}" r="3"
        fill="${c.stroke}" opacity="${isActive ? 1 : 0.5}"/>`;
        }

        // "+N more" label if truncated
        const actual = getActualNodeCount(s);
        if (actual !== null && actual > MAX_NODES) {
            const lastY = CY + ((n - 1) - (n - 1) / 2) * NODE_GAP + NODE_R + 11;
            out += `<text x="${cx}" y="${lastY}" text-anchor="middle"
        fill="${c.text}" font-size="8" font-family="JetBrains Mono,monospace">+${actual - MAX_NODES}</text>`;
        }

        // Layer name (bottom)
        out += `<text x="${cx}" y="${SVG_H - 8}" text-anchor="middle"
      fill="${c.text}" font-size="9" font-weight="600" font-family="Inter,sans-serif">${s.type}</text>`;

        // Layer number (top)
        out += `<text x="${cx}" y="12" text-anchor="middle"
      fill="${c.stroke}" font-size="8" font-family="Inter,sans-serif">${i + 1}</text>`;
    }

    svg.innerHTML = out;
}

// ── Render layer list (right panel) ──────────────────────────────────
function renderLayerList() {
    const el = document.getElementById('layer-list');
    el.innerHTML = layers.map((s, i) => {
        const c = COLORS[s.type] || COLORS.relu;
        const desc = describeLayer(s);
        return `
    <div class="layer-item" id="li-${i}">
      <div class="layer-num" style="background:${c.stroke}">${i + 1}</div>
      <div class="layer-info">
        <div class="layer-name">${s.type}</div>
        ${desc ? `<div class="layer-params">${desc}</div>` : ''}
        ${renderInlineEditor(s, i)}
      </div>
      <button class="layer-del" onclick="removeLayer(${i})" title="Remove">✕</button>
    </div>`;
    }).join('');
}

function renderInlineEditor(spec, idx) {
    const t = spec.type;
    const inp = (field, val, step = 1, min = 1) =>
        `<input type="number" class="inline-num" value="${val}" step="${step}" min="${min}"
      onchange="updateParam(${idx},'${field}',+this.value)"
      onclick="event.stopPropagation()"
      style="width:46px;margin-right:2px;background:var(--surface);border:1px solid var(--border);
             border-radius:4px;color:var(--text);font-size:.65rem;padding:1px 4px;">`;

    if (t === 'conv2d')
        return `<div style="margin-top:3px;font-size:.65rem;color:var(--muted)">
      in:${inp('in_channels', spec.in_channels)}
      out:${inp('out_channels', spec.out_channels)}
      k:${inp('kernel_size', spec.kernel_size)}
      p:${inp('padding', spec.padding, 1, 0)}</div>`;
    if (t === 'linear')
        return `<div style="margin-top:3px;font-size:.65rem;color:var(--muted)">
      in:${inp('in_features', spec.in_features)}
      out:${inp('out_features', spec.out_features)}</div>`;
    if (t === 'dropout')
        return `<div style="margin-top:3px;font-size:.65rem;color:var(--muted)">
      p:${inp('p', spec.p, 0.05, 0)}</div>`;
    if (t === 'batchnorm2d')
        return `<div style="margin-top:3px;font-size:.65rem;color:var(--muted)">
      features:${inp('num_features', spec.num_features)}</div>`;
    if (t === 'maxpool2d')
        return `<div style="margin-top:3px;font-size:.65rem;color:var(--muted)">
      k:${inp('kernel_size', spec.kernel_size)}</div>`;
    return '';
}

// ── Master render ─────────────────────────────────────────────────────
function renderAll() {
    renderLayerList();
    renderDiagram('net-svg');
    document.getElementById('layer-count').textContent = layers.length;
    const p = estimateParams();
    document.getElementById('param-count').textContent =
        p > 0 ? `~${p.toLocaleString()} params` : '— params';
    document.getElementById('empty-state').style.display = layers.length ? 'none' : 'flex';
}

// ── Public API ────────────────────────────────────────────────────────
window.addLayer = (type) => { layers.push({ type, ...DEFAULTS[type] }); renderAll(); };
window.removeLayer = (idx) => { layers.splice(idx, 1); renderAll(); };
window.updateParam = (idx, field, val) => { layers[idx][field] = val; renderAll(); };
window.clearLayers = () => { layers = []; renderAll(); };
window.getLayers = () => layers;
window.renderDiagram = renderDiagram; // used by train.js

// ── Quick Templates ───────────────────────────────────────────────────
window.loadTemplate = function (name) {
    if (name === 'mlp') {
        document.getElementById('dataset').value = 'mnist';
        onDatasetChange('mnist');
        layers = [
            { type: 'flatten' },
            { type: 'linear', in_features: 784, out_features: 256 },
            { type: 'relu' },
            { type: 'linear', in_features: 256, out_features: 128 },
            { type: 'relu' },
            { type: 'dropout', p: 0.3 },
            { type: 'linear', in_features: 128, out_features: 10 },
        ];
    } else if (name === 'cnn') {
        document.getElementById('dataset').value = 'cifar10';
        onDatasetChange('cifar10');
        layers = [
            { type: 'conv2d', in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1 },
            { type: 'relu' },
            { type: 'maxpool2d', kernel_size: 2 },
            { type: 'conv2d', in_channels: 32, out_channels: 64, kernel_size: 3, padding: 1 },
            { type: 'relu' },
            { type: 'maxpool2d', kernel_size: 2 },
            { type: 'flatten' },
            { type: 'linear', in_features: 4096, out_features: 128 },
            { type: 'relu' },
            { type: 'linear', in_features: 128, out_features: 10 },
        ];
    } else if (name === 'deep') {
        document.getElementById('dataset').value = 'cifar10';
        onDatasetChange('cifar10');
        layers = [
            { type: 'conv2d', in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1 },
            { type: 'batchnorm2d', num_features: 32 },
            { type: 'relu' },
            { type: 'conv2d', in_channels: 32, out_channels: 64, kernel_size: 3, padding: 1 },
            { type: 'batchnorm2d', num_features: 64 },
            { type: 'relu' },
            { type: 'maxpool2d', kernel_size: 2 },
            { type: 'conv2d', in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1 },
            { type: 'batchnorm2d', num_features: 128 },
            { type: 'relu' },
            { type: 'maxpool2d', kernel_size: 2 },
            { type: 'flatten' },
            { type: 'linear', in_features: 8192, out_features: 256 },
            { type: 'relu' },
            { type: 'dropout', p: 0.4 },
            { type: 'linear', in_features: 256, out_features: 10 },
        ];
    }
    renderAll();
};

// ── Dataset change handler ────────────────────────────────────────────
window.onDatasetChange = function (val) {
    document.getElementById('custom-ds-area').style.display =
        val === 'custom' ? 'block' : 'none';
};

// ── Drag & drop for ZIP upload zone ──────────────────────────────────
window._droppedFile = null;

window.dzDragOver = (e) => {
    e.preventDefault();
    document.getElementById('drop-zone').classList.add('dragover');
};
window.dzDragLeave = () => {
    document.getElementById('drop-zone').classList.remove('dragover');
};
window.dzDrop = (e) => {
    e.preventDefault();
    document.getElementById('drop-zone').classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
        window._droppedFile = file;
        document.getElementById('drop-label').textContent = `📦 ${file.name}`;
    }
};
window.onFileSelected = (input) => {
    window._droppedFile = null; // clear drop override — file input takes precedence
    if (input.files[0]) {
        document.getElementById('drop-label').textContent = `📦 ${input.files[0].name}`;
    }
};

// ── Panel switcher ────────────────────────────────────────────────────
window.showPanel = function (name, btn) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${name}`).classList.add('active');
    btn.classList.add('active');
};

// Init
renderAll();
