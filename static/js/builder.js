/**
 * builder.js — Universal Neural Network Builder
 *
 * Features:
 * - Supports ANY torch.nn module with skip/residual connections
 * - Dual viz: Node-based SVG + Architecture block diagram (with skip arrows)
 * - Categorized palette, drag-to-reorder, inline editors
 * - Gemini AI integration
 */

// ═══════ State ═══════
let layers = [];
let _lastGeminiResult = null;
let _viewMode = 'nodes';

// ═══════ SVG Icon Helpers ═══════
const ICONS = {
    conv: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    pool: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><path d="M3 3h18v18H3z"/><path d="M12 3v18M3 12h18"/></svg>',
    linear: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
    flat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><rect x="3" y="10" width="18" height="4" rx="1"/></svg>',
    act: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><polyline points="4 18 10 18 14 6 20 6"/></svg>',
    drop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>',
    bn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/></svg>',
    transformer: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>',
    rnn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>',
    utility: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><circle cx="12" cy="12" r="10"/><path d="M12 8v8M8 12h8"/></svg>',
    input: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><rect x="4" y="4" width="16" height="16" rx="2"/><polyline points="9 12 12 15 15 12"/><line x1="12" y1="7" x2="12" y2="15"/></svg>',
    output: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><rect x="4" y="4" width="16" height="16" rx="2"/><polyline points="9 12 12 9 15 12"/><line x1="12" y1="9" x2="12" y2="17"/></svg>',
    skip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="licon"><path d="M7 17l9.2-9.2M17 17V7H7"/></svg>',
};

// ═══════ Palette ═══════
const PALETTE = [
    { name: 'Convolution', open: true, items: [['Conv1d', 'conv'], ['Conv2d', 'conv'], ['ConvTranspose2d', 'conv']] },
    { name: 'Pooling', open: true, items: [['MaxPool2d', 'pool'], ['AvgPool2d', 'pool'], ['AdaptiveAvgPool2d', 'pool'], ['MaxPool1d', 'pool']] },
    { name: 'Linear & Utility', open: true, items: [['Linear', 'linear'], ['Flatten', 'flat'], ['Embedding', 'utility'], ['Unflatten', 'flat']] },
    { name: 'Activations', open: true, items: [['ReLU', 'act'], ['LeakyReLU', 'act'], ['GELU', 'act'], ['SiLU', 'act'], ['Sigmoid', 'act'], ['Tanh', 'act'], ['Softmax', 'act'], ['LogSoftmax', 'act']] },
    { name: 'Normalization', open: true, items: [['BatchNorm1d', 'bn'], ['BatchNorm2d', 'bn'], ['LayerNorm', 'bn'], ['GroupNorm', 'bn']] },
    { name: 'Regularization', open: true, items: [['Dropout', 'drop'], ['Dropout2d', 'drop'], ['AlphaDropout', 'drop']] },
    { name: 'Transformer', open: false, items: [['TransformerEncoderLayer', 'transformer'], ['TransformerDecoderLayer', 'transformer'], ['TransformerEncoder', 'transformer'], ['MultiheadAttention', 'transformer']] },
    { name: 'Recurrent', open: false, items: [['LSTM', 'rnn'], ['GRU', 'rnn'], ['RNN', 'rnn']] },
];

const DEFAULTS = {
    Conv1d: { in_channels: 1, out_channels: 32, kernel_size: 3, padding: 1 },
    Conv2d: { in_channels: 1, out_channels: 32, kernel_size: 3, padding: 1 },
    ConvTranspose2d: { in_channels: 32, out_channels: 16, kernel_size: 3, padding: 1 },
    MaxPool2d: { kernel_size: 2 }, MaxPool1d: { kernel_size: 2 }, AvgPool2d: { kernel_size: 2 },
    AdaptiveAvgPool2d: { output_size: 1 },
    Flatten: {}, Unflatten: { dim: 1, unflattened_size: [1, 1] },
    Linear: { in_features: 128, out_features: 64 },
    Embedding: { num_embeddings: 10000, embedding_dim: 128 },
    ReLU: {}, LeakyReLU: {}, GELU: {}, SiLU: {}, Sigmoid: {}, Tanh: {},
    Softmax: { dim: 1 }, LogSoftmax: { dim: 1 },
    Dropout: { p: 0.5 }, Dropout2d: { p: 0.5 }, AlphaDropout: { p: 0.5 },
    BatchNorm1d: { num_features: 64 }, BatchNorm2d: { num_features: 32 },
    LayerNorm: { normalized_shape: [64] }, GroupNorm: { num_groups: 8, num_channels: 32 },
    TransformerEncoderLayer: { d_model: 128, nhead: 8 },
    TransformerDecoderLayer: { d_model: 128, nhead: 8 },
    TransformerEncoder: {},
    MultiheadAttention: { embed_dim: 128, num_heads: 8 },
    LSTM: { input_size: 128, hidden_size: 64, batch_first: true, _mode: 'rnn' },
    GRU: { input_size: 128, hidden_size: 64, batch_first: true, _mode: 'rnn' },
    RNN: { input_size: 128, hidden_size: 64, batch_first: true, _mode: 'rnn' },
};

const CATMAP = {
    Conv1d: 'conv', Conv2d: 'conv', ConvTranspose2d: 'conv',
    MaxPool2d: 'pool', MaxPool1d: 'pool', AvgPool2d: 'pool', AdaptiveAvgPool2d: 'pool',
    Flatten: 'flat', Unflatten: 'flat', Linear: 'linear', Embedding: 'utility',
    ReLU: 'act', LeakyReLU: 'act', GELU: 'act', SiLU: 'act', Sigmoid: 'act', Tanh: 'act', Softmax: 'act', LogSoftmax: 'act',
    Dropout: 'drop', Dropout2d: 'drop', AlphaDropout: 'drop',
    BatchNorm1d: 'bn', BatchNorm2d: 'bn', LayerNorm: 'bn', GroupNorm: 'bn',
    TransformerEncoderLayer: 'transformer', TransformerDecoderLayer: 'transformer',
    TransformerEncoder: 'transformer', TransformerDecoder: 'transformer', MultiheadAttention: 'transformer',
    LSTM: 'rnn', GRU: 'rnn', RNN: 'rnn',
};

const COLORS = {
    conv: { fill: '#dbeafe', stroke: '#3b82f6', text: '#1d4ed8', ring: '#60a5fa' },
    pool: { fill: '#e0e7ff', stroke: '#6366f1', text: '#3730a3', ring: '#818cf8' },
    flat: { fill: '#ede9fe', stroke: '#8b5cf6', text: '#6d28d9', ring: '#a78bfa' },
    linear: { fill: '#d1fae5', stroke: '#10b981', text: '#047857', ring: '#34d399' },
    utility: { fill: '#f3e8ff', stroke: '#a855f7', text: '#7e22ce', ring: '#c084fc' },
    act: { fill: '#fef3c7', stroke: '#f59e0b', text: '#b45309', ring: '#fbbf24' },
    drop: { fill: '#fee2e2', stroke: '#ef4444', text: '#b91c1c', ring: '#f87171' },
    bn: { fill: '#ccfbf1', stroke: '#14b8a6', text: '#0f766e', ring: '#2dd4bf' },
    transformer: { fill: '#ede9fe', stroke: '#8b5cf6', text: '#5b21b6', ring: '#a78bfa' },
    rnn: { fill: '#cffafe', stroke: '#06b6d4', text: '#0e7490', ring: '#22d3ee' },
};

function getCat(s) { return CATMAP[s.type] || 'linear'; }
function getColor(s) { return COLORS[getCat(s)] || COLORS.linear; }

// ═══════ Palette Render ═══════
function renderPalette() {
    const el = document.getElementById('palette-container');
    let h = '';
    PALETTE.forEach(cat => {
        const open = cat.open;
        h += `<div class="palette-cat" data-open="${open}">
      <div class="palette-cat-hdr" onclick="toggleCategory(this)">${open ? '▾' : '▸'} ${cat.name}</div>
      <div class="palette-cat-body" style="display:${open ? 'flex' : 'none'}">
        ${cat.items.map(([type, cls]) => {
            let l = type; if (l.length > 14) l = l.replace('Transformer', 'TF').replace('Layer', 'L');
            return `<div class="layer-chip ${cls}" onclick="addLayer('${type}')">${l}</div>`;
        }).join('')}
      </div></div>`;
    });
    el.innerHTML = h;
}
window.toggleCategory = function (hdr) {
    const cat = hdr.parentElement, body = cat.querySelector('.palette-cat-body');
    const isOpen = cat.dataset.open === 'true';
    cat.dataset.open = isOpen ? 'false' : 'true'; body.style.display = isOpen ? 'none' : 'flex';
    hdr.textContent = hdr.textContent.replace(isOpen ? '▾' : '▸', isOpen ? '▸' : '▾');
};

// ═══════ Helpers ═══════
const MAX_NODES = 6;
function getActualNodeCount(s) {
    if (['Conv1d', 'Conv2d', 'ConvTranspose2d'].includes(s.type)) return s.out_channels || 32;
    if (s.type === 'Linear') return s.out_features || 64;
    if (['BatchNorm1d', 'BatchNorm2d'].includes(s.type)) return s.num_features || 32;
    if (s.type === 'Embedding') return Math.min(s.embedding_dim || 128, 64);
    if (['LSTM', 'GRU', 'RNN'].includes(s.type)) return s.hidden_size || 64;
    if (['TransformerEncoderLayer', 'TransformerDecoderLayer'].includes(s.type)) return s.d_model || 128;
    if (s.type === 'MultiheadAttention') return s.embed_dim || 128;
    return null;
}
function computeNodeCounts() {
    const c = []; let prev = 4;
    for (const s of layers) {
        const a = getActualNodeCount(s); let v;
        if (a !== null) { v = Math.min(a, MAX_NODES); prev = v; }
        else if (s.type === 'Flatten') { v = Math.min(6, MAX_NODES); prev = v; }
        else if (['MaxPool2d', 'MaxPool1d', 'AvgPool2d'].includes(s.type)) { v = Math.max(2, Math.ceil(prev * .67)); prev = v; }
        else if (s.type === 'AdaptiveAvgPool2d') { v = 3; prev = v; }
        else v = prev;
        c.push(v);
    } return c;
}
function describeLayer(s) {
    const t = s.type;
    if (['Conv1d', 'Conv2d', 'ConvTranspose2d'].includes(t)) return `${s.in_channels}→${s.out_channels}, k=${s.kernel_size}`;
    if (t === 'Linear') return `${s.in_features}→${s.out_features}`;
    if (['Dropout', 'Dropout2d', 'AlphaDropout'].includes(t)) return `p=${s.p}`;
    if (['BatchNorm1d', 'BatchNorm2d'].includes(t)) return `f=${s.num_features}`;
    if (t === 'LayerNorm') return `shape=${JSON.stringify(s.normalized_shape)}`;
    if (t === 'GroupNorm') return `g=${s.num_groups}, c=${s.num_channels}`;
    if (['MaxPool2d', 'MaxPool1d', 'AvgPool2d'].includes(t)) return `k=${s.kernel_size}`;
    if (t === 'AdaptiveAvgPool2d') return `out=${s.output_size}`;
    if (t === 'Embedding') return `${s.num_embeddings}×${s.embedding_dim}`;
    if (['LSTM', 'GRU', 'RNN'].includes(t)) return `${s.input_size}→${s.hidden_size}`;
    if (['TransformerEncoderLayer', 'TransformerDecoderLayer'].includes(t)) return `d=${s.d_model}, h=${s.nhead}`;
    if (t === 'MultiheadAttention') return `d=${s.embed_dim}, h=${s.num_heads}`;
    if (t === 'Softmax' || t === 'LogSoftmax') return `dim=${s.dim}`;
    if (t === 'Custom') return s._custom_name || 'Custom';
    return '';
}
function estimateParams() {
    let t = 0;
    for (const s of layers) {
        if (['Conv1d', 'Conv2d', 'ConvTranspose2d'].includes(s.type)) t += (s.out_channels || 0) * (s.in_channels || 0) * (s.kernel_size || 1) * (s.kernel_size || 1) + (s.out_channels || 0);
        else if (s.type === 'Linear') t += (s.in_features || 0) * (s.out_features || 0) + (s.out_features || 0);
        else if (['BatchNorm1d', 'BatchNorm2d'].includes(s.type)) t += (s.num_features || 0) * 2;
        else if (s.type === 'Embedding') t += (s.num_embeddings || 0) * (s.embedding_dim || 0);
        else if (['LSTM', 'GRU'].includes(s.type)) { const i = s.input_size || 0, h = s.hidden_size || 0, g = s.type === 'LSTM' ? 4 : 3; t += g * (i * h + h * h + h); }
    } return t;
}

// ═══════ SVG Node Diagram ═══════
function renderDiagram(svgId, activeIdx = -1) {
    const svg = document.getElementById(svgId);
    if (!svg) return;
    if (!layers.length) { svg.setAttribute('width', 10); svg.setAttribute('height', 10); svg.innerHTML = ''; return; }
    const R = 9, GAP = 22, CW = 84, H = 280, M = 36, CY = H / 2;
    const nc = computeNodeCounts();
    const W = M * 2 + layers.length * CW;
    svg.setAttribute('width', W); svg.setAttribute('height', H);

    let o = `<defs>
    <filter id="gs" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="1.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>`;

    // Edges
    for (let i = 0; i < layers.length - 1; i++) {
        const n1 = nc[i], n2 = nc[i + 1], x1 = M + (i + .5) * CW + R, x2 = M + (i + 1.5) * CW - R, mx = (x1 + x2) / 2;
        const isAc = i === activeIdx || i + 1 === activeIdx;
        const op = isAc ? .5 : .15;
        const cc = isAc ? getColor(layers[i + 1]).stroke : '#cbd5e1';
        for (let a = 0; a < n1; a++) {
            const y1 = CY + (a - (n1 - 1) / 2) * GAP;
            for (let b = 0; b < n2; b++) {
                const y2 = CY + (b - (n2 - 1) / 2) * GAP;
                o += `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="${cc}" stroke-width=".8" opacity="${op}"/>`;
            }
        }
    }

    // Skip connection arcs
    for (let i = 0; i < layers.length; i++) {
        const sf = layers[i]._skip_from;
        if (sf !== undefined && sf !== null && sf !== '' && sf >= 0 && sf < i) {
            const fromX = M + (+sf + .5) * CW, toX = M + (i + .5) * CW;
            const arcY = H - 22;
            o += `<path d="M${fromX},${CY + nc[sf] * GAP / 2 + R + 6} Q${(fromX + toX) / 2},${arcY + 20} ${toX},${CY + nc[i] * GAP / 2 + R + 6}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4,3" opacity=".7"/>`;
            o += `<text x="${(fromX + toX) / 2}" y="${arcY + 16}" text-anchor="middle" fill="#d97706" font-size="7" font-weight="700" font-family="Inter">skip</text>`;
        }
    }

    // Nodes
    for (let i = 0; i < layers.length; i++) {
        const s = layers[i], c = getColor(s), isAc = i === activeIdx, n = nc[i], cx = M + (i + .5) * CW;
        if (isAc) {
            const rh = (n - 1) * GAP + R * 2 + 12, ry = CY - (n - 1) / 2 * GAP - R - 6;
            o += `<rect x="${cx - 18}" y="${ry}" width="36" height="${rh}" rx="16" fill="${c.ring}22" stroke="${c.ring}" stroke-width="1" opacity=".6"/>`;
        }
        for (let j = 0; j < n; j++) {
            const cy = CY + (j - (n - 1) / 2) * GAP;
            o += `<circle cx="${cx}" cy="${cy}" r="${R}" fill="${c.fill}" stroke="${isAc ? c.text : c.stroke}" stroke-width="${isAc ? 2.5 : 1.2}" filter="url(#gs)"/>`;
            o += `<circle cx="${cx}" cy="${cy}" r="3" fill="${c.stroke}" opacity="${isAc ? 1 : .5}"/>`;
        }
        const ac = getActualNodeCount(s);
        if (ac !== null && ac > MAX_NODES) {
            const ly = CY + ((n - 1) - (n - 1) / 2) * GAP + R + 11;
            o += `<text x="${cx}" y="${ly}" text-anchor="middle" fill="${c.text}" font-size="8" font-family="JetBrains Mono">+${ac - MAX_NODES}</text>`;
        }
        let dn = s.type; if (dn.length > 10) dn = dn.replace('Transformer', 'TF').replace('Layer', 'L');
        o += `<text x="${cx}" y="${H - 8}" text-anchor="middle" fill="${c.text}" font-size="9" font-weight="600" font-family="Inter">${dn}</text>`;
        o += `<text x="${cx}" y="12" text-anchor="middle" fill="${c.stroke}" font-size="8" font-family="Inter">${i + 1}</text>`;
    }
    svg.innerHTML = o;
}

// ═══════ Architecture Block Diagram ═══════
function renderArchDiagram() {
    const el = document.getElementById('arch-diagram');
    if (!el) return;
    if (!layers.length) { el.innerHTML = ''; return; }

    // Build the block list, then overlay skip connections with SVG
    let html = '<div class="arch-block" id="arch-block-inner" style="position:relative">';

    // Input
    html += `<div class="arch-layer" style="border-color:#c7d2fe;background:#eef2ff" data-arch-idx="input">
    <div class="arch-icon" style="background:#e0e7ff;color:#4f46e5">${ICONS.input}</div>
    <div class="arch-info"><div class="arch-type" style="color:#4f46e5">Input</div><div class="arch-detail">Data tensor</div></div></div>`;

    layers.forEach((s, i) => {
        const c = getColor(s), cat = getCat(s), desc = describeLayer(s);
        const hasSkip = s._skip_from !== undefined && s._skip_from !== null && s._skip_from !== '' && s._skip_from >= 0;
        let displayName = s.type === 'Custom' ? (s._custom_name || 'Custom') : s.type;

        html += `<div class="arch-arrow"></div>`;
        html += `<div class="arch-layer" data-arch-idx="${i}" style="border-color:${c.stroke}30;background:${c.fill}">
      <div class="arch-icon" style="background:${c.stroke}18;color:${c.text}">${ICONS[cat] || ICONS.utility}</div>
      <div class="arch-info">
        <div class="arch-type" style="color:${c.text}">${displayName}</div>
        ${desc ? `<div class="arch-detail">${desc}</div>` : ''}
        ${hasSkip ? `<div style="margin-top:2px"><span class="skip-badge">${ICONS.skip} skip from #${+s._skip_from + 1}</span></div>` : ''}
      </div>
      <span style="font-size:.58rem;color:${c.stroke};font-weight:700">${i + 1}</span>
    </div>`;
    });

    // Output
    html += `<div class="arch-arrow"></div>`;
    html += `<div class="arch-layer" data-arch-idx="output" style="border-color:#a7f3d030;background:#d1fae5">
    <div class="arch-icon" style="background:#a7f3d0;color:#047857">${ICONS.output}</div>
    <div class="arch-info"><div class="arch-type" style="color:#047857">Output</div><div class="arch-detail">Predictions</div></div></div>`;

    html += '</div>';
    el.innerHTML = html;

    // Draw skip connection SVG overlay
    requestAnimationFrame(() => {
        const container = document.getElementById('arch-block-inner');
        if (!container) return;
        const skips = [];
        layers.forEach((s, i) => {
            if (s._skip_from !== undefined && s._skip_from !== null && s._skip_from !== '' && +s._skip_from >= 0 && +s._skip_from < i) {
                skips.push({ from: +s._skip_from, to: i });
            }
        });
        if (!skips.length) return;

        const allBlocks = container.querySelectorAll('[data-arch-idx]');
        const blockMap = {};
        allBlocks.forEach(b => { blockMap[b.dataset.archIdx] = b; });

        const cRect = container.getBoundingClientRect();
        let svgH = '';
        skips.forEach(({ from, to }, si) => {
            const fromEl = blockMap[from], toEl = blockMap[to];
            if (!fromEl || !toEl) return;
            const fRect = fromEl.getBoundingClientRect(), tRect = toEl.getBoundingClientRect();
            const fy = fRect.top + fRect.height / 2 - cRect.top;
            const ty = tRect.top + tRect.height / 2 - cRect.top;
            const x = cRect.width - 8;
            const cx = x + 25 + si * 12;
            svgH += `<path d="M${x},${fy} C${cx},${fy} ${cx},${ty} ${x},${ty}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrowSkip)"/>`;
            svgH += `<text x="${cx + 4}" y="${(fy + ty) / 2 + 3}" fill="#d97706" font-size="8" font-weight="700" font-family="Inter" transform="rotate(-90,${cx + 4},${(fy + ty) / 2})">skip</text>`;
        });

        const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svgEl.setAttribute('class', 'arch-skip-svg');
        svgEl.setAttribute('style', `position:absolute;left:0;top:0;width:${cRect.width + 60}px;height:${cRect.height}px;pointer-events:none;overflow:visible`);
        svgEl.innerHTML = `<defs><marker id="arrowSkip" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto"><polygon points="0 0, 6 2.5, 0 5" fill="#f59e0b"/></marker></defs>` + svgH;
        container.appendChild(svgEl);
    });
}

// ═══════ View Mode Toggle ═══════
window.setViewMode = function (mode, btn) {
    _viewMode = mode;
    document.querySelectorAll('.vmode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const svg = document.getElementById('net-svg'), arch = document.getElementById('arch-diagram');
    if (mode === 'nodes') { svg.classList.remove('hidden'); arch.classList.add('hidden'); renderDiagram('net-svg'); }
    else { svg.classList.add('hidden'); arch.classList.remove('hidden'); renderArchDiagram(); }
};

// ═══════ Layer List ═══════
let dragSrcIndex = -1;

function renderLayerList() {
    const el = document.getElementById('layer-list');
    let h = '';
    for (let i = 0; i <= layers.length; i++) {
        h += `<div class="layer-drop-zone" id="dz-${i}" ondragover="dzOver(event,${i})" ondragleave="dzLeave(event,${i})" ondrop="dzDrop(event,${i})">
      <div class="dz-line"></div><div class="dz-plus" onclick="togglePicker(${i})">＋</div>
      <div class="dz-picker" id="dz-picker-${i}">${renderMiniPalette(i)}</div></div>`;

        if (i < layers.length) {
            const s = layers[i], c = getColor(s), desc = describeLayer(s), cat = getCat(s);
            const hasSkip = s._skip_from !== undefined && s._skip_from !== null && s._skip_from !== '';
            h += `<div class="layer-item" id="li-${i}" draggable="true" ondragstart="onDragStart(event,${i})" ondragend="onDragEnd(event)">
        <div class="layer-drag-handle" title="Drag">⠿</div>
        <div class="layer-num" style="background:${c.stroke}">${i + 1}</div>
        <div style="flex:1;min-width:0">
          <div class="layer-name">${s.type === 'Custom' ? (s._custom_name || 'Custom') : s.type}</div>
          ${desc ? `<div class="layer-params">${desc}</div>` : ''}
          ${renderInlineEditor(s, i)}
          <div style="margin-top:2px;display:flex;align-items:center;gap:4px">
            <span style="font-size:.55rem;color:#94a3b8;font-weight:600">SKIP:</span>
            <select class="skip-select" onchange="updateSkip(${i},this.value)" onclick="event.stopPropagation()">
              <option value=""${!hasSkip ? ' selected' : ''}>none</option>
              ${Array.from({ length: i }, (_, j) => `<option value="${j}"${+s._skip_from === j ? ' selected' : ''}>Layer ${j + 1}</option>`).join('')}
            </select>
          </div>
        </div>
        <button class="layer-help-btn" onclick="event.stopPropagation();geminiHelpLayer(${i})" title="AI Help">✦</button>
        <button class="layer-del" onclick="removeLayer(${i})" title="Remove">✕</button>
      </div>`;
        }
    }
    el.innerHTML = h;
}

function renderMiniPalette(at) {
    const items = [['Conv2d', 'conv'], ['Linear', 'linear'], ['Flatten', 'flat'], ['ReLU', 'act'], ['GELU', 'act'], ['Dropout', 'drop'], ['BN2d', 'BatchNorm2d', 'bn'], ['LSTM', 'LSTM', 'rnn'], ['TF Enc', 'TransformerEncoderLayer', 'transformer']];
    return items.map(([label, type, cls]) => {
        if (!cls) { cls = type; type = label; }
        return `<div class="layer-chip ${cls}" style="padding:.16rem .35rem;font-size:.58rem" onclick="insertLayer('${type}',${at});hidePickers()">${label}</div>`;
    }).join('');
}

function togglePicker(i) { const p = document.getElementById(`dz-picker-${i}`); const o = p.classList.contains('open'); hidePickers(); if (!o) { p.classList.add('open'); document.getElementById(`dz-${i}`).classList.add('picker-open'); } }
function hidePickers() { document.querySelectorAll('.dz-picker').forEach(p => p.classList.remove('open')); document.querySelectorAll('.layer-drop-zone').forEach(z => z.classList.remove('picker-open')); }

// Drag & Drop
function onDragStart(e, i) { dragSrcIndex = i; e.dataTransfer.effectAllowed = 'move'; e.dataTransfer.setData('text/plain', i); setTimeout(() => e.target.classList.add('dragging'), 0); }
function onDragEnd(e) { e.target.classList.remove('dragging'); dragSrcIndex = -1; document.querySelectorAll('.layer-drop-zone').forEach(z => z.classList.remove('drag-over')); }
function dzOver(e, i) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; document.getElementById(`dz-${i}`)?.classList.add('drag-over'); }
function dzLeave(e, i) { document.getElementById(`dz-${i}`)?.classList.remove('drag-over'); }
function dzDrop(e, to) { e.preventDefault(); document.getElementById(`dz-${to}`)?.classList.remove('drag-over'); if (dragSrcIndex < 0 || dragSrcIndex === to || dragSrcIndex + 1 === to) return; const m = layers.splice(dragSrcIndex, 1)[0]; const at = dragSrcIndex < to ? to - 1 : to; layers.splice(at, 0, m); dragSrcIndex = -1; renderAll(); }

let _currentActiveIdx = -1;
window.setActiveLayerInList = function (idx) {
    _currentActiveIdx = idx;
    renderDiagram('net-svg-train', idx);
    document.querySelectorAll('.layer-item').forEach((el, i) => { el.classList.toggle('active-layer', i === idx); if (i === idx) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
};

// ═══════ Inline Editor ═══════
function renderInlineEditor(spec, idx) {
    const t = spec.type;
    const inp = (f, v, step = 1, min = 0) =>
        `<input type="number" class="inline-input" value="${v ?? ''}" step="${step}" min="${min}" onchange="updateParam(${idx},'${f}',+this.value)" onclick="event.stopPropagation()">`;
    const boolInp = (f, v) =>
        `<select class="inline-input" style="width:48px" onchange="updateParam(${idx},'${f}',this.value==='true')" onclick="event.stopPropagation()"><option value="true"${v ? ' selected' : ''}>T</option><option value="false"${!v ? ' selected' : ''}>F</option></select>`;
    const w = c => `<div style="margin-top:2px;font-size:.58rem;color:#94a3b8">${c}</div>`;

    if (['Conv1d', 'Conv2d', 'ConvTranspose2d'].includes(t)) return w(`in:${inp('in_channels', spec.in_channels)} out:${inp('out_channels', spec.out_channels)} k:${inp('kernel_size', spec.kernel_size)} p:${inp('padding', spec.padding || 0, 1, 0)}`);
    if (t === 'Linear') return w(`in:${inp('in_features', spec.in_features)} out:${inp('out_features', spec.out_features)}`);
    if (['Dropout', 'Dropout2d', 'AlphaDropout'].includes(t)) return w(`p:${inp('p', spec.p, .05, 0)}`);
    if (['BatchNorm1d', 'BatchNorm2d'].includes(t)) return w(`f:${inp('num_features', spec.num_features)}`);
    if (t === 'LayerNorm') return w(`shape:${inp('normalized_shape', Array.isArray(spec.normalized_shape) ? spec.normalized_shape[0] : spec.normalized_shape)}`);
    if (t === 'GroupNorm') return w(`g:${inp('num_groups', spec.num_groups)} ch:${inp('num_channels', spec.num_channels)}`);
    if (['MaxPool2d', 'MaxPool1d', 'AvgPool2d'].includes(t)) return w(`k:${inp('kernel_size', spec.kernel_size)}`);
    if (t === 'AdaptiveAvgPool2d') return w(`out:${inp('output_size', spec.output_size)}`);
    if (t === 'Embedding') return w(`vocab:${inp('num_embeddings', spec.num_embeddings)} dim:${inp('embedding_dim', spec.embedding_dim)}`);
    if (['LSTM', 'GRU', 'RNN'].includes(t)) return w(`in:${inp('input_size', spec.input_size)} hid:${inp('hidden_size', spec.hidden_size)} bf:${boolInp('batch_first', spec.batch_first)}`);
    if (['TransformerEncoderLayer', 'TransformerDecoderLayer'].includes(t)) return w(`d:${inp('d_model', spec.d_model)} h:${inp('nhead', spec.nhead)}`);
    if (t === 'MultiheadAttention') return w(`dim:${inp('embed_dim', spec.embed_dim)} h:${inp('num_heads', spec.num_heads)}`);
    if (t === 'Softmax' || t === 'LogSoftmax') return w(`dim:${inp('dim', spec.dim)}`);
    return '';
}

// ═══════ Master Render ═══════
function renderAll() {
    hidePickers();
    renderLayerList();
    if (_viewMode === 'nodes') renderDiagram('net-svg');
    else renderArchDiagram();
    document.getElementById('layer-count').textContent = layers.length;
    const p = estimateParams();
    document.getElementById('param-count').textContent = p > 0 ? `~${p.toLocaleString()} params` : '— params';
    document.getElementById('empty-state').style.display = layers.length ? 'none' : 'flex';
    document.getElementById('net-svg').style.display = layers.length && _viewMode === 'nodes' ? 'block' : 'none';
}

// ═══════ Public API ═══════
window.addLayer = (type) => { layers.push({ type, ...JSON.parse(JSON.stringify(DEFAULTS[type] || {})) }); renderAll(); };
window.insertLayer = (type, at) => { layers.splice(at, 0, { type, ...JSON.parse(JSON.stringify(DEFAULTS[type] || {})) }); renderAll(); };
window.removeLayer = (i) => { layers.splice(i, 1); renderAll(); };
window.updateParam = (i, f, v) => { if (f === 'normalized_shape' && typeof v === 'number') layers[i][f] = [v]; else layers[i][f] = v; renderAll(); };
window.updateSkip = (i, v) => { if (v === '') delete layers[i]._skip_from; else layers[i]._skip_from = parseInt(v); renderAll(); };
window.clearLayers = () => { layers = []; renderAll(); };
window.getLayers = () => layers;
window.renderDiagram = renderDiagram;
window.togglePicker = togglePicker; window.hidePickers = hidePickers;
window.dzOver = dzOver; window.dzLeave = dzLeave; window.dzDrop = dzDrop;
window.onDragStart = onDragStart; window.onDragEnd = onDragEnd;

document.addEventListener('click', e => { if (!e.target.closest('.layer-drop-zone')) hidePickers(); });

// ═══════ Templates ═══════
window.loadTemplate = function (name) {
    if (name === 'mlp') {
        document.getElementById('dataset').value = 'mnist'; document.getElementById('task-type').value = 'classification';
        onDatasetChange('mnist'); onTaskTypeChange('classification');
        layers = [{ type: 'Flatten' }, { type: 'Linear', in_features: 784, out_features: 256 }, { type: 'ReLU' }, { type: 'Linear', in_features: 256, out_features: 128 }, { type: 'ReLU' }, { type: 'Dropout', p: .3 }, { type: 'Linear', in_features: 128, out_features: 10 }];
    } else if (name === 'cnn') {
        document.getElementById('dataset').value = 'cifar10'; document.getElementById('task-type').value = 'classification';
        onDatasetChange('cifar10'); onTaskTypeChange('classification');
        layers = [{ type: 'Conv2d', in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1 }, { type: 'ReLU' }, { type: 'MaxPool2d', kernel_size: 2 }, { type: 'Conv2d', in_channels: 32, out_channels: 64, kernel_size: 3, padding: 1 }, { type: 'ReLU' }, { type: 'MaxPool2d', kernel_size: 2 }, { type: 'Flatten' }, { type: 'Linear', in_features: 4096, out_features: 128 }, { type: 'ReLU' }, { type: 'Linear', in_features: 128, out_features: 10 }];
    } else if (name === 'resnet') {
        document.getElementById('dataset').value = 'cifar10'; document.getElementById('task-type').value = 'classification';
        onDatasetChange('cifar10'); onTaskTypeChange('classification');
        // ResNet-style block with skip connections
        _viewMode = 'arch'; document.getElementById('mode-btn-arch').classList.add('active'); document.getElementById('mode-btn-nodes').classList.remove('active');
        document.getElementById('net-svg').classList.add('hidden'); document.getElementById('arch-diagram').classList.remove('hidden');
        layers = [
            { type: 'Conv2d', in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1 },
            { type: 'BatchNorm2d', num_features: 64 },
            { type: 'ReLU' },
            // Residual block 1
            { type: 'Conv2d', in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1 },
            { type: 'BatchNorm2d', num_features: 64 },
            { type: 'ReLU', _skip_from: 2 }, // skip from layer 2 (after first ReLU)
            // Residual block 2
            { type: 'Conv2d', in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1 },
            { type: 'BatchNorm2d', num_features: 64 },
            { type: 'ReLU', _skip_from: 5 }, // skip from layer 5 (end of block 1)
            { type: 'AdaptiveAvgPool2d', output_size: 1 },
            { type: 'Flatten' },
            { type: 'Linear', in_features: 64, out_features: 10 }
        ];
    } else if (name === 'transformer') {
        document.getElementById('dataset').value = 'mnist'; document.getElementById('task-type').value = 'classification';
        onDatasetChange('mnist'); onTaskTypeChange('classification');
        _viewMode = 'arch'; document.getElementById('mode-btn-arch').classList.add('active'); document.getElementById('mode-btn-nodes').classList.remove('active');
        document.getElementById('net-svg').classList.add('hidden'); document.getElementById('arch-diagram').classList.remove('hidden');
        layers = [{ type: 'Flatten' }, { type: 'Linear', in_features: 784, out_features: 128 }, { type: 'ReLU' }, { type: 'LayerNorm', normalized_shape: [128] }, { type: 'Linear', in_features: 128, out_features: 64 }, { type: 'ReLU' }, { type: 'Dropout', p: .1 }, { type: 'Linear', in_features: 64, out_features: 10 }];
    } else if (name === 'regression') {
        document.getElementById('task-type').value = 'regression'; onTaskTypeChange('regression');
        layers = [{ type: 'Linear', in_features: 10, out_features: 64 }, { type: 'ReLU' }, { type: 'BatchNorm1d', num_features: 64 }, { type: 'Dropout', p: .2 }, { type: 'Linear', in_features: 64, out_features: 32 }, { type: 'ReLU' }, { type: 'Linear', in_features: 32, out_features: 1 }];
    }
    renderAll();
};

// ═══════ Dataset & Task ═══════
window.onDatasetChange = function (v) { document.getElementById('custom-ds-area').className = (v === 'custom' || v === 'csv') ? 'mt-2' : 'hidden'; };
window.onTaskTypeChange = function (v) { document.getElementById('loss').value = v === 'regression' ? 'MSELoss' : 'CrossEntropyLoss'; };

// File uploads
window._droppedFile = null;
window.dzDragOver = e => { e.preventDefault(); document.getElementById('drop-zone').classList.add('dragover'); };
window.dzDragLeave = () => { document.getElementById('drop-zone').classList.remove('dragover'); };
window.dzDrop_file = e => { e.preventDefault(); document.getElementById('drop-zone').classList.remove('dragover'); if (e.dataTransfer.files[0]) { window._droppedFile = e.dataTransfer.files[0]; document.getElementById('drop-label').textContent = `📦 ${e.dataTransfer.files[0].name}`; } };
window.onFileSelected = inp => { window._droppedFile = null; if (inp.files[0]) document.getElementById('drop-label').textContent = `📦 ${inp.files[0].name}`; };

// Panel switcher
window.showPanel = function (name, btn) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${name}`).classList.add('active');
    btn.classList.add('active');
};

// ═══════ Gemini AI ═══════
window.toggleGeminiPanel = function () {
    const b = document.getElementById('gemini-body'), t = document.getElementById('gemini-toggle');
    const open = b.style.display !== 'none'; b.style.display = open ? 'none' : 'block'; t.textContent = open ? '▾' : '▴';
};
function geminiSetOutput(html, showApply = false) {
    const out = document.getElementById('gemini-output'), body = document.getElementById('gemini-output-body'), ab = document.getElementById('gemini-apply-btn');
    out.classList.remove('hidden'); body.innerHTML = html; ab.classList.toggle('hidden', !showApply);
}
window.geminiSuggest = async function () {
    const desc = document.getElementById('gemini-prompt').value.trim();
    if (!desc) { alert('Please describe your task first.'); return; }
    const tt = document.getElementById('task-type').value;
    geminiSetOutput('<div class="gemini-loading">✦ Thinking...</div>');
    try {
        const r = await fetch('/api/gemini/suggest/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ description: desc, task_type: tt }) });
        const d = await r.json();
        if (d.error) { geminiSetOutput(`<div class="gemini-error">✗ ${d.error}</div>`); return; }
        _lastGeminiResult = d;
        let h = ''; if (d.reasoning) h += `<div class="gemini-reasoning">${d.reasoning}</div>`;
        h += `<div style="font-size:.78rem"><strong>${d.layers?.length || 0} layers</strong> · Loss: ${d.loss || 'N/A'} · Opt: ${d.optimizer || 'N/A'}`;
        const skipCount = (d.layers || []).filter(l => l._skip_from !== undefined).length;
        if (skipCount) h += ` · <span style="color:#d97706;font-weight:600">${skipCount} skip</span>`;
        h += `<div class="gemini-layer-list">`; (d.layers || []).forEach((l, i) => {
            const sk = l._skip_from !== undefined ? ` ⤴${+l._skip_from + 1}` : '';
            h += `<span class="gemini-layer-tag">${i + 1}. ${l.type}${sk}</span>`;
        });
        h += `</div></div>`;
        geminiSetOutput(h, true);
    } catch (e) { geminiSetOutput(`<div class="gemini-error">✗ ${e.message}</div>`); }
};

window._pdfFile = null;
window.pdfDragOver = e => { e.preventDefault(); document.getElementById('pdf-drop').classList.add('dragover'); };
window.pdfDragLeave = () => { document.getElementById('pdf-drop').classList.remove('dragover'); };
window.pdfDrop = e => { e.preventDefault(); document.getElementById('pdf-drop').classList.remove('dragover'); const f = e.dataTransfer.files[0]; if (f && f.name.toLowerCase().endsWith('.pdf')) { window._pdfFile = f; document.getElementById('pdf-drop-label').textContent = `📄 ${f.name}`; } };
window.pdfFileSelected = inp => { if (inp.files[0]) { window._pdfFile = inp.files[0]; document.getElementById('pdf-drop-label').textContent = `📄 ${inp.files[0].name}`; } };

window.geminiParsePdf = async function () {
    const f = window._pdfFile || document.getElementById('pdf-file').files[0];
    if (!f) { alert('Please select a PDF first.'); return; }
    geminiSetOutput('<div class="gemini-loading">✦ Analyzing PDF...</div>');
    try {
        const fd = new FormData(); fd.append('file', f);
        const r = await fetch('/api/gemini/upload-pdf/', { method: 'POST', body: fd }); const d = await r.json();
        if (d.error) { geminiSetOutput(`<div class="gemini-error">✗ ${d.error}</div>`); return; }
        _lastGeminiResult = d;
        let h = ''; if (d.reasoning) h += `<div class="gemini-reasoning">${d.reasoning}</div>`;
        h += `<div style="font-size:.78rem"><strong>${d.layers?.length || 0} layers</strong>`;
        h += `<div class="gemini-layer-list">`; (d.layers || []).forEach((l, i) => { h += `<span class="gemini-layer-tag">${i + 1}. ${l.type}</span>`; });
        h += `</div></div>`; geminiSetOutput(h, true);
    } catch (e) { geminiSetOutput(`<div class="gemini-error">✗ ${e.message}</div>`); }
};

window.applyGeminiArchitecture = function () {
    if (!_lastGeminiResult?.layers) return;
    layers = _lastGeminiResult.layers.map(l => ({ ...l }));
    if (_lastGeminiResult.task_type) { document.getElementById('task-type').value = _lastGeminiResult.task_type; onTaskTypeChange(_lastGeminiResult.task_type); }
    if (_lastGeminiResult.loss) { const ls = document.getElementById('loss'); if (Array.from(ls.options).find(o => o.value === _lastGeminiResult.loss)) ls.value = _lastGeminiResult.loss; }
    if (_lastGeminiResult.optimizer) { const os = document.getElementById('optimizer'); if (Array.from(os.options).find(o => o.value === _lastGeminiResult.optimizer.toLowerCase())) os.value = _lastGeminiResult.optimizer.toLowerCase(); }
    if (_lastGeminiResult.lr) document.getElementById('lr').value = _lastGeminiResult.lr;
    // Auto-switch to arch view for complex architectures or those with skip connections
    const hasComplex = layers.some(l => ['TransformerEncoderLayer', 'TransformerDecoderLayer', 'TransformerEncoder', 'MultiheadAttention', 'LSTM', 'GRU', 'RNN', 'Custom'].includes(l.type) || l._skip_from !== undefined);
    if (hasComplex) { _viewMode = 'arch'; document.getElementById('mode-btn-arch').classList.add('active'); document.getElementById('mode-btn-nodes').classList.remove('active'); document.getElementById('net-svg').classList.add('hidden'); document.getElementById('arch-diagram').classList.remove('hidden'); }
    renderAll();
    geminiSetOutput('<div class="gemini-success">✓ Architecture applied!</div>');
};

window.geminiHelpLayer = async function (idx) {
    const s = layers[idx]; if (!s) return;
    const b = document.getElementById('gemini-body'); if (b.style.display === 'none') toggleGeminiPanel();
    geminiSetOutput(`<div class="gemini-loading">✦ Getting help for ${s.type}...</div>`);
    try {
        const r = await fetch('/api/gemini/help-layer/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ layer_type: s.type, context: JSON.stringify(s) }) });
        const d = await r.json();
        if (d.error) { geminiSetOutput(`<div class="gemini-error">✗ ${d.error}</div>`); return; }
        let h = `<div style="font-weight:700;font-size:.82rem;color:#4f46e5;margin-bottom:.25rem">nn.${s.type}</div>`;
        h += `<div style="font-size:.76rem;line-height:1.5;color:#475569">${(d.help || '').replace(/\n/g, '<br>')}</div>`;
        if (d.suggested_params) h += `<div style="margin-top:.3rem;font-size:.68rem;font-family:JetBrains Mono;color:#059669;padding:.25rem .4rem;background:#f0fdf4;border-radius:4px;border:1px solid #bbf7d0"><strong>Suggested:</strong> ${JSON.stringify(d.suggested_params)}</div>`;
        geminiSetOutput(h);
    } catch (e) { geminiSetOutput(`<div class="gemini-error">✗ ${e.message}</div>`); }
};

// ═══════ Init ═══════
renderPalette();
renderAll();
