/**
 * Factory Layout WebUI - Main Application
 */

// State
let sessionId = null;
let socket = null;
let currentState = null;
let hoveredCandidate = null;
let selectedCandidate = null;
let isDarkMode = false;

// Canvas
const canvas = document.getElementById('layout-canvas');
const ctx = canvas.getContext('2d');

// Settings
const settings = {
    showCandidates: true,
    showFlow: true,
    showForbidden: false,
    showConstraintZones: false,
    showScores: true,
    showVisits: false,
    showLabels: true,
    showGrid: false,
};

// Candidate 정렬 상태
let candidateSortColumn = 'q';  // 'index', 'pos', 'q', 'p', 'n'
let candidateSortDesc = true;   // true: 내림차순, false: 오름차순

// Colors (theme-aware)
function getColors() {
    if (isDarkMode) {
        return {
            background: '#161b22',
            grid: '#30363d',
            placed: {
                fill: 'rgba(255, 165, 0, 0.6)',
                stroke: '#ffa500',
            },
            candidate: {
                valid: '#4ade80',
                invalid: '#f87171',
                hovered: '#60a5fa',
                selected: '#ff7f0e',
            },
            text: '#ffffff',
            flow: 'rgba(96, 165, 250, 0.75)',
            forbidden: 'rgba(255, 100, 100, 0.2)',
            constraintZone: 'rgba(30, 144, 255, 0.15)',
        };
    }
    return {
        background: '#ffffff',
        grid: '#e0e0e0',
        placed: {
            fill: 'rgba(255, 165, 0, 0.6)',
            stroke: '#000000',
        },
        candidate: {
            valid: '#2ca02c',
            invalid: '#d62728',
            hovered: '#1f77b4',
            selected: '#ff7f0e',
        },
        text: '#000000',
        flow: 'rgba(31, 119, 180, 0.65)',
        forbidden: 'rgba(255, 0, 0, 0.15)',
        constraintZone: 'rgba(30, 144, 255, 0.12)',
    };
}

// Shortcut for current colors
let COLORS = getColors();

// ============================================================
// Initialization
// ============================================================

function init() {
    setupEventListeners();
    loadConfigs();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        toggleTheme();
    }
}

function toggleTheme() {
    isDarkMode = !isDarkMode;
    COLORS = getColors();
    
    if (isDarkMode) {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.getElementById('theme-toggle').textContent = '☀️';
        localStorage.setItem('theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
        document.getElementById('theme-toggle').textContent = '🌙';
        localStorage.setItem('theme', 'light');
    }
    
    render();
}

function setupEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    
    // Session controls
    document.getElementById('btn-create').addEventListener('click', createSession);
    document.getElementById('btn-reset').addEventListener('click', resetSession);
    document.getElementById('btn-undo').addEventListener('click', undoStep);
    document.getElementById('btn-redo').addEventListener('click', redoStep);
    document.getElementById('btn-search').addEventListener('click', runSearch);

    // Layer toggles
    const layerToggles = [
        ['show-candidates', 'showCandidates'],
        ['show-flow', 'showFlow'],
        ['show-forbidden', 'showForbidden'],
        ['show-constraint-zones', 'showConstraintZones'],
        ['show-scores', 'showScores'],
        ['show-visits', 'showVisits'],
        ['show-labels', 'showLabels'],
        ['show-grid', 'showGrid'],
    ];
    
    layerToggles.forEach(([id, setting]) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', (e) => {
                settings[setting] = e.target.checked;
                render();
            });
        }
    });
    
    // Candidate 컬럼 헤더 클릭 정렬
    const sortColumns = ['col-index', 'col-pos', 'col-q', 'col-p', 'col-n'];
    const columnMap = { 'col-index': 'index', 'col-pos': 'pos', 'col-q': 'q', 'col-p': 'p', 'col-n': 'n' };
    
    sortColumns.forEach(colClass => {
        const headerEl = document.querySelector(`.candidate-header .${colClass}`);
        if (headerEl) {
            headerEl.addEventListener('click', () => {
                const col = columnMap[colClass];
                if (candidateSortColumn === col) {
                    candidateSortDesc = !candidateSortDesc;  // 같은 컬럼 클릭 시 방향 토글
                } else {
                    candidateSortColumn = col;
                    candidateSortDesc = true;  // 새 컬럼은 내림차순으로 시작
                }
                updateCandidateList();
            });
        }
    });

    // Canvas interactions
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('mouseleave', onCanvasMouseLeave);
    
    // Settings panel buttons
    document.querySelectorAll('.btn-settings').forEach(btn => {
        btn.addEventListener('click', () => openSettingsPanel(btn.dataset.panel));
    });
    document.getElementById('settings-close').addEventListener('click', closeSettingsPanel);
    
    // Mode change listeners - 동적 파라미터 로딩
    document.getElementById('wrapper-mode').addEventListener('change', () => loadParams('wrapper'));
    document.getElementById('agent-mode').addEventListener('change', () => loadParams('agent'));
    document.getElementById('search-mode').addEventListener('change', () => loadParams('search'));
    
    // 초기 파라미터 로딩
    loadParams('wrapper');
    loadParams('agent');
    loadParams('search');
}

// 파라미터 캐시 (API 호출 최소화)
const paramsCache = {};

async function loadParams(type) {
    const selectId = type === 'wrapper' ? 'wrapper-mode' : 
                     type === 'agent' ? 'agent-mode' : 'search-mode';
    const name = document.getElementById(selectId).value;
    const cacheKey = `${type}-${name}`;
    
    // 캐시 확인
    if (!paramsCache[cacheKey]) {
        try {
            const res = await fetch(`/api/params/${type}/${name}`);
            const data = await res.json();
            paramsCache[cacheKey] = data.params;
        } catch (e) {
            console.error(`Failed to load params for ${type}/${name}:`, e);
            paramsCache[cacheKey] = {};
        }
    }
    
    renderParams(type, name, paramsCache[cacheKey]);
}

function renderParams(type, name, params) {
    const container = document.getElementById(`${type}-params-container`);
    const titleEl = document.getElementById(`${type}-title`);
    
    // 제목 업데이트
    const titles = {
        wrapper: { greedy: 'Greedy V1', greedyv2: 'Greedy V2', greedyv3: 'Greedy V3', alphachip: 'AlphaChip', maskplace: 'MaskPlace' },
        agent: { greedy: 'Greedy Agent', alphachip: 'AlphaChip Agent', maskplace: 'MaskPlace Agent' },
        search: { none: 'No Search', mcts: 'MCTS', beam: 'Beam Search' }
    };
    titleEl.textContent = (titles[type] && titles[type][name]) || name;
    
    // 컨테이너 초기화
    container.innerHTML = '';
    
    if (!params || Object.keys(params).length === 0) {
        container.innerHTML = '<p class="text-muted">No configurable parameters.</p>';
        return;
    }
    
    // 파라미터별 input 생성
    for (const [paramName, info] of Object.entries(params)) {
        const div = document.createElement('div');
        div.className = 'form-group';
        
        const label = document.createElement('label');
        label.textContent = formatParamName(paramName) + (info.required ? ' *' : '') + ':';
        label.setAttribute('for', `param-${type}-${paramName}`);
        if (info.required) {
            label.classList.add('required-label');
        }
        
        const input = createParamInput(type, paramName, info);
        
        div.appendChild(label);
        div.appendChild(input);
        container.appendChild(div);
    }
}

function formatParamName(name) {
    // snake_case -> Title Case
    return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function createParamInput(type, name, info) {
    const input = document.createElement('input');
    input.id = `param-${type}-${name}`;
    input.dataset.paramType = type;
    input.dataset.paramName = name;
    
    const paramType = info.type || 'str';
    const defaultVal = info.default;
    const isRequired = info.required === true;
    
    if (isRequired) {
        input.required = true;
        input.classList.add('required-input');
    }
    
    if (paramType === 'int') {
        input.type = 'number';
        input.step = '1';
        if (isRequired && defaultVal === null) {
            input.value = '';
            input.placeholder = '(required)';
        } else {
            input.value = defaultVal !== null ? defaultVal : 0;
        }
    } else if (paramType === 'float') {
        input.type = 'number';
        input.step = defaultVal !== null && defaultVal < 1 ? '0.01' : '0.1';
        if (isRequired && defaultVal === null) {
            input.value = '';
            input.placeholder = '(required)';
        } else {
            input.value = defaultVal !== null ? defaultVal : 0;
        }
    } else if (paramType === 'bool') {
        input.type = 'checkbox';
        input.checked = defaultVal === true;
    } else {
        input.type = 'text';
        input.value = defaultVal !== null ? defaultVal : '';
    }
    
    return input;
}

function openSettingsPanel(panelType) {
    const panel = document.getElementById('settings-panel');
    const titleEl = document.getElementById('settings-title');
    
    // 모든 패널 컨텐츠 숨기기
    document.querySelectorAll('.panel-content').forEach(el => el.style.display = 'none');
    
    // 선택된 패널 컨텐츠 표시
    document.getElementById(`panel-${panelType}`).style.display = 'block';
    
    // 제목 설정
    const titles = { wrapper: 'Wrapper Settings', agent: 'Agent Settings', search: 'Search Settings' };
    titleEl.textContent = titles[panelType] || 'Settings';
    
    // 패널 열기
    panel.classList.add('open');
}

function closeSettingsPanel() {
    const panel = document.getElementById('settings-panel');
    panel.classList.remove('open');
}

async function loadConfigs() {
    try {
        const res = await fetch('/api/configs');
        const data = await res.json();
        const select = document.getElementById('env-config');
        select.innerHTML = '';
        data.configs.forEach(config => {
            const opt = document.createElement('option');
            opt.value = config;
            opt.textContent = config;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load configs:', e);
    }
}

function resizeCanvas() {
    const container = canvas.parentElement;
    const maxW = container.clientWidth - 40;
    const maxH = container.clientHeight - 40;
    
    if (currentState) {
        const aspectRatio = currentState.grid_width / currentState.grid_height;
        if (maxW / aspectRatio <= maxH) {
            canvas.width = maxW;
            canvas.height = maxW / aspectRatio;
        } else {
            canvas.height = maxH;
            canvas.width = maxH * aspectRatio;
        }
    } else {
        canvas.width = Math.min(maxW, 800);
        canvas.height = Math.min(maxH, 600);
    }
    
    render();
}

// ============================================================
// Session Management
// ============================================================

function collectParams() {
    // 동적으로 생성된 파라미터 input에서 값 수집
    const params = {};
    document.querySelectorAll('input[data-param-type]').forEach(input => {
        const type = input.dataset.paramType;
        const name = input.dataset.paramName;
        const key = `${type}_${name}`;
        
        if (input.type === 'checkbox') {
            params[key] = input.checked;
        } else if (input.type === 'number') {
            const val = input.value;
            if (val === '' || val === null) {
                // 빈 값은 포함하지 않음 (백엔드 기본값 사용)
                return;
            }
            params[key] = val.includes('.') ? parseFloat(val) : parseInt(val);
        } else {
            params[key] = input.value || null;
        }
    });
    return params;
}

function validateRequiredParams() {
    // 필수 파라미터 검증
    const requiredInputs = document.querySelectorAll('input.required-input');
    for (const input of requiredInputs) {
        if (input.value === '' || input.value === null) {
            const paramName = input.dataset.paramName;
            showSessionMessage(`"${paramName}" is required.`, 'error');
            input.focus();
            return false;
        }
    }
    return true;
}

function showSessionMessage(message, type = 'success') {
    const el = document.getElementById('session-message');
    el.textContent = message;
    el.className = 'session-message ' + type;
}

function clearSessionMessage() {
    const el = document.getElementById('session-message');
    el.textContent = '';
    el.className = 'session-message';
}

async function createSession() {
    clearSessionMessage();
    
    // 필수 파라미터 검증
    if (!validateRequiredParams()) {
        return;
    }
    
    const params = collectParams();
    
    const req = {
        env_json: document.getElementById('env-config').value,
        collision_check: document.getElementById('collision-check').value,
        wrapper_mode: document.getElementById('wrapper-mode').value,
        agent_mode: document.getElementById('agent-mode').value,
        search_mode: document.getElementById('search-mode').value,
        params: params,  // 동적 파라미터
    };

    try {
        const res = await fetch('/api/session/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req),
        });
        const data = await res.json();
        
        if (data.session_id) {
            sessionId = data.session_id;
            document.getElementById('session-id').textContent = sessionId;
            updateState(data.state);
            connectWebSocket();
            enableControls(true);
            showSessionMessage('Session created successfully!', 'success');
        } else if (data.detail) {
            showSessionMessage('Error: ' + data.detail, 'error');
        }
    } catch (e) {
        console.error('Failed to create session:', e);
        showSessionMessage('Failed to create session: ' + e.message, 'error');
    }
}

async function resetSession() {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/reset`, { method: 'POST' });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to reset:', e);
    }
}

async function undoStep() {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/undo`, { method: 'POST' });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to undo:', e);
    }
}

async function redoStep() {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/redo`, { method: 'POST' });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to redo:', e);
    }
}

async function stepAction(action) {
    if (!sessionId) return;
    
    try {
        const res = await fetch(`/api/session/${sessionId}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action }),
        });
        const data = await res.json();
        updateState(data.state);
    } catch (e) {
        console.error('Failed to step:', e);
    }
}

async function runSearch() {
    if (!sessionId) return;
    
    // 동적으로 생성된 search params에서 simulations 가져오기
    const simsInput = document.getElementById('param-search-num_simulations');
    const sims = simsInput ? parseInt(simsInput.value) || 50 : 50;
    const interval = parseInt(document.getElementById('search-interval').value);
    
    document.getElementById('btn-search').disabled = true;
    const progressBar = document.getElementById('search-progress');
    progressBar.style.display = 'block';
    
    // Progress bar 초기화 (이전 search 값이 남아있는 문제 해결)
    progressBar.querySelector('.progress-fill').style.width = '0%';
    progressBar.querySelector('.progress-text').textContent = '0% (0/' + sims + ')';
    
    try {
        const res = await fetch(`/api/session/${sessionId}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ simulations: sims, broadcast_interval: interval }),
        });
        const data = await res.json();
        console.log('Search result:', data);
    } catch (e) {
        console.error('Failed to run search:', e);
    } finally {
        document.getElementById('btn-search').disabled = false;
        progressBar.style.display = 'none';
    }
}

function enableControls(enabled) {
    document.getElementById('btn-reset').disabled = !enabled;
    document.getElementById('btn-search').disabled = !enabled;
    updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
    if (currentState) {
        document.getElementById('btn-undo').disabled = !currentState.can_undo;
        document.getElementById('btn-redo').disabled = !currentState.can_redo;
    }
}

// ============================================================
// WebSocket
// ============================================================

function connectWebSocket() {
    if (socket) {
        socket.close();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);
    
    socket.onopen = () => {
        document.getElementById('connection-status').className = 'status-connected';
    };
    
    socket.onclose = () => {
        document.getElementById('connection-status').className = 'status-disconnected';
    };
    
    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'state') {
            updateState(msg.state);
        } else if (msg.type === 'search_progress') {
            updateSearchProgress(msg.progress);
        }
    };
    
    socket.onerror = (e) => {
        console.error('WebSocket error:', e);
    };
}

// ============================================================
// State Updates
// ============================================================

function updateState(state) {
    currentState = state;
    resizeCanvas();
    render();
    updateInfoPanel();
    updateCandidateList();
    updateUndoRedoButtons();
}

function updateSearchProgress(progress) {
    // Update progress bar
    const percent = Math.round((progress.simulation / progress.total) * 100);
    const progressBar = document.getElementById('search-progress');
    progressBar.querySelector('.progress-fill').style.width = `${percent}%`;
    progressBar.querySelector('.progress-text').textContent = `${percent}% (${progress.simulation}/${progress.total})`;
    
    // Update candidates with visits/values
    if (currentState && progress.candidates) {
        currentState.candidates = progress.candidates;
        render();
        updateCandidateList();
    }
}

function updateInfoPanel() {
    if (!currentState) return;
    
    document.getElementById('info-step').textContent = currentState.step;
    document.getElementById('info-cost').textContent = currentState.cost.toFixed(2);
    document.getElementById('info-value').textContent = currentState.value.toFixed(4);
    document.getElementById('info-remaining').textContent = currentState.remaining.length;
    document.getElementById('info-current').textContent = currentState.current_gid || '-';
}

function updateSortIndicators() {
    const columns = {
        'index': '#',
        'pos': 'Pos',
        'q': 'Q',
        'p': 'P',
        'n': 'N'
    };
    const classMap = {
        'index': 'col-index',
        'pos': 'col-pos',
        'q': 'col-q',
        'p': 'col-p',
        'n': 'col-n'
    };
    
    Object.keys(columns).forEach(col => {
        const el = document.querySelector(`.candidate-header .${classMap[col]}`);
        if (el) {
            const arrow = col === candidateSortColumn ? (candidateSortDesc ? ' ▼' : ' ▲') : '';
            el.textContent = columns[col] + arrow;
        }
    });
}

function updateCandidateList() {
    if (!currentState) return;
    
    const list = document.getElementById('candidate-list');
    const countSpan = document.getElementById('candidate-count');
    
    const validCount = currentState.candidates.filter(c => c.valid).length;
    countSpan.textContent = `(${validCount}/${currentState.candidates.length})`;
    
    // 헤더 정렬 표시 업데이트
    updateSortIndicators();
    
    list.innerHTML = '';
    
    // 컬럼 기준 정렬 (0/N/A는 정렬 방향과 무관하게 항상 맨 뒤로)
    const sorted = [...currentState.candidates].sort((a, b) => {
        let cmp = 0;
        switch (candidateSortColumn) {
            case 'index':
                cmp = a.index - b.index;
                break;
            case 'pos':
                cmp = (a.x + a.y * 10000) - (b.x + b.y * 10000);  // y 우선, x 보조
                break;
            case 'q':
                // N/A (q_value === 0)는 정렬 방향과 무관하게 항상 맨 뒤로
                if (a.q_value === 0 && b.q_value === 0) return 0;
                if (a.q_value === 0) return 1;
                if (b.q_value === 0) return -1;
                cmp = a.q_value - b.q_value;
                break;
            case 'p':
                // score === 0은 정렬 방향과 무관하게 항상 맨 뒤로
                if (a.score === 0 && b.score === 0) return 0;
                if (a.score === 0) return 1;
                if (b.score === 0) return -1;
                cmp = a.score - b.score;
                break;
            case 'n':
                // visits === 0은 정렬 방향과 무관하게 항상 맨 뒤로
                if (a.visits === 0 && b.visits === 0) return 0;
                if (a.visits === 0) return 1;
                if (b.visits === 0) return -1;
                cmp = a.visits - b.visits;
                break;
            default:
                cmp = a.index - b.index;
        }
        return candidateSortDesc ? -cmp : cmp;
    });
    
    sorted.forEach(cand => {
        const item = document.createElement('div');
        item.className = 'candidate-item' + (cand.valid ? '' : ' invalid');
        if (selectedCandidate === cand.index) item.classList.add('selected');
        
        const qVal = cand.q_value !== 0 ? cand.q_value.toFixed(3) : 'N/A';
        const pVal = cand.score.toFixed(3);
        const nVal = cand.visits;
        
        item.innerHTML = `
            <span class="col-index">#${cand.index}</span>
            <span class="col-pos">(${cand.x.toFixed(0)}, ${cand.y.toFixed(0)})</span>
            <span class="col-q">${qVal}</span>
            <span class="col-p">${pVal}</span>
            <span class="col-n">${nVal}</span>
        `;
        
        item.addEventListener('click', () => {
            if (cand.valid) {
                stepAction(cand.index);
            }
        });
        
        item.addEventListener('mouseenter', () => {
            hoveredCandidate = cand.index;
            // updateCandidateInfoPanel(cand);  // 컬럼 헤더로 대체
            render();
        });
        
        item.addEventListener('mouseleave', () => {
            hoveredCandidate = null;
            // updateCandidateInfoPanel(null);  // 컬럼 헤더로 대체
            render();
        });
        
        list.appendChild(item);
    });
}

// 컬럼 헤더로 대체되어 주석 처리
// function updateCandidateInfoPanel(cand) {
//     const posEl = document.getElementById('cand-pos');
//     const qvalEl = document.getElementById('cand-qvalue');
//     const visitsEl = document.getElementById('cand-visits');
//     
//     if (!cand) {
//         posEl.textContent = '-';
//         qvalEl.textContent = '-';
//         visitsEl.textContent = '-';
//         return;
//     }
//     
//     posEl.textContent = `(${cand.x.toFixed(1)}, ${cand.y.toFixed(1)})`;
//     qvalEl.textContent = cand.q_value !== 0 ? cand.q_value.toFixed(4) : '-';
//     visitsEl.textContent = cand.visits > 0 ? cand.visits : '-';
// }

// ============================================================
// Canvas Rendering
// ============================================================

function render() {
    if (!ctx) return;
    
    // Clear
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    if (!currentState) {
        // Draw placeholder
        ctx.fillStyle = '#666';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Create a session to start', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    const scaleX = canvas.width / currentState.grid_width;
    const scaleY = canvas.height / currentState.grid_height;
    
    // Draw grid
    if (settings.showGrid) {
        drawGrid(scaleX, scaleY);
    }
    
    // Draw zones (back to front)
    if (settings.showConstraintZones) drawConstraintZones(scaleX, scaleY, currentState.constraint_zones, COLORS.constraintZone, '#1e90ff');
    if (settings.showForbidden) drawZones(scaleX, scaleY, currentState.forbidden_areas, COLORS.forbidden, '#d62728', '');
    
    // Draw candidates (before placed so they appear behind)
    if (settings.showCandidates) {
        drawCandidates(scaleX, scaleY);
    }
    
    // Draw placed facilities
    drawPlaced(scaleX, scaleY);
    
    // Draw flow arrows (on top)
    if (settings.showFlow) {
        drawFlow(scaleX, scaleY);
    }
}

function drawGrid(scaleX, scaleY) {
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    
    // Draw every 50 grid units
    const step = 50;
    
    for (let x = 0; x <= currentState.grid_width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x * scaleX, 0);
        ctx.lineTo(x * scaleX, canvas.height);
        ctx.stroke();
    }
    
    for (let y = 0; y <= currentState.grid_height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, canvas.height - y * scaleY);
        ctx.lineTo(canvas.width, canvas.height - y * scaleY);
        ctx.stroke();
    }
}

function drawZones(scaleX, scaleY, zones, fillColor, strokeColor, prefix) {
    if (!zones || zones.length === 0) return;
    
    zones.forEach(zone => {
        const x = zone.x0 * scaleX;
        const y = canvas.height - zone.y1 * scaleY;
        const w = (zone.x1 - zone.x0) * scaleX;
        const h = (zone.y1 - zone.y0) * scaleY;
        
        // Fill
        ctx.fillStyle = fillColor;
        ctx.fillRect(x, y, w, h);
        
        // Stroke
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w, h);
        
        // Label
        if (settings.showLabels && (zone.value !== null || zone.id !== null)) {
            let label = '';
            if (zone.id) label = zone.id;
            else if (zone.value !== null) label = prefix + zone.value;
            
            if (label) {
                ctx.fillStyle = strokeColor;
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, x + w / 2, y + h / 2);
            }
        }
    });
}

function drawConstraintZones(scaleX, scaleY, constraints, fillColor, strokeColor) {
    if (!constraints || typeof constraints !== 'object') return;
    for (const zones of Object.values(constraints)) {
        drawZones(scaleX, scaleY, zones, fillColor, strokeColor, '');
    }
}

function drawFlow(scaleX, scaleY) {
    if (!currentState.flow_edges) return;
    
    ctx.strokeStyle = COLORS.flow;
    ctx.lineWidth = 1.8;
    
    currentState.flow_edges.forEach(edge => {
        // Only draw if both endpoints are placed
        if (edge.src_x === null || edge.dst_x === null) return;
        
        const sx = edge.src_x * scaleX;
        const sy = canvas.height - edge.src_y * scaleY;
        const dx = edge.dst_x * scaleX;
        const dy = canvas.height - edge.dst_y * scaleY;
        
        // Draw line
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(dx, dy);
        ctx.stroke();
        
        // Draw arrowhead
        const angle = Math.atan2(dy - sy, dx - sx);
        const arrowLen = 10;
        ctx.beginPath();
        ctx.moveTo(dx, dy);
        ctx.lineTo(
            dx - arrowLen * Math.cos(angle - Math.PI / 6),
            dy - arrowLen * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(dx, dy);
        ctx.lineTo(
            dx - arrowLen * Math.cos(angle + Math.PI / 6),
            dy - arrowLen * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
    });
}

function drawPlaced(scaleX, scaleY) {
    currentState.placed.forEach(fac => {
        const x = fac.x * scaleX;
        const y = canvas.height - (fac.y + fac.h) * scaleY;
        const w = fac.w * scaleX;
        const h = fac.h * scaleY;
        
        // Fill
        ctx.fillStyle = COLORS.placed.fill;
        ctx.fillRect(x, y, w, h);
        
        // Stroke
        ctx.strokeStyle = COLORS.placed.stroke;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, w, h);
        
        // Label
        if (settings.showLabels) {
            ctx.fillStyle = COLORS.text;
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // Truncate long names
            let label = fac.gid;
            if (label.length > 15) {
                label = label.substring(0, 12) + '..';
            }
            ctx.fillText(label, x + w / 2, y + h / 2);
        }
    });
}

function drawCandidates(scaleX, scaleY) {
    const candidates = currentState.candidates;
    if (!candidates || candidates.length === 0) return;
    
    // Check if search has been performed (any Q value exists)
    const hasAnyQ = candidates.some(c => c.q_value !== 0);
    
    // Filter: search 후에는 Q값 있는 것만, search 전에는 valid만
    const visibleCandidates = hasAnyQ 
        ? candidates.filter(c => c.q_value !== 0)
        : candidates.filter(c => c.valid);
    
    if (visibleCandidates.length === 0) return;
    
    // Value for color mapping: Q if searched, P otherwise
    const getDisplayValue = (cand) => hasAnyQ ? cand.q_value : cand.score;
    
    // Find value range for color mapping
    const values = visibleCandidates.map(c => getDisplayValue(c));
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = maxValue - minValue || 1;
    
    visibleCandidates.forEach(cand => {
        const cx = cand.x * scaleX;
        const cy = canvas.height - cand.y * scaleY;
        
        // Determine color
        let color;
        if (cand.index === hoveredCandidate) {
            color = COLORS.candidate.hovered;
        } else if (cand.index === selectedCandidate) {
            color = COLORS.candidate.selected;
        } else if (!cand.valid) {
            color = COLORS.candidate.invalid;
        } else {
            // Color by value (green = high, red = low)
            const displayVal = getDisplayValue(cand);
            const t = (displayVal - minValue) / valueRange;
            color = scoreToColor(t);
        }
        
        // Size by visits if showing
        let radius = 5;
        if (settings.showVisits && cand.visits > 0) {
            radius = Math.min(15, 5 + Math.log(cand.visits + 1) * 2);
        }
        
        // Draw point
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        
        // Highlight hovered/selected
        if (cand.index === hoveredCandidate || cand.index === selectedCandidate) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

function scoreToColor(t) {
    // t: 0 (low) -> 1 (high)
    // Red (low) -> Yellow (mid) -> Green (high)
    const r = t < 0.5 ? 255 : Math.round(255 * (1 - (t - 0.5) * 2));
    const g = t < 0.5 ? Math.round(255 * t * 2) : 255;
    const b = 50;
    return `rgb(${r}, ${g}, ${b})`;
}

// ============================================================
// Canvas Interactions
// ============================================================

function onCanvasMouseMove(e) {
    if (!currentState) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const scaleX = canvas.width / currentState.grid_width;
    const scaleY = canvas.height / currentState.grid_height;
    
    // Find nearest candidate
    let nearest = null;
    let minDist = 20; // Max distance to consider
    
    currentState.candidates.forEach(cand => {
        const cx = cand.x * scaleX;
        const cy = canvas.height - cand.y * scaleY;
        const dist = Math.sqrt((mouseX - cx) ** 2 + (mouseY - cy) ** 2);
        
        if (dist < minDist) {
            minDist = dist;
            nearest = cand;
        }
    });
    
    if (nearest !== null && nearest.index !== hoveredCandidate) {
        hoveredCandidate = nearest.index;
        showHoverInfo(e.clientX, e.clientY, nearest);
        render();
    } else if (nearest === null && hoveredCandidate !== null) {
        hoveredCandidate = null;
        hideHoverInfo();
        render();
    }
}

function onCanvasClick(e) {
    if (!currentState || hoveredCandidate === null) return;
    
    const cand = currentState.candidates.find(c => c.index === hoveredCandidate);
    if (cand && cand.valid) {
        stepAction(cand.index);
    }
}

function onCanvasMouseLeave() {
    hoveredCandidate = null;
    hideHoverInfo();
    render();
}

function showHoverInfo(x, y, cand) {
    const info = document.getElementById('hover-info');
    info.style.display = 'block';
    info.style.left = `${x + 15}px`;
    info.style.top = `${y + 15}px`;
    info.innerHTML = `
        <div><strong>#${cand.index}</strong> ${cand.valid ? '✓' : '✗'}</div>
        <div>Pos: (${cand.x.toFixed(1)}, ${cand.y.toFixed(1)})</div>
        <div>Prior: ${cand.score.toFixed(4)}</div>
        ${cand.visits > 0 ? `<div>Visits: ${cand.visits}</div>` : ''}
        ${cand.q_value !== 0 ? `<div>Q-value: ${cand.q_value.toFixed(4)}</div>` : ''}
    `;
}

function hideHoverInfo() {
    document.getElementById('hover-info').style.display = 'none';
}

// ============================================================
// Start
// ============================================================

init();
