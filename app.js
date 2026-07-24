// Helper function to generate Gaussian (Normal) distributed random numbers
// Using the Box-Muller transform
function randomNormal(mean, stdDev) {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); // Convert [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return num * stdDev + mean;
}

// Well Simulator Class in JavaScript
class WellSimulator {
    constructor(initialChoke = 30.0) {
        this.choke = initialChoke;
        
        // Initial steady-state conditions at choke = 30%
        this.Q = 90.0;
        this.WHP = 250.0;
        this.FLP = 180.0;
        this.BHP = 3000.0;
        
        // Model Parameters: (a, b, c, noise_sigma)
        this.params = {
            Q:   [0.82366,  0.32006,   6.93175, 0.73],
            WHP: [0.88924, -0.17564,  35.37564, 0.66],
            FLP: [0.86253, -0.13556,  30.14338, 0.52],
            BHP: [0.92574, -0.62250, 253.76057, 2.91]
        };
    }

    step(chokePosition, addNoise = true) {
        // Clip choke to physical limits [0, 100]%
        chokePosition = Math.max(0.0, Math.min(100.0, chokePosition));
        
        // Generate noise
        const n_q = addNoise ? randomNormal(0, this.params.Q[3]) : 0.0;
        const n_w = addNoise ? randomNormal(0, this.params.WHP[3]) : 0.0;
        const n_f = addNoise ? randomNormal(0, this.params.FLP[3]) : 0.0;
        const n_b = addNoise ? randomNormal(0, this.params.BHP[3]) : 0.0;
        
        // Update states based on previous choke position
        this.Q = this.params.Q[0] * this.Q + this.params.Q[1] * this.choke + this.params.Q[2] + n_q;
        this.WHP = this.params.WHP[0] * this.WHP + this.params.WHP[1] * this.choke + this.params.WHP[2] + n_w;
        this.FLP = this.params.FLP[0] * this.FLP + this.params.FLP[1] * this.choke + this.params.FLP[2] + n_f;
        this.BHP = this.params.BHP[0] * this.BHP + this.params.BHP[1] * this.choke + this.params.BHP[2] + n_b;
        
        // Save current choke position for the next interval
        this.choke = chokePosition;
        
        return {
            Q: this.Q,
            WHP: this.WHP,
            FLP: this.FLP,
            BHP: this.BHP
        };
    }
}

// MPC Controller Class in JavaScript
class MPCController {
    constructor(Hp = 30, lambda_u = 0.5, whp_min = 220.0, flp_min = 150.0, bhp_min = 2900.0) {
        this.Hp = Hp;
        this.lambda_u = lambda_u;
        this.whp_min = whp_min;
        this.flp_min = flp_min;
        this.bhp_min = bhp_min;
        
        // Model Parameters for prediction (deterministic part)
        this.params = {
            Q:   [0.82366,  0.32006,   6.93175],
            WHP: [0.88924, -0.17564,  35.37564],
            FLP: [0.86253, -0.13556,  30.14338],
            BHP: [0.92574, -0.62250, 253.76057]
        };
    }

    predictTrajectory(u_cand, Q_init, WHP_init, FLP_init, BHP_init) {
        const Q_pred = new Array(this.Hp + 1);
        const WHP_pred = new Array(this.Hp + 1);
        const FLP_pred = new Array(this.Hp + 1);
        const BHP_pred = new Array(this.Hp + 1);
        
        Q_pred[0] = Q_init;
        WHP_pred[0] = WHP_init;
        FLP_pred[0] = FLP_init;
        BHP_pred[0] = BHP_init;
        
        for (let j = 1; j <= this.Hp; j++) {
            Q_pred[j] = this.params.Q[0] * Q_pred[j-1] + this.params.Q[1] * u_cand + this.params.Q[2];
            WHP_pred[j] = this.params.WHP[0] * WHP_pred[j-1] + this.params.WHP[1] * u_cand + this.params.WHP[2];
            FLP_pred[j] = this.params.FLP[0] * FLP_pred[j-1] + this.params.FLP[1] * u_cand + this.params.FLP[2];
            BHP_pred[j] = this.params.BHP[0] * BHP_pred[j-1] + this.params.BHP[1] * u_cand + this.params.BHP[2];
        }
        
        // Return trajectories (excluding initial measurement)
        return {
            Q: Q_pred.slice(1),
            WHP: WHP_pred.slice(1),
            FLP: FLP_pred.slice(1),
            BHP: BHP_pred.slice(1)
        };
    }

    calculateControl(Q_meas, WHP_meas, FLP_meas, BHP_meas, current_choke, Q_target) {
        // Choke ramp rate limit is +/- 5% per hour
        const u_min = Math.max(0.0, current_choke - 5.0);
        const u_max = Math.min(100.0, current_choke + 5.0);
        
        // Evaluate 101 candidates between u_min and u_max
        let best_u = current_choke;
        let min_cost = Infinity;
        
        let num_safe = 0;
        let num_whp_viol = 0;
        let num_flp_viol = 0;
        let num_bhp_viol = 0;
        let rejection_example = "None";
        let best_trajectory_info = {};
        
        const w_Q = 1.0;
        const w_viol = 1e6; // Large penalty weight for constraint violations
        
        for (let i = 0; i <= 100; i++) {
            const u_cand = u_min + (u_max - u_min) * (i / 100);
            
            const traj = this.predictTrajectory(u_cand, Q_meas, WHP_meas, FLP_meas, BHP_meas);
            
            // Check constraint violations over prediction horizon Hp
            let is_whp_viol = false;
            let is_flp_viol = false;
            let is_bhp_viol = false;
            
            let viol_whp = 0;
            let viol_flp = 0;
            let viol_bhp = 0;
            
            for (let j = 0; j < this.Hp; j++) {
                const whp_viol_amt = Math.max(0.0, this.whp_min - traj.WHP[j]);
                const flp_viol_amt = Math.max(0.0, this.flp_min - traj.FLP[j]);
                const bhp_viol_amt = Math.max(0.0, this.bhp_min - traj.BHP[j]);
                
                viol_whp += whp_viol_amt * whp_viol_amt;
                viol_flp += flp_viol_amt * flp_viol_amt;
                viol_bhp += bhp_viol_amt * bhp_viol_amt;
                
                if (whp_viol_amt > 0.0) {
                    is_whp_viol = true;
                    if (rejection_example === "None") {
                        rejection_example = `Choke ${u_cand.toFixed(1)}% rejected: WHP predicted ${traj.WHP[j].toFixed(1)} psi (limit >= ${this.whp_min.toFixed(1)} psi)`;
                    }
                }
                if (flp_viol_amt > 0.0) {
                    is_flp_viol = true;
                    if (rejection_example === "None") {
                        rejection_example = `Choke ${u_cand.toFixed(1)}% rejected: FLP predicted ${traj.FLP[j].toFixed(1)} psi (limit >= ${this.flp_min.toFixed(1)} psi)`;
                    }
                }
                if (bhp_viol_amt > 0.0) {
                    is_bhp_viol = true;
                    if (rejection_example === "None") {
                        rejection_example = `Choke ${u_cand.toFixed(1)}% rejected: BHP predicted ${traj.BHP[j].toFixed(1)} psi (limit >= ${this.bhp_min.toFixed(1)} psi)`;
                    }
                }
            }
            
            if (!is_whp_viol && !is_flp_viol && !is_bhp_viol) {
                num_safe++;
            } else {
                if (is_whp_viol) num_whp_viol++;
                if (is_flp_viol) num_flp_viol++;
                if (is_bhp_viol) num_bhp_viol++;
            }
            
            // Calculate tracking error (sum of squares over prediction horizon)
            let tracking_err = 0;
            for (let j = 0; j < this.Hp; j++) {
                tracking_err += (traj.Q[j] - Q_target) * (traj.Q[j] - Q_target);
            }
            
            // Choke movement penalty
            const choke_move_penalty = this.lambda_u * (u_cand - current_choke) * (u_cand - current_choke);
            
            const cost = w_Q * tracking_err + w_viol * (viol_whp + viol_flp + viol_bhp) + choke_move_penalty;
            
            if (cost < min_cost) {
                min_cost = cost;
                best_u = u_cand;
                best_trajectory_info = {
                    expected_flow: traj.Q[this.Hp - 1],
                    expected_whp: traj.WHP[this.Hp - 1],
                    expected_flp: traj.FLP[this.Hp - 1],
                    expected_bhp: traj.BHP[this.Hp - 1],
                };
            }
        }
        
        return {
            recommended_choke: best_u,
            diagnostics: {
                expected_flow: best_trajectory_info.expected_flow,
                expected_whp: best_trajectory_info.expected_whp,
                expected_flp: best_trajectory_info.expected_flp,
                expected_bhp: best_trajectory_info.expected_bhp,
                num_safe: num_safe,
                num_rejected: 101 - num_safe,
                rejection_example: rejection_example,
                status: (101 - num_safe) === 0 ? 'SAFE' : ((101 - num_safe) < 101 ? 'LIMIT ACTIVE' : 'UNSAFE_SYSTEM_LIMIT')
            }
        };
    }
}

// ----------------------------------------------------
// UI Logic & Chart rendering
// ----------------------------------------------------

let simulator = null;
let controller = null;
let current_hour = 0;
let sim_interval_id = null;
let is_running = false;
let max_scenario_steps = 60;

// Data History arrays
let t_history = [];
let target_history = [];
let Q_history = [];
let choke_history = [];
let whp_history = [];
let flp_history = [];
let bhp_history = [];

// Chart references
let chartRate = null;
let chartChoke = null;
let chartPressures = null;
let chartBHP = null;

// DOM Element references
const scenarioSelect = document.getElementById("scenario-select");
const manualGroup = document.querySelector(".manual-only");
const targetSlider = document.getElementById("target-slider");
const targetValueLabel = document.getElementById("target-value");

const paramHpSlider = document.getElementById("param-hp");
const valueHp = document.getElementById("value-hp");
const paramLambdaSlider = document.getElementById("param-lambda");
const valueLambda = document.getElementById("value-lambda");
const paramWhpSlider = document.getElementById("param-whp");
const valueWhp = document.getElementById("value-whp");
const paramFlpSlider = document.getElementById("param-flp");
const valueFlp = document.getElementById("value-flp");
const paramBhpSlider = document.getElementById("param-bhp");
const valueBhp = document.getElementById("value-bhp");
const paramNoiseCheckbox = document.getElementById("param-noise");

const btnPlay = document.getElementById("btn-play");
const btnStep = document.getElementById("btn-step");
const btnReset = document.getElementById("btn-reset");
const btnClearConsole = document.getElementById("btn-clear-console");
const consoleBox = document.getElementById("console-box");

const kpiTarget = document.getElementById("kpi-target");
const kpiFlow = document.getElementById("kpi-flow");
const kpiChoke = document.getElementById("kpi-choke");
const kpiStatus = document.getElementById("kpi-status");
const kpiHour = document.getElementById("kpi-hour");

// SVG Elements
const flowLine = document.getElementById("flow-line");
const valveHandwheel = document.getElementById("valve-handwheel");
const valvePct = document.getElementById("valve-pct");
const labelWHP = document.getElementById("label-whp");
const labelFLP = document.getElementById("label-flp");
const labelBHP = document.getElementById("label-bhp");

// Initialize Charts
function initCharts() {
    const configRate = {
        type: 'line',
        data: {
            labels: t_history,
            datasets: [
                {
                    label: 'Target Oil Rate',
                    data: target_history,
                    borderColor: '#ef4444',
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Actual Oil Flow Rate',
                    data: Q_history,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: (context) => context.dataIndex === Q_history.length - 1 ? 4 : 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Oil Flow Rate (Q vs Target)', color: '#fafafa', font: { weight: 'bold', size: 12 } },
                legend: { labels: { color: '#a1a1aa', font: { size: 10 } } }
            },
            scales: {
                x: { grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } },
                y: { min: 40, max: 260, grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } }
            }
        }
    };

    const configChoke = {
        type: 'line',
        data: {
            labels: t_history,
            datasets: [{
                label: 'Choke Position',
                data: choke_history,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: (context) => context.dataIndex === choke_history.length - 1 ? 4 : 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Choke Opening (%)', color: '#fafafa', font: { weight: 'bold', size: 12 } },
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } },
                y: { min: 20, max: 80, grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } }
            }
        }
    };

    const configPressures = {
        type: 'line',
        data: {
            labels: t_history,
            datasets: [
                {
                    label: 'WHP',
                    data: whp_history,
                    borderColor: '#a78bfa',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'FLP',
                    data: flp_history,
                    borderColor: '#22d3ee',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'WHP & FLP (psi)', color: '#fafafa', font: { weight: 'bold', size: 12 } },
                legend: { labels: { color: '#a1a1aa', font: { size: 10 } } }
            },
            scales: {
                x: { grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } },
                y: { min: 100, max: 300, grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } }
            }
        }
    };

    const configBHP = {
        type: 'line',
        data: {
            labels: t_history,
            datasets: [{
                label: 'BHP',
                data: bhp_history,
                borderColor: '#fb7185',
                backgroundColor: 'rgba(251, 113, 133, 0.05)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: (context) => context.dataIndex === bhp_history.length - 1 ? 4 : 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Bottom Hole Pressure (BHP, psi)', color: '#fafafa', font: { weight: 'bold', size: 12 } },
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } },
                y: { min: 2850, max: 3150, grid: { color: '#27272a' }, ticks: { color: '#a1a1aa' } }
            }
        }
    };

    chartRate = new Chart(document.getElementById('chart-rate'), configRate);
    chartChoke = new Chart(document.getElementById('chart-choke'), configChoke);
    chartPressures = new Chart(document.getElementById('chart-pressures-top'), configPressures);
    chartBHP = new Chart(document.getElementById('chart-bhp'), configBHP);
}

// Update charts dataset values and re-render
function updateCharts() {
    chartRate.update();
    chartChoke.update();
    chartPressures.update();
    chartBHP.update();
}

// Reset data structures
function resetSimulation() {
    stopSimulation();
    
    simulator = new WellSimulator(30.0);
    
    // Read advanced parameters from UI
    const Hp = parseInt(paramHpSlider.value);
    const lambda = parseFloat(paramLambdaSlider.value);
    const whp_min = parseFloat(paramWhpSlider.value);
    const flp_min = parseFloat(paramFlpSlider.value);
    const bhp_min = parseFloat(paramBhpSlider.value);
    controller = new MPCController(Hp, lambda, whp_min, flp_min, bhp_min);
    
    current_hour = 0;
    t_history.length = 0;
    target_history.length = 0;
    Q_history.length = 0;
    choke_history.length = 0;
    whp_history.length = 0;
    flp_history.length = 0;
    bhp_history.length = 0;
    
    // Add initial state to history
    t_history.push(0);
    target_history.push(getSetpoint(0));
    Q_history.push(simulator.Q);
    choke_history.push(simulator.choke);
    whp_history.push(simulator.WHP);
    flp_history.push(simulator.FLP);
    bhp_history.push(simulator.BHP);
    
    // Update KPI panels
    kpiTarget.textContent = getSetpoint(0).toFixed(1);
    kpiFlow.textContent = simulator.Q.toFixed(1);
    kpiChoke.textContent = simulator.choke.toFixed(1);
    kpiStatus.textContent = "SAFE";
    kpiStatus.className = "kpi-val status-safe";
    kpiHour.textContent = "Hour 0";
    
    // Update SVG values
    valvePct.textContent = "30.0%";
    labelWHP.textContent = `WHP: ${simulator.WHP.toFixed(1)} psi`;
    labelFLP.textContent = `FLP: ${simulator.FLP.toFixed(1)} psi`;
    labelBHP.textContent = `BHP: ${simulator.BHP.toFixed(1)} psi`;
    
    flowLine.style.animationPlayState = "paused";
    valveHandwheel.setAttribute("class", "");
    
    clearConsole();
    logToConsole("System initialized. Parameters loaded.", "system");
    logToConsole(`Initial states: Flow Q=${simulator.Q.toFixed(1)} bbl/hr, BHP=${simulator.BHP.toFixed(1)} psi, WHP=${simulator.WHP.toFixed(1)} psi.`, "system");
    
    updateCharts();
}

// Helper to determine target setpoint based on selected scenario and hour
function getSetpoint(hour) {
    const sc = scenarioSelect.value;
    if (sc === "scenario_a") {
        max_scenario_steps = 40;
        return 120.0;
    } else if (sc === "scenario_b") {
        max_scenario_steps = 60;
        return hour < 20 ? 100.0 : 150.0;
    } else if (sc === "scenario_c") {
        max_scenario_steps = 60;
        return hour < 20 ? 100.0 : 200.0;
    } else {
        max_scenario_steps = 100;
        return parseFloat(targetSlider.value);
    }
}

// Single step execution (1 hour)
function executeStep() {
    if (current_hour >= max_scenario_steps) {
        stopSimulation();
        logToConsole(`Scenario completed.`, "system");
        
        // Print Summary block if Scenario C finishes
        if (scenarioSelect.value === "scenario_c") {
            // Compute final 10 hour average rate
            const final_rates = Q_history.slice(-10);
            const avg_rate = final_rates.reduce((a, b) => a + b, 0) / final_rates.length;
            
            logToConsole("====================================================", "system");
            logToConsole("Scenario C Summary: Infeasible Target Analysis", "system");
            logToConsole("====================================================", "system");
            logToConsole("Target Requested:        200.0 bbl/hr", "system");
            logToConsole(`Maximum Safe Production: ${avg_rate.toFixed(1)} bbl/hr`, "system");
            logToConsole("Reason:                  BHP constraint active (limit >= 2900 psi)", "warning");
            logToConsole("                         WHP constraint active (limit >= 220 psi)", "warning");
            logToConsole("Controller Status:       Settled at maximum safe operating point.", "safe");
            logToConsole("====================================================", "system");
        }
        return;
    }
    
    current_hour++;
    const target = getSetpoint(current_hour);
    const Q = simulator.Q;
    const WHP = simulator.WHP;
    const FLP = simulator.FLP;
    const BHP = simulator.BHP;
    const choke = simulator.choke;
    
    // Call controller
    const controlResult = controller.calculateControl(Q, WHP, FLP, BHP, choke, target);
    const nextChoke = controlResult.recommended_choke;
    const diag = controlResult.diagnostics;
    
    // Step Simulator
    const addNoise = paramNoiseCheckbox.checked;
    const newStates = simulator.step(nextChoke, addNoise);
    
    // Append to histories
    t_history.push(current_hour);
    target_history.push(target);
    Q_history.push(newStates.Q);
    choke_history.push(nextChoke);
    whp_history.push(newStates.WHP);
    flp_history.push(newStates.FLP);
    bhp_history.push(newStates.BHP);
    
    // Update KPIs
    kpiTarget.textContent = target.toFixed(1);
    kpiFlow.textContent = newStates.Q.toFixed(1);
    kpiChoke.textContent = nextChoke.toFixed(1);
    kpiHour.textContent = `Hour ${current_hour}`;
    
    // Safety status indicator color
    let statusClass = "safe";
    if (diag.status === "LIMIT ACTIVE") {
        statusClass = "warning";
        kpiStatus.textContent = "LIMIT ACTIVE";
    } else if (diag.status === "UNSAFE_SYSTEM_LIMIT") {
        statusClass = "danger";
        kpiStatus.textContent = "UNSAFE";
    } else {
        kpiStatus.textContent = "SAFE";
    }
    kpiStatus.className = `kpi-val status-${statusClass}`;
    
    // Update SVG layout values
    valvePct.textContent = `${nextChoke.toFixed(1)}%`;
    labelWHP.textContent = `WHP: ${newStates.WHP.toFixed(1)} psi`;
    labelFLP.textContent = `FLP: ${newStates.FLP.toFixed(1)} psi`;
    labelBHP.textContent = `BHP: ${newStates.BHP.toFixed(1)} psi`;
    
    // Toggle active valve rotating class if choke is changing
    const choke_diff = Math.abs(nextChoke - choke);
    if (choke_diff > 0.05) {
        valveHandwheel.setAttribute("class", "valve-rotating");
    } else {
        valveHandwheel.setAttribute("class", "");
    }
    
    // Terminal logging
    const logText = `Hour ${current_hour.toString().padStart(2)} | Target: ${target.toFixed(1)} | Rec Choke: ${nextChoke.toFixed(1)}% | Expected Flow: ${diag.expected_flow.toFixed(1)} | Expected BHP: ${diag.expected_bhp.toFixed(0)} | Status: ${diag.status} | Candidates: 101 | Safe: ${diag.num_safe} | Rejected: ${diag.num_rejected}`;
    logToConsole(logText, statusClass);
    
    if (diag.num_rejected > 0) {
        logToConsole(`  +- Example rejection: ${diag.rejection_example}`, "sub");
    }
    
    updateCharts();
}

// Log line printing in dashboard box
function logToConsole(text, type = "system") {
    const el = document.createElement("div");
    el.className = `log-line ${type}`;
    el.textContent = text;
    consoleBox.appendChild(el);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

function clearConsole() {
    consoleBox.innerHTML = "";
}

// Simulation loop runner control functions
function startSimulation() {
    if (is_running) return;
    is_running = true;
    btnPlay.textContent = "⏸ Pause Simulation";
    btnPlay.className = "btn btn-secondary";
    flowLine.style.animationPlayState = "running";
    
    sim_interval_id = setInterval(() => {
        executeStep();
    }, 250); // Fast simulation execution speed (4 steps per second)
}

function stopSimulation() {
    if (!is_running) return;
    is_running = false;
    btnPlay.textContent = "▶ Run Scenario";
    btnPlay.className = "btn btn-primary";
    flowLine.style.animationPlayState = "paused";
    valveHandwheel.setAttribute("class", "");
    
    if (sim_interval_id) {
        clearInterval(sim_interval_id);
        sim_interval_id = null;
    }
}

// Set up UI Event listeners
function setupListeners() {
    scenarioSelect.addEventListener("change", () => {
        const sc = scenarioSelect.value;
        if (sc === "manual") {
            manualGroup.style.display = "block";
        } else {
            manualGroup.style.display = "none";
        }
        resetSimulation();
    });

    targetSlider.addEventListener("input", () => {
        targetValueLabel.textContent = `${targetSlider.value} bbl/hr`;
        if (scenarioSelect.value === "manual") {
            kpiTarget.textContent = parseFloat(targetSlider.value).toFixed(1);
        }
    });

    // Event listeners to update slider badges
    paramHpSlider.addEventListener("input", () => {
        valueHp.textContent = `${paramHpSlider.value}h`;
        resetSimulation();
    });
    paramLambdaSlider.addEventListener("input", () => {
        valueLambda.textContent = paramLambdaSlider.value;
        resetSimulation();
    });
    paramWhpSlider.addEventListener("input", () => {
        valueWhp.textContent = `${paramWhpSlider.value} psi`;
        resetSimulation();
    });
    paramFlpSlider.addEventListener("input", () => {
        valueFlp.textContent = `${paramFlpSlider.value} psi`;
        resetSimulation();
    });
    paramBhpSlider.addEventListener("input", () => {
        valueBhp.textContent = `${paramBhpSlider.value} psi`;
        resetSimulation();
    });

    btnPlay.addEventListener("click", () => {
        if (is_running) {
            stopSimulation();
        } else {
            startSimulation();
        }
    });

    btnStep.addEventListener("click", () => {
        stopSimulation();
        executeStep();
    });

    btnReset.addEventListener("click", () => {
        resetSimulation();
    });

    btnClearConsole.addEventListener("click", () => {
        clearConsole();
    });
}

// Window Onload Page trigger
window.addEventListener("DOMContentLoaded", () => {
    initCharts();
    setupListeners();
    resetSimulation();
});
