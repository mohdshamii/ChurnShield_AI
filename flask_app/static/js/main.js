/**
 * ChurnShield AI - Interactive Client-Side Engine
 * Implements AJAX real-time predictions, What-If simulation,
 * dynamic risk gauges, and UI interactivity.
 */

document.addEventListener("DOMContentLoaded", function () {
  // Initialize Toast Notifications container
  initToastContainer();

  // Single Customer Predict Form
  const predictForm = document.getElementById("predict-form");
  if (predictForm) {
    predictForm.addEventListener("submit", handlePredictSubmit);
  }

  // Preset Buttons (High Risk, Loyal, Enterprise)
  const presetButtons = document.querySelectorAll(".btn-preset");
  presetButtons.forEach(btn => {
    btn.addEventListener("click", function () {
      applyPreset(this.dataset.preset);
    });
  });

  // What-If Simulator Inputs
  const whatIfForm = document.getElementById("whatif-form");
  if (whatIfForm) {
    const inputs = whatIfForm.querySelectorAll("select, input");
    inputs.forEach(input => {
      input.addEventListener("change", runWhatIfSimulation);
    });
  }

  // Save Prediction to DB button
  const saveBtn = document.getElementById("btn-save-prediction");
  if (saveBtn) {
    saveBtn.addEventListener("click", handleSavePrediction);
  }

  // Copy Buttons for Retention Templates
  const copyButtons = document.querySelectorAll(".btn-copy-text");
  copyButtons.forEach(btn => {
    btn.addEventListener("click", function () {
      const targetId = this.dataset.target;
      const text = document.getElementById(targetId).innerText;
      navigator.clipboard.writeText(text).then(() => {
        showToast("Copied to clipboard!", "success");
      });
    });
  });
});

// Toast notification helper
function initToastContainer() {
  if (!document.getElementById("toast-container")) {
    const container = document.createElement("div");
    container.id = "toast-container";
    container.style.position = "fixed";
    container.style.bottom = "24px";
    container.style.right = "24px";
    container.style.zIndex = "9999";
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "10px";
    document.body.appendChild(container);
  }
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.style.background = type === "success" ? "#00ffaa" : (type === "danger" ? "#ff3d57" : "#0095ff");
  toast.style.color = "#0b0f19";
  toast.style.padding = "12px 20px";
  toast.style.borderRadius = "8px";
  toast.style.fontWeight = "600";
  toast.style.boxShadow = "0 8px 24px rgba(0,0,0,0.5)";
  toast.style.transition = "all 0.3s ease";
  toast.style.fontFamily = "'Inter', sans-serif";
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// Preset application
const PRESETS = {
  high_risk: {
    customer_id: "CUST-RISK-882",
    gender: "Female",
    SeniorCitizen: "0",
    Partner: "No",
    Dependents: "No",
    tenure: "2",
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: "94.8",
    TotalCharges: "189.6"
  },
  loyal: {
    customer_id: "CUST-LOYAL-104",
    gender: "Male",
    SeniorCitizen: "0",
    Partner: "Yes",
    Dependents: "Yes",
    tenure: "64",
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Two year",
    PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)",
    MonthlyCharges: "58.4",
    TotalCharges: "3737.6"
  },
  enterprise: {
    customer_id: "CUST-CORP-450",
    gender: "Female",
    SeniorCitizen: "0",
    Partner: "Yes",
    Dependents: "No",
    tenure: "28",
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Fiber optic",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "One year",
    PaperlessBilling: "Yes",
    PaymentMethod: "Bank transfer (automatic)",
    MonthlyCharges: "108.5",
    TotalCharges: "3038.0"
  }
};

function applyPreset(name) {
  const data = PRESETS[name];
  if (!data) return;
  const form = document.getElementById("predict-form");
  for (const [key, value] of Object.entries(data)) {
    const field = form.elements[key];
    if (field) {
      field.value = value;
    }
  }
  showToast(`Applied ${name.toUpperCase().replace('_', ' ')} profile preset!`, "info");
}

// AJAX Prediction Handler
async function handlePredictSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  const jsonPayload = {};
  formData.forEach((val, key) => jsonPayload[key] = val);

  const submitBtn = form.querySelector("button[type='submit']");
  const origText = submitBtn.innerHTML;
  submitBtn.innerHTML = `<span>Computing SHAP Risk Score...</span>`;
  submitBtn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest"
      },
      body: JSON.stringify(jsonPayload)
    });

    const result = await res.json();
    submitBtn.innerHTML = origText;
    submitBtn.disabled = false;

    if (!result.success) {
      showToast(result.error || "Prediction failed.", "danger");
      return;
    }

    renderPredictionResults(result, jsonPayload);
    showToast("Prediction and SHAP analysis computed successfully!", "success");

  } catch (err) {
    submitBtn.innerHTML = origText;
    submitBtn.disabled = false;
    showToast("Network error executing prediction.", "danger");
    console.error(err);
  }
}

// Global state holding current prediction
let currentPredictionState = null;

function renderPredictionResults(res, inputPayload) {
  currentPredictionState = { ...res, raw_input: inputPayload };

  const resultsContainer = document.getElementById("prediction-results-area");
  if (resultsContainer) {
    resultsContainer.style.display = "block";
    resultsContainer.scrollIntoView({ behavior: "smooth" });
  }

  // Update Radial Gauge
  const gaugeCircle = document.getElementById("gauge-circle");
  const gaugeVal = document.getElementById("gauge-val");
  const riskBadge = document.getElementById("risk-badge");
  const riskUrgency = document.getElementById("risk-urgency");

  const prob = res.churn_prob;
  gaugeVal.innerText = `${prob}%`;
  riskBadge.innerText = `${res.risk_meta.level} RISK`;
  riskBadge.className = `badge ${res.risk_meta.badge_class}`;
  riskUrgency.innerText = res.risk_meta.urgency;

  // Gradient conic fill
  const color = res.risk_meta.color;
  gaugeCircle.style.background = `conic-gradient(${color} 0% ${prob}%, rgba(255,255,255,0.06) ${prob}% 100%)`;
  gaugeCircle.style.boxShadow = `0 0 25px ${res.risk_meta.glow_color}`;

  // Update Financials
  document.getElementById("disp-annual-rev").innerText = `$${res.clv_roi.annual_revenue.toLocaleString()}`;
  document.getElementById("disp-rev-risk").innerText = `$${res.clv_roi.revenue_at_risk.toLocaleString()}`;
  document.getElementById("disp-projected-clv").innerText = `$${res.clv_roi.projected_clv.toLocaleString()}`;
  document.getElementById("disp-roi").innerText = `+${res.clv_roi.retention_roi_percent}% ROI`;

  // Render Feature Impact / SHAP bars
  renderFeatureBars(res.explanation.impact_list);

  // Render Retention Playbook
  renderRetentionPlaybook(res.retention_plan);

  // Sync baseline for What-If simulator
  syncWhatIfBaseline(inputPayload, res.churn_prob);
}

function renderFeatureBars(impactList) {
  const container = document.getElementById("feature-impacts-list");
  if (!container) return;
  container.innerHTML = "";

  const maxAbs = Math.max(...impactList.map(item => Math.abs(item.impact)), 0.01);

  impactList.forEach(item => {
    const widthPct = Math.min(100, Math.round((Math.abs(item.impact) / maxAbs) * 100));
    const isRisk = item.direction === "INCREASE_RISK";
    const barColor = isRisk ? "#ff3d57" : "#00ffaa";
    const sign = item.impact > 0 ? "+" : "";

    const row = document.createElement("div");
    row.className = "factor-bar-row";
    row.innerHTML = `
      <div class="factor-name" title="${item.factor}">${item.factor}</div>
      <div class="factor-bar-track">
        <div class="factor-bar-fill" style="width: ${widthPct}%; background: ${barColor};"></div>
      </div>
      <div class="factor-val" style="color: ${barColor};">${sign}${item.impact}</div>
    `;
    container.appendChild(row);
  });
}

function renderRetentionPlaybook(plan) {
  const stratContainer = document.getElementById("strategies-list");
  if (stratContainer) {
    stratContainer.innerHTML = "";
    plan.strategies.forEach(st => {
      const card = document.createElement("div");
      card.className = "strategy-item";
      card.style.background = "rgba(15, 23, 42, 0.6)";
      card.style.padding = "14px";
      card.style.borderRadius = "8px";
      card.style.marginBottom = "10px";
      card.style.borderLeft = "3px solid #00ffaa";
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
          <strong style="color: #fff;">${st.title}</strong>
          <span class="badge badge-success" style="font-size: 0.7rem;">${st.tag}</span>
        </div>
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">${st.detail}</p>
      `;
      stratContainer.appendChild(card);
    });
  }

  // Update Timeline
  const timeContainer = document.getElementById("timeline-list");
  if (timeContainer) {
    timeContainer.innerHTML = "";
    plan.timeline.forEach(t => {
      const li = document.createElement("li");
      li.style.marginBottom = "8px";
      li.style.fontSize = "0.85rem";
      li.style.color = "#cbd5e1";
      li.innerHTML = `<strong style="color: #00ffaa;">${t.day}:</strong> ${t.action}`;
      timeContainer.appendChild(li);
    });
  }

  // Update templates
  const emailElem = document.getElementById("template-email-body");
  if (emailElem) emailElem.innerText = plan.email_template;

  const smsElem = document.getElementById("template-sms-body");
  if (smsElem) smsElem.innerText = plan.sms_template;
}

// What-If Simulator Handler
function syncWhatIfBaseline(inputPayload, baselineProb) {
  const whatIfBox = document.getElementById("whatif-panel");
  if (whatIfBox) whatIfBox.style.display = "block";

  const form = document.getElementById("whatif-form");
  if (form) {
    form.elements["Contract"].value = inputPayload["Contract"] || "Month-to-month";
    form.elements["TechSupport"].value = inputPayload["TechSupport"] || "No";
    form.elements["OnlineSecurity"].value = inputPayload["OnlineSecurity"] || "No";
    form.elements["PaymentMethod"].value = inputPayload["PaymentMethod"] || "Electronic check";
  }

  document.getElementById("sim-baseline-val").innerText = `${baselineProb}%`;
  document.getElementById("sim-scenario-val").innerText = `${baselineProb}%`;
  document.getElementById("sim-delta-badge").innerText = "0.0% Delta";
  document.getElementById("sim-delta-badge").className = "delta-badge";
}

async function runWhatIfSimulation() {
  if (!currentPredictionState) return;
  const form = document.getElementById("whatif-form");
  const baseline = currentPredictionState.raw_input;

  const scenario = { ...baseline };
  scenario["Contract"] = form.elements["Contract"].value;
  scenario["TechSupport"] = form.elements["TechSupport"].value;
  scenario["OnlineSecurity"] = form.elements["OnlineSecurity"].value;
  scenario["PaymentMethod"] = form.elements["PaymentMethod"].value;

  try {
    const res = await fetch("/api/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baseline, scenario })
    });

    const data = await res.json();
    if (!data.success) return;

    document.getElementById("sim-scenario-val").innerText = `${data.scenario_prob}%`;
    const deltaBadge = document.getElementById("sim-delta-badge");

    if (data.delta_prob < 0) {
      deltaBadge.innerText = `${data.delta_prob}% (${data.percent_risk_reduction}% Risk Mitigated!)`;
      deltaBadge.className = "delta-badge delta-positive";
    } else if (data.delta_prob > 0) {
      deltaBadge.innerText = `+${data.delta_prob}% Risk Increase`;
      deltaBadge.className = "delta-badge delta-negative";
    } else {
      deltaBadge.innerText = "0.0% Change";
      deltaBadge.className = "delta-badge";
    }

  } catch (err) {
    console.error("Simulation error:", err);
  }
}

// Save Prediction to DB
async function handleSavePrediction() {
  if (!currentPredictionState) return;
  const notes = document.getElementById("customer-notes-input").value;

  try {
    const res = await fetch("/api/save-prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer_id: currentPredictionState.customer_id,
        churn_prob: currentPredictionState.churn_prob,
        risk_level: currentPredictionState.risk_meta.level,
        notes: notes,
        raw_input: currentPredictionState.raw_input
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast("Saved customer audit to SQLite database!", "success");
    } else {
      showToast("Failed to save to database.", "danger");
    }
  } catch (err) {
    console.error(err);
    showToast("Error saving to database.", "danger");
  }
}
