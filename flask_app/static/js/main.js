/**
 * ChurnShield AI - Production Client-Side Engine
 * Stripe-Inspired UI, Dark Mode Toggle, Chart Theming,
 * Real-Time AJAX Prediction, What-If Simulator, and Persistence.
 */

// -------------------------------------------------------------
// 1. Theme Manager (Light / Dark Mode with Persistence)
// -------------------------------------------------------------
function getSavedTheme() {
  const saved = localStorage.getItem("churnshield_theme");
  if (saved) return saved;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("churnshield_theme", theme);
  
  const toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    toggleBtn.innerHTML = theme === "dark" 
      ? '<i class="fa-solid fa-sun"></i>' 
      : '<i class="fa-solid fa-moon"></i>';
    toggleBtn.setAttribute("title", `Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`);
  }

  // Notify any active Chart.js instances to refresh grid/tick colors
  if (window.Chart && Chart.instances) {
    const isDark = theme === "dark";
    const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
    const textColor = isDark ? "#94a3b8" : "#64748b";

    Object.values(Chart.instances).forEach(chart => {
      if (chart.options.scales) {
        Object.values(chart.options.scales).forEach(scale => {
          if (scale.grid) scale.grid.color = gridColor;
          if (scale.ticks) scale.ticks.color = textColor;
          if (scale.title) scale.title.color = textColor;
        });
      }
      if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
        chart.options.plugins.legend.labels.color = isDark ? "#f8fafc" : "#0f172a";
      }
      chart.update();
    });
  }
}

// -------------------------------------------------------------
// 2. DOM Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", function () {
  // Initialize Theme
  const currentTheme = getSavedTheme();
  applyTheme(currentTheme);

  const toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", function () {
      const active = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(active);
    });
  }

  // Mobile Menu Toggle
  const mobileBtn = document.getElementById("mobile-menu-btn");
  const sidebar = document.querySelector(".sidebar");
  if (mobileBtn && sidebar) {
    mobileBtn.addEventListener("click", function () {
      sidebar.classList.toggle("show");
    });
    // Close sidebar on click outside
    document.addEventListener("click", function (e) {
      if (!sidebar.contains(e.target) && !mobileBtn.contains(e.target) && sidebar.classList.contains("show")) {
        sidebar.classList.remove("show");
      }
    });
  }

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
      const targetElem = document.getElementById(targetId);
      if (targetElem) {
        const text = targetElem.innerText;
        navigator.clipboard.writeText(text).then(() => {
          showToast("Copied template to clipboard!", "success");
        });
      }
    });
  });
});

// -------------------------------------------------------------
// 3. Toast Notifications (Stripe Style)
// -------------------------------------------------------------
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
    container.style.gap = "8px";
    container.style.maxWidth = "380px";
    document.body.appendChild(container);
  }
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const bg = isDark ? "#1f2937" : "#ffffff";
  const text = isDark ? "#f9fafb" : "#0f172a";
  const border = isDark ? "#374151" : "#e2e8f0";

  let iconHtml = '<i class="fa-solid fa-circle-info" style="color: #635bff;"></i>';
  if (type === "success") iconHtml = '<i class="fa-solid fa-circle-check" style="color: #10b981;"></i>';
  if (type === "danger") iconHtml = '<i class="fa-solid fa-triangle-exclamation" style="color: #ef4444;"></i>';

  toast.style.background = bg;
  toast.style.color = text;
  toast.style.border = `1px solid ${border}`;
  toast.style.padding = "10px 16px";
  toast.style.borderRadius = "8px";
  toast.style.fontSize = "0.85rem";
  toast.style.fontWeight = "500";
  toast.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.08)";
  toast.style.display = "flex";
  toast.style.alignItems = "center";
  toast.style.gap = "10px";
  toast.style.transition = "all 0.25s ease";
  toast.style.fontFamily = "'Inter', sans-serif";
  
  toast.innerHTML = `${iconHtml}<span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(6px)";
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}

// -------------------------------------------------------------
// 4. Presets
// -------------------------------------------------------------
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
  if (!form) return;
  for (const [key, value] of Object.entries(data)) {
    const field = form.elements[key];
    if (field) {
      field.value = value;
    }
  }
  showToast(`Loaded ${name.replace('_', ' ').toUpperCase()} customer preset`, "info");
}

// -------------------------------------------------------------
// 5. AJAX Prediction Handling
// -------------------------------------------------------------
let currentPredictionState = null;

async function handlePredictSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  const jsonPayload = {};
  formData.forEach((val, key) => jsonPayload[key] = val);

  const submitBtn = form.querySelector("button[type='submit']");
  const origText = submitBtn.innerHTML;
  submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Computing SHAP Risk Score...</span>`;
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
      showToast(result.error || "Prediction computation failed.", "danger");
      return;
    }

    renderPredictionResults(result, jsonPayload);
    showToast("Ensemble prediction computed successfully", "success");

  } catch (err) {
    submitBtn.innerHTML = origText;
    submitBtn.disabled = false;
    showToast("Network error executing prediction.", "danger");
    console.error(err);
  }
}

function renderPredictionResults(res, inputPayload) {
  currentPredictionState = { ...res, raw_input: inputPayload };

  const emptyState = document.getElementById("empty-state-notice");
  if (emptyState) emptyState.style.display = "none";

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
  if (gaugeVal) gaugeVal.innerText = `${prob}%`;
  
  if (riskBadge) {
    riskBadge.innerText = `${res.risk_meta.level} RISK`;
    riskBadge.className = `badge ${res.risk_meta.badge_class}`;
  }
  if (riskUrgency) {
    riskUrgency.innerText = res.risk_meta.urgency;
  }

  // Stripe styled Conic fill (clean, no harsh neon glow)
  let gaugeColor = "#10b981"; // emerald
  if (res.risk_meta.level === "HIGH") gaugeColor = "#ef4444"; // rose
  else if (res.risk_meta.level === "MEDIUM") gaugeColor = "#f59e0b"; // amber

  if (gaugeCircle) {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const trackColor = isDark ? "#1e293b" : "#e2e8f0";
    gaugeCircle.style.background = `conic-gradient(${gaugeColor} 0% ${prob}%, ${trackColor} ${prob}% 100%)`;
  }

  // Financial Metrics
  const elAnnual = document.getElementById("disp-annual-rev");
  if (elAnnual) elAnnual.innerText = `$${res.clv_roi.annual_revenue.toLocaleString()}`;
  
  const elRisk = document.getElementById("disp-rev-risk");
  if (elRisk) elRisk.innerText = `$${res.clv_roi.revenue_at_risk.toLocaleString()}`;
  
  const elClv = document.getElementById("disp-projected-clv");
  if (elClv) elClv.innerText = `$${res.clv_roi.projected_clv.toLocaleString()}`;
  
  const elRoi = document.getElementById("disp-roi");
  if (elRoi) elRoi.innerText = `+${res.clv_roi.retention_roi_percent}% ROI`;

  // Render SHAP Feature Impact Bars
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
    const barColor = isRisk ? "#ef4444" : "#10b981";
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
      card.style.background = "var(--bg-subtle)";
      card.style.border = "1px solid var(--border-color)";
      card.style.borderLeft = "3px solid var(--primary)";
      card.style.padding = "12px 14px";
      card.style.borderRadius = "8px";
      card.style.marginBottom = "8px";
      card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
          <strong style="color: var(--text-primary); font-size: 0.88rem;">${st.title}</strong>
          <span class="badge badge-success" style="font-size: 0.68rem;">${st.tag}</span>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.82rem; margin: 0;">${st.detail}</p>
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
      li.style.marginBottom = "6px";
      li.style.fontSize = "0.82rem";
      li.style.color = "var(--text-secondary)";
      li.innerHTML = `<strong style="color: var(--primary);">${t.day}:</strong> ${t.action}`;
      timeContainer.appendChild(li);
    });
  }

  // Update Templates
  const emailElem = document.getElementById("template-email-body");
  if (emailElem) emailElem.innerText = plan.email_template;

  const smsElem = document.getElementById("template-sms-body");
  if (smsElem) smsElem.innerText = plan.sms_template;
}

// -------------------------------------------------------------
// 6. Interactive What-If Simulator
// -------------------------------------------------------------
function syncWhatIfBaseline(inputPayload, baselineProb) {
  const whatIfBox = document.getElementById("whatif-panel");
  if (whatIfBox) whatIfBox.style.display = "block";

  const form = document.getElementById("whatif-form");
  if (form) {
    if (form.elements["Contract"]) form.elements["Contract"].value = inputPayload["Contract"] || "Month-to-month";
    if (form.elements["TechSupport"]) form.elements["TechSupport"].value = inputPayload["TechSupport"] || "No";
    if (form.elements["OnlineSecurity"]) form.elements["OnlineSecurity"].value = inputPayload["OnlineSecurity"] || "No";
    if (form.elements["PaymentMethod"]) form.elements["PaymentMethod"].value = inputPayload["PaymentMethod"] || "Electronic check";
  }

  const baseVal = document.getElementById("sim-baseline-val");
  if (baseVal) baseVal.innerText = `${baselineProb}%`;

  const scenVal = document.getElementById("sim-scenario-val");
  if (scenVal) scenVal.innerText = `${baselineProb}%`;

  const badge = document.getElementById("sim-delta-badge");
  if (badge) {
    badge.innerText = "0.0% Delta";
    badge.className = "delta-badge";
  }
}

async function runWhatIfSimulation() {
  if (!currentPredictionState) return;
  const form = document.getElementById("whatif-form");
  if (!form) return;
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

    const scenVal = document.getElementById("sim-scenario-val");
    if (scenVal) scenVal.innerText = `${data.scenario_prob}%`;
    
    const deltaBadge = document.getElementById("sim-delta-badge");
    if (deltaBadge) {
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
    }

  } catch (err) {
    console.error("Simulation error:", err);
  }
}

// -------------------------------------------------------------
// 7. Save Prediction to DB
// -------------------------------------------------------------
async function handleSavePrediction() {
  if (!currentPredictionState) return;
  const notesElem = document.getElementById("customer-notes-input");
  const notes = notesElem ? notesElem.value : "";

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
      showToast("Saved customer audit to SQLite persistence store", "success");
    } else {
      showToast("Failed to save record to database.", "danger");
    }
  } catch (err) {
    console.error(err);
    showToast("Error saving to database.", "danger");
  }
}
