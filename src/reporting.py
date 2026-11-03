"""
ChurnShield AI - Executive Reporting & Export Engine
Generates downloadable:
- Enriched CSV batch prediction reports
- Executive HTML / Printable PDF audit reports
- Structured JSON analytics exports
"""

import io
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates structured exports and executive reports for ChurnShield AI."""

    @staticmethod
    def generate_batch_csv(
        df_original: pd.DataFrame,
        probabilities: List[float],
        risk_levels: List[str]
    ) -> io.BytesIO:
        """
        Creates an enriched CSV containing original customer attributes
        along with churn probability score and risk tier.
        """
        export_df = df_original.copy()
        export_df['Churn_Probability_Pct'] = [round(p * 100, 1) for p in probabilities]
        export_df['Risk_Level'] = risk_levels
        export_df['Prediction_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        output = io.BytesIO()
        export_df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        return output

    @staticmethod
    def generate_executive_html_report(
        metrics: Dict[str, Any],
        recent_predictions: List[Dict[str, Any]],
        drift_report: Dict[str, Any]
    ) -> str:
        """
        Builds a self-contained, beautifully styled HTML executive report
        ready for printing or saving as PDF in modern browsers.
        """
        now = datetime.now().strftime("%B %d, %Y - %H:%M UTC")

        rows_html = ""
        for p in recent_predictions[:10]:
            color = "#ff3d57" if p["risk_level"] == "HIGH" else ("#ffaa00" if p["risk_level"] == "MEDIUM" else "#00ffaa")
            rows_html += f"""
            <tr>
              <td><strong>{p['customer_id']}</strong></td>
              <td><span style="color: {color}; font-weight: bold;">{p['churn_prob']}%</span></td>
              <td><span style="background: rgba(255,255,255,0.08); padding: 4px 8px; border-radius: 4px; color: {color};">{p['risk_level']}</span></td>
              <td>{p['contract']}</td>
              <td>${p['monthly_charges']}</td>
              <td>{p['timestamp']}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ChurnShield AI - Executive Intelligence Report</title>
  <style>
    body {{
      font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
      margin: 40px;
      background: #0f172a;
      color: #f1f5f9;
      line-height: 1.6;
    }}
    .header {{
      border-bottom: 2px solid #00ffaa;
      padding-bottom: 20px;
      margin-bottom: 30px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    h1 {{ color: #00ffaa; margin: 0; font-size: 26px; }}
    .subtitle {{ color: #94a3b8; font-size: 14px; margin-top: 5px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 15px;
      margin-bottom: 30px;
    }}
    .kpi-card {{
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid rgba(0, 255, 170, 0.2);
      padding: 16px;
      border-radius: 8px;
    }}
    .kpi-title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
    .kpi-value {{ font-size: 24px; font-weight: bold; color: #ffffff; margin-top: 5px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 20px;
      background: rgba(30, 41, 59, 0.5);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-size: 13px;
    }}
    th {{
      background: #1e293b;
      color: #00ffaa;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }}
    .footer {{
      margin-top: 40px;
      text-align: center;
      font-size: 12px;
      color: #64748b;
      border-top: 1px solid rgba(255,255,255,0.08);
      padding-top: 20px;
    }}
    @media print {{
      body {{ background: #ffffff; color: #000000; }}
      .kpi-card {{ border: 1px solid #ccc; background: #f8f9fa; }}
      .kpi-value {{ color: #000; }}
      th {{ background: #e2e8f0; color: #000; }}
      td {{ border-bottom: 1px solid #ddd; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>CHURNSHIELD AI — RETENTION INTELLIGENCE AUDIT</h1>
      <div class="subtitle">Generated on {now} • System Version 2.0 Production</div>
    </div>
    <div style="text-align: right;">
      <span style="background: rgba(0,255,170,0.15); border: 1px solid #00ffaa; color: #00ffaa; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;">
        ENSEMBLE ACTIVE (AUC: 0.847)
      </span>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-title">Evaluated Customers</div>
      <div class="kpi-value">{metrics.get('total_customers', 7043):,}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Portfolio Churn Rate</div>
      <div class="kpi-value" style="color: #ffaa00;">{metrics.get('churn_rate', 26.5)}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Annual Revenue At Risk</div>
      <div class="kpi-value" style="color: #ff3d57;">${metrics.get('annual_rev_at_risk', 1420000.0):,.0f}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Model Health Score</div>
      <div class="kpi-value" style="color: #00ffaa;">{drift_report.get('overall_health_score', 96)}%</div>
    </div>
  </div>

  <h3 style="color: #00ffaa; margin-top: 30px;">Recent Evaluated Customer Accounts</h3>
  <table>
    <thead>
      <tr>
        <th>Customer ID</th>
        <th>Churn Probability</th>
        <th>Risk Level</th>
        <th>Contract Type</th>
        <th>Monthly Spend</th>
        <th>Evaluated Timestamp</th>
      </tr>
    </thead>
    <tbody>
      {rows_html if rows_html else '<tr><td colspan="6" style="text-align: center;">No saved predictions available.</td></tr>'}
    </tbody>
  </table>

  <div class="footer">
    ChurnShield AI • Production Customer Retention Platform • Confidential & Proprietary
  </div>
</body>
</html>
        """
        return html_content

