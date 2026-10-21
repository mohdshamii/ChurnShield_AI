"""
ChurnShield AI - Actionable Retention & Financial ROI Engine
Calculates risk categorization, Customer Lifetime Value (CLV) preservation,
cost-benefit retention ROI, and generates customized intervention playbooks.
"""

from typing import Dict, Any, List
from src import config


class RetentionStrategyEngine:
    """Generates financial models, risk assessments, and targeted retention strategies."""

    @staticmethod
    def assess_risk(churn_prob: float) -> Dict[str, Any]:
        """Categorizes churn probability into risk tiers with styling attributes."""
        if churn_prob >= config.RISK_THRESHOLD_HIGH:
            return {
                "level": "HIGH",
                "badge_class": "badge-danger",
                "color": "#ff3d57",
                "glow_color": "rgba(255, 61, 87, 0.4)",
                "urgency": "Immediate Critical Action Required (<24 Hours)",
                "action_header": "High-Risk Customer: Swift Retention Protocol Activated"
            }
        elif churn_prob >= config.RISK_THRESHOLD_MEDIUM:
            return {
                "level": "MEDIUM",
                "badge_class": "badge-warning",
                "color": "#ffaa00",
                "glow_color": "rgba(255, 170, 0, 0.4)",
                "urgency": "Proactive Intervention Recommended (<7 Days)",
                "action_header": "Medium-Risk Customer: Proactive Value Nurture"
            }
        else:
            return {
                "level": "LOW",
                "badge_class": "badge-success",
                "color": "#00ffaa",
                "glow_color": "rgba(0, 255, 170, 0.4)",
                "urgency": "Standard Monitoring & Upsell Opportunities",
                "action_header": "Low-Risk Customer: Loyalty & Relationship Expansion"
            }

    @staticmethod
    def calculate_clv_and_roi(
        monthly_charges: float,
        tenure_months: float,
        churn_prob: float,
        intervention_cost: float = 60.0,
        expected_success_rate: float = config.INTERVENTION_SUCCESS_RATE
    ) -> Dict[str, Any]:
        """
        Computes financial metrics including Customer Lifetime Value (CLV),
        Annualized Revenue At Risk, and Retention Campaign Return on Investment (ROI).
        """
        # Annualized Revenue
        annual_revenue = monthly_charges * 12.0
        revenue_at_risk = annual_revenue * churn_prob

        # Estimated remaining tenure based on inverse hazard / current tenure
        baseline_tenure = max(tenure_months, 6.0)
        estimated_future_months = max(3.0, (1.0 - churn_prob) * 36.0 + 6.0)
        projected_clv = monthly_charges * estimated_future_months

        # Financial value saved by retention intervention
        gross_value_saved = revenue_at_risk * expected_success_rate
        net_value_preserved = gross_value_saved - intervention_cost
        roi_percentage = (net_value_preserved / max(intervention_cost, 1.0)) * 100.0

        return {
            "annual_revenue": round(annual_revenue, 2),
            "revenue_at_risk": round(revenue_at_risk, 2),
            "projected_clv": round(projected_clv, 2),
            "intervention_cost": round(intervention_cost, 2),
            "gross_value_saved": round(gross_value_saved, 2),
            "net_value_preserved": round(net_value_preserved, 2),
            "retention_roi_percent": round(roi_percentage, 1)
        }

    @staticmethod
    def generate_recommendations(
        customer_data: Dict[str, Any],
        churn_prob: float,
        risk_catalysts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes customer profile, probability, and risk drivers into
        a tailored retention playbook and communication templates.
        """
        risk_meta = RetentionStrategyEngine.assess_risk(churn_prob)
        monthly_charges = float(customer_data.get('MonthlyCharges', customer_data.get('monthly_charges', 70.0)))
        contract = str(customer_data.get('Contract', customer_data.get('contract', 'Month-to-month')))
        payment = str(customer_data.get('PaymentMethod', customer_data.get('payment_method', 'Electronic check')))
        tech_support = str(customer_data.get('TechSupport', customer_data.get('tech_support', 'No')))
        online_security = str(customer_data.get('OnlineSecurity', customer_data.get('online_security', 'No')))

        strategies: List[Dict[str, str]] = []

        # 1. Contract Strategy
        if "month-to-month" in contract.lower():
            discount_amount = round(monthly_charges * 0.15, 2)
            strategies.append({
                "title": "Annual Commitment Incentive",
                "tag": "High Impact",
                "detail": f"Offer a 15% discount (${discount_amount}/mo savings) in exchange for transitioning to a 1-year agreement."
            })
        elif "one year" in contract.lower() and churn_prob > 0.4:
            strategies.append({
                "title": "Early Renewal Loyalty Lock",
                "tag": "Contract Perk",
                "detail": "Provide a 2-year renewal lock guaranteeing zero price increases with free speed upgrade."
            })

        # 2. Payment Method Strategy
        if "electronic check" in payment.lower():
            strategies.append({
                "title": "Automated Billing Upgrade Perk",
                "tag": "Friction Reduction",
                "detail": "Credit $25 to customer account upon setting up Automatic Credit Card or Bank Transfer billing."
            })

        # 3. Value-Added Service Bundling
        if tech_support.lower() != "yes" or online_security.lower() != "yes":
            strategies.append({
                "title": "Complimentary Shield Protection Bundle",
                "tag": "Service Stickiness",
                "detail": "Bundle Premium Tech Support & Online Security free for 90 days to enhance product utility."
            })

        # 4. Senior / High-tier Care
        if monthly_charges > 85.0:
            strategies.append({
                "title": "VIP Executive Support Routing",
                "tag": "Relationship",
                "detail": "Assign direct escalation route to a dedicated retention specialist with zero queue times."
            })

        # Fallback general strategy if list is short
        if len(strategies) < 3:
            strategies.append({
                "title": "Quarterly Value Realization Review",
                "tag": "Proactive Care",
                "detail": "Schedule a 10-minute proactive check-in call to demonstrate utilization savings and resolve minor grievances."
            })

        # 14-Day Timeline
        if risk_meta["level"] == "HIGH":
            timeline = [
                {"day": "Day 1 (Within 4h)", "action": "Dedicated account manager outreach via priority phone/SMS."},
                {"day": "Day 3", "action": "Personalized 15% discount contract lock delivered via email."},
                {"day": "Day 7", "action": "Follow-up consultation to verify service satisfaction & resolve friction points."},
                {"day": "Day 14", "action": "Finalize 1-year contract extension and credit loyalty rewards to account."}
            ]
        elif risk_meta["level"] == "MEDIUM":
            timeline = [
                {"day": "Day 2", "action": "Automated satisfaction survey with instant $10 bill credit upon completion."},
                {"day": "Day 5", "action": "Educational email highlighting underutilized bundle features (Security & Cloud)."},
                {"day": "Day 10", "action": "Special offer: upgrade to annual plan with first 2 months at 20% off."},
                {"day": "Day 14", "action": "Retention representative check-in call if offer hasn't been claimed."}
            ]
        else:
            timeline = [
                {"day": "Month 1", "action": "Customer appreciation note with complimentary speed or feature boost."},
                {"day": "Quarter 1", "action": "Annual usage report showing total savings and uptime reliability."},
                {"day": "Quarter 2", "action": "Refer-a-friend incentive: $50 credit for both referrer and referee."}
            ]

        # Communication Templates
        email_template = f"""Subject: Mohd Shami, a special thank you from your ChurnShield VIP Team

Dear Valued Customer,

We noticed you've been with us and we want to ensure you are getting the absolute best experience from your connection.

Because you are a priority member of our network, we have pre-approved an exclusive loyalty benefit for your account:
- 15% Monthly Savings on your plan
- Free 3-Month Premium Tech Support & Cyber Shield
- Zero price increases guaranteed

To activate your preferred rate, simply reply to this email or speak with your dedicated care specialist at 1-800-CHURNSHIELD.

Warm regards,
ChurnShield Retention & Customer Success Team"""

        sms_template = f"Hi from ChurnShield! As a thank you for your loyalty, we've applied a $25 credit toward your next bill. Tap here to lock in your discounted rate: https://churnshield.ai/perks"

        return {
            "risk_meta": risk_meta,
            "strategies": strategies,
            "timeline": timeline,
            "email_template": email_template,
            "sms_template": sms_template
        }
