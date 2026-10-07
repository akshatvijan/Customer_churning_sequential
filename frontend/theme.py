import gradio as gr

CUSTOM_CSS = """
/* Enterprise Theme Styles for Churn Intelligence Platform */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.main-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}

.main-subtitle {
    font-size: 1.05rem !important;
    color: #94a3b8 !important;
    margin-top: 8px !important;
    line-height: 1.5;
}

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}

.risk-badge-high {
    background-color: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
}

.risk-badge-medium {
    background-color: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
}

.risk-badge-low {
    background-color: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    display: inline-block;
}

.action-card {
    border-left: 4px solid #3b82f6;
    background: rgba(30, 41, 59, 0.7);
    padding: 16px;
    border-radius: 0 8px 8px 0;
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: #38bdf8;
}
"""

def get_theme():
    """Create curated Gradio theme."""
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate"
    )
