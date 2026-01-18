import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List


def create_radar_chart(features: Dict[str, Any]) -> go.Figure:
    """Create radar chart for features"""

    # Normalize features for radar
    radar_data = {
        "Density": min(features.get("vessel_density", 0) * 10, 1.0),
        "Thin Ratio": min(features.get("thin_ratio", 0) * 2, 1.0),
        "Thick Ratio": min(features.get("thick_ratio", 0) * 3, 1.0),
        "Branching": min(features.get("branching_points", 0) / 500, 1.0),
        "Tortuosity": min(features.get("tortuosity", 1) / 3, 1.0),
        "Complexity": min(features.get("fractal_dimension", 1) / 2, 1.0)
    }

    categories = list(radar_data.keys())
    values = list(radar_data.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(13, 148, 136, 0.3)',
        line_color='rgb(13, 148, 136)',
        line_width=2
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=400
    )

    return fig


def create_width_histogram(widths: List[float]) -> go.Figure:
    """Create histogram of vessel widths"""

    if not widths:
        fig = go.Figure()
        fig.update_layout(title="No width data available", height=300)
        return fig

    fig = go.Figure(data=[
        go.Histogram(
            x=widths,
            nbinsx=20,
            marker_color='#0d9488',
            opacity=0.7
        )
    ])

    mean_width = np.mean(widths)
    fig.add_vline(
        x=mean_width,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_width:.2f}"
    )

    fig.update_layout(
        title="Vessel Width Distribution",
        xaxis_title="Width (pixels)",
        yaxis_title="Frequency",
        height=300
    )

    return fig