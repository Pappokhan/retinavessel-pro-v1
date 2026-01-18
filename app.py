import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from config import Config
from src.model.inference import ModelInference
from src.model.registry import ModelRegistry
from src.features.extractor import FeatureExtractor
from src.features.clinical import ClinicalInterpreter
from src.utils.image_utils import ImageProcessor, create_overlay
from src.utils.validation import validate_image
from src.utils.visualization import create_radar_chart, create_width_histogram
from src.mlops.monitor import PerformanceMonitor
from src.mlops.tracker import ExperimentTracker

# Set page configuration
st.set_page_config(
    page_title="RetinaVessel Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load custom CSS
def load_css():
    css_path = Path(__file__).parent / "static" / "css" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = Config()

    if 'model' not in st.session_state:
        try:
            st.session_state.model = ModelInference(st.session_state.config)
            st.session_state.model_loaded = True
        except Exception as e:
            st.session_state.model_loaded = False
            st.session_state.model_error = str(e)

    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = FeatureExtractor(st.session_state.config)

    if 'clinical_interpreter' not in st.session_state:
        st.session_state.clinical_interpreter = ClinicalInterpreter(st.session_state.config)

    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor(st.session_state.config)

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()

    if 'experiment_tracker' not in st.session_state:
        st.session_state.experiment_tracker = ExperimentTracker()


# Main application
def main():
    # Initialize
    init_session_state()
    load_css()

    # App header
    st.markdown("""
    <div class="header">
        <h1>👁️ RetinaVessel Pro</h1>
        <p class="subtitle">AI-Powered Retinal Vessel Analysis System</p>
        <p class="tagline">Research-Grade Quantitative Morphology Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Analyze",
        "📊 Results Dashboard",
        "🔬 Advanced Analytics",
        "📈 MLops Monitor"
    ])

    with tab1:
        render_upload_analysis()

    with tab2:
        if st.session_state.analysis_results:
            render_results_dashboard()
        else:
            st.info("👈 Upload an image to begin analysis")

    with tab3:
        if st.session_state.analysis_results:
            render_advanced_analytics()
        else:
            st.info("👈 Upload an image to view advanced analytics")

    with tab4:
        render_mlops_monitor()


def render_sidebar():
    """Render sidebar content"""
    st.markdown("### ⚙️ Analysis Settings")

    # Threshold slider
    threshold = st.slider(
        "Segmentation Threshold",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.config.THRESHOLD,
        step=0.05,
        help="Adjust sensitivity for vessel detection"
    )

    st.session_state.config.THRESHOLD = threshold

    st.markdown("---")
    st.markdown("### 📊 Analysis Options")

    st.session_state.enable_features = st.checkbox("Extract Vessel Features", value=True)
    st.session_state.enable_clinical = st.checkbox("Clinical Interpretation", value=True)
    st.session_state.enable_mlops = st.checkbox("Enable MLops Tracking", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ System Information")

    col1, col2 = st.columns(2)
    with col1:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device)
    with col2:
        st.metric("Model", "Attention U-Net")

    # Model status
    if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not loaded")
        if hasattr(st.session_state, 'model_error'):
            st.error(f"Error: {st.session_state.model_error}")

    st.markdown("---")

    # Performance metrics
    if hasattr(st.session_state, 'performance_monitor'):
        metrics = st.session_state.performance_monitor.get_summary()
        st.metric("Total Analyses", metrics.get('total_analyses', 0))
        st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.1%}")

    st.markdown("---")
    st.warning("""
    ⚠️ **RESEARCH USE ONLY**

    This tool is for research purposes only.
    Not for clinical diagnosis.
    All results should be validated by medical professionals.
    """)


def render_upload_analysis():
    """Render upload and analysis section"""
    st.markdown("### 📤 Upload Retinal Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=st.session_state.config.ALLOWED_EXTENSIONS,
        label_visibility="collapsed",
        help=f"Supported formats: {', '.join(st.session_state.config.ALLOWED_EXTENSIONS).upper()}"
    )

    if uploaded_file:
        try:
            # Validate image
            is_valid, message = validate_image(
                uploaded_file,
                st.session_state.config.ALLOWED_EXTENSIONS,
                st.session_state.config.MAX_UPLOAD_SIZE_MB
            )

            if not is_valid:
                st.error(f"❌ {message}")
                return

            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                # Image info
                st.markdown("#### 📋 Image Details")
                st.metric("Resolution", f"{image.width} × {image.height}")
                st.metric("Format", uploaded_file.type.split("/")[-1].upper())
                st.metric("Mode", image.mode)

                # Quality assessment
                quality = st.session_state.image_processor.assess_quality(image)
                quality_level = quality.get('quality', 'Unknown')
                quality_color = {
                    'Excellent': 'green',
                    'Good': 'blue',
                    'Fair': 'orange',
                    'Poor': 'red'
                }.get(quality_level, 'gray')

                st.markdown(f"**Quality:** <span style='color:{quality_color}'>{quality_level}</span>",
                            unsafe_allow_html=True)

            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing image..."):
                    try:
                        # Run analysis
                        results = perform_analysis(image, uploaded_file.name)

                        # Store results
                        st.session_state.analysis_results = results

                        # Track in MLops
                        if st.session_state.enable_mlops:
                            track_analysis(results)

                        st.success(f"✅ Analysis complete! Confidence: {results['confidence']:.1%}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

    else:
        # Show upload prompt
        st.markdown("""
        <div class="upload-prompt">
            <div style="text-align: center; padding: 4rem; border: 3px dashed #0d9488; border-radius: 15px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);">
                <div style="font-size: 4rem; margin-bottom: 1rem; color: #0d9488;">📤</div>
                <p style="font-size: 1.5rem; margin-bottom: 1rem; color: #0d9488; font-weight: 600;">Upload Retinal Image</p>
                <p style="color: #475569; margin-bottom: 2rem; font-size: 1.1rem;">
                    Drag and drop or click to browse files
                </p>
                <div style="background: white; padding: 1.5rem; border-radius: 10px; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 400px;">
                    <p style="margin: 0 0 1rem 0; font-weight: 600; color: #1e293b; font-size: 1.1rem;">📋 For best results:</p>
                    <ul style="text-align: left; margin: 0; padding-left: 1.5rem; color: #475569;">
                        <li>Use macula- or disc-centered images</li>
                        <li>Ensure good focus and illumination</li>
                        <li>Minimum resolution: 512×512 pixels</li>
                        <li>Clear optic disc and macula visibility</li>
                        <li>Red-free or color fundus images</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Features showcase
        st.markdown("---")
        st.markdown("### 🎯 What You Get")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🔍</div>
                <h4 style="margin: 0 0 0.5rem 0;">AI Segmentation</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Precise vessel detection using Attention U-Net</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                <h4 style="margin: 0 0 0.5rem 0;">15+ Metrics</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Comprehensive morphological analysis</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🩺</div>
                <h4 style="margin: 0 0 0.5rem 0;">Clinical Insights</h4>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Research-grade interpretation</p>
            </div>
            """, unsafe_allow_html=True)


def perform_analysis(image, filename):
    """Perform complete analysis on image"""
    # Run model inference
    mask, confidence = st.session_state.model.predict(image, st.session_state.config.THRESHOLD)

    # Prepare results
    results = {
        "filename": filename,
        "original_image": image,
        "binary_mask": mask,
        "confidence": confidence,
        "threshold": st.session_state.config.THRESHOLD,
        "timestamp": datetime.now().isoformat(),
        "image_size": image.size,
        "analysis_id": f"analysis_{int(datetime.now().timestamp())}"
    }

    # Extract features if enabled
    if st.session_state.enable_features:
        features = st.session_state.feature_extractor.extract(mask)
        results["features"] = features

        # Clinical interpretation if enabled
        if st.session_state.enable_clinical:
            clinical = st.session_state.clinical_interpreter.interpret(features)
            results["clinical"] = clinical

    # Update performance monitor
    st.session_state.performance_monitor.record_analysis(
        analysis_id=results["analysis_id"],
        confidence=confidence,
        features=results.get("features", {}),
        timestamp=results["timestamp"]
    )

    return results


def track_analysis(results):
    """Track analysis in MLops systems"""
    # Track in experiment tracker
    st.session_state.experiment_tracker.log_experiment(
        experiment_name="retinal_analysis",
        parameters={
            "threshold": results["threshold"],
            "image_size": results["image_size"]
        },
        metrics={
            "confidence": results["confidence"],
            "vessel_density": results.get("features", {}).get("vessel_density", 0)
        },
        tags=["production", "inference"],
        timestamp=results["timestamp"]
    )

    # Log to file for monitoring
    log_entry = {
        "analysis_id": results["analysis_id"],
        "timestamp": results["timestamp"],
        "confidence": results["confidence"],
        "filename": results["filename"]
    }

    log_file = Path("data/outputs/analysis_log.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def render_results_dashboard():
    """Render results dashboard"""
    results = st.session_state.analysis_results
    features = results.get("features", {})
    clinical = results.get("clinical", {})

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Vessel Density",
            value=f"{features.get('vessel_density', 0):.3f}",
            delta="Normal" if 0.08 < features.get('vessel_density', 0) < 0.15 else "Check",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Mean Width",
            value=f"{features.get('mean_width', 0):.2f} px",
            delta="Normal" if 2.0 < features.get('mean_width', 0) < 3.5 else "Note"
        )

    with col3:
        st.metric(
            label="Tortuosity",
            value=f"{features.get('tortuosity', 1):.2f}",
            delta="Normal" if features.get('tortuosity', 1) < 1.8 else "Elevated"
        )

    with col4:
        confidence_color = "green" if results["confidence"] > 0.8 else "orange" if results[
                                                                                       "confidence"] > 0.6 else "red"
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; font-weight: 600; color: #475569;">Confidence</div>
            <div style="font-size: 2rem; font-weight: 700; color: {confidence_color};">
                {results['confidence']:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visual results
    st.markdown("### 🖼️ Visual Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(results["original_image"], caption="Original Image", use_container_width=True)

    with col2:
        mask_img = Image.fromarray((results["binary_mask"] * 255).astype(np.uint8))
        st.image(mask_img, caption=f"Vessel Segmentation", use_container_width=True)

    with col3:
        overlay = create_overlay(
            results["original_image"],
            results["binary_mask"],
            color=st.session_state.config.DEFAULT_COLOR,
            alpha=st.session_state.config.OVERLAY_ALPHA
        )
        st.image(overlay, caption="Vessel Overlay", use_container_width=True)

    st.markdown("---")

    # Features display
    if features:
        st.markdown("### 📊 Quantitative Features")

        # Create tabs for different feature categories
        tab1, tab2, tab3 = st.tabs(["📈 Density & Width", "🌀 Morphology", "🗺️ Regional"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Density Metrics")
                st.metric("Overall Density", f"{features.get('vessel_density', 0):.4f}")
                st.metric("Thin Vessel Ratio", f"{features.get('thin_ratio', 0):.2%}")
                st.metric("Thick Vessel Ratio", f"{features.get('thick_ratio', 0):.2%}")

            with col2:
                st.markdown("#### Width Metrics")
                st.metric("Mean Width", f"{features.get('mean_width', 0):.2f} px")
                st.metric("Std Width", f"{features.get('std_width', 0):.2f} px")
                st.metric("Median Width", f"{features.get('median_width', 0):.2f} px")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Complexity Metrics")
                st.metric("Tortuosity", f"{features.get('tortuosity', 1):.3f}")
                st.metric("Fractal Dimension", f"{features.get('fractal_dimension', 1):.3f}")
                st.metric("Branching Points", features.get('branching_points', 0))

            with col2:
                st.markdown("#### Network Metrics")
                st.metric("Vessel Length", f"{features.get('vessel_length', 0):.0f}")
                st.metric("Components", features.get('num_components', 0))
                st.metric("Avg Component Area", f"{features.get('avg_component_area', 0):.0f}")

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Regional Density")
                st.metric("Central Density", f"{features.get('central_density', 0):.4f}")
                st.metric("Peripheral Density", f"{features.get('peripheral_density', 0):.4f}")

            with col2:
                central = features.get('central_density', 0)
                peripheral = features.get('peripheral_density', 0)
                ratio = central / peripheral if peripheral > 0 else 0
                st.markdown("#### Regional Ratio")
                st.metric("C:P Ratio", f"{ratio:.2f}")
                if ratio > 1.2:
                    st.info("Central dominance")
                elif ratio < 0.8:
                    st.info("Peripheral dominance")
                else:
                    st.success("Balanced distribution")

    # Clinical interpretation
    if clinical:
        st.markdown("---")
        st.markdown("### 🩺 Clinical Interpretation")

        # Severity indicator
        severity = clinical.get("severity", "Unknown")
        severity_info = {
            "Normal": {"emoji": "🟢", "color": "#10b981", "bg": "#d1fae5"},
            "Borderline": {"emoji": "🟡", "color": "#f59e0b", "bg": "#fef3c7"},
            "Mild": {"emoji": "🟠", "color": "#f97316", "bg": "#ffedd5"},
            "Moderate": {"emoji": "🔴", "color": "#ef4444", "bg": "#fee2e2"},
            "Severe": {"emoji": "🔴", "color": "#dc2626", "bg": "#fee2e2"}
        }.get(severity, {"emoji": "⚪", "color": "#6b7280", "bg": "#f3f4f6"})

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"""
            <div style="background: {severity_info['bg']}; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {severity_info['color']};">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 2rem;">{severity_info['emoji']}</span>
                    <div>
                        <h3 style="margin: 0; color: {severity_info['color']};">{severity}</h3>
                        <p style="margin: 0.25rem 0 0 0; color: #6b7280;">Clinical Assessment</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Findings
            if clinical.get("findings"):
                st.markdown("#### 🔍 Key Findings")
                for finding in clinical["findings"]:
                    st.markdown(f"• {finding}")

        with col2:
            st.metric(
                label="Follow-up",
                value=clinical.get("followup", "N/A"),
                delta="Timeline"
            )

        with col3:
            confidence = clinical.get("confidence", 0)
            st.metric(
                label="Confidence",
                value=f"{confidence:.0%}",
                delta="High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
            )

        # Recommendations and differentials
        col1, col2 = st.columns(2)

        with col1:
            if clinical.get("recommendations"):
                with st.expander("📋 Recommendations", expanded=True):
                    for rec in clinical["recommendations"]:
                        st.info(f"• {rec}")

        with col2:
            if clinical.get("differentials"):
                with st.expander("🔍 Differential Considerations", expanded=True):
                    for diff in clinical["differentials"]:
                        st.warning(f"• {diff}")

    # Export section
    st.markdown("---")
    st.markdown("### 💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export as JSON
        if st.button("📥 Export JSON", use_container_width=True):
            export_data = {
                "analysis_id": results["analysis_id"],
                "timestamp": results["timestamp"],
                "confidence": results["confidence"],
                "features": features,
                "clinical": clinical
            }

            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"retinal_analysis_{results['analysis_id']}.json",
                mime="application/json",
                use_container_width=True
            )

    with col2:
        # Export as CSV
        if st.button("📊 Export CSV", use_container_width=True):
            if features:
                df = pd.DataFrame([features])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"retinal_features_{results['analysis_id']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with col3:
        # New analysis button
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analysis_results = None
            st.rerun()


def render_advanced_analytics():
    """Render advanced analytics section"""
    results = st.session_state.analysis_results
    features = results.get("features", {})

    st.markdown("### 📈 Advanced Analytics")

    # Vessel width distribution
    if "width_distribution" in features and features["width_distribution"]:
        st.markdown("#### 📏 Vessel Width Distribution")
        fig = create_width_histogram(features["width_distribution"])
        st.plotly_chart(fig, use_container_width=True)

    # Feature radar chart
    st.markdown("#### 📊 Feature Radar Chart")
    fig = create_radar_chart(features)
    st.plotly_chart(fig, use_container_width=True)

    # Network complexity analysis
    st.markdown("---")
    st.markdown("#### 🌀 Network Complexity Analysis")

    complexity_score = (
            features.get('fractal_dimension', 1) * 0.4 +
            min(features.get('branching_points', 0) / 500, 1) * 0.3 +
            min(features.get('std_width', 0) / 3, 1) * 0.3
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Complexity Score", f"{complexity_score:.3f}")

    with col2:
        st.metric("Fractal Dimension", f"{features.get('fractal_dimension', 1):.3f}")

    with col3:
        st.metric("Branching Density",
                  f"{features.get('branching_points', 0) / 1000:.3f}" if features.get('branching_points',
                                                                                      0) > 0 else "0.000")

    with col4:
        st.metric("Width Heterogeneity", f"{features.get('std_width', 0):.3f}")

    # Regional heatmap
    st.markdown("---")
    st.markdown("#### 🗺️ Regional Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        central = features.get('central_density', 0)
        st.metric("Central Zone", f"{central:.4f}")

    with col2:
        peripheral = features.get('peripheral_density', 0)
        st.metric("Peripheral Zone", f"{peripheral:.4f}")

    with col3:
        ratio = central / peripheral if peripheral > 0 else 0
        st.metric(
            "Central:Peripheral Ratio",
            f"{ratio:.2f}",
            delta="Central Dominant" if ratio > 1.2 else "Peripheral Dominant" if ratio < 0.8 else "Balanced"
        )


def render_mlops_monitor():
    """Render MLops monitoring section"""
    st.markdown("### 📈 MLops Monitoring Dashboard")

    # Performance metrics
    if hasattr(st.session_state, 'performance_monitor'):
        metrics = st.session_state.performance_monitor.get_summary()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Analyses", metrics.get('total_analyses', 0))

        with col2:
            st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.1%}")

        with col3:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")

        with col4:
            st.metric("Avg Processing Time", f"{metrics.get('avg_processing_time', 0):.2f}s")

        # Recent analyses
        st.markdown("---")
        st.markdown("#### 📋 Recent Analyses")

        recent_analyses = metrics.get('recent_analyses', [])
        if recent_analyses:
            df = pd.DataFrame(recent_analyses)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No analyses recorded yet")

    # Model information
    st.markdown("---")
    st.markdown("#### 🤖 Model Information")

    if hasattr(st.session_state, 'model') and st.session_state.model_loaded:
        model_info = st.session_state.model.get_model_info()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Details:**")
            st.markdown(f"- Architecture: {model_info.get('architecture', 'Unknown')}")
            st.markdown(f"- Input Size: {model_info.get('input_size', 'Unknown')}")
            st.markdown(f"- Parameters: {model_info.get('parameters', 0):,}")
            st.markdown(f"- Device: {model_info.get('device', 'Unknown')}")

        with col2:
            st.markdown("**Performance:**")
            st.markdown(f"- Threshold: {model_info.get('threshold', 0.5)}")
            if 'metadata' in model_info:
                metadata = model_info['metadata']
                st.markdown(f"- Version: {metadata.get('version', 'Unknown')}")
                st.markdown(f"- Training Date: {metadata.get('training_date', 'Unknown')}")

    # System health
    st.markdown("---")
    st.markdown("#### 🩺 System Health")

    import psutil
    import platform

    col1, col2, col3 = st.columns(3)

    with col1:
        cpu_percent = psutil.cpu_percent()
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(cpu_percent / 100)

    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent:.1f}%")
        st.progress(memory.percent / 100)

    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent:.1f}%")
        st.progress(disk.percent / 100)

    # System info
    st.markdown(f"""
    **System Information:**
    - Python: {platform.python_version()}
    - OS: {platform.system()} {platform.release()}
    - Processor: {platform.processor()}
    """)


if __name__ == "__main__":
    main()