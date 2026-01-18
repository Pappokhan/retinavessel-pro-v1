## App Link: https://retinavessel-pro-v1.streamlit.app/

# 👁️ RetinaVessel Pro

**Advanced AI-Powered Retinal Vessel Analysis System with Full MLops Implementation**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![MLops](https://img.shields.io/badge/MLops-Complete-green)
![License](https://img.shields.io/badge/License-Research%20Use%20Only-lightgrey)

## 🎯 Overview

RetinaVessel Pro is a **complete web application** for research-grade retinal vessel analysis using **Attention U-Net deep learning**. It combines **AI segmentation, quantitative feature extraction, clinical interpretation, and full MLops infrastructure** in a single, production-ready package.

**⚠️ RESEARCH USE ONLY** - Not for clinical diagnosis or medical decision-making.

## ✨ Features

### 🤖 **AI & Deep Learning**
- **Attention U-Net** for precise vessel segmentation
- **Real-time inference** with confidence scoring
- **15+ morphological features** extraction
- **Adjustable sensitivity** with threshold control

### 📊 **Quantitative Analysis**
- **Vessel Density**: Overall, central, and peripheral
- **Width Analysis**: Mean, median, thin/thick ratios
- **Morphology**: Tortuosity, fractal dimension, branching
- **Regional Distribution**: Central vs. peripheral patterns

### 🩺 **Clinical Interpretation**
- **5-level severity scoring** (Normal → Severe)
- **Automated findings** and recommendations
- **Differential diagnosis** considerations
- **Follow-up timeframes** guidance

### 🔬 **MLops Infrastructure**
- **Model Registry**: Version control & management
- **Performance Monitor**: Real-time metrics tracking
- **Experiment Tracker**: ML experiment logging
- **System Health**: CPU, memory, disk monitoring

### 🎨 **Professional UI**
- **Interactive dashboard** with Plotly charts
- **Real-time visualizations** (overlay, histograms, radar)
- **Export capabilities** (JSON, CSV, images)
- **Responsive design** for all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Trained Attention U-Net model (.pth file)
- 4GB+ RAM (GPU optional but recommended)

### Installation

```bash
# 1. Clone/download the project
git clone <repository-url>
cd retinavessel-pro

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your trained model
cp your_model.pth models/attention_unet.pth

# 6. Run the application
streamlit run app.py

# 7. Open browser: http://localhost:8501
