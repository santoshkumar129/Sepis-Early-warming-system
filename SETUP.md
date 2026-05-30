# 🚀 Setup Guide - Sepsis Early Warning System

## Quick Start (5 minutes)

### Step 1: Clone & Enter Directory
```bash
git clone https://github.com/santosh-ai/sepsis-early-warning-system.git
cd sepsis-early-warning-system
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter
```bash
jupyter notebook notebooks/sepsis_prediction.ipynb
```

### Step 5: Run All Cells
In Jupyter: `Cell → Run All` (or press `Ctrl+Shift+Enter`)

---

## 📋 Detailed Setup

### Prerequisites Check
```bash
python --version    # Should be 3.9+
nvidia-smi          # Check GPU (optional but recommended)
```

### GPU Setup (Optional but Recommended)
```bash
# Check CUDA version
nvcc --version

# Install GPU-enabled TensorFlow
pip install tensorflow-gpu

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### For ClinicalBERT (Generation 3)
```bash
# HuggingFace transformers (already in requirements.txt)
pip install transformers datasets

# The model will auto-download on first run:
# emilyalsentzer/Bio_ClinicalBERT (~400MB)
```

---

## 📁 What Each File Does

| File | Purpose |
|------|---------|
| `notebooks/sepsis_prediction.ipynb` | **Main notebook** - Run this! Contains all 3 generations |
| `notebooks/sepsis_prediction.py` | Python script version of the notebook |
| `README.md` | Project overview, architecture, results |
| `requirements.txt` | Python package dependencies |
| `report/Sepsis_EWS_Report.html` | Full 5-section analytical report (open in browser) |
| `report/report.tex` | LaTeX source for PDF generation |
| `images/` | Generated plots and figures |
| `models/` | Saved model weights (created after training) |
| `results/` | CSV comparison tables (created after training) |

---

## ⚡ Expected Runtime

| Hardware | Total Time |
|----------|-----------|
| CPU only (no GPU) | ~30 minutes |
| GPU (GTX 1650) | ~8 minutes |
| GPU (RTX 3060) | ~5 minutes |
| GPU (RTX 4090) | ~2 minutes |

*ClinicalBERT training adds ~10-15 minutes depending on hardware*

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'transformers'`
```bash
pip install transformers datasets
```

### Issue: `CUDA out of memory`
```python
# In the notebook, reduce batch size:
batch_size = 8  # instead of 32 or 64
```

### Issue: Jupyter not found
```bash
pip install jupyter
python -m notebook
```

### Issue: `SSL certificate verify failed` (on Mac)
```bash
# Install certificates
/Applications/Python\ 3.9/Install\ Certificates.command
```

### Issue: Slow training on CPU
```python
# Reduce dataset size for testing
n_patients = 500  # instead of 1000
```

---

## 🎯 What You'll See After Running

1. **Data Loading**: "Dataset shape: (24000, 43)" etc.
2. **Gen 1 DNN**: Loss curves comparing Adam vs SGD
3. **Gen 2 RNNs**: LSTM, Bi-LSTM, GRU training curves
4. **Gen 3 ClinicalBERT**: Tokenization and fine-tuning (if transformers installed)
5. **Comparison Table**: All 6 models side by side
6. **Confusion Matrices**: Visual comparison of all models
7. **Saved Files**: Models in `models/`, plots in `images/`, results in `results/`

---

## 📊 Interpreting Results

### Key Metrics Explained
- **Accuracy**: Overall correct predictions (can be misleading with imbalanced data)
- **Precision**: Of predicted sepsis cases, how many were actually sepsis?
- **Recall**: Of actual sepsis cases, how many did we catch? **MOST IMPORTANT**
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (1.0 = perfect, 0.5 = random)

### Clinical Interpretation
```
High Recall = We catch most sepsis cases ✓ (patients live)
Low Recall = We miss sepsis cases ✗ (patients may die)
```

---

## 🔄 Reproducibility Notes

- All random seeds are fixed (`np.random.seed(42)`, `tf.random.set_seed(42)`)
- Synthetic data generation is deterministic
- Results may vary slightly due to:
  - Different TensorFlow versions
  - GPU vs CPU numerical precision
  - Operating system differences

---

## 📚 Next Steps

1. **Get real data**: Apply for PhysioNet credentials
2. **Hyperparameter tuning**: Use KerasTuner or Optuna
3. **Deploy**: Convert to TensorFlow Serving or ONNX
4. **Monitor**: Track model drift in production

---

## 💬 Questions?

- GitHub Issues: [github.com/santosh-ai/sepsis-early-warning-system/issues](https://github.com/santosh-ai/sepsis-early-warning-system/issues)
- Email: santosh.ai@university.edu

**Good luck with your submission! 🎓**
