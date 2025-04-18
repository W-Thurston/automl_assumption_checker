# AutoML Assumption Checker 🔍
An interactive tool for checking and visualizing the assumptions of statistical learning models — built for learning, teaching, and robust modeling.

## 🎯 Purpose
Most AutoML tools skip statistical assumptions. This project flips the script:
- Step-by-step assumption checking
- Visual diagnostics and interpretation helpers
- Built for students, bloggers, and practitioners

## 🧠 What It Does
✅ Upload your dataset or simulate one  
✅ Walk through common linear model assumptions:  
- Linearity
- Homoskedasticity
- Normality
- Multicollinearity
- Outliers & Influential Points

✅ Visual + statistical tests side-by-side  
✅ Model summaries with plain-English diagnostics  

## 🧰 Tech Stack
- Python 3.11+
- Streamlit (or FastAPI/Gradio TBD)
- Pandas / Statsmodels / Scikit-learn
- Matplotlib / Seaborn / Plotly

## 🚀 Getting Started
```bash
# Clone the repo
git clone https://github.com/W-Thurston/automl_assumption_checker.git
cd automl_assumption_checker

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app (placeholder)
python app/main.py
```

# 🧪 Tests
```bash
pytest tests/
```

# 📦 Docker (coming soon)
Want full reproducibility? A containerized version will be available shortly.

# 📜 License
MIT License © 2025 [Will Thurston](https://github.com/W-Thurston)

