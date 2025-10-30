# 🧩 AI Project Template

A reusable, professional template for ML/AI projects. Clone or use as a GitHub _Template_ to start new projects quickly.

## 📁 Structure
```
ai-project-template/
├── data/            # raw & processed data (kept out of git via .gitignore rules)
├── models/          # saved models, checkpoints
├── notebooks/       # Jupyter notebooks
├── scripts/         # CLI utilities (train/evaluate/etc.)
├── src/             # reusable modules (feature_eng.py, model.py, etc.)
├── requirements.txt # project deps
├── .gitignore
├── setup_venv.sh    # Linux/macOS setup
├── setup_venv.ps1   # Windows PowerShell setup
└── README.md
```

## 🚀 Quick start
**Linux/macOS (or Ubuntu):**
```bash
bash setup_venv.sh
# then
jupyter notebook
```
**Windows PowerShell:**
```powershell
.\setup_venv.ps1
# then
jupyter notebook
```

## ✍️ Notes
- The virtual environment (`.venv/`) is **not** committed to git.
- Update `requirements.txt` as you add libs: `pip freeze --exclude-editable > requirements.txt`.
- Register your kernel once per project so notebooks pick the right env.
