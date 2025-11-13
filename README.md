# 📘 **README.md — CalcBERT**

````md
# CalcBERT  
**Offline Transaction Categoriser — GHCI Hackathon**

CalcBERT is a lightweight, offline-first transaction categorisation system.  
It includes:
- A FastAPI backend (TF-IDF and DistilBERT models)
- A Streamlit UI
- A simple ML pipeline designed for speed, clarity, and hackathon-friendly iteration

---

# 🚀 Quickstart (Developers)

## 1) Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/CalcBERT.git
cd CalcBERT
````

## 2) Create Python environment

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the project

## Start Backend (FastAPI)

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Backend will be at:
👉 **[http://localhost:8000](http://localhost:8000)**

Test it:

```
GET http://localhost:8000/health
```

---

## Start UI (Streamlit)

Open a new terminal:

```bash
cd ui
streamlit run app.py --server.port 8501
```

UI runs at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

# 📂 Project Structure

```
CalcBERT/
├── backend/
│   ├── app.py
│   ├── model_adapter.py
│   └── routes/
│       ├── predict.py
│       ├── feedback.py
│       └── retrain.py
├── ml/
│   ├── data_pipeline.py
│   ├── normalize_map.json
│   ├── tfidf_pipeline.py
│   ├── train_tfidf.py
│   ├── evaluate_tfidf.py
│   ├── feedback_handler.py
│   └── distilbert_model.py
├── ui/
│   ├── app.py
│   └── components/
│       └── explain_card.py
├── data/
│   ├── demo_train_small.csv
│   ├── train.csv
│   └── categories.json
├── saved_models/
│   ├── tfidf/
│   └── distilbert/
├── tests/
│   ├── test_data_pipeline.py
│   └── test_tfidf_pipeline.py
├── metrics/
│   └── tfidf_metrics.json
├── scripts/
│   ├── start_all.sh
│   └── save_weights.sh
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 👥 Team Workflow (Hackathon-Optimised)

### Rules

* **Never commit directly to `main`**
* Always work on **feature branches**
* Branch format:
  `feature/<name>-<task>`
  → Example: `feature/neha-tfidf`

### Start a feature

```bash
git checkout -b feature/<name>-<task>
```

### Save progress

```bash
git add .
git commit -m "short message"
git push -u origin feature/<name>-<task>
```

### Open a Pull Request (PR)

* Go to GitHub → Repo
* You'll see *“Compare & pull request”*
* Assign a teammate
* Merge after **1 approval**

---

# 🧩 Team Roles (suggested)

**Neha (ML)**

* `ml/data_pipeline.py`
* `ml/tfidf_pipeline.py`
* Add `data/demo_train_small.csv`

**Adya (ML)**

* `ml/distilbert_model.py`
* Optional: fusion/ensemble logic

**Dhyey (Backend)**

* `backend/app.py`
* `backend/model_adapter.py`
* `backend/routes/predict.py` + `feedback.py`

**Suchet (UI)**

* `ui/app.py`
* `ui/components/explain_card.py`

---

# 🧪 Basic Testing

To run tests:

```bash
pytest
```

To quickly check backend:

```
GET http://localhost:8000/health
```

---

# 🧱 Model Weights Handling

To avoid large files messing up the repo:

* `saved_models/distilbert/` is ignored in `.gitignore`
* Store large `.pt` or `.bin` files in Google Drive
* Add download script in `scripts/save_weights.sh`

---

# 🩹 Troubleshooting

**Can’t push?**

```bash
git pull origin main
# resolve conflicts
git add .
git commit -m "fix: merge conflicts"
git push
```

**Backend not reachable?**

* Ensure backend is running on port 8000
* UI should call: `http://localhost:8000/predict`

**Package missing?**

```bash
pip install -r requirements.txt
```

**Accidentally committed big model files?**
Remove them:

```bash
git rm --cached file.pt
echo "file.pt" >> .gitignore
```

---

# 📝 PR Checklist

Before merging a PR:

* [ ] Code runs locally (backend + UI)
* [ ] PR description is clear
* [ ] No breaking changes unless documented
* [ ] New files follow folder structure
* [ ] No model weights or large files committed

---

# 🎯 Ready for Hackathon

You now have:

* A clean repo
* Clear workflow
* Working backend and UI stubs
* Precise team responsibilities
* Easy local setup

Let’s build CalcBERT 🚀

```
