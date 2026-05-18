# Explainable Machine Learning for Hotel Booking Cancellations: The Role of Public Holidays

**Master's Thesis**  
**Author:** Nurbek Suvonov  
**University:** Graduate School of Business IT, Kookmin University  
**Advisor:** Prof. Hyunchul Ahn  
**Year:** 2026

---

## Abstract

This thesis develops explainable machine learning models to predict hotel booking cancellations and investigates whether public holidays influence cancellation patterns. Using data from two Portuguese hotels (one resort, one city) covering 25 months, we compare Logistic Regression, Random Forest, and XGBoost models across multiple evaluation strategies. Our key finding: tree-based ensemble models achieve excellent predictive performance (AUC > 0.95), but the engineered public holiday feature has negligible impact on prediction accuracy. Cross-hotel SHAP analysis reveals distinct cancellation drivers.

---

## Project Overview

## Research Questions

- **RQ1:** How accurately can LR, RF, and XGBoost predict cancellations for resort vs. city hotels?
- **RQ2:** Which model performs best per hotel type, and why?
- **RQ3:** What are the key factors driving cancellations (via SHAP explainability)?
- **RQ4:** How do public holidays influence cancellation patterns?

---

## Key Findings

- 🎯 **Best Models:** XGBoost for resort (AUC 0.958), Random Forest for city (AUC 0.953)
- 📊 **Holiday Effect:** Negligible (ΔAUC ≤ 0.0005, SHAP rank 31–43 of 64, Cramér's V < 0.05)
- 🔑 **Top Drivers:** Guest origin (country_PRT), lead time, special requests, parking/rate
- ⏰ **Temporal Drift:** ~10-point AUC drop when testing on future data (2017)

---

## Dataset

**Source:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Antonio et al., 2019)

- **Period:** July 2015 – August 2017 (~45K resort, ~95K city bookings)
- **Target:** `is_canceled` (binary)
- **Features:** 64 numeric (pre-cleaned, one-hot encoded)

To reproduce: Download the dataset from Kaggle and place as `cleaned_dataset.csv` in the root directory.

## Project Structure

```
├── analysis.ipynb                      # Main analysis notebook
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git exclusions
│
├── results/                            # Output CSVs
├── figures/                            # Visualizations
├── shap_outputs/                       # SHAP analysis outputs
│
├── TECHNICAL_DETAILS.md                # Full technical documentation
├── KEY_FINDINGS.md                     # Summary of key findings
└── CRITICAL_REVIEW.md                  # Peer review & limitations
```

## How to Run

```bash
# Clone and setup
git clone https://github.com/yourusername/hotel-cancellation-ml.git
cd hotel-cancellation-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset and place as cleaned_dataset.csv

# Run analysis
jupyter notebook analysis.ipynb
```


## Key Results

| Hotel | Best Model | AUC (No Holiday) | AUC (With Holiday) | Δ AUC |
|---|---|---|---|---|
| Resort | XGBoost | 0.9582 | 0.9581 | −0.0001 |
| City | Random Forest | 0.9526 | 0.9531 | +0.0005 |

**Evaluation:** Random split (80/20) + temporal split (2015–16 train, 2017 test) + bootstrap CIs.

---

## Technologies Used

- **Data:** Pandas, NumPy
- **ML:** Scikit-learn (LR, RF), XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter

---

## Known Limitations

1. **Single property per type** — Generalizability limited to Portuguese market
2. **Temporal drift** — 10-point AUC drop in temporal validation; models require periodic retraining
3. **Potential data leakage** — ART_* (assigned room type) features may be post-check-in placeholders for canceled bookings
4. **Holiday scope** — Portuguese holidays applied uniformly; ~60% of guests are non-Portuguese

See `CRITICAL_REVIEW.md` for detailed discussion.

---

## References

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49.

---

## Citation

```bibtex
@mastersthesis{suvonov2026,
  author = {Suvonov, Nurbek},
  title = {Explainable Machine Learning for Hotel Booking Cancellations: The Role of Public Holidays},
  school = {Kookmin University},
  year = {2026}
}
```
