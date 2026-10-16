# Validation

Checks run during packaging in an isolated environment:
- scikit-learn 1.6.1, pandas 2.3.3, numpy 2.0.2, joblib 1.5.3, Streamlit 1.66.0.
- Original supplied pipeline loads successfully.
- Predictions recomputed for all 737 journey rows match the supplied CSV (numeric tolerance).
- Absolute-error values and elapsed-time calculation agree with source columns.
- CSV contains one vehicle–trip pair.
- Every final notebook code cell passes Python syntax parsing.
- Streamlit AppTest: no exceptions; eight numeric inputs; prediction button returns 0.002521; journey metrics and table render.
- Journey MAE: 0.000146208195.

The 5-million-row source CSV was not supplied, so training, sample membership and vehicle-disjoint separation were not independently rerun. Overall test metrics and group counts in README are reported from the user's Kaggle run. Notebook outputs are cleared rather than reconstructed. Browser visual layout was previously confirmed in the supplied screenshots, before the packaging changes; AppTest checks execution rather than pixel appearance.
