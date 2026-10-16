# Vehicle Energy & Telemetry Prediction

A lightweight end-to-end machine learning prototype that estimates the dataset's Energy_Consumption target from eight telemetry and road-context inputs, with a local Streamlit interface.

## Data
- Source: https://www.kaggle.com/datasets/yashdev01/vehicle-energy-and-telemetry-dataset
- Original CSV inspected: 5,000,000 rows, 35 columns.
- Rows with an available target: 1,846,643; modeling sample: 200,000 rows, random_state=42.
- Sample contains 308 vehicle IDs.
- Inputs: vehicle speed, MAF, engine RPM, absolute load, outside air temperature, smoothed elevation, gradient and speed limit.

## Method and evaluation
Vehicle-disjoint GroupShuffleSplit (test_size=0.2, random_state=42): 162,839 training rows from 246 vehicles; 37,161 test rows from 62 vehicles; zero shared vehicle IDs. Median imputation is fitted only on training data inside a scikit-learn Pipeline.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.001575 | 0.002887 | 0.669055 |
| Random Forest | 0.000670 | 0.001604 | 0.897888 |

Final Random Forest: 15 trees, max_depth=10, random_state=42. Compressed pipeline: approximately 0.681 MB. Save/load prediction equivalence was checked in Kaggle. A local Streamlit smoke test produced 0.002521 for the default inputs.

Impurity-based importance: MAF 65.57%, speed 27.76%, outside temperature 2.67%, load 2.57%, RPM 1.02%, elevation 0.21%, gradient 0.14%, speed limit 0.07%. These are model-specific importances, not causal effects.

## Run locally
Place these files together:
- app.py (Streamlit application)
- vehicle_energy_pipeline.joblib (downloaded from Kaggle)
- requirements.txt
- trip_review.csv (example journey)

In a terminal opened in that folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep the terminal open while using the app. To stop, press Ctrl+C. To restart later, run the Streamlit command again from the same folder.

The app must load vehicle_energy_pipeline.joblib, rather than the older vehicle_energy_model.joblib. Use the same scikit-learn version as the training environment for model compatibility. The training versions are pinned: scikit-learn 1.6.1, pandas 2.3.3, numpy 2.0.2 and joblib 1.5.3. Streamlit 1.66.0 was validated during packaging and is pinned in this final package. Python 3.11 is a recommended setup target.

## Limitations
- The supplied Kaggle description identifies a curated subset but does not establish the original collection source, target unit or target derivation. Independent provenance verification remains open.
- Predictions and error metrics use the target's dataset scale; do not label them as litres or kWh.
- It remains unknown whether the target was derived from inputs such as MAF. A high score could partly reflect reconstruction of a target formula.
- Target missingness in the original CSV is approximately 63.07%; conclusions apply to the available-target sample.
- Evaluation uses one vehicle-disjoint split of one dataset, without repeated splits or external validation.
- One absolute-load value is approximately 13,143; it was retained pending source verification. No validated sensor-quality filtering was performed.
- This estimates the contemporaneous target from telemetry snapshots; it does not establish future consumption forecasting.
- The app is a local prototype, without live vehicle integration or measured energy savings. Manually chosen inputs can be outside the training distribution.
- The final model remains fitted on the training partition used for the reported test evaluation.

## Possible next work
Verify provenance and target construction, investigate telemetry quality, evaluate repeated vehicle-disjoint splits and external data, then consider time-based forecasting or live telemetry integration.

## Journey Review Dashboard

The Streamlit application also displays actual and predicted
target values for an example journey from a held-out vehicle.
It shows the journey MAE and the 10 records with the largest
absolute prediction errors.

The journey contains 737 records with available target values.
Its MAE is approximately 0.000146 in the dataset target scale.
This illustrative journey result is separate from the overall
held-out test evaluation. Large residuals do not establish faults
or inefficiency.

Keep trip_review.csv in the same directory as app.py.

## Final package
- `app.py`: manual prediction and journey review. Files resolve relative to the app directory; journey predictions are recalculated with the loaded model.
- `vehicle_energy_pipeline.joblib`: original supplied final pipeline, unchanged.
- `trip_review.csv`: supplied 737-row illustrative journey, unchanged.
- `vehicle_energy_final.ipynb`: cleaned final training/export workflow, with outputs cleared. Legacy random-row models and large exports are omitted.
- `requirements.txt`: app dependencies with recorded training-library versions.
- `requirements-notebook.txt`: additional notebook dependencies.
- `BASLA_TR.md`: Turkish setup and presentation guide.
- `VALIDATION.md`: packaging checks and limits.

The full source dataset is not bundled. Attach it on Kaggle to rerun the notebook; no full-data retraining was performed during packaging. Review the dataset's current license before publishing redistributed data or model artifacts. No license for third-party data is granted by this package.

To run the final notebook locally, update its `file_path` to the downloaded source CSV and install `requirements-notebook.txt`.
