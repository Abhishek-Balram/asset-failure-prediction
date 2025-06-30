# Asset-Failure Prediction

Anticipating when industrial equipment will fail **before** it actually does saves money, increases safety and reduces downtime.  This repo is my end-to-end proof-of-concept that walks from raw sensor data ➜ trained ML model ➜ containerised API demo

---

## 1. Why it matters
Keeping machines healthy is an important and often mission-critical goal across many industries incuding: **Energy**, **Agriculture**, **Aerospace**, and **Manufacturing/logistics**

The common theme accross all these sectors is *continuous sensor data* + *high cost of failure*.  Predictive-maintenance ML models turn that data into early warnings so engineers can act proactively.

---

## 2. What this repo delivers
1. **Training pipeline**: `training-pipeline.py` cleans data, trains a PyTorch model, logs metrics and saves artefacts.
2. **Inference service**: `serve.py` is a Flask API with a minimalistic web UI for manual testing.
3. **Dockerfile**: one-command container build so it runs identically on my laptop and in the cloud.

---

## 3. Quick start (running it locally)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (takes ~30s on CPU)
python training-pipeline.py

# 3. Serve
python serve.py   # open http://localhost:8000
```
Model weights and the fitted preprocessing pipeline are saved to `artifacts/` after training so the API can load them on start-up.

---

## 4. Live demo
A containerised build of this project is deployed on Google Cloud Run. You can try it out here:
https://asset-failure-prediction-896255560923.europe-west1.run.app/

*(The above link points to the public instance; feel free to raise an issue or email me if it's down).*  
For local testing you can still `curl` or use the web form at `localhost:8000`.

---

## 5. Technical challenges


### 5.1 Challenge: No access to proprietary data
Real industrial condition-monitoring data is almost always locked behind NDAs because it can reveal production volumes, proprietary processes or even safety incidents.

**My solution**: I used the public (albeit synthetic) *AI4I 2020* dataset.  While the signals are simulated, the generators were calibrated on real bearings and motors so the failure modes and feature interactions remain realistic enough to prototype on.  Using a public dataset also means anyone can reproduce my results without special permissions.

### 5.2 Challenge: Target leakage in the raw data
During exploratory analysis I noticed five columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) that directly flag specific failure modes.  Leaving them in would let the model "cheat" because they are derived from the same ground-truth label I'm trying to predict.

**My solution**: these columns are programmatically dropped inside an sklearn `ColumnTransformer`. 

### 5.3 Challenge: Heterogeneous data types
The raw data mixes a string-based `Type` column with numeric sensor readings.  Neural networks expect numerical tensors, so categorical text must be converted without exploding dimensionality.

**My solution**: use a `OneHotEncoder` inside a `ColumnTransformer` to turn the three machine types (L, M, H) into three binary columns.  This keeps the information while preserving a dense mathematical representation suitable for linear algebra operations.

### 5.4 Challenge: Feature scale imbalance
Numeric sensors live on very different ranges (`Torque [Nm]` ≈ 0–100 vs `Rotational speed [rpm]` ≈ 0–3000).  Un-scaled inputs produce gradients proportional to their magnitude, so high-range features dominate the loss surface and drown out subtle ones. I.e., the model "pays too much attention" to Rotational speed and not enough to Torque when making predictions

**My solution**: standard-scale every numeric column to zero-mean, unit-variance via the same `ColumnTransformer`.  The model now "sees" each feature on equal footing, which speeds up convergence and improves stability.

### 5.5 Challenge: Training–serving skew
It is easy to accidentally preprocess data one way in training and a *slightly* different way in production, leading to silent performance drops.

**My solution**: the entire sklearn `Pipeline` (column removal → encoding → scaling) is serialised with `joblib` after training.  The Flask service deserialises the object at boot, so the exact same linear algebra (matrix multiplications, mean/variance, one-hot indices) is applied in production.  This eliminates drifting definitions of "what a feature means".

### 5.6 Challenge: Reproducibility
ML code that works only in the author's environment is a recurring headache.

**My solution**: fixed random seeds across NumPy, Torch and sklearn; pinned package versions in `requirements.txt`; and wrapped the whole stack in a Docker image.  A GitHub **Continuous Integration (CI)** workflow rebuilds the image and runs unit tests on every push to the main branch

### 5.7 Challenge: Low-latency inference
A real-time dashboard needs sub-second responses

**My solution**: artefacts are loaded **once at process start-up** (`serve.py` initialises `PREPROCESSOR, MODEL = load_artifacts()` before the Flask app handles traffic).  They live as global, read-only objects (~1 MB) for the lifetime of the worker. This reduces latency because the app doesn't need to re-load the artefacts for every request

---

## 6. What I learned 

* **Leakage hides in plain sight**: catching the five leaking columns early saved me from celebrating inflated metrics.
* **You don't always need a GPU**: a tiny CPU-friendly network reached a respectable 0.97 ROC-AUC.
* **The last mile (serving) is more than half the work**: Figuring out how to serve the model from an API and then deploy the API took considerably longer than training the model in the first place
* **Infrastructure skills**:  Docker, CI and Google Cloud Run once felt like intimidating DevOps tools. Now I can easily see myself using them again in future projects.
---

## 7. Limitations & roadmap 

| Limitation | Why it's a problem | How I could address it in the future |
|------------|---------------|----------------|
| **Use of synthetic data** | Models tuned on synthetic signals may over-fit simulator quirks and under-perform on physical sensors. | Approach local firms for anonymised logs; alternatively scrape publicly available datasets and fine-tune. |
| **Static web form** | The manual entry in my web demo doesn't showcase real-time monitoring and alerting. | Build a small Kafka → Flask bridge that streams sensor JSON and live-updates a dashboard. |
| **No drift / health monitoring** | Data distributions shift and if left unmonitored the models performance can silently degrade over time. | Research methods for drift detection and health monitoring. Try to implement some of them |
| **Minimal API error-handling** | Bad requests or spikes could crash the service. | Adopt Pydantic schemas, implement circuit-breakers & exponential back-off, etc... |


---

## 8. Repository map
```
├── ai4i2020.csv            # raw data
├── training-pipeline.py    # training script
├── serve.py                # Flask API + web UI
├── artifacts/              # saved model & preprocessing
├── templates/index.html    # minimal front-end
└── Dockerfile              # container recipe
```

---

## 9. References
* S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, 2020.
* UCI Machine-Learning Repository, AI4I 2020 dataset.

---
