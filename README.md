# Asset-Failure Prediction

Anticipating when industrial equipment will fail **before** it actually does saves money, increases safety and reduces downtime.  This repo is my end-to-end proof-of-concept that walks from raw sensor data --> trained ML model --> containerised API demo

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
See the public demo at https://predictive-maintenance.abhishekbalram.com/


Cold start note: To stay within the free tier I run the API in Google Cloud Run with `min instances = 0`. The first request after inactivity can take ~10–15s while the container cold‑starts, but any subsequent requests are fast. Setting `min instances = 1` would get rid of cold starts by always keeping at least one instance alive, but that would push costs above the free tier. 

*(If the public instance is down, please raise an issue or email me: abhishek.balram@icloud.com)*

---

## 5. Technical challenges


### 5.1 Challenge: No access to proprietary data
Real industrial condition-monitoring data is almost always locked behind NDAs 

**My solution**: I used the public (albeit synthetic) *AI4I 2020* dataset.  While the signals are simulated, the generators were calibrated on real bearings and motors so the failure modes and feature interactions remain realistic enough to prototype on.  Using a public dataset also means anyone can reproduce my results.

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
Other people need to be able to reproduce and verify my results

**My solution**: fixed random seeds across NumPy, Torch and sklearn; pinned package versions in `requirements.txt`; and wrapped the whole stack in a Docker image.  A GitHub **Continuous Integration (CI)** workflow rebuilds the image on every push to the main branch

### 5.7 Challenge: Low-latency inference
Real-time monitoring needs low-latency responses

**My solution**: artefacts are loaded **once at process start-up** (`serve.py` initialises `PREPROCESSOR, MODEL = load_artifacts()` before the Flask app handles traffic).  They live as global, read-only objects (~1 MB) for the lifetime of the worker. This reduces latency because the app doesn't need to re-load the artefacts for every request

### 5.8 Challenge: Cost vs. cold starts on Cloud Run
When deploying to Google Cloud Run, I faced a dilema regarding the minimum number of instances (`min instances`) to use. Keeping `min instances = 1` avoids cold starts by keeping one instance alive at all times but comes with a monthly cost of ~20 NZD per month. `min instances = 0` fits within the free tier but the first request after idle can take ~10–15 seconds because it has to spin up a new instance.

**My solution**: Since this is a proof-of-concept, I chose to prioritise keeping costs low and set `min instances = 0`. However I also split the frontend and backend, allowing me to host the frontend as a separate static site with no cold start, and deploy only the API to Cloud Run with `min instances = 0`. The first API call may be slow after idle, but overall cost remains ~$0. 

---

## 6. Limitations & roadmap 

| Limitation | Why it's a problem | How I could address it in the future |
|------------|---------------|----------------|
| **Use of synthetic data** | Models tuned on synthetic signals may over-fit simulator quirks and under-perform on physical sensors. | Approach local firms for anonymised logs; alternatively scrape publicly available datasets and fine-tune. |
| **Limited to known failure modes** | The model is a supervised classifier that can only predict failures it has seen in the training data. It will likely miss entirely new or rare types of malfunctions, offering no warning for "unknown unknowns". | Complement this classifier with an unsupervised anomaly detection model (e.g., an Autoencoder or Isolation Forest). The anomaly model would learn the signature of normal operation and could flag any deviation, providing a much broader safety net and an earlier warning system for all types of operational issues, not just catastrophic failure.|
| **Static web form** | The manual entry in my web demo doesn't showcase real-time monitoring and alerting. | Build a small Kafka --> Flask bridge that streams sensor JSON and live-updates a dashboard. |
| **No drift / health monitoring** | Data distributions shift and if left unmonitored the models performance can silently degrade over time. | Research methods for drift detection and health monitoring. Try to implement some of them |
| **Minimal API error-handling** | Bad requests or spikes could crash the service. | Adopt Pydantic schemas, implement circuit-breakers & exponential back-off, etc... |


---

## 7. Repository map
```
├── ai4i2020.csv            # raw data
├── training-pipeline.py    # training script
├── serve.py                # Flask API + web UI
├── artifacts/              # saved model & preprocessing
├── templates/index.html    # minimal front-end
└── Dockerfile              # container recipe
```

---

## 8. References
* S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, 2020.
* UCI Machine-Learning Repository, AI4I 2020 dataset.

---
