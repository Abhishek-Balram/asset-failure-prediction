# Asset Failure Prediction - Project Documentation 

## Training Pipeline

The goal of the training pipeline is to produce a production-ready PyTorch model that predicts machine failure for the *AI4I 2020 Predictive Maintenance* dataset.

### 1. Dataset
The dataset is in the `ai4i2020.csv` file. It was downloaded from the UCI Machine Learning Repository:  https://doi.org/10.24432/C5HS5C


### 2. Pre-processing
| Step | Details |
|------|---------|
| **Column removal** | `UDI` and`Product ID` columns were dropped because they are identifiers with no predictive value.<br/> <br/> `TWF`, `HDF`, `PWF`, `OSF`, `RNF` columns were dropped because they are individual failure-type flags that are direct components of the target, creating a target-leakage risks. |
| **Categorical encoding** | One-hot encoding was done on the `Type` column using `sklearn.preprocessing.OneHotEncoder`  |
| **Numerical scaling** | Standardised (zero-mean, unit-variance) all numeric features via `StandardScaler`. |
| **Pipeline** | `ColumnTransformer` + `Pipeline` wrap preprocessing steps for clean fit/transform and are persisted to `artifacts/preprocessing.pkl` |
| **Splits** | Stratified 80 / 10 / 10 train / validation / test splits. Random seed = 42 for reproducibility. |

### 3. Model
Simple feed-forward neural network architecture implemented in PyTorch:
  * Input → 64 → 32 → 1 units.
  * ReLU activation
  * BatchNorm + Dropout regularisation.
* BCEWithLogitsLoss (binary classification).
* Adam Optimiser (default β parameters)
* Configurable learning-rate  (default = 1e-3).
* Best validation loss checkpoint is kept and restored at the end of training.

### 4. Metrics & Monitoring
During each epoch the script prints:
* Training Loss
* Validation Loss
* Validation Accuracy
* Validation F1-score
* Validation ROC-AUC

After training it evaluates on the test set and adds a Confusion Matrix to visualise FP/FN trade-offs.

### 5. Artifacts
**`artifacts/model.pt`** – Best-performing model weights.

**`artifacts/preprocessing.pkl`** – Fitted `sklearn` preprocessing pipeline.

Both artifacts are required for serving the model


### 6. Usage
```bash
# Install required packages
pip install -r requirements.txt

# Train the model (default settings)
python training-pipeline.py

```

---

## Serving the Model 

After training completes and the required artifacts are producted, a lightweight inference service can be spun up with Flask:

```bash
python serve.py
```

### API Endpoint

`POST /predict` — expects a JSON payload with a `data` key containing a list of records. Each record must include the same feature columns used in training.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "data": [
             {
               "Type": "L",
               "Air temperature [K]": 298.1,
               "Process temperature [K]": 308.5,
               "Rotational speed [rpm]": 1550,
               "Torque [Nm]": 40.5,
               "Tool wear [min]": 120
             }
           ]
         }'
```

Sample response:

```json
{
  "predictions": [
    {
      "failure_probability": 0.1234,
      "predicted_label": 0
    }
  ]
}
```

## Interactive Web App  

A minimal yet polished front-end is bundled into the same Flask service so you can explore the model without crafting raw JSON.

### 1. Start the server
```bash
python serve.py  
```

### 2. Using the UI
Open your browser at <http://localhost:8000>. You will see a minimalistic form.  

Either fill in the six feature fields manually **or** click one of the _Load sample_ buttons:  
  * **Non-failure sample** – typical operating conditions the model considers healthy.  
  * **Failure sample** – stressed conditions that lead to a high failure probability.

Hit **Predict** and the page will display the predicted probability and binary label in real time.


