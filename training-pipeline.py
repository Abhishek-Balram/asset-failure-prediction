import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


DATA_CSV = "ai4i2020.csv"
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AI4IDataset(Dataset):
    """PyTorch Dataset for the AI4I 2020 predictive maintenance data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # BCEWithLogits expects (N,1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess(csv_path: str = DATA_CSV, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load CSV, preprocess, and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, input_dim
    """
    df = pd.read_csv(csv_path)

    # Target column
    y = df["Machine failure"].values

    # Drop identifiers and columns that create target leakage
    X_df = df.drop(
        columns=[
            "UDI",  # unique identifier
            "Product ID",  # essentially categorical with 1000s levels; drop for simplicity
            "Machine failure",  # target
            "TWF",  # failure type column -> leakage
            "HDF",  # failure type column -> leakage    
            "PWF",  # failure type column -> leakage
            "OSF",  # failure type column -> leakage
            "RNF",  # failure type column -> leakage
        ]
    )

    # Identify categorical and numeric columns
    categorical_cols = ["Type"]
    numeric_cols = [col for col in X_df.columns if col not in categorical_cols]

    # Create transformers: one-hot for categorical, standard scaler for numeric
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Build full sklearn pipeline to fit/transform in one go (useful later for inference)
    pipeline = SklearnPipeline(steps=[("preprocessor", preprocessor)])

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED, stratify=y_train_val
    )

    # Fit on training data and transform all splits
    X_train_np = pipeline.fit_transform(X_train)
    X_val_np = pipeline.transform(X_val)
    X_test_np = pipeline.transform(X_test)

    # Persist the sklearn pipeline for future inference
    os.makedirs("artifacts", exist_ok=True)
    import joblib

    joblib.dump(pipeline, "artifacts/preprocessing.pkl")

    # Create PyTorch datasets and loaders
    train_dataset = AI4IDataset(X_train_np, y_train)
    val_dataset = AI4IDataset(X_val_np, y_val)
    test_dataset = AI4IDataset(X_test_np, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    input_dim = X_train_np.shape[1]
    return train_loader, val_loader, test_loader, input_dim


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 1),  # output logits
        )

    def forward(self, x):
        return self.net(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate on val set
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y_batch.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_accuracy = accuracy_score(all_targets, np.round(all_preds))
        val_f1 = f1_score(all_targets, np.round(all_preds))
        val_auc = roc_auc_score(all_targets, all_preds)

        print(
            f"Epoch {epoch:02d}/{epochs} - "
            f"train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f} - "
            f"val_acc: {val_accuracy:.4f} - val_f1: {val_f1:.4f} - val_auc: {val_auc:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

    # restore best state
    model.load_state_dict(best_state)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_targets, np.round(all_preds))
    f1 = f1_score(all_targets, np.round(all_preds))
    auc = roc_auc_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, np.round(all_preds))

    return avg_loss, accuracy, f1, auc, cm


def main(args):
    set_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, input_dim = load_and_preprocess(args.csv, args.test_size, args.val_size)

    model = FeedForwardNet(input_dim).to(device)

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    test_loss, test_acc, test_f1, test_auc, cm = evaluate(model, test_loader, device)
    print(
        "\nTest metrics:\n"
        f"  loss: {test_loss:.4f}\n"
        f"  accuracy: {test_acc:.4f}\n"
        f"  f1-score: {test_f1:.4f}\n"
        f"  roc-auc: {test_auc:.4f}\n"
        f"\nConfusion Matrix (rows=true, cols=pred):\n{cm}"
    )

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    print("Model and preprocessing pipeline saved under ./artifacts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch model on the AI4I 2020 dataset.")
    parser.add_argument("--csv", type=str, default=DATA_CSV, help="Path to ai4i2020 CSV file.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for test set.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of training data for validation.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available.")
    args = parser.parse_args()

    main(args)
