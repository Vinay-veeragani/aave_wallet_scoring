import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.extract_features import extract_features

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def create_labels(df):
    # Simulate labels: "Good" = no liquidations, borrow_repay_ratio >= 0.95, and has borrows
    cond = (df["num_liquidations"] == 0) & (df["borrow_repay_ratio"] >= 0.95) & (df["num_borrows"] > 0)
    df["label"] = cond.astype(int)
    print("Label distribution:", dict(df["label"].value_counts()))
    return df

def main(
    json_path="data/user-wallet-transactions.json",
    output_path="wallet_scores_ml.csv",
    plot_path="score_distribution_ml.png"
    ):
    print("Load and process data...")
    txs = load_json(json_path)
    df = extract_features(txs)

    # For demonstration, auto-label (ideally, use real outcomes)
    df = create_labels(df)

    feature_cols = [
        'num_transactions','num_deposits','num_borrows',
        'num_repays','num_redeems','num_liquidations',
        'unique_actions','num_assets',
        'borrow_repay_ratio', 'liquidation_ratio', 'tx_per_day'
    ]
    X = df[feature_cols]
    y = df["label"]

    if len(y.unique()) < 2:
        print("Only one class present. Assigning max credit score to all.")
        df["credit_score"] = 1000
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)

        # Score probabilities on all wallets
        df["ml_prob"] = clf.predict_proba(X)[:, 1]  # Probability "good"
        # Map ML probability to 0-1000
        scaler = MinMaxScaler(feature_range=(0, 1000))
        df["credit_score"] = scaler.fit_transform(df[["ml_prob"]]).astype(int)

        # Optional: Test set classification report
        try:
            from sklearn.metrics import classification_report
            y_pred = clf.predict(X_test)
            print("Test set classification report:")
            print(classification_report(y_test, y_pred))
        except:
            pass

    df[["wallet", "credit_score"] + feature_cols].to_csv(output_path, index=False)
    print(f"Credit scores written to {output_path}")

    plt.hist(df["credit_score"], bins=20, edgecolor='black')
    plt.title("ML Credit Score Distribution")
    plt.xlabel("Score (0-1000)")
    plt.ylabel("# Wallets")
    plt.savefig(plot_path)
    print(f"Distribution plot saved to {plot_path}")

if __name__ == "__main__":
    main()
