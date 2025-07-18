import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def extract_features(transactions):
    wallets = defaultdict(list)
    for tx in transactions:
        wallet = tx.get("userWallet")
        if wallet:
            wallets[wallet].append(tx)
    features = []
    for wallet, txs in tqdm(wallets.items(), desc="Extracting features"):
        action = [t["action"] for t in txs]
        f = {
            "wallet": wallet,
            "num_transactions": len(txs),
            "num_deposits": action.count("deposit"),
            "num_borrows": action.count("borrow"),
            "num_repays": action.count("repay"),
            "num_redeems": action.count("redeemunderlying"),
            "num_liquidations": action.count("liquidationcall"),
            "unique_actions": len(set(action)),
        }
        # Transaction time features
        timestamps = sorted([int(t.get("timestamp", 0)) for t in txs if "timestamp" in t])
        f["active_days"] = (max(timestamps)-min(timestamps))/(60*60*24)+1 if timestamps else 1
        f["tx_per_day"] = f["num_transactions"]/f["active_days"] if f["active_days"] > 0 else f["num_transactions"]

        # Asset diversity
        assets = set([t["actionData"].get("assetSymbol") for t in txs if 
                      "actionData" in t and isinstance(t["actionData"], dict) and "assetSymbol" in t["actionData"]])
        f["num_assets"] = len(assets)

        # Borrow/Repay ratio (amounts)
        borrow_amt = sum(float(t["actionData"].get("amount", 0))
                         for t in txs if t["action"] == "borrow" and "actionData" in t)
        repay_amt = sum(float(t["actionData"].get("amount", 0))
                        for t in txs if t["action"] == "repay" and "actionData" in t)
        f["borrow_repay_ratio"] = repay_amt / borrow_amt if borrow_amt > 0 else 0

        # Liquidation ratio
        f["liquidation_ratio"] = f["num_liquidations"] / f["num_borrows"] if f["num_borrows"] > 0 else 0

        features.append(f)
    return pd.DataFrame(features)
