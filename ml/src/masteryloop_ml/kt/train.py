"""Train SAKT against the same origins BKT saw."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml


from masteryloop_ml.data import splits
from masteryloop_ml.kt import bkt_baseline, encoding, evaluate, sakt

def load_params():
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as f:
            return yaml.safe_load(f)
    return {}

def flatten_train_interactions(train_sequences: dict) -> list[dict]:
    out = []
    for seq in train_sequences.values():
        out.extend(seq)
    return out

def train_sakt():
    params = load_params()
    kt_cfg = params.get("kt_data", {})
    sakt_cfg = params.get("sakt", {})
    bkt_cfg = params.get("bkt", {})

    seed = sakt_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training SAKT using device: {device}")

    # 1. Load Data
    data_file = Path("data/processed/interactions.parquet")
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found. Run prepare_ednet first.")
    
    df = pd.read_parquet(data_file)
    interactions = df.to_dict(orient="records")

    # Item mapping
    all_items = sorted(df["item_id"].unique().tolist())
    item_map = {item: idx for idx, item in enumerate(all_items)}
    n_items = len(all_items)
    print(f"Total unique items: {n_items}")

    # 2. Build Learner-wise Split & Truncated Origin Points
    val_fraction = kt_cfg.get("val_fraction", 0.2)
    min_history = kt_cfg.get("min_history", 5)
    max_len = kt_cfg.get("max_len", 200)

    print("Building learner-wise splits and truncated origins...")
    split = splits.build_split(interactions, val_fraction=val_fraction, min_history=min_history)
    print(f"Split: {split['report']['train_learners']} train learners / "
          f"{split['report']['val_learners']} val learners / "
          f"{split['report']['val_points']} prediction points")

    # 3. Fit BKT Baseline on exact same training set
    print("Fitting per-skill BKT baseline...")
    train_rows = flatten_train_interactions(split["train_sequences"])
    bkt = bkt_baseline.fit_per_skill(train_rows, iterations=bkt_cfg.get("em_iterations", 30))
    print(f"Per-skill BKT fitted for {len(bkt)} skills")

    # Save BKT params
    bkt_dir = Path("artifacts/bkt")
    bkt_dir.mkdir(parents=True, exist_ok=True)
    with open(bkt_dir / "bkt_params.json", "w") as f:
        json.dump({s: {"p_init": p.p_init, "p_transit": p.p_transit,
                       "p_guess": p.p_guess, "p_slip": p.p_slip}
                  for s, p in bkt.items()}, f, indent=2)

    # 4. Score BKT on identical validation prediction points
    print("Scoring BKT on validation points...")
    val_points = split["val_points"]
    bkt_preds = []
    val_labels = []
    val_skills = []

    for p in val_points:
        tgt = p["target"]
        sid = str(tgt["skill_id"])
        val_skills.append(sid)
        val_labels.append(int(tgt["correct"]))
        
        bp = bkt.get(sid, bkt_baseline.BktParams())
        hist_skill = [h for h in p["history"] if str(h["skill_id"]) == sid]
        m = bp.p_init
        for h in hist_skill:
            m = bkt_baseline.update(m, bool(h["correct"]), bp)
        bkt_preds.append(bkt_baseline.predict(m, bp))

    bkt_auc = evaluate.auc(val_labels, bkt_preds)
    print(f"BKT Baseline Val AUC: {bkt_auc:.4f}")

    # 5. Prepare SAKT Training Dataset
    print("Encoding training sequences...")
    train_inters, train_nxts, train_targets = [], [], []
    for seq in split["train_sequences"].values():
        items = [item_map[r["item_id"]] for r in seq if r["item_id"] in item_map]
        corrects = [bool(r["correct"]) for r in seq if r["item_id"] in item_map]
        if len(items) < 2:
            continue
        stride = 100
        for start in range(0, max(1, len(items) - 10), stride):
            sub_items = items[start:start + max_len]
            sub_corrects = corrects[start:start + max_len]
            if len(sub_items) >= 2:
                inter, nxt, tgt = encoding.encode_sequence(sub_items, sub_corrects, n_items, max_len=max_len)
                train_inters.append(inter)
                train_nxts.append(nxt)
                train_targets.append(tgt)

    train_ds = TensorDataset(torch.tensor(np.array(train_inters), dtype=torch.long),
                             torch.tensor(np.array(train_nxts), dtype=torch.long),
                             torch.tensor(np.array(train_targets), dtype=torch.long))
    batch_size = sakt_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 6. Prepare SAKT Validation Dataset (Batched)
    print("Encoding validation prediction points...")
    val_inters, val_nxts = [], []
    for p in val_points:
        hist_items = [item_map[h["item_id"]] for h in p["history"] if h["item_id"] in item_map] + [item_map[p["target"]["item_id"]]]
        hist_corrects = [bool(h["correct"]) for h in p["history"] if h["item_id"] in item_map] + [bool(p["target"]["correct"])]
        inter, nxt, _ = encoding.encode_sequence(hist_items, hist_corrects, n_items, max_len=max_len)
        val_inters.append(inter)
        val_nxts.append(nxt)

    val_ds = TensorDataset(torch.tensor(np.array(val_inters), dtype=torch.long),
                           torch.tensor(np.array(val_nxts), dtype=torch.long))
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # 7. Initialize SAKT Model
    dim = sakt_cfg.get("dim", 128)
    heads = sakt_cfg.get("heads", 8)
    dropout = sakt_cfg.get("dropout", 0.2)
    lr = sakt_cfg.get("lr", 0.001)
    epochs = sakt_cfg.get("epochs", 30)
    patience = sakt_cfg.get("patience", 4)
    min_auc = sakt_cfg.get("min_auc", 0.77)
    min_lift = sakt_cfg.get("min_lift_points", 0.04)

    model = sakt.Sakt(n_items=n_items, max_len=max_len, dim=dim, heads=heads, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sakt_dir = Path("artifacts/sakt")
    sakt_dir.mkdir(parents=True, exist_ok=True)
    with open(sakt_dir / "item_map.json", "w") as f:
        json.dump(item_map, f)

    # 8. Training Loop with Early Stopping on Validation AUC
    best_auc = 0.0
    best_preds = None
    no_improve = 0

    print(f"Beginning SAKT training ({epochs} epochs max, patience {patience})...")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for b_inter, b_nxt, b_tgt in train_loader:
            b_inter, b_nxt, b_tgt = b_inter.to(device), b_nxt.to(device), b_tgt.to(device)
            optimizer.zero_grad()
            logits = model(b_inter, b_nxt)
            mask = b_tgt != -100
            loss = F.binary_cross_entropy_with_logits(logits[mask], b_tgt[mask].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation evaluation
        model.eval()
        preds = []
        with torch.no_grad():
            for b_inter, b_nxt in val_loader:
                b_inter, b_nxt = b_inter.to(device), b_nxt.to(device)
                logits = model(b_inter, b_nxt)
                p = torch.sigmoid(logits[:, -1]).cpu().numpy()
                preds.extend(p)

        val_auc = evaluate.auc(val_labels, preds)
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d} val auc {val_auc:.4f} (train loss: {total_loss/max(n_batches,1):.4f}, {elapsed:.1f}s)")

        if val_auc > best_auc:
            best_auc = val_auc
            best_preds = list(preds)
            no_improve = 0
            torch.save(model.state_dict(), sakt_dir / "model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch:02d} (patience {patience})")
                break

    # 9. Comparison & Report Generation
    print("Computing final evaluation comparison...")
    report = evaluate.compare(val_labels, bkt_preds, best_preds, skills=val_skills,
                              min_auc=min_auc, min_lift=min_lift)
    report["split"] = split["report"]
    report["bkt_auc"] = float(bkt_auc)
    report["sakt_auc"] = float(best_auc)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "m3_kt.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"KT evaluation report saved to {report_file}")
    for g in report["gates"]:
        status = "OK " if g["pass"] else "XX "
        print(f"[{status}] {g['gate']} ({g['detail']})")
    print(f"calibration: brier {report['sakt_calibration']['brier']:.4f} "
          f"ece {report['sakt_calibration']['ece']:.4f} | "
          f"worst scored skill auc {report.get('sakt_per_skill', {}).get('worst_skill_auc')}")
    print(f"recommendation: {report['recommendation']}")

if __name__ == "__main__":
    train_sakt()

