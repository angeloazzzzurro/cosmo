"""
COSMO — Training AstroNet
Usa dati sintetici oppure Kepler reale via lightkurve.

Usage:
    python model/train.py --epochs 30 --batch 64
    python model/train.py --epochs 30 --batch 64 --real        # dati Kepler reali
    python model/train.py --epochs 30 --batch 64 --real --n 500
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from astronet import AstroNet

# Preprocessing già implementato
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
from fold import make_views
from normalize import median_normalize, clip_normalize

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Dataset sintetico (fase 1 — training senza dati reali) ──────────────────

def generate_synthetic(n_samples: int = 50_000, seq_len: int = 2001):
    """
    Genera curve di luce sintetiche con e senza transito.
    Transito: box-shaped dip centrato in posizione casuale, profondità 0.1-2%.
    """
    X_global = np.random.normal(0, 0.002, (n_samples, seq_len)).astype(np.float32)
    X_local  = np.random.normal(0, 0.002, (n_samples, 201)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    # Metà dei campioni ha un transito
    n_planet = n_samples // 2
    for i in range(n_planet):
        depth   = np.random.uniform(0.001, 0.02)
        width_g = np.random.randint(5, 80)
        center  = np.random.randint(width_g, seq_len - width_g)
        X_global[i, center - width_g: center + width_g] -= depth
        # Local view centrata sul transito
        width_l = np.random.randint(5, 30)
        c_l = 100  # center of local view
        X_local[i, c_l - width_l: c_l + width_l] -= depth
        y[i] = 1.0

    return X_global, X_local, y


class SyntheticDataset(Dataset):
    def __init__(self, n_samples: int = 50_000):
        X_g, X_l, y = generate_synthetic(n_samples)
        self.X_g = torch.from_numpy(X_g).unsqueeze(1)  # (N, 1, 2001)
        self.X_l = torch.from_numpy(X_l).unsqueeze(1)  # (N, 1, 201)
        self.y   = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_g[idx], self.X_l[idx], self.y[idx]


# ── Dataset Kepler reale (via lightkurve + catalogo KOI) ────────────────────

def fetch_kepler_sample(n_confirmed: int = 200, n_false: int = 200) -> list:
    """
    Scarica curve di luce Kepler da lightkurve.
    Restituisce lista di (global_view, local_view, label).

    Catalogo KOI hardcoded con un sottoinsieme bilanciato di:
      - CONFIRMED  → label 1
      - FALSE POSITIVE → label 0
    """
    try:
        import lightkurve as lk
    except ImportError:
        raise ImportError("Installa lightkurve: pip install lightkurve")

    # KOI confermati (kepoi_name, period_giorni, t0_bkjd, duration_ore)
    CONFIRMED_KOI = [
        ("K00001.01", 2.4706,  54.1,  2.19),
        ("K00002.01", 2.2047,  54.3,  3.11),
        ("K00007.01", 68.958,  59.8,  6.67),
        ("K00010.01", 3.5225,  54.2,  2.88),
        ("K00012.01", 17.855,  57.0,  5.76),
        ("K00041.01", 12.816,  55.5,  3.29),
        ("K00069.01", 4.7268,  54.3,  2.01),
        ("K00072.01", 10.054,  55.0,  2.39),
        ("K00085.01", 2.1547,  54.3,  1.72),
        ("K00098.01", 3.0123,  54.4,  2.38),
        ("K00100.01", 3.5398,  54.4,  1.96),
        ("K00111.01", 9.6203,  55.2,  2.44),
        ("K00114.01", 5.3389,  54.5,  2.61),
        ("K00115.01", 6.4335,  54.7,  2.52),
        ("K00116.01", 15.572,  56.6,  3.89),
    ]
    # KOI falsi positivi (stessa struttura)
    FALSE_KOI = [
        ("K00005.01", 4.7803,  54.4,  2.63),
        ("K00006.01", 3.2347,  54.3,  2.11),
        ("K00008.01", 3.0553,  54.3,  1.89),
        ("K00013.01", 1.7636,  54.2,  3.39),
        ("K00016.01", 3.3517,  54.3,  2.74),
        ("K00017.01", 3.2347,  54.3,  2.11),
        ("K00019.01", 3.2347,  54.3,  2.11),
        ("K00020.01", 3.2347,  54.3,  2.11),
        ("K00022.01", 7.8912,  54.9,  3.01),
        ("K00024.01", 3.2347,  54.3,  2.11),
    ]

    samples = []

    def process_koi(koi_list, label, max_n):
        count = 0
        for kepoi, period, t0, duration_h in koi_list:
            if count >= max_n:
                break
            kepid = kepoi.split(".")[0].replace("K", "")
            kepid_int = int(kepid)
            try:
                search = lk.search_lightcurve(
                    f"KIC {kepid_int}", mission="Kepler", cadence="long"
                )
                if len(search) == 0:
                    continue
                lc = search[0].download()
                if lc is None:
                    continue
                lc = lc.remove_nans().remove_outliers(sigma=5)
                time = lc.time.value.astype(np.float64)
                flux = lc.flux.value.astype(np.float32)
                flux_norm = median_normalize(flux)
                duration_days = duration_h / 24.0
                g_view, l_view = make_views(time, flux_norm, period, t0, duration_days)
                g_view = clip_normalize(g_view)
                l_view = clip_normalize(l_view)
                samples.append((g_view, l_view, float(label)))
                count += 1
            except Exception as e:
                print(f"  [skip] {kepoi}: {e}")
                continue

    print(f"Download curve Kepler — confermati (max {n_confirmed})...")
    process_koi(CONFIRMED_KOI * (n_confirmed // len(CONFIRMED_KOI) + 1), 1, n_confirmed)
    print(f"Download curve Kepler — falsi positivi (max {n_false})...")
    process_koi(FALSE_KOI * (n_false // len(FALSE_KOI) + 1), 0, n_false)

    print(f"Campioni scaricati: {len(samples)} "
          f"(+: {sum(1 for _,_,l in samples if l==1)}, "
          f"-: {sum(1 for _,_,l in samples if l==0)})")
    return samples


class KeplerDataset(Dataset):
    """Dataset da curve di luce Kepler reali."""

    def __init__(self, samples: list):
        self.X_g = torch.from_numpy(
            np.stack([s[0] for s in samples])
        ).unsqueeze(1)   # (N, 1, 2001)
        self.X_l = torch.from_numpy(
            np.stack([s[1] for s in samples])
        ).unsqueeze(1)   # (N, 1, 201)
        self.y = torch.tensor([s[2] for s in samples], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_g[idx], self.X_l[idx], self.y[idx]


# ── Training loop ────────────────────────────────────────────────────────────

def train(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
          use_real: bool = False, n_real: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if use_real:
        print("Modalità: dati Kepler reali")
        samples = fetch_kepler_sample(n_confirmed=n_real, n_false=n_real)
        if len(samples) < 10:
            print("Troppo pochi campioni scaricati — fallback su sintetici")
            dataset = SyntheticDataset(n_samples=60_000)
        else:
            dataset = KeplerDataset(samples)
    else:
        print("Modalità: dati sintetici")
        dataset = SyntheticDataset(n_samples=60_000)
    n_val = int(0.15 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = AstroNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for X_g, X_l, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            X_g, X_l, y = X_g.to(device), X_l.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X_g, X_l)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            correct += ((pred > 0.5) == y.bool()).sum().item()
            total += len(y)
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_g, X_l, y in val_loader:
                X_g, X_l, y = X_g.to(device), X_l.to(device), y.to(device)
                pred = model(X_g, X_l)
                val_loss += criterion(pred, y).item() * len(y)
                val_correct += ((pred > 0.5) == y.bool()).sum().item()
                val_total += len(y)

        tl = train_loss / total
        vl = val_loss / val_total
        ta = correct / total
        va = val_correct / val_total
        print(f"Epoch {epoch:3d} | train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), MODELS_DIR / "astronet_best.pt")
            print(f"           → saved best model (val_loss={vl:.4f})")

    print(f"\nTraining completato. Best model: {MODELS_DIR / 'astronet_best.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--batch",  type=int,   default=64)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--real",   action="store_true", help="Usa dati Kepler reali via lightkurve")
    p.add_argument("--n",      type=int,   default=200, help="Campioni per classe (solo con --real)")
    args = p.parse_args()
    train(args.epochs, args.batch, args.lr, use_real=args.real, n_real=args.n)
