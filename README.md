# COSMO — Mappa Interattiva Esopianeti

Atlante stellare interattivo che combina:
- **CNN PyTorch** (AstroNet dual-stream) per rilevare esopianeti da curve di luce Kepler/TESS
- **Mappa del cielo** interattiva (Plotly) con overlay pianeti confermati
- **HYG Database** (119k stelle) + **Kepler KOI** catalog

## Quick Start

```bash
source ~/projects/claude-codex/deepML1/.venv/bin/activate
cd ~/projects/claude-codex/cosmo
pip install -r requirements.txt

# Mappa demo immediata (dati sintetici, nessun download)
python atlas/map.py --demo

# Con dati reali
python download_data.py         # scarica HYG + KOI (~15 MB)
python atlas/map.py             # genera cosmo_map.html
```

## Training CNN

```bash
cd ~/projects/claude-codex/cosmo
python model/train.py --epochs 30 --batch 64
```

## Struttura

```
cosmo/
├── model/
│   ├── astronet.py       ← CNN PyTorch dual-stream (AstroNet)
│   └── train.py          ← training loop + dataset sintetico
├── preprocessing/
│   ├── fold.py           ← phase folding curve di luce
│   └── normalize.py      ← normalizzazione mediana
├── atlas/
│   └── map.py            ← mappa interattiva Plotly
├── download_data.py      ← scarica HYG + KOI
└── data/                 ← dataset (gitignored)
```

## Pipeline Completa

```
Kepler .fits → phase fold → normalize → AstroNet CNN → score
                                                           ↓
HYG catalog ←──── cross-match RA/Dec ──────────────────────┘
     ↓
Plotly map (stelle + overlay pianeti + curva di luce)
```
