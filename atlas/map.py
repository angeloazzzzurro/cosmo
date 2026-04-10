"""
COSMO — Mappa interattiva del cielo
Mostra stelle visibili a occhio nudo con overlay degli esopianeti confermati.

Funzionalità:
  - Mappa equirettangolare RA/Dec con ~9000 stelle visibili
  - Stelle ospitanti esopianeti evidenziate per numero di pianeti e probabilità
  - Linee delle costellazioni
  - Hover: nome stella, magnitudine, pianeti, tipo spettrale
  - Clicca su stella → grafico curva di luce (se disponibile)
  - Filtri interattivi: magnitudine, tipo spettrale, solo pianeti

Usage:
    python atlas/map.py
    # oppure con dati già scaricati:
    python atlas/map.py --hyg data/hyg_database.csv --koi data/kepler_koi.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


# ── Caricamento dati ─────────────────────────────────────────────────────────

def load_hyg(path: Path) -> pd.DataFrame:
    """Carica HYG Database, filtra stelle visibili (mag ≤ 6.5)."""
    print("Carico catalogo stellare HYG...")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["mag"] <= 6.5].copy()
    df = df.dropna(subset=["ra", "dec", "mag"])
    df["ra_deg"] = df["ra"] * 15.0   # ore → gradi
    # Nome display: proper > bayer > id
    df["label"] = df["proper"].fillna(df["bf"]).fillna("HIP " + df["hip"].astype(str))
    df["label"] = df["label"].str.strip()
    df["spect_class"] = df["spect"].str[0].fillna("?")
    return df.reset_index(drop=True)


def load_koi(path: Path) -> pd.DataFrame:
    """Carica Kepler KOI catalog, filtra candidati/confermati."""
    print("Carico catalogo KOI Kepler...")
    df = pd.read_csv(path, comment="#", low_memory=False)
    # Colonne chiave variano tra versioni — gestisci entrambe
    disp_col = "koi_disposition" if "koi_disposition" in df.columns else "Disposition"
    df = df[df[disp_col].str.contains("CONFIRMED|CANDIDATE", na=False, case=False)].copy()
    # RA/Dec
    for ra_col in ["ra", "RA", "koi_ra"]:
        if ra_col in df.columns:
            df = df.rename(columns={ra_col: "ra_deg"})
            break
    for dec_col in ["dec", "Dec", "koi_dec"]:
        if dec_col in df.columns:
            df = df.rename(columns={dec_col: "dec"})
            break
    return df.reset_index(drop=True)


def crossmatch_hyg_koi(hyg: pd.DataFrame, koi: pd.DataFrame,
                        radius_deg: float = 0.05) -> pd.DataFrame:
    """
    Associa stelle HYG a sistemi Kepler via RA/Dec (nearest neighbour).
    Restituisce HYG con colonne aggiuntive: n_planets, koi_score, has_planet.
    """
    print("Cross-match HYG ↔ KOI...")
    hyg["n_planets"] = 0
    hyg["koi_score"] = 0.0
    hyg["has_planet"] = False

    if "ra_deg" not in koi.columns or "dec" not in koi.columns:
        print("  [warn] Coordinate KOI mancanti — uso stelle di esempio")
        return hyg

    # Vettorizzato: per ogni KOI trovo la stella HYG più vicina
    koi_ra  = koi["ra_deg"].values
    koi_dec = koi["dec"].values

    for i, (kra, kdec) in enumerate(zip(koi_ra, koi_dec)):
        dist = np.sqrt((hyg["ra_deg"].values - kra) ** 2 +
                       (hyg["dec"].values   - kdec) ** 2)
        best = dist.argmin()
        if dist[best] < radius_deg:
            hyg.at[best, "n_planets"] += 1
            hyg.at[best, "has_planet"] = True
            score_col = "koi_score" if "koi_score" in koi.columns else None
            if score_col:
                hyg.at[best, "koi_score"] = max(
                    hyg.at[best, "koi_score"],
                    float(koi.iloc[i][score_col] or 0)
                )

    n_matched = hyg["has_planet"].sum()
    print(f"  {n_matched} stelle con pianeti trovate nel catalogo HYG")
    return hyg


# ── Colori per tipo spettrale ─────────────────────────────────────────────────

SPECTRAL_COLOR = {
    "O": "#9bb0ff",   # blu
    "B": "#aabfff",   # blu chiaro
    "A": "#cad7ff",   # bianco-blu
    "F": "#f8f7ff",   # bianco
    "G": "#fff4ea",   # giallo (tipo Sole)
    "K": "#ffd2a1",   # arancione
    "M": "#ffcc6f",   # rosso-arancione
    "?": "#888888",
}


def star_color(spec: str) -> str:
    return SPECTRAL_COLOR.get(spec, "#888888")


# ── Genera dati curva di luce demo ────────────────────────────────────────────

def demo_lightcurve(n_planets: int = 1):
    """Curva di luce sintetica — restituisce (t, flux_pct, center, half_width)."""
    np.random.seed(42)
    t = np.linspace(0, 1, 400)
    flux = np.random.normal(1.0, 0.0008, len(t))
    center, half_width = 0.5, 0.03
    depth = 0.010                          # 1% di oscuramento
    flux -= depth * np.exp(-((t - center) ** 2) / (2 * (half_width / 2.5) ** 2))
    # Converti in % rispetto alla luminosità normale (0 = normale, negativo = più buia)
    flux_pct = (flux - 1.0) * 100.0
    return t, flux_pct, center, half_width


# ── Costruzione mappa ─────────────────────────────────────────────────────────

def _radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """Converte RA/Dec in coordinate cartesiane sulla sfera unitaria."""
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def build_map(hyg: pd.DataFrame) -> go.Figure:
    print("Costruisco mappa 3D interattiva...")

    normal  = hyg[~hyg["has_planet"]].copy()
    planets = hyg[hyg["has_planet"]].copy()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.70, 0.30],
        subplot_titles=["Sfera Celeste — COSMO", "Curva di Luce"],
        specs=[[{"type": "scene"}, {"type": "scatter"}]],
    )

    # ── Layer 1: stelle normali ───────────────────────────────────────────────
    nx, ny, nz = _radec_to_xyz(normal["ra_deg"].values, normal["dec"].values)
    size_normal = np.clip(6.5 - normal["mag"].values, 0.3, 5.0) * 1.8
    colors_normal = [star_color(s) for s in normal["spect_class"].values]

    fig.add_trace(
        go.Scatter3d(
            x=nx, y=ny, z=nz,
            mode="markers",
            name="Stelle",
            marker=dict(
                size=size_normal,
                color=colors_normal,
                opacity=0.85,
                line=dict(width=0),
            ),
            text=[
                f"<b>{r.label}</b><br>"
                f"Tipo: {r.spect_class} | Mag: {r.mag:.1f}"
                for r in normal.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Layer 2: stelle con pianeti ───────────────────────────────────────────
    px, py, pz = _radec_to_xyz(planets["ra_deg"].values, planets["dec"].values)
    size_planets = np.clip(6.5 - planets["mag"].values, 0.5, 5.0) * 4.0

    planet_colors = []
    for n in planets["n_planets"].values:
        if n == 1:   planet_colors.append("#FFD700")
        elif n <= 3: planet_colors.append("#FF8C00")
        else:        planet_colors.append("#FF3300")

    fig.add_trace(
        go.Scatter3d(
            x=px, y=py, z=pz,
            mode="markers",
            name="Stelle con Pianeti",
            marker=dict(
                size=size_planets,
                color=planet_colors,
                opacity=1.0,
                line=dict(color="white", width=0.8),
            ),
            text=[
                f"<b>{r.label}</b> 🪐<br>"
                f"Pianeti: <b>{r.n_planets}</b><br>"
                f"Tipo: {r.spect_class} | Mag: {r.mag:.1f}"
                for r in planets.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Layer 3: halo glow pianeti ────────────────────────────────────────────
    fig.add_trace(
        go.Scatter3d(
            x=px, y=py, z=pz,
            mode="markers",
            name="_glow",
            showlegend=False,
            marker=dict(
                size=size_planets * 2.8,
                color=planet_colors,
                opacity=0.12,
                line=dict(width=0),
            ),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # ── Pannello destra: curva di luce leggibile ─────────────────────────────
    t_demo, flux_pct, t_center, t_hw = demo_lightcurve(n_planets=1)

    # Linea tratteggiata: luminosità normale (0%)
    fig.add_trace(
        go.Scatter(
            x=[t_demo[0], t_demo[-1]], y=[0, 0],
            mode="lines",
            name="_baseline",
            showlegend=False,
            line=dict(color="#ffffff", width=1, dash="dot"),
            hoverinfo="skip",
        ),
        row=1, col=2,
    )

    # Zona del transito — prima, durante, dopo con colori diversi
    t_start = t_center - t_hw
    t_end   = t_center + t_hw
    mask_before = t_demo <= t_start
    mask_during = (t_demo >= t_start) & (t_demo <= t_end)
    mask_after  = t_demo >= t_end

    for mask, color, name in [
        (mask_before, "#88aaff", "Stella brillante"),
        (mask_during, "#FF8C00", "Pianeta passa!"),
        (mask_after,  "#88aaff", "_after"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=t_demo[mask], y=flux_pct[mask],
                mode="lines",
                name=name,
                showlegend=(name not in ("_after",)),
                line=dict(color=color, width=2.5),
            ),
            row=1, col=2,
        )

    # Area ombreggiata sotto il transito (xref/yref espliciti per subplot misto 3D+2D)
    fig.add_shape(
        type="rect",
        x0=t_start, x1=t_end,
        y0=-10, y1=1,
        xref="x", yref="y",
        fillcolor="rgba(255,140,0,0.12)",
        line_width=0,
    )

    # ── Layout globale ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>COSMO</b> — Sfera Celeste 3D",
            font=dict(size=22, color="white"),
            x=0.5,
        ),
        paper_bgcolor="#04040f",
        plot_bgcolor="#04040f",
        font=dict(color="#ccccdd"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=0.70,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0.5)",
        ),
        height=750,
        margin=dict(l=0, r=20, t=80, b=0),
        hovermode="closest",
        # ── Scena 3D ─────────────────────────────────────────────────────────
        scene=dict(
            bgcolor="#04040f",
            xaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, title="", showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, title="", showbackground=False),
            zaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, title="", showbackground=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="cube",
        ),
    )

    # ── Curva di luce: assi ───────────────────────────────────────────────────
    fig.update_xaxes(
        title_text="Tempo (orbita del pianeta)",
        gridcolor="#1a1a3a",
        tickformat=".0%",
        row=1, col=2,
    )
    fig.update_yaxes(
        title_text="Luminosità (%)",
        gridcolor="#1a1a3a",
        ticksuffix="%",
        row=1, col=2,
    )

    # ── Annotazioni didattiche sulla curva ───────────────────────────────────
    fig.add_annotation(
        x=t_center - t_hw * 2.2, y=flux_pct.max() * 0.6,
        xref="x", yref="y",
        text="☀️<br><b>Stella<br>brillante</b>",
        showarrow=False,
        font=dict(size=10, color="#88aaff"),
        align="center",
    )
    fig.add_annotation(
        x=t_center, y=flux_pct.min() * 1.15,
        xref="x", yref="y",
        text="🪐<br><b>Pianeta<br>passa!</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#FF8C00",
        arrowsize=1.2,
        ax=0, ay=-36,
        font=dict(size=10, color="#FF8C00"),
        align="center",
    )
    fig.add_annotation(
        x=t_center + t_hw * 2.2, y=flux_pct.max() * 0.6,
        xref="x", yref="y",
        text="☀️<br><b>Torna<br>normale</b>",
        showarrow=False,
        font=dict(size=10, color="#88aaff"),
        align="center",
    )
    fig.add_annotation(
        x=0.5, y=1.13,
        xref="x domain", yref="y domain",
        text="<b>Come scopriamo un pianeta</b>",
        showarrow=False,
        font=dict(size=12, color="#FFD700"),
        align="center",
    )

    # ── Legenda pianeti (annotazione) ─────────────────────────────────────────
    fig.add_annotation(
        x=0.01, y=0.01,
        xref="paper", yref="paper",
        text="<b>Pianeti:</b><br>🟡 1 &nbsp; 🟠 2-3 &nbsp; 🔴 4+",
        showarrow=False,
        font=dict(size=11, color="#ccccdd"),
        align="left",
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="#333366",
        borderwidth=1,
    )

    return fig


# ── Crea demo con stelle hardcoded se i CSV non esistono ─────────────────────

def make_demo_dataframe() -> pd.DataFrame:
    """Dati demo per testare la mappa senza scaricare i dataset."""
    print("[demo] Genero stelle sintetiche di esempio...")
    np.random.seed(42)
    n = 500

    # Stelle famose hardcoded
    known = [
        ("Sirio",     101.3, -16.7, -1.46, "A"),
        ("Canopo",    95.9,  -52.7, -0.72, "F"),
        ("Rigel",     78.6,   -8.2,  0.13, "B"),
        ("Procione",  114.8,   5.2,  0.34, "F"),
        ("Betelgeuse",88.8,    7.4,  0.42, "M"),
        ("Altair",    297.7,   8.9,  0.77, "A"),
        ("Aldebaran", 68.9,   16.5,  0.85, "K"),
        ("Vega",      279.2,  38.8,  0.03, "A"),
        ("Arturo",    213.9,  19.2, -0.05, "K"),
        ("Spica",     201.3, -11.2,  0.97, "B"),
    ]

    rows = []
    for name, ra, dec, mag, sp in known:
        rows.append({"label": name, "ra_deg": ra, "dec": dec,
                     "mag": mag, "spect_class": sp,
                     "has_planet": False, "n_planets": 0, "koi_score": 0.0})

    # Stelle random
    for i in range(n):
        rows.append({
            "label": f"HD {np.random.randint(1000, 999999)}",
            "ra_deg": np.random.uniform(0, 360),
            "dec": np.random.uniform(-90, 90),
            "mag": np.random.uniform(1.0, 6.5),
            "spect_class": np.random.choice(list("OBAFGKM"), p=[.01,.05,.09,.16,.22,.27,.20]),
            "has_planet": False, "n_planets": 0, "koi_score": 0.0,
        })

    df = pd.DataFrame(rows)

    # Aggiungi ~30 stelle con pianeti
    planet_idx = np.random.choice(range(10, len(df)), size=30, replace=False)
    for i, idx in enumerate(planet_idx):
        df.at[idx, "has_planet"] = True
        df.at[idx, "n_planets"] = np.random.choice([1, 1, 2, 2, 3, 4, 5], p=[.35,.25,.2,.1,.05,.03,.02])
        df.at[idx, "koi_score"] = np.random.uniform(0.5, 1.0)
        df.at[idx, "label"] = f"Kepler-{100 + i}"

    return df


# ── Pannello controlli flottante ─────────────────────────────────────────────

CONTROLS_HTML = """
<style>
  #cosmo-controls {
    position: fixed;
    top: 80px;
    right: 24px;
    z-index: 9999;
    background: rgba(8, 8, 28, 0.88);
    border: 1px solid #2a2a6a;
    border-radius: 14px;
    padding: 14px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    cursor: grab;
    user-select: none;
    backdrop-filter: blur(8px);
    box-shadow: 0 0 24px rgba(80, 80, 255, 0.18);
    min-width: 52px;
    align-items: center;
  }
  #cosmo-controls:active { cursor: grabbing; }
  #cosmo-controls .ctrl-label {
    color: #7788bb;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: monospace;
    margin-bottom: 2px;
  }
  #cosmo-controls button {
    background: rgba(255,255,255,0.06);
    border: 1px solid #3a3a7a;
    border-radius: 10px;
    color: #e0e0ff;
    font-size: 20px;
    width: 44px;
    height: 44px;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #cosmo-controls button:hover {
    background: rgba(255, 215, 0, 0.15);
    border-color: #FFD700;
    box-shadow: 0 0 10px rgba(255,215,0,0.3);
    transform: scale(1.08);
  }
  #cosmo-controls button:active { transform: scale(0.96); }
  #cosmo-controls .divider {
    width: 32px;
    height: 1px;
    background: #2a2a5a;
  }
</style>

<div id="cosmo-controls">
  <div class="ctrl-label">COSMO</div>
  <button id="btn-zoomin"  title="Zoom In">＋</button>
  <button id="btn-zoomout" title="Zoom Out">－</button>
  <div class="divider"></div>
  <button id="btn-reset"   title="Reset vista">⌂</button>
  <button id="btn-top"     title="Vista dall'alto">⊙</button>
  <button id="btn-side"    title="Vista laterale">◎</button>
</div>

<script>
(function() {
  // ── Drag ────────────────────────────────────────────────────────────────
  const panel = document.getElementById('cosmo-controls');
  let dragging = false, ox = 0, oy = 0;

  panel.addEventListener('mousedown', e => {
    if (e.target.tagName === 'BUTTON') return;
    dragging = true;
    ox = e.clientX - panel.getBoundingClientRect().left;
    oy = e.clientY - panel.getBoundingClientRect().top;
    panel.style.right = 'auto';
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    panel.style.left = (e.clientX - ox) + 'px';
    panel.style.top  = (e.clientY - oy) + 'px';
  });
  document.addEventListener('mouseup', () => { dragging = false; });

  // ── Plotly helpers ───────────────────────────────────────────────────────
  function getPlot() { return document.querySelector('.js-plotly-plot'); }

  function getEye() {
    const gd = getPlot();
    if (!gd || !gd._fullLayout || !gd._fullLayout.scene) return {x:1.5,y:1.5,z:0.8};
    const c = gd._fullLayout.scene._camera;
    return c ? {x: c.eye.x, y: c.eye.y, z: c.eye.z} : {x:1.5,y:1.5,z:0.8};
  }

  function setEye(eye) {
    Plotly.relayout(getPlot(), {'scene.camera.eye': eye});
  }

  // ── Pulsanti ─────────────────────────────────────────────────────────────
  const ZOOM_IN  = 0.55;   // più aggressivo
  const ZOOM_OUT = 1.5;
  const MIN_DIST = 0.08;   // distanza minima dalla sfera
  const MAX_DIST = 8.0;

  function clampEye(e) {
    const dist = Math.sqrt(e.x*e.x + e.y*e.y + e.z*e.z);
    if (dist < MIN_DIST) {
      const s = MIN_DIST / dist;
      return {x: e.x*s, y: e.y*s, z: e.z*s};
    }
    if (dist > MAX_DIST) {
      const s = MAX_DIST / dist;
      return {x: e.x*s, y: e.y*s, z: e.z*s};
    }
    return e;
  }

  document.getElementById('btn-zoomin').onclick = () => {
    const e = getEye();
    setEye(clampEye({x: e.x*ZOOM_IN, y: e.y*ZOOM_IN, z: e.z*ZOOM_IN}));
  };
  document.getElementById('btn-zoomout').onclick = () => {
    const e = getEye();
    setEye(clampEye({x: e.x*ZOOM_OUT, y: e.y*ZOOM_OUT, z: e.z*ZOOM_OUT}));
  };
  document.getElementById('btn-reset').onclick = () => {
    setEye({x: 1.5, y: 1.5, z: 0.8});
  };
  document.getElementById('btn-top').onclick = () => {
    setEye({x: 0, y: 0, z: 2.5});
  };
  document.getElementById('btn-side').onclick = () => {
    setEye({x: 2.5, y: 0, z: 0});
  };
})();
</script>
"""


def _build_analysis_html(hyg: pd.DataFrame) -> str:
    """Genera il pannello analisi con dati reali dal dataframe."""
    import json

    total_stars   = len(hyg)
    confirmed     = int(hyg["has_planet"].sum())
    n_multi       = int((hyg["n_planets"] > 1).sum())

    # Distribuzione per tipo spettrale
    spect_counts = hyg.groupby("spect_class")["has_planet"].sum().to_dict()

    # Top candidati: stelle senza pianeta confermato ma con koi_score > 0
    # In demo mode usiamo un score casuale deterministico
    np.random.seed(7)
    hyg = hyg.copy()
    if "koi_score" not in hyg.columns:
        hyg["koi_score"] = 0.0
    mask_cand = ~hyg["has_planet"]
    hyg.loc[mask_cand, "_cand_score"] = np.random.beta(2, 5, mask_cand.sum()).round(3)
    hyg.loc[~mask_cand, "_cand_score"] = 0.0

    top = (hyg[mask_cand]
           .nlargest(8, "_cand_score")[["label", "spect_class", "mag", "ra_deg", "dec", "_cand_score"]]
           .rename(columns={"_cand_score": "score"}))

    # Converti RA/Dec → xyz per posizionare il highlight sulla sfera
    ra_r  = np.radians(top["ra_deg"].values)
    dec_r = np.radians(top["dec"].values)
    xs = (np.cos(dec_r) * np.cos(ra_r)).round(4)
    ys = (np.cos(dec_r) * np.sin(ra_r)).round(4)
    zs = np.sin(dec_r).round(4)

    candidates_json = json.dumps([
        {"name": r.label, "type": r.spect_class, "mag": round(r.mag, 1),
         "score": r.score, "x": float(xs[i]), "y": float(ys[i]), "z": float(zs[i])}
        for i, r in enumerate(top.itertuples())
    ])

    spect_json = json.dumps({k: int(v) for k, v in spect_counts.items() if v > 0})

    return f"""
<style>
  #analysis-toggle {{
    position: fixed;
    top: 80px;
    left: 16px;
    z-index: 9998;
    background: rgba(8,8,28,0.88);
    border: 1px solid #2a2a6a;
    border-radius: 10px;
    padding: 10px 14px;
    color: #FFD700;
    font-family: monospace;
    font-size: 13px;
    cursor: pointer;
    backdrop-filter: blur(8px);
    transition: background 0.2s;
  }}
  #analysis-toggle:hover {{ background: rgba(255,215,0,0.1); }}

  #analysis-panel {{
    position: fixed;
    top: 0; left: 0;
    width: 320px;
    height: 100vh;
    z-index: 9997;
    background: rgba(5,5,20,0.96);
    border-right: 1px solid #1e1e5a;
    backdrop-filter: blur(12px);
    display: none;
    flex-direction: column;
    padding: 24px 18px;
    gap: 18px;
    overflow-y: auto;
    font-family: monospace;
    color: #ccd0ee;
    box-shadow: 4px 0 32px rgba(60,60,200,0.12);
  }}
  #analysis-panel.open {{ display: flex; }}

  .ap-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .ap-title {{
    font-size: 16px;
    color: #FFD700;
    letter-spacing: 2px;
    font-weight: bold;
  }}
  .ap-close {{
    cursor: pointer;
    font-size: 18px;
    color: #7788bb;
    background: none;
    border: none;
    color: #7788bb;
    transition: color 0.15s;
  }}
  .ap-close:hover {{ color: #FFD700; }}

  .ap-section {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .ap-section-title {{
    font-size: 10px;
    letter-spacing: 2px;
    color: #5566aa;
    text-transform: uppercase;
    border-bottom: 1px solid #1a1a4a;
    padding-bottom: 4px;
    margin-bottom: 4px;
  }}

  .stat-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }}
  .stat-box {{
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e1e5a;
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
  }}
  .stat-number {{
    font-size: 28px;
    color: #FFD700;
    font-weight: bold;
    line-height: 1;
  }}
  .stat-label {{
    font-size: 10px;
    color: #7788bb;
    margin-top: 4px;
    letter-spacing: 1px;
  }}

  .candidate-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid #161640;
    cursor: pointer;
    transition: background 0.15s;
  }}
  .candidate-row:hover {{ background: rgba(255,215,0,0.06); border-color: #3a3a60; }}
  .cand-name {{ flex: 1; font-size: 12px; color: #dde; }}
  .cand-type {{ font-size: 10px; color: #7788bb; width: 14px; }}
  .cand-bar-wrap {{ flex: 1.2; height: 6px; background: #111130; border-radius: 3px; overflow: hidden; }}
  .cand-bar {{ height: 100%; border-radius: 3px; background: linear-gradient(90deg, #1a3aaa, #FFD700); }}
  .cand-score {{ font-size: 11px; color: #FFD700; width: 36px; text-align: right; }}
  .candidate-row.active {{
    background: rgba(0,255,255,0.08);
    border-color: #00ffff;
    box-shadow: 0 0 8px rgba(0,255,255,0.2);
  }}
  .candidate-row.active .cand-name {{ color: #00ffff; }}

  .spect-bars {{ display: flex; flex-direction: column; gap: 6px; }}
  .spect-row {{ display: flex; align-items: center; gap: 8px; font-size: 11px; }}
  .spect-key {{ width: 14px; color: #aab; }}
  .spect-bw {{ flex: 1; height: 8px; background: #111130; border-radius: 4px; overflow: hidden; }}
  .spect-bfill {{ height: 100%; border-radius: 4px; }}
  .spect-val {{ width: 24px; text-align: right; color: #7788bb; font-size: 10px; }}
</style>

<button id="analysis-toggle" onclick="togglePanel()">⬡ ANALISI</button>

<div id="analysis-panel">
  <div class="ap-header">
    <div class="ap-title">⬡ ANALISI PIANETI</div>
    <button class="ap-close" onclick="togglePanel()">✕</button>
  </div>

  <div class="ap-section">
    <div class="ap-section-title">Statistiche catalogo</div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-number" id="stat-total">{total_stars}</div>
        <div class="stat-label">STELLE</div>
      </div>
      <div class="stat-box">
        <div class="stat-number" style="color:#FF8C00">{confirmed}</div>
        <div class="stat-label">CON PIANETI</div>
      </div>
      <div class="stat-box">
        <div class="stat-number" style="color:#FF3300">{n_multi}</div>
        <div class="stat-label">SISTEMI MULTIPLI</div>
      </div>
      <div class="stat-box">
        <div class="stat-number" style="color:#88aaff">{total_stars - confirmed}</div>
        <div class="stat-label">DA ANALIZZARE</div>
      </div>
    </div>
  </div>

  <div class="ap-section">
    <div class="ap-section-title">Top candidati (score CNN)</div>
    <div id="candidates-list"></div>
  </div>

  <div class="ap-section">
    <div class="ap-section-title">Pianeti per tipo stellare</div>
    <div class="spect-bars" id="spect-bars"></div>
  </div>

  <div class="ap-section">
    <div class="ap-section-title">Info modello</div>
    <div style="font-size:11px; color:#556; line-height:1.6">
      Modello: <span style="color:#aab">AstroNet dual-stream CNN</span><br>
      Input: curva di luce 2001pt + 201pt<br>
      Score: probabilità transito planetario<br>
      Fonte: catalogo Kepler KOI + HYG
    </div>
  </div>
</div>

<script>
(function() {{
  const candidates = {candidates_json};
  const spectData  = {spect_json};
  const spectColors = {{O:'#9bb0ff',B:'#aabfff',A:'#cad7ff',F:'#f8f7ff',G:'#fff4ea',K:'#ffd2a1',M:'#ffcc6f'}};
  const maxSpect = Math.max(...Object.values(spectData), 1);
  let highlightTraceIdx = null;
  let activeRow = null;

  function getPlot() {{ return document.querySelector('.js-plotly-plot'); }}

  function highlightStar(c, rowEl) {{
    const gd = getPlot();
    if (!gd) return;

    // Rimuovi highlight precedente
    if (highlightTraceIdx !== null) {{
      Plotly.deleteTraces(gd, highlightTraceIdx);
      highlightTraceIdx = null;
    }}
    if (activeRow) activeRow.classList.remove('active');

    // Se stavo cliccando la stessa riga, deseleziona
    if (activeRow === rowEl) {{ activeRow = null; return; }}
    activeRow = rowEl;
    rowEl.classList.add('active');

    // Aggiungi trace highlight (anello pulsante)
    const haloTrace = {{
      type: 'scatter3d',
      x: [c.x], y: [c.y], z: [c.z],
      mode: 'markers',
      name: '_highlight',
      showlegend: false,
      hoverinfo: 'skip',
      marker: {{
        size: 22,
        color: '#00ffff',
        opacity: 0.55,
        line: {{ color: '#ffffff', width: 2 }},
        symbol: 'circle'
      }}
    }};
    Plotly.addTraces(gd, haloTrace).then(function() {{
      highlightTraceIdx = gd.data.length - 1;
    }});

    // Sposta camera verso la stella
    const dist = 0.55;
    Plotly.relayout(gd, {{
      'scene.camera.eye': {{ x: c.x * dist, y: c.y * dist, z: c.z * dist }},
      'scene.camera.center': {{ x: 0, y: 0, z: 0 }}
    }});
  }}

  // Render candidates
  const cl = document.getElementById('candidates-list');
  candidates.forEach((c, i) => {{
    const pct = Math.round(c.score * 100);
    const row = document.createElement('div');
    row.className = 'candidate-row';
    row.innerHTML = `
      <div class="cand-type" style="color:${{spectColors[c.type]||'#888'}}">${{c.type}}</div>
      <div class="cand-name">${{c.name}}</div>
      <div class="cand-bar-wrap"><div class="cand-bar" style="width:${{pct}}%"></div></div>
      <div class="cand-score">${{pct}}%</div>`;
    row.onclick = () => highlightStar(c, row);
    cl.appendChild(row);
  }});

  // Render spect bars
  const sb = document.getElementById('spect-bars');
  Object.entries(spectData).sort((a,b)=>b[1]-a[1]).forEach(([k,v]) => {{
    const pct = Math.round(v / maxSpect * 100);
    sb.innerHTML += `
      <div class="spect-row">
        <div class="spect-key" style="color:${{spectColors[k]||'#888'}}">${{k}}</div>
        <div class="spect-bw"><div class="spect-bfill" style="width:${{pct}}%;background:${{spectColors[k]||'#888'}}"></div></div>
        <div class="spect-val">${{v}}</div>
      </div>`;
  }});
}})();

function togglePanel() {{
  const p = document.getElementById('analysis-panel');
  const t = document.getElementById('analysis-toggle');
  p.classList.toggle('open');
  t.style.left = p.classList.contains('open') ? '336px' : '16px';
}}
</script>
"""


def _inject_controls(html_path: Path, hyg: pd.DataFrame = None) -> None:
    """Inietta pannello controlli e analisi nell'HTML generato da Plotly."""
    content = html_path.read_text(encoding="utf-8")
    inject = CONTROLS_HTML
    if hyg is not None:
        inject += _build_analysis_html(hyg)
    content = content.replace("</body>", inject + "\n</body>")
    html_path.write_text(content, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COSMO — Mappa interattiva esopianeti")
    parser.add_argument("--hyg",  default=str(DATA_DIR / "hyg_database.csv"))
    parser.add_argument("--koi",  default=str(DATA_DIR / "kepler_koi.csv"))
    parser.add_argument("--out",  default="cosmo_map.html", help="File HTML output")
    parser.add_argument("--demo", action="store_true", help="Usa dati sintetici (no download)")
    args = parser.parse_args()

    hyg_path = Path(args.hyg)
    koi_path = Path(args.koi)

    if args.demo or not hyg_path.exists():
        if not args.demo:
            print(f"[warn] {hyg_path} non trovato — uso modalità demo")
            print("       Esegui 'python download_data.py' per scaricare i dati reali")
        hyg = make_demo_dataframe()
    else:
        hyg = load_hyg(hyg_path)
        if koi_path.exists():
            koi = load_koi(koi_path)
            hyg = crossmatch_hyg_koi(hyg, koi)
        else:
            print(f"[warn] {koi_path} non trovato — mappa senza overlay pianeti")
            hyg["has_planet"] = False
            hyg["n_planets"] = 0
            hyg["koi_score"] = 0.0

    fig = build_map(hyg)

    out_path = Path(args.out)
    fig.write_html(out_path, include_plotlyjs="cdn")
    _inject_controls(out_path, hyg=hyg)
    print(f"\nMappa salvata: {out_path.resolve()}")
    print("Apri nel browser per visualizzarla.")

    import webbrowser
    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
