from __future__ import annotations

import json
import copy
import hashlib
import re
import unicodedata
from html import escape as html_escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Balón de Oro · Puntuador estadístico",
    page_icon="⚽",
    layout="wide",
)

DATASET_PATH = Path("dataset_ml_base.csv")
OUTPUT_DIR = Path("puntuador")
WEIGHTS_PATH = OUTPUT_DIR / "pesos_puntuador.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Se mantienen las variables y restricciones de signo del script de ML.
NEGATIVE_STATS = {
    "bigChancesMissed",
    "dispossessed",
    "dribbledPast",
    "errorLeadToGoal",
    "errorLeadToShot",
    "goalsConceded",
    "ownGoals",
    "penaltyConceded",
}

ADDITIVE_STATS_REQUESTED = [
    "goalsAssistsSum",
    "bigChancesCreated",
    "bigChancesMissed",
    "keyPasses",
    "successfulDribbles",
    "accuratePasses",
    "totalAttemptAssist",
    "aerialDuelsWon",
    "totalDuelsWon",
    "tacklesWon",
    "interceptions",
    "outfielderBlocks",
    "ballRecovery",
    "dispossessed",
    "dribbledPast",
    "clearances",
    "errorLeadToGoal",
    "errorLeadToShot",
    "saves",
    "goalsPrevented",
    "goalsConceded",
    "highClaims",
    "punches",
    "successfulRunsOut",
    "cleanSheet",
    "ownGoals",
    "penaltyConceded",
    "possessionWonAttThird",
]

PERCENT_PAIRS = {
    "accuratePassesPercentage": ("accuratePasses", "totalPasses"),
    "successfulDribblesPercentage": ("successfulDribbles", "totalDribbles"),
    "aerialDuelsWonPercentage": ("aerialDuelsWon", "totalAerialDuels"),
    "totalDuelsWonPercentage": ("totalDuelsWon", "totalDuels"),
    "tacklesWonPercentage": ("tacklesWon", "totalTackles"),
    "goalConversionPercentage": ("goals", "totalShots"),
}

GOALKEEPER_ALLOWED_STATS = {
    "saves",
    "goalsPrevented",
    "goalsConceded",
    "highClaims",
    "punches",
    "successfulRunsOut",
    "cleanSheet",
    "ownGoals",
    "penaltyConceded",
    "errorLeadToGoal",
    "errorLeadToShot",
    "accuratePassesPercentage",
}
GOALKEEPER_EXCLUSIVE = {
    "saves",
    "goalsPrevented",
    "highClaims",
    "punches",
    "successfulRunsOut",
}

# Misma conversión ordinal que utiliza el modelo adjunto.
RESULT_ORDINAL_MAP = {
    "Ganador": 1.00,
    "Final": 0.90,
    "Partido por el tercer puesto": 0.80,
    "Semifinales": 0.80,
    "Cuartos de final": 0.70,
    "Octavos de final": 0.60,
    "Dieciseisavos de final": 0.50,
    "Ronda intermedia": 0.40,
    "Repechaje": 0.40,
    "Fase de Grupos": 0.30,
    "7ª Ronda": 0.20,
    "6ª Ronda": 0.20,
    "5ª Ronda": 0.10,
    "4ª Ronda": 0.10,
    "3ª Ronda": 0.05,
    "2ª Ronda": 0.05,
    "1": 1.00,
    "2": 0.90,
    "3": 0.80,
    "4": 0.70,
    "5": 0.60,
    "6": 0.50,
    "7": 0.40,
    "8": 0.30,
    "9": 0.20,
    "10": 0.10,
    "11": 0.075,
    "17": 0.05,
    "20": 0.025,
}

# Clasificación final proporcionada para 2025/26. Se aplica al puesto real
# del dataset para poder evaluar y comparar el ranking estadístico.
# Los nombres se reconocen de forma flexible, ignorando mayúsculas y tildes.
OFFICIAL_2025_26_RESULTS = [
    (1, ("kane",)),
    (2, ("mbappe",)),
    (3, ("lamine",)),
    (4, ("rodri",)),
    (5, ("messi",)),
    (6, ("kvaratskhelia",)),
    (7, ("dembele",)),
    (8, ("haaland",)),
    (9, ("olise",)),
    (10, ("fabian",)),
    (11, ("bellingham",)),
    (12, ("cubarsi",)),
    (13, ("achraf",)),
    (14, ("vitinha",)),
    (15, ("declan rice",)),
    (16, ("ferran",)),
    (17, ("vinicius",)),
    (18, ("cucurella",)),
    (19, ("joao neves",)),
    (20, ("pacho",)),
    (21, ("gabriel magalhaes",)),
    (22, ("lautaro",)),
    (23, ("nuno mendes",)),
    (24, ("luis diaz",)),
    (25, ("marquinhos",)),
    (26, ("upamecano",)),
    (27, ("bruno fernandes",)),
    (28, ("quinones",)),
    (29, ("saliba",)),
    (30, ("mane",)),
]


def official_2025_26_position(player_name: Any) -> float:
    """Devuelve el puesto oficial 2025/26 por nombre, o NaN si no está en el Top 30."""
    normalized = re.sub(r"[^a-z0-9]+", " ", fold_text(player_name)).strip()
    padded = f" {normalized} "

    # Estos dos nombres cortos son ambiguos, así que solo se aceptan con apellido
    # o cuando el valor del dataset contiene únicamente el nombre.
    if " gabriel magalhaes " in padded or normalized == "gabriel":
        return 21.0
    if " bruno fernandes " in padded or normalized == "bruno":
        return 27.0

    for rank, aliases in OFFICIAL_2025_26_RESULTS:
        for alias in aliases:
            alias_norm = re.sub(r"[^a-z0-9]+", " ", fold_text(alias)).strip()
            if alias_norm and f" {alias_norm} " in padded:
                return float(rank)
    return np.nan


def apply_official_2025_26_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Sobrescribe la posición real 2025/26 para los nombres del Top 30 indicado."""
    df = df.copy()
    if not {"temporada", "nombre", "posicion_balon_oro"}.issubset(df.columns):
        return df
    mask = df["temporada"].map(season_norm).eq("2025/26")
    if mask.any():
        official = df.loc[mask, "nombre"].map(official_2025_26_position)
        matched = official.notna()
        matched_idx = official.index[matched]
        df.loc[matched_idx, "posicion_balon_oro"] = official.loc[matched_idx]
    return df


# Columnas que se muestran en las fichas y en el comparador, además de las
# variables exactas de puntuación. Se usan solo si existen en el dataset.
DISPLAY_EXTRA_STATS = [
    # Goals y assists se muestran en las fichas; xG y xA se excluyen de la vista.
    "goals", "assists", "appearances", "minutesPlayed",
]

IDENTITY_COLUMNS = [
    "temporada", "nombre", "club", "posicion", "nacionalidad",
    "id_sofascore", "posicion_balon_oro", "dataset",
]

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
      .small-note {opacity: .75; font-size: .9rem;}
      .ranking-card {
        border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
        padding: 13px 16px; margin: 7px 0;
        background: rgba(128,128,128,.045);
      }
      .ranking-card-top {
        display: grid; grid-template-columns: 54px minmax(160px,1fr) 130px 130px;
        gap: 12px; align-items: center;
      }
      .ranking-rank {font-size: 1.3rem; font-weight: 750; text-align:center;}
      .ranking-player-name {font-size: 1.03rem; font-weight: 650; line-height:1.25;}
      .ranking-player-meta, .ranking-label {opacity:.68; font-size:.78rem; line-height:1.35;}
      .ranking-top-value {text-align:right; overflow-wrap:anywhere;}
      .ranking-top-value strong {display:block; font-size:.94rem; font-variant-numeric:tabular-nums;}
      .ranking-card-metrics {
        display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px;
        margin:12px 0 0 66px; padding-top:10px;
        border-top:1px solid rgba(128,128,128,.16);
      }
      .ranking-metric strong {display:block; margin-top:2px; font-size:.9rem; overflow-wrap:anywhere;}
      @media(max-width:700px) {
        .ranking-card-top {grid-template-columns:40px minmax(0,1fr) 85px; gap:8px;}
        .ranking-real-position {display:none;}
        .ranking-card-metrics {grid-template-columns:repeat(2,minmax(0,1fr)); margin-left:48px;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Balón de Oro · Puntuador estadístico")
st.caption(
    "Ranking transparente basado en z-scores y pesos configurables por posición y competición. "
    "No es un modelo de Machine Learning: las puntuaciones se calculan mediante sumas ponderadas."
)


# ============================================================
# UTILIDADES DE CARGA Y NORMALIZACIÓN
# ============================================================

def season_norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("-", "/").replace("_", "/")
    match = re.match(r"^(\d{4})/(\d{2}|\d{4})$", text)
    if match:
        return f"{match.group(1)}/{match.group(2)[-2:]}"
    return text


def season_order(value: Any) -> int:
    match = re.match(r"^\s*(\d{4})", str(value))
    return int(match.group(1)) if match else 10**9


def fold_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).casefold().strip()


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).replace("**", "").strip() for column in frame.columns]
    return frame.loc[:, ~pd.Index(frame.columns).duplicated()]


@st.cache_data(show_spinner="Cargando y preparando el dataset...")
def load_dataset(path_string: str, file_version: str) -> pd.DataFrame:
    del file_version  # forma parte de la clave de caché para detectar cambios del archivo
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {path}")
    try:
        raw = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = clean_columns(raw)

    required = {"temporada", "nombre", "competicion", "posicion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el dataset: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["temporada", "nombre", "competicion", "posicion"]).copy()
    df["temporada"] = df["temporada"].map(season_norm)
    for column in ["nombre", "club", "posicion", "competicion", "nacionalidad", "dataset"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    if "club" not in df:
        df["club"] = ""
    if "posicion_balon_oro" in df.columns:
        df["posicion_balon_oro"] = pd.to_numeric(df["posicion_balon_oro"], errors="coerce")
    else:
        df["posicion_balon_oro"] = np.nan

    # En el dataset del modelo, los puestos <= 0 representan ranking aún desconocido.
    df.loc[df["posicion_balon_oro"].le(0), "posicion_balon_oro"] = np.nan

    # Se incorporan los puestos finales 1–30 facilitados para la temporada 2025/26.
    df = apply_official_2025_26_positions(df)

    candidate_numeric = set(ADDITIVE_STATS_REQUESTED) | set(PERCENT_PAIRS) | set(DISPLAY_EXTRA_STATS)
    candidate_numeric |= {column for pair in PERCENT_PAIRS.values() for column in pair}
    candidate_numeric |= {"totalPasses", "totalDribbles", "totalAerialDuels", "totalDuels", "totalTackles", "totalShots"}
    for column in candidate_numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "goalsAssistsSum" not in df.columns and {"goals", "assists"}.issubset(df.columns):
        df["goalsAssistsSum"] = df[["goals", "assists"]].sum(axis=1, min_count=1)

    # Inferir denominadores ausentes cuando existen numerador y porcentaje, como en el ML.
    for pct, (num, den) in PERCENT_PAIRS.items():
        if num not in df.columns or pct not in df.columns:
            continue
        percentage = pd.to_numeric(df[pct], errors="coerce")
        valid_pct = percentage.dropna()
        if not valid_pct.empty and valid_pct.max() <= 1.0:
            percentage = percentage * 100.0
        numerator = pd.to_numeric(df[num], errors="coerce")
        inferred = numerator / (percentage / 100.0).where(percentage > 0)
        zero_mask = numerator.eq(0) & percentage.eq(0) & numerator.notna() & percentage.notna()
        inferred = inferred.mask(zero_mask, 0.0)
        if den not in df.columns:
            df[den] = inferred
        else:
            df[den] = pd.to_numeric(df[den], errors="coerce").fillna(inferred)

    df["__resultado_ordinal"] = np.nan
    if "resultado_competicion" in df.columns:
        result_text = df["resultado_competicion"].astype(str).str.strip()
        df["__resultado_ordinal"] = result_text.map(RESULT_ORDINAL_MAP).astype(float)

    # No se normalizan ni agrupan los nombres de las competiciones: cada valor original
    # de 'competicion' conserva su peso individual.
    return df.reset_index(drop=True)


# ============================================================
# FEATURES EXACTAS DEL MODELO + ESTADÍSTICAS PARA FICHA
# ============================================================

def available_model_stats(df: pd.DataFrame) -> list[str]:
    additive = [stat for stat in ADDITIVE_STATS_REQUESTED if stat in df.columns]
    percentages = [
        pct for pct, (num, den) in PERCENT_PAIRS.items()
        if num in df.columns and den in df.columns
    ]
    rates = ["goalsAssistsPerMins"] if {"goalsAssistsSum", "minutesPlayed"}.issubset(df.columns) else []
    return list(dict.fromkeys(additive + percentages + rates + ["resultado_competicion"]))


def stat_sign(stat: str) -> int:
    return -1 if stat in NEGATIVE_STATS else 1


def stats_allowed_for_position(position: str, stats: list[str]) -> list[str]:
    if position == "Goalkeeper":
        return [stat for stat in stats if stat == "resultado_competicion" or stat in GOALKEEPER_ALLOWED_STATS]
    # Los cinco indicadores exclusivos de portero no puntúan para jugadores de campo.
    return [stat for stat in stats if stat not in GOALKEEPER_EXCLUSIVE]


@st.cache_data(show_spinner=False)
def build_player_competition_table(df: pd.DataFrame, model_stats: list[str]) -> pd.DataFrame:
    keys = ["temporada", "nombre", "competicion"]
    group = df.groupby(keys, sort=False, dropna=False)
    result = group.size().rename("__filas_fuente").to_frame()

    for col in ["club", "posicion", "nacionalidad", "id_sofascore", "dataset", "posicion_balon_oro"]:
        if col in df.columns:
            result[col] = group[col].first()

    # Columnas acumulables necesarias para sumar los indicadores y reconstruir porcentajes.
    additive_source_cols = set(ADDITIVE_STATS_REQUESTED) | set(DISPLAY_EXTRA_STATS)
    additive_source_cols |= {column for pair in PERCENT_PAIRS.values() for column in pair}
    additive_source_cols |= {"totalPasses", "totalDribbles", "totalAerialDuels", "totalDuels", "totalTackles", "totalShots"}
    for col in sorted(additive_source_cols):
        if col in df.columns:
            result[col] = group[col].sum(min_count=1)

    # Porcentajes ponderados por su denominador agregado, nunca medias simples de porcentajes.
    for pct, (num, den) in PERCENT_PAIRS.items():
        if num in result.columns and den in result.columns:
            numerator = pd.to_numeric(result[num], errors="coerce")
            denominator = pd.to_numeric(result[den], errors="coerce")
            result[pct] = np.where(
                denominator > 0,
                100.0 * numerator / denominator,
                np.where(denominator.eq(0), 0.0, np.nan),
            )

    if "goalsAssistsSum" not in result.columns and {"goals", "assists"}.issubset(result.columns):
        result["goalsAssistsSum"] = result[["goals", "assists"]].sum(axis=1, min_count=1)
    if {"goalsAssistsSum", "minutesPlayed"}.issubset(result.columns):
        minutes = pd.to_numeric(result["minutesPlayed"], errors="coerce").replace(0, np.nan)
        result["goalsAssistsPerMins"] = pd.to_numeric(result["goalsAssistsSum"], errors="coerce") / minutes

    if "__resultado_ordinal" in df.columns:
        result["resultado_competicion"] = group["__resultado_ordinal"].max()
    else:
        result["resultado_competicion"] = np.nan

    result = result.reset_index()
    if "club" not in result.columns:
        result["club"] = ""
    if "posicion" not in result.columns:
        result["posicion"] = ""
    if "posicion_balon_oro" not in result.columns:
        result["posicion_balon_oro"] = np.nan
    result["posicion_balon_oro"] = pd.to_numeric(result["posicion_balon_oro"], errors="coerce")

    # Aplicar la restricción del script ML: los porteros solo conservan su lista permitida.
    gk_mask = result["posicion"].astype(str).eq("Goalkeeper")
    for stat in model_stats:
        if stat == "resultado_competicion" or stat in GOALKEEPER_ALLOWED_STATS:
            continue
        if stat in result.columns:
            result.loc[gk_mask, stat] = np.nan

    # Para las otras posiciones no se calculan las cinco estadísticas exclusivas de portero.
    outfield_mask = ~gk_mask
    for stat in GOALKEEPER_EXCLUSIVE:
        if stat in result.columns:
            result.loc[outfield_mask, stat] = np.nan

    return result


@st.cache_data(show_spinner=False)
def calculate_zscores(player_comp: pd.DataFrame, model_stats: list[str]) -> pd.DataFrame:
    """Z-score por temporada, competición individual y estadística observada.

    Los jugadores que no participaron en una competición no se crean como filas artificiales:
    su contribución a esa competición es cero. Los NaN individuales también reciben z=0
    (neutro), pero se excluyen de la media y desviación estándar.
    """
    data = player_comp.copy()
    for stat in model_stats:
        data[f"z__{stat}"] = 0.0

    for (_, _), indices in data.groupby(["temporada", "competicion"], sort=False).groups.items():
        idx = list(indices)
        sub = data.loc[idx]
        for stat in model_stats:
            if stat not in data.columns:
                continue
            values = pd.to_numeric(sub[stat], errors="coerce")
            if stat in GOALKEEPER_EXCLUSIVE:
                # Referencia entre porteros observados, sin incluir a jugadores de campo como ceros.
                eligible = sub["posicion"].astype(str).eq("Goalkeeper") & values.notna()
            else:
                eligible = values.notna()
            observed = values.loc[eligible]
            if observed.empty:
                continue
            mean = float(observed.mean())
            std = float(observed.std(ddof=0))
            if np.isfinite(std) and std > 1e-12:
                valid_values = values.notna()
                data.loc[sub.index[valid_values.to_numpy()], f"z__{stat}"] = (
                    (values.loc[valid_values] - mean) / std
                ).to_numpy(dtype=float)
            # Si no hay variación, todos conservan z=0.
    return data


# ============================================================
# PESOS: ESTADÍSTICAS Y COMPETICIONES POR POSICIÓN
# ============================================================

def default_weight_maps(positions: list[str], stats: list[str], competitions: list[str]):
    """Pesos iniciales (1.0) para cada posición, estadística y competición."""
    stat_weights = {
        position: {stat: 1.0 for stat in stats_allowed_for_position(position, stats)}
        for position in positions
    }
    competition_weights = {
        position: {competition: 1.0 for competition in competitions}
        for position in positions
    }
    return stat_weights, competition_weights


def clamp_weight(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return float(np.clip(number, 0.0, 10.0))


def create_default_weight_configs(
    seasons: list[str], positions: list[str], stats: list[str], competitions: list[str]
) -> dict[str, Any]:
    stat_defaults, comp_defaults = default_weight_maps(positions, stats, competitions)
    return {
        "version": 2,
        "stat_weights_by_season": {
            season: copy.deepcopy(stat_defaults) for season in seasons
        },
        "competition_weights_by_season": {
            season: copy.deepcopy(comp_defaults) for season in seasons
        },
    }


def load_weight_configs(
    seasons: list[str], positions: list[str], stats: list[str], competitions: list[str]
) -> dict[str, Any]:
    """Carga los pesos por temporada y migra el antiguo JSON global si lo encuentra."""
    configs = create_default_weight_configs(seasons, positions, stats, competitions)
    if not WEIGHTS_PATH.exists():
        return configs

    try:
        saved = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.warning("No se pudo leer el JSON de pesos guardado. Se usarán pesos iniciales de 1.")
        return configs

    saved_stats_by_season = saved.get("stat_weights_by_season", {})
    saved_comps_by_season = saved.get("competition_weights_by_season", {})

    # Compatibilidad: versión antigua con una sola configuración para todas las temporadas.
    legacy_stats = saved.get("stat_weights", {})
    legacy_comps = saved.get("competition_weights", {})

    for season in seasons:
        season_stats = saved_stats_by_season.get(season, {})
        season_comps = saved_comps_by_season.get(season, {})
        for position in positions:
            allowed = stats_allowed_for_position(position, stats)
            stats_source = season_stats.get(position, {})
            comps_source = season_comps.get(position, {})
            if not stats_source and legacy_stats:
                stats_source = legacy_stats.get(position, {})
            if not comps_source and legacy_comps:
                comps_source = legacy_comps.get(position, {})
            configs["stat_weights_by_season"][season][position] = {
                stat: clamp_weight(stats_source.get(stat, 1.0)) for stat in allowed
            }
            configs["competition_weights_by_season"][season][position] = {
                comp: clamp_weight(comps_source.get(comp, 1.0)) for comp in competitions
            }
    return configs


def save_weight_configs(configs: dict[str, Any]) -> None:
    payload = {
        "version": 2,
        "description": (
            "Pesos manuales independientes por temporada, posición, estadística y competición. "
            "No son coeficientes aprendidos por Machine Learning."
        ),
        "stat_weights_by_season": configs.get("stat_weights_by_season", {}),
        "competition_weights_by_season": configs.get("competition_weights_by_season", {}),
    }
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def autosave_weight_configs(configs: dict[str, Any]) -> None:
    """Guarda cambios aplicados sin romper la app si el sistema de archivos no es escribible."""
    try:
        save_weight_configs(configs)
        st.session_state["puntuador_weights_save_error"] = ""
        st.session_state["puntuador_weights_saved_notice"] = True
    except OSError as exc:
        st.session_state["puntuador_weights_save_error"] = str(exc)


def stat_weight_widget_key(season: str, position: str, stat: str) -> str:
    version = int(st.session_state.get("puntuador_weight_widget_versions", {}).get(season, 0))
    return f"wstat::{season}::{position}::{stat}::v{version}"


def competition_weight_widget_key(season: str, position: str, competition: str) -> str:
    version = int(st.session_state.get("puntuador_weight_widget_versions", {}).get(season, 0))
    return f"wcomp::{season}::{position}::{competition}::v{version}"


def weight_configs_for_season(season: str) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    configs = st.session_state["puntuador_weight_configs"]
    stat_by_season = configs["stat_weights_by_season"]
    comp_by_season = configs["competition_weights_by_season"]
    return (
        copy.deepcopy(stat_by_season.get(season, {})),
        copy.deepcopy(comp_by_season.get(season, {})),
    )


def apply_weight_inputs(season: str, position: str, stats: list[str], competitions: list[str]) -> None:
    """Aplica los spin buttons de una única posición para reducir widgets por rerun."""
    configs = copy.deepcopy(st.session_state["puntuador_weight_configs"])
    stat_season = configs["stat_weights_by_season"].setdefault(season, {})
    comp_season = configs["competition_weights_by_season"].setdefault(season, {})
    stat_season.setdefault(position, {})
    comp_season.setdefault(position, {})
    for stat in stats_allowed_for_position(position, stats):
        key = stat_weight_widget_key(season, position, stat)
        old_value = stat_season[position].get(stat, 1.0)
        stat_season[position][stat] = clamp_weight(st.session_state.get(key, old_value))
    for competition in competitions:
        key = competition_weight_widget_key(season, position, competition)
        old_value = comp_season[position].get(competition, 1.0)
        comp_season[position][competition] = clamp_weight(st.session_state.get(key, old_value))

    st.session_state["puntuador_weight_configs"] = configs
    autosave_weight_configs(configs)
    st.session_state["puntuador_weights_applied_notice"] = season


def copy_weights_between_seasons(
    source_season: str, target_season: str, positions: list[str], stats: list[str], competitions: list[str]
) -> None:
    """Copia todas las ponderaciones de una temporada a otra y recrea sus widgets."""
    if source_season == target_season:
        return
    configs = copy.deepcopy(st.session_state["puntuador_weight_configs"])
    source_stat = configs["stat_weights_by_season"].get(source_season, {})
    source_comp = configs["competition_weights_by_season"].get(source_season, {})
    target_stat, target_comp = default_weight_maps(positions, stats, competitions)
    for position in positions:
        target_stat[position] = {
            stat: clamp_weight(source_stat.get(position, {}).get(stat, 1.0))
            for stat in stats_allowed_for_position(position, stats)
        }
        target_comp[position] = {
            comp: clamp_weight(source_comp.get(position, {}).get(comp, 1.0))
            for comp in competitions
        }
    configs["stat_weights_by_season"][target_season] = target_stat
    configs["competition_weights_by_season"][target_season] = target_comp
    st.session_state["puntuador_weight_configs"] = configs
    autosave_weight_configs(configs)
    versions = dict(st.session_state.get("puntuador_weight_widget_versions", {}))
    versions[target_season] = int(versions.get(target_season, 0)) + 1
    st.session_state["puntuador_weight_widget_versions"] = versions
    st.session_state["puntuador_weights_applied_notice"] = target_season
    st.session_state["puntuador_weights_copy_notice"] = (source_season, target_season)


def reset_season_weights(season: str, positions: list[str], stats: list[str], competitions: list[str]) -> None:
    configs = copy.deepcopy(st.session_state["puntuador_weight_configs"])
    stat_defaults, comp_defaults = default_weight_maps(positions, stats, competitions)
    configs["stat_weights_by_season"][season] = stat_defaults
    configs["competition_weights_by_season"][season] = comp_defaults
    st.session_state["puntuador_weight_configs"] = configs
    autosave_weight_configs(configs)
    versions = dict(st.session_state.get("puntuador_weight_widget_versions", {}))
    versions[season] = int(versions.get(season, 0)) + 1
    st.session_state["puntuador_weight_widget_versions"] = versions
    st.session_state["puntuador_weights_applied_notice"] = season


def calculate_ranking(
    zdata: pd.DataFrame,
    season: str,
    model_stats: list[str],
    positions: list[str],
    stat_weights: dict[str, dict[str, float]],
    comp_weights: dict[str, dict[str, float]],
    include_detail: bool = False,
):
    """Calcula el ranking mediante grupos vectorizados por competición y posición.

    Por defecto solo calcula las puntuaciones. El desglose completo por estadística
    se genera únicamente cuando se solicita expresamente para descargarlo.
    """
    season_data = zdata.loc[zdata["temporada"].eq(season)].copy().reset_index(drop=True)
    if season_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "__player_key" not in season_data.columns:
        season_data["__player_key"] = season_data["nombre"].map(fold_text)
    identity_cols = [
        c for c in ["temporada", "nombre", "club", "posicion", "nacionalidad", "posicion_balon_oro"]
        if c in season_data.columns
    ]
    candidates = season_data[identity_cols].drop_duplicates(["temporada", "nombre"]).copy().reset_index(drop=True)
    candidates["__player_key"] = candidates["nombre"].map(fold_text)
    score_by_player: dict[str, float] = {}
    detail_parts: list[pd.DataFrame] = []

    grouped = season_data.groupby(["competicion", "posicion"], sort=False, dropna=False).indices
    for (competition, position), row_indices in grouped.items():
        competition = str(competition)
        position = str(position)
        sub = season_data.iloc[np.asarray(row_indices, dtype=int)].reset_index(drop=True)
        allowed_stats = stats_allowed_for_position(position, model_stats)
        if not allowed_stats:
            continue

        active_stats = [
            stat for stat in allowed_stats
            if stat in sub.columns and pd.to_numeric(sub[stat], errors="coerce").notna().any()
        ]
        active_set = set(active_stats)
        weights_map = stat_weights.get(position, {})
        stat_weight_vector = np.asarray(
            [max(0.0, float(weights_map.get(stat, 1.0))) for stat in allowed_stats],
            dtype=np.float64,
        )
        active_mask = np.asarray([stat in active_set for stat in allowed_stats], dtype=bool)
        denominator = float(stat_weight_vector[active_mask].sum()) if active_mask.any() else 0.0
        signs = np.asarray([stat_sign(stat) for stat in allowed_stats], dtype=np.float64)
        z_columns = [f"z__{stat}" for stat in allowed_stats]
        z_matrix = np.column_stack([
            pd.to_numeric(sub[z_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if z_col in sub.columns else np.zeros(len(sub), dtype=np.float64)
            for z_col in z_columns
        ])

        if denominator > 1e-12:
            contributions_before_comp = (
                z_matrix * signs[None, :] * stat_weight_vector[None, :] / denominator
            )
            contributions_before_comp[:, ~active_mask] = 0.0
        else:
            contributions_before_comp = np.zeros_like(z_matrix, dtype=np.float64)

        competition_scores = contributions_before_comp.sum(axis=1)
        comp_weight = max(0.0, float(comp_weights.get(position, {}).get(competition, 1.0)))
        competition_contributions = competition_scores * comp_weight
        group_scores = pd.DataFrame({
            "__player_key": sub["__player_key"].astype(str).to_numpy(),
            "score": competition_contributions,
        }).groupby("__player_key", sort=False)["score"].sum()
        for player_key, player_score in group_scores.items():
            score_by_player[player_key] = score_by_player.get(player_key, 0.0) + float(player_score)

        if include_detail:
            n_rows, n_stats = z_matrix.shape
            raw_matrix = np.column_stack([
                pd.to_numeric(sub[stat], errors="coerce").to_numpy(dtype=np.float64)
                if stat in sub.columns else np.full(n_rows, np.nan, dtype=np.float64)
                for stat in allowed_stats
            ])
            detail_parts.append(pd.DataFrame({
                "temporada": season,
                "nombre": np.repeat(sub["nombre"].astype(str).to_numpy(), n_stats),
                "posicion": position,
                "competicion": competition,
                "estadistica": np.tile(np.asarray(allowed_stats, dtype=object), n_rows),
                "valor_agregado": raw_matrix.reshape(-1),
                "zscore": z_matrix.reshape(-1),
                "orientacion": np.tile(np.where(signs < 0, "Negativa", "Positiva"), n_rows),
                "peso_estadistica": np.tile(stat_weight_vector, n_rows),
                "contribucion_antes_peso_competicion": contributions_before_comp.reshape(-1),
                "peso_competicion": comp_weight,
                "puntuacion_competicion": np.repeat(competition_scores, n_stats),
                "contribucion_competicion": np.repeat(competition_contributions, n_stats),
                "contribucion_final": (contributions_before_comp * comp_weight).reshape(-1),
            }))

    candidates["score_puntuacion"] = candidates["__player_key"].map(score_by_player).fillna(0.0)
    candidates = candidates.drop(columns=["__player_key"])
    candidates = candidates.sort_values(
        ["score_puntuacion", "nombre"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    candidates["posicion_predicha"] = np.arange(1, len(candidates) + 1)
    candidates = candidates.rename(columns={"posicion_balon_oro": "posicion_real"})
    ranking_cols = [
        c for c in ["temporada", "posicion_predicha", "posicion_real", "nombre", "club", "posicion", "score_puntuacion"]
        if c in candidates.columns
    ]
    candidates = candidates[ranking_cols]
    detail_df = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return candidates, detail_df


def weight_signature(stat_weights: dict, comp_weights: dict) -> str:
    """Firma compacta para reutilizar rankings cuando no cambian los pesos."""
    payload = json.dumps(
        {"stats": stat_weights, "competitions": comp_weights},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def get_cached_ranking(
    zdata: pd.DataFrame,
    season: str,
    model_stats: list[str],
    positions: list[str],
    stat_weights: dict,
    comp_weights: dict,
    data_signature: str,
) -> pd.DataFrame:
    """Caché por sesión: evita volver a hashear el DataFrame completo en cada rerun."""
    cache = st.session_state.setdefault("puntuador_ranking_cache", {})
    key = (data_signature, season, weight_signature(stat_weights, comp_weights))
    if key not in cache:
        ranking, _ = calculate_ranking(
            zdata, season, model_stats, positions, stat_weights, comp_weights, include_detail=False
        )
        cache[key] = ranking
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
    return cache[key]


def calculate_player_score_detail(
    zdata: pd.DataFrame,
    season: str,
    player_name: str,
    model_stats: list[str],
    stat_weights: dict[str, dict[str, float]],
    comp_weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Desglose de contribuciones de un jugador; no calcula detalles de todo el ranking."""
    season_data = zdata.loc[zdata["temporada"].eq(season)].copy()
    if season_data.empty:
        return pd.DataFrame()
    if "__player_key" not in season_data.columns:
        season_data["__player_key"] = season_data["nombre"].map(fold_text)
    player_key = fold_text(player_name)
    player_rows = season_data.loc[season_data["__player_key"].eq(player_key)]
    if player_rows.empty:
        return pd.DataFrame()

    # Índice de población para identificar variables observadas por competición y posición.
    populations = {
        (str(comp), str(pos)): group
        for (comp, pos), group in season_data.groupby(["competicion", "posicion"], sort=False, dropna=False)
    }
    detail_rows = []
    for row in player_rows.itertuples(index=False):
        row_data = row._asdict()
        competition = str(row_data.get("competicion", ""))
        position = str(row_data.get("posicion", ""))
        allowed_stats = stats_allowed_for_position(position, model_stats)
        population = populations.get((competition, position))
        if population is None or not allowed_stats:
            continue
        active_stats = [
            stat for stat in allowed_stats
            if stat in population.columns and pd.to_numeric(population[stat], errors="coerce").notna().any()
        ]
        active = set(active_stats)
        weights = stat_weights.get(position, {})
        denominator = sum(max(0.0, float(weights.get(stat, 1.0))) for stat in active_stats)
        comp_weight = max(0.0, float(comp_weights.get(position, {}).get(competition, 1.0)))
        contributions = []
        stat_values = []
        for stat in allowed_stats:
            try:
                weight = max(0.0, float(weights.get(stat, 1.0)))
            except (TypeError, ValueError):
                weight = 1.0
            sign = stat_sign(stat)
            z_value = row_data.get(f"z__{stat}", 0.0)
            z_value = float(z_value) if pd.notna(z_value) and np.isfinite(float(z_value)) else 0.0
            raw_value = row_data.get(stat, np.nan)
            contribution = sign * z_value * weight / denominator if denominator > 1e-12 and stat in active else 0.0
            contributions.append(contribution)
            stat_values.append({
                "temporada": season,
                "nombre": player_name,
                "posicion": position,
                "competicion": competition,
                "estadistica": stat,
                "valor_agregado": raw_value,
                "zscore": z_value,
                "orientacion": "Negativa" if sign < 0 else "Positiva",
                "peso_estadistica": weight,
                "contribucion_antes_peso_competicion": contribution,
            })
        competition_score = float(sum(contributions))
        competition_contribution = competition_score * comp_weight
        for item in stat_values:
            item["peso_competicion"] = comp_weight
            item["puntuacion_competicion"] = competition_score
            item["contribucion_competicion"] = competition_contribution
            item["contribucion_final"] = item["contribucion_antes_peso_competicion"] * comp_weight
        detail_rows.extend(stat_values)
    return pd.DataFrame(detail_rows)


def get_cached_player_detail(
    zdata: pd.DataFrame,
    season: str,
    player_name: str,
    model_stats: list[str],
    stat_weights: dict,
    comp_weights: dict,
    data_signature: str,
) -> pd.DataFrame:
    cache = st.session_state.setdefault("puntuador_player_detail_cache", {})
    key = (data_signature, season, fold_text(player_name), weight_signature(stat_weights, comp_weights))
    if key not in cache:
        cache[key] = calculate_player_score_detail(
            zdata, season, player_name, model_stats, stat_weights, comp_weights
        )
        while len(cache) > 24:
            cache.pop(next(iter(cache)))
    return cache[key]


# ============================================================
# ESTADÍSTICAS DE FICHA Y COMPARADOR
# ============================================================

def find_player_rows(raw_df: pd.DataFrame, season: str, name: str) -> pd.DataFrame:
    """Busca por un índice creado al cargar el dataset, evitando recorrerlo entero."""
    lookup = st.session_state.get("puntuador_raw_row_indices", {})
    indices = lookup.get((season, fold_text(name)))
    if indices is not None:
        return raw_df.iloc[indices].copy()
    mask = raw_df["temporada"].eq(season) & raw_df["nombre"].map(fold_text).eq(fold_text(name))
    return raw_df.loc[mask].copy()


def find_player_comp_rows(player_comp: pd.DataFrame, season: str, name: str) -> pd.DataFrame:
    """Busca la tabla ya agregada a jugador-temporada-competición."""
    lookup = st.session_state.get("puntuador_comp_row_indices", {})
    indices = lookup.get((season, fold_text(name)))
    if indices is not None:
        return player_comp.iloc[indices].copy()
    mask = player_comp["temporada"].eq(season) & player_comp["nombre"].map(fold_text).eq(fold_text(name))
    return player_comp.loc[mask].copy()


def safe_num_sum(rows: pd.DataFrame, column: str):
    if column not in rows.columns:
        return np.nan
    return pd.to_numeric(rows[column], errors="coerce").sum(min_count=1)


def percentage_from_rows(rows: pd.DataFrame, numerator: str, denominator: str):
    if numerator not in rows.columns or denominator not in rows.columns:
        return np.nan
    num = safe_num_sum(rows, numerator)
    den = safe_num_sum(rows, denominator)
    if pd.isna(num) or pd.isna(den):
        return np.nan
    if den > 0:
        return 100.0 * num / den
    if den == 0:
        return 0.0
    return np.nan


def aggregate_display_stats(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {}
    values: dict[str, Any] = {}
    display_sum_stats = list(dict.fromkeys(ADDITIVE_STATS_REQUESTED + DISPLAY_EXTRA_STATS))
    # Mostrar columnas numéricas de estadísticas que existan en los datos.
    for stat in display_sum_stats:
        if stat in rows.columns:
            values[stat] = safe_num_sum(rows, stat)
    for pct, (num, den) in PERCENT_PAIRS.items():
        if num in rows.columns and den in rows.columns:
            values[pct] = percentage_from_rows(rows, num, den)
    if "goalsAssistsSum" in rows.columns and "minutesPlayed" in rows.columns:
        ga = safe_num_sum(rows, "goalsAssistsSum")
        minutes = safe_num_sum(rows, "minutesPlayed")
        values["goalsAssistsPerMins"] = ga / minutes if pd.notna(ga) and pd.notna(minutes) and minutes else np.nan
    elif {"goals", "assists", "minutesPlayed"}.issubset(rows.columns):
        goals = safe_num_sum(rows, "goals")
        assists = safe_num_sum(rows, "assists")
        minutes = safe_num_sum(rows, "minutesPlayed")
        values["goalsAssistsPerMins"] = (goals + assists) / minutes if pd.notna(goals) and pd.notna(assists) and pd.notna(minutes) and minutes else np.nan
    return values


def safe_filename(value: Any) -> str:
    return re.sub(r"\W+", "_", str(value)).strip("_") or "datos"


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}".replace(",", "X").replace(".", ",").replace("X", ".") if abs(value) < 100 else f"{float(value):,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)


def display_stats_table(rows: pd.DataFrame, title: str, key: str, include_result: bool = True):
    vals = aggregate_display_stats(rows)
    table_rows = []
    for stat, value in vals.items():
        if pd.notna(value):
            table_rows.append({"Estadística": stat, "Valor": format_value(value)})
    if include_result and "resultado_competicion" in rows.columns:
        result_values = rows["resultado_competicion"].dropna().astype(str).str.strip()
        result_values = list(dict.fromkeys(v for v in result_values if v and v.casefold() != "nan"))
        if result_values:
            table_rows.append({"Estadística": "resultado_competicion", "Valor": " · ".join(result_values)})
    st.markdown(f"**{title}**")
    if not table_rows:
        st.info("No hay estadísticas observadas para mostrar.")
        return
    table = pd.DataFrame(table_rows)
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(620, 40 + len(table) * 34))
    st.download_button(
        "Descargar estadísticas CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"estadisticas_{safe_filename(title)}.csv",
        mime="text/csv",
        key=key,
    )


def show_player_detail(raw_df: pd.DataFrame, season: str, name: str, detail_df: pd.DataFrame, key_prefix: str):
    rows = find_player_rows(raw_df, season, name)
    if rows.empty:
        st.info("No se han encontrado filas del jugador en el dataset original.")
        return
    competitions = sorted(rows["competicion"].dropna().astype(str).unique().tolist())
    tabs = st.tabs(["🌐 Global"] + competitions)
    with tabs[0]:
        st.caption("Los porcentajes se recalculan con numeradores y denominadores agregados; resultado_competicion se muestra por torneo.")
        display_stats_table(rows, f"Estadísticas globales · {name}", f"{key_prefix}_global")
        if not detail_df.empty:
            comp_score = detail_df[["competicion", "peso_competicion", "puntuacion_competicion", "contribucion_competicion"]].drop_duplicates()
            st.markdown("**Contribución de cada competición a la puntuación**")
            st.dataframe(comp_score.sort_values("contribucion_competicion", ascending=False), use_container_width=True, hide_index=True)
    for idx, competition in enumerate(competitions, start=1):
        with tabs[idx]:
            subset = rows[rows["competicion"].astype(str).eq(competition)]
            display_stats_table(subset, f"Estadísticas · {competition}", f"{key_prefix}_comp_{idx}")
            player_comp_detail = detail_df[detail_df["competicion"].eq(competition)] if not detail_df.empty else pd.DataFrame()
            if not player_comp_detail.empty:
                detail_cols = [
                    "estadistica", "valor_agregado", "zscore", "orientacion", "peso_estadistica", "contribucion_final"
                ]
                st.markdown("**Desglose de la puntuación en esta competición**")
                shown = player_comp_detail[detail_cols].rename(columns={
                    "estadistica": "Estadística", "valor_agregado": "Valor", "zscore": "Z-score",
                    "orientacion": "Orientación", "peso_estadistica": "Peso estadístico",
                    "contribucion_final": "Contribución final",
                })
                st.dataframe(shown, use_container_width=True, hide_index=True)


def toggle_player_profile(season: str, player_key: str) -> None:
    """Abre o cierra la ficha solicitada antes del rerun, sin renderizar fichas anteriores."""
    state_key = f"puntuador_open_profile::{season}"
    st.session_state[state_key] = None if st.session_state.get(state_key) == player_key else player_key


def show_ranking_cards(
    ranking: pd.DataFrame,
    raw_df: pd.DataFrame,
    zdata: pd.DataFrame,
    season: str,
    model_stats: list[str],
    stat_weights: dict,
    comp_weights: dict,
    data_signature: str,
):
    if ranking.empty:
        st.warning("No hay candidatos disponibles para esta temporada.")
        return
    q = st.text_input("Buscar jugador, club o posición", key=f"search_{season}", placeholder="Ej. Messi, Real Madrid, Forward…")
    filtered = ranking.copy()
    if q.strip():
        query = fold_text(q)
        match = pd.Series(False, index=filtered.index)
        for col in ["nombre", "club", "posicion"]:
            if col in filtered.columns:
                match |= filtered[col].fillna("").map(fold_text).str.contains(re.escape(query), regex=True, na=False)
        filtered = filtered[match]
    st.caption(f"Mostrando {len(filtered):,} jugadores · temporada {season}".replace(",", "."))

    opened_key = f"puntuador_open_profile::{season}"
    opened_player = st.session_state.get(opened_key)
    for row in filtered.itertuples(index=False):
        row_dict = row._asdict()
        rank_value = row_dict.get("posicion_predicha", np.nan)
        actual_value = row_dict.get("posicion_real", np.nan)
        rank = "—" if pd.isna(rank_value) else str(int(rank_value))
        actual = "—" if pd.isna(actual_value) else str(int(actual_value))
        raw_name = str(row_dict.get("nombre", "Jugador desconocido"))
        name = html_escape(raw_name)
        club = html_escape(str(row_dict.get("club", "—") or "—"))
        position = html_escape(str(row_dict.get("posicion", "—") or "—"))
        score = float(row_dict.get("score_puntuacion", 0.0))
        card = f"""
        <div class="ranking-card">
          <div class="ranking-card-top">
            <div class="ranking-rank">{rank}</div>
            <div><div class="ranking-player-name">{name}</div><div class="ranking-player-meta">{club} · {position}</div></div>
            <div class="ranking-top-value ranking-real-position"><span class="ranking-label">Puesto real</span><strong>{actual}</strong></div>
            <div class="ranking-top-value"><span class="ranking-label">Puntuación</span><strong>{score:.4f}</strong></div>
          </div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)
        player_key = fold_text(raw_name)
        button_label = "🙈 Ocultar estadísticas" if opened_player == player_key else "📊 Ver estadísticas"
        st.button(
            button_label,
            key=f"profile_btn_{safe_filename(season)}_{safe_filename(raw_name)}_{rank}",
            on_click=toggle_player_profile,
            args=(season, player_key),
        )

        # Solo se genera/renderiza la ficha abierta. Los otros acordeones no procesan
        # sus estadísticas en cada rerun, lo que reduce mucho el coste del ranking.
        if st.session_state.get(opened_key) == player_key:
            with st.expander(f"Estadísticas de {raw_name}", expanded=True):
                player_detail = get_cached_player_detail(
                    zdata, season, raw_name, model_stats, stat_weights, comp_weights, data_signature
                )
                show_player_detail(raw_df, season, raw_name, player_detail, f"detail_{season}_{safe_filename(raw_name)}")


def comparison_tab(
    raw_df: pd.DataFrame,
    player_comp: pd.DataFrame,
    season: str,
    ranking: pd.DataFrame,
):
    st.subheader("Comparador de jugadores")
    season_rows = raw_df.loc[raw_df["temporada"].eq(season), ["nombre", "club", "posicion"]]
    if season_rows.empty:
        st.info("No hay datos para esta temporada.")
        return
    players = season_rows.drop_duplicates("nombre").copy()
    players["club"] = players["club"].replace("", "Club desconocido")
    players["label"] = players["nombre"].astype(str) + " — " + players["club"].astype(str) + " (" + players["posicion"].astype(str) + ")"
    labels = players["label"].tolist()
    chosen = st.multiselect("Elige dos o más jugadores", labels, default=labels[:2], key=f"compare_players_{season}")
    if len(chosen) < 2:
        st.info("Selecciona al menos dos jugadores para comparar.")
        return

    player_map = dict(zip(players["label"], players["nombre"]))
    output = []
    players_rows: dict[str, pd.DataFrame] = {}
    player_comp_frames: dict[str, pd.DataFrame] = {}
    for label in chosen:
        name = player_map[label]
        raw_rows = find_player_rows(raw_df, season, name)
        stat_rows = find_player_comp_rows(player_comp, season, name)
        players_rows[label] = raw_rows
        player_comp_frames[label] = stat_rows
        vals = aggregate_display_stats(stat_rows if not stat_rows.empty else raw_rows)
        item = {"Jugador": label}
        item.update({stat: value for stat, value in vals.items() if pd.notna(value)})
        if "resultado_competicion" in raw_rows.columns:
            results = raw_rows[["competicion", "resultado_competicion"]].dropna().drop_duplicates()
            item["resultado_competicion"] = " · ".join(
                f"{r.competicion}: {r.resultado_competicion}" for r in results.itertuples(index=False)
            ) or "—"
        output.append(item)

    comparison = pd.DataFrame(output).set_index("Jugador")
    available_stats = list(comparison.columns)
    recommended = [s for s in ["goals", "assists", "goalsAssistsSum", "minutesPlayed", "appearances", "goalConversionPercentage", "accuratePassesPercentage", "resultado_competicion"] if s in available_stats]
    chosen_stats = st.multiselect(
        "Estadísticas globales a comparar", available_stats,
        default=recommended or available_stats[:8], key=f"compare_stats_{season}"
    )
    if chosen_stats:
        st.markdown("#### Comparación global")
        global_table = comparison[chosen_stats].T.rename_axis("Estadística").reset_index()
        st.dataframe(global_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar comparación global CSV", global_table.to_csv(index=False).encode("utf-8-sig"),
            f"comparacion_global_{season.replace('/', '-')}.csv", "text/csv", key=f"cmp_dl_{season}"
        )

    competitions = sorted({str(comp) for player_frame in player_comp_frames.values() for comp in player_frame.get("competicion", pd.Series(dtype=str)).dropna().unique()})
    selected_competitions = st.multiselect(
        "Competiciones individuales", competitions, default=[], key=f"compare_comps_{season}",
        help="Selecciona solo las competiciones que quieras comparar. Se dejan sin seleccionar inicialmente para que el comparador cargue más rápido."
    )
    for idx, comp in enumerate(selected_competitions):
        with st.expander(comp, expanded=False):
            comp_result = []
            for label, comp_frame in player_comp_frames.items():
                subset = comp_frame.loc[comp_frame["competicion"].astype(str).eq(comp)]
                if subset.empty:
                    continue
                vals = aggregate_display_stats(subset)
                item = {"Jugador": label}
                for stat in chosen_stats:
                    if stat == "resultado_competicion":
                        raw_subset = players_rows[label].loc[players_rows[label]["competicion"].astype(str).eq(comp)]
                        result_values = raw_subset["resultado_competicion"].dropna().astype(str).drop_duplicates().tolist() if "resultado_competicion" in raw_subset else []
                        item[stat] = " · ".join(result_values) if result_values else "—"
                    else:
                        item[stat] = vals.get(stat, np.nan)
                comp_result.append(item)
            if comp_result:
                st.dataframe(pd.DataFrame(comp_result), use_container_width=True, hide_index=True)
            else:
                st.caption("Ninguno de los jugadores seleccionados tiene datos en esta competición.")

    if not ranking.empty:
        st.markdown("#### Puntuación del ranking para los jugadores seleccionados")
        selected_names = [player_map[label] for label in chosen]
        subset_rank = ranking[ranking["nombre"].isin(selected_names)]
        st.dataframe(subset_rank, use_container_width=True, hide_index=True)


def build_row_indices(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Índice de posiciones de filas por (temporada, nombre normalizado)."""
    keys = frame["nombre"].map(fold_text)
    groups = frame.groupby([frame["temporada"], keys], sort=False, dropna=False).indices
    return {
        (str(key[0]), str(key[1])): np.asarray(indices, dtype=np.int64)
        for key, indices in groups.items()
    }


@st.cache_data(show_spinner=False)
def read_file_bytes(path_string: str, modified_ns: int, file_size: int) -> bytes:
    """Lee bytes de descargas una sola vez por versión del fichero."""
    del modified_ns, file_size
    return Path(path_string).read_bytes()


# ============================================================
# MAIN
# ============================================================

if not DATASET_PATH.exists():
    st.error(
        f"No se encuentra `{DATASET_PATH}`. Coloca dataset_ml_base.csv en la raíz "
        "del proyecto y vuelve a ejecutar Streamlit."
    )
    st.stop()

file_stat = DATASET_PATH.stat()
data_signature = f"{DATASET_PATH.resolve()}::{file_stat.st_mtime_ns}::{file_stat.st_size}"
bundle = st.session_state.get("puntuador_prepared_data_bundle")
if bundle is None or bundle.get("signature") != data_signature:
    try:
        dataset = load_dataset(str(DATASET_PATH), data_signature)
        seasons = sorted(dataset["temporada"].dropna().unique().tolist(), key=season_order)
        positions = sorted(dataset["posicion"].dropna().astype(str).str.strip().unique().tolist())
        competitions = sorted(dataset["competicion"].dropna().astype(str).str.strip().unique().tolist())
        model_stats = available_model_stats(dataset)
        if not seasons or not positions or not competitions or not model_stats:
            st.error("El dataset no contiene temporadas, posiciones, competiciones o estadísticas suficientes.")
            st.stop()
        player_comp = build_player_competition_table(dataset, model_stats)
        player_comp["__player_key"] = player_comp["nombre"].map(fold_text)
        zdata = calculate_zscores(player_comp, model_stats)
        bundle = {
            "signature": data_signature,
            "dataset": dataset,
            "player_comp": player_comp,
            "zdata": zdata,
            "seasons": seasons,
            "positions": positions,
            "competitions": competitions,
            "model_stats": model_stats,
            "raw_row_indices": build_row_indices(dataset),
            "comp_row_indices": build_row_indices(player_comp),
        }
        st.session_state["puntuador_prepared_data_bundle"] = bundle
        # Si el CSV ha cambiado, no se deben reutilizar rankings calculados con datos anteriores.
        st.session_state["puntuador_ranking_cache"] = {}
        st.session_state["puntuador_player_detail_cache"] = {}
    except Exception as exc:
        st.error(f"No se pudo cargar/preparar el dataset: {exc}")
        st.stop()
else:
    dataset = bundle["dataset"]
    player_comp = bundle["player_comp"]
    zdata = bundle["zdata"]
    seasons = bundle["seasons"]
    positions = bundle["positions"]
    competitions = bundle["competitions"]
    model_stats = bundle["model_stats"]

# Los índices también se restauran en variables accesibles por find_player_rows.
st.session_state["puntuador_raw_row_indices"] = bundle["raw_row_indices"]
st.session_state["puntuador_comp_row_indices"] = bundle["comp_row_indices"]

# Cada temporada tiene su configuración independiente de pesos.
if "puntuador_weight_configs" not in st.session_state:
    st.session_state["puntuador_weight_configs"] = load_weight_configs(
        seasons, positions, model_stats, competitions
    )
if "puntuador_weight_widget_versions" not in st.session_state:
    st.session_state["puntuador_weight_widget_versions"] = {season: 0 for season in seasons}
else:
    versions = dict(st.session_state["puntuador_weight_widget_versions"])
    for season in seasons:
        versions.setdefault(season, 0)
    st.session_state["puntuador_weight_widget_versions"] = versions

season_default = "2025/26" if "2025/26" in seasons else seasons[-1]
selected_season = st.selectbox(
    "Temporada que quieres puntuar", seasons,
    index=seasons.index(season_default), key="selected_season",
    help="Puedes seleccionar cualquier temporada que aparezca en dataset_ml_base.csv.",
)
st.caption(
    f"Dataset: {len(dataset):,} filas · {len(seasons)} temporadas · {len(positions)} posiciones · "
    f"{len(competitions)} competiciones individuales · {len(model_stats)} variables de puntuación".replace(",", ".")
)

stat_weights, competition_weights = weight_configs_for_season(selected_season)

# A diferencia de st.tabs, este selector solo construye la sección activa. Así el
# comparador no renderiza cientos de tarjetas y controles de pesos en cada cambio.
section = st.radio(
    "Sección de la aplicación",
    ["🏆 Ranking", "⚙️ Pesos por posición", "🔎 Comparador"],
    horizontal=True,
    key="puntuador_main_section",
    label_visibility="collapsed",
)

ranking = pd.DataFrame()
if section == "🏆 Ranking":
    ranking = get_cached_ranking(
        zdata, selected_season, model_stats, positions,
        stat_weights, competition_weights, data_signature,
    )
    st.subheader(f"Ranking estadístico · {selected_season}")
    st.write(
        "La puntuación suma las contribuciones de cada competición. En cada competición, "
        "los z-scores se multiplican por los pesos de las estadísticas de la posición; "
        "el total se divide por la suma fija de los pesos estadísticos activos para esa posición "
        "y se multiplica por el peso de la competición."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Jugadores", f"{len(ranking):,}".replace(",", "."))
    m2.metric("Competiciones", len(competitions))
    m3.metric("Estadísticas", len(model_stats))
    d1, d2 = st.columns(2)
    ranking_csv = ranking.to_csv(index=False).encode("utf-8-sig")
    d1.download_button(
        "Descargar ranking CSV", ranking_csv,
        f"ranking_puntuador_{selected_season.replace('/', '-')}.csv", "text/csv",
        key=f"dl_ranking_{selected_season}",
    )
    export_key = (data_signature, selected_season, weight_signature(stat_weights, competition_weights))
    export_state = st.session_state.get("puntuador_detail_export", {})
    if st.button("Preparar desglose completo de contribuciones CSV", key=f"prepare_detail_{safe_filename(selected_season)}"):
        with st.spinner("Generando el desglose completo..."):
            _, full_detail = calculate_ranking(
                zdata, selected_season, model_stats, positions,
                stat_weights, competition_weights, include_detail=True,
            )
            st.session_state["puntuador_detail_export"] = {
                "key": export_key,
                "data": full_detail.to_csv(index=False).encode("utf-8-sig") if not full_detail.empty else b"",
            }
            export_state = st.session_state["puntuador_detail_export"]
    if export_state.get("key") == export_key and export_state.get("data") is not None:
        d2.download_button(
            "Descargar desglose completo CSV", export_state["data"],
            f"desglose_puntuador_{selected_season.replace('/', '-')}.csv", "text/csv",
            key=f"dl_detail_{selected_season}_{export_key[2][:8]}",
        )
    else:
        d2.caption("El desglose completo se genera solo cuando lo solicitas.")

    if "posicion_real" in ranking.columns and ranking["posicion_real"].notna().any():
        evaluated = ranking.dropna(subset=["posicion_real"]).copy()
        if not evaluated.empty:
            predicted = evaluated["posicion_predicha"].astype(float)
            actual = evaluated["posicion_real"].astype(float)
            mae = float(np.mean(np.abs(predicted - actual)))
            top5_pred = set(evaluated.nsmallest(5, "posicion_predicha")["nombre"])
            top5_real = set(evaluated.nsmallest(5, "posicion_real")["nombre"])
            top10_pred = set(evaluated.nsmallest(10, "posicion_predicha")["nombre"])
            top10_real = set(evaluated.nsmallest(10, "posicion_real")["nombre"])
            a, b, c = st.columns(3)
            a.metric("MAE de puesto", f"{mae:.2f}")
            b.metric("Coincidencias Top 5", f"{len(top5_pred & top5_real)}/5")
            c.metric("Coincidencias Top 10", f"{len(top10_pred & top10_real)}/10")
            st.caption("Estas métricas solo se muestran cuando hay puestos reales. No intervienen en la puntuación.")

    show_ranking_cards(
        ranking, dataset, zdata, selected_season, model_stats,
        stat_weights, competition_weights, data_signature,
    )

elif section == "⚙️ Pesos por posición":
    st.subheader(f"Pesos de la temporada {selected_season}")
    st.info(
        "Cada temporada guarda su propia configuración. Modifica los controles numéricos (+/−) y pulsa "
        "‘Aplicar pesos’ para guardar los cambios de la posición seleccionada. Los pesos van de 0 a 10, "
        "la orientación positiva/negativa es fija y cada competición conserva su nombre individual. "
        "Para no perder cambios, aplica los pesos antes de cambiar de posición o temporada."
    )

    other_seasons = [season for season in seasons if season != selected_season]
    if other_seasons:
        copy_col, copy_button_col = st.columns([3, 1.5])
        with copy_col:
            source_season = st.selectbox(
                "Copiar pesos de esta temporada", other_seasons,
                index=other_seasons.index("2025/26") if "2025/26" in other_seasons else 0,
                key=f"copy_weights_source_{safe_filename(selected_season)}",
                help="Elige la temporada de origen. Se copian los pesos de todas las posiciones, estadísticas y competiciones.",
            )
        with copy_button_col:
            st.markdown(" ")
            st.button(
                f"📋 Copiar a {selected_season}",
                key=f"copy_weights_button_{safe_filename(selected_season)}",
                on_click=copy_weights_between_seasons,
                args=(source_season, selected_season, positions, model_stats, competitions),
                use_container_width=True,
            )
        st.caption("Se copia la configuración aplicada de origen. Aplica primero cualquier cambio pendiente.")

    copied_notice = st.session_state.get("puntuador_weights_copy_notice")
    if copied_notice and copied_notice[1] == selected_season:
        st.success(f"Pesos copiados desde {copied_notice[0]} a {selected_season} y guardados automáticamente.")
    if st.session_state.get("puntuador_weights_applied_notice") == selected_season:
        st.success(f"Pesos aplicados para {selected_season}; el ranking usará estos valores al abrir la sección Ranking.")
    if st.session_state.get("puntuador_weights_saved_notice"):
        st.caption(f"Configuraciones guardadas en `{WEIGHTS_PATH}`.")
    save_error = st.session_state.get("puntuador_weights_save_error", "")
    if save_error:
        st.warning(f"No se pudieron guardar automáticamente los pesos: {save_error}.")

    selected_position = st.selectbox(
        "Posición cuyos pesos quieres editar", positions,
        key=f"weight_position_{safe_filename(selected_season)}",
    )
    configs = st.session_state["puntuador_weight_configs"]
    season_stat_config = configs["stat_weights_by_season"].get(selected_season, {})
    season_comp_config = configs["competition_weights_by_season"].get(selected_season, {})
    form_key = f"puntuador_weights_form_{safe_filename(selected_season)}_{safe_filename(selected_position)}"
    with st.form(form_key, clear_on_submit=False):
        stats_col, comp_col = st.columns(2)
        allowed_stats = stats_allowed_for_position(selected_position, model_stats)
        with stats_col:
            with st.expander("⚙️ Pesos de las estadísticas", expanded=True):
                st.caption("Las estadísticas negativas se mantienen negativas; el peso solo controla su magnitud.")
                for stat in allowed_stats:
                    label_col, spin_col = st.columns([2.4, 1])
                    with label_col:
                        sign = "−" if stat_sign(stat) < 0 else "+"
                        st.markdown(f"{sign} {stat}")
                    with spin_col:
                        st.number_input(
                            f"Peso de {stat}", min_value=0.0, max_value=10.0, step=0.1,
                            value=clamp_weight(season_stat_config.get(selected_position, {}).get(stat, 1.0)),
                            format="%.1f",
                            key=stat_weight_widget_key(selected_season, selected_position, stat),
                            label_visibility="collapsed",
                            help="Peso entre 0 y 10. Usa + y − para ajustarlo.",
                        )
        with comp_col:
            with st.expander("🏆 Pesos de las competiciones", expanded=True):
                st.caption("Cada competición conserva su nombre y su peso individual.")
                for competition in competitions:
                    label_col, spin_col = st.columns([2.4, 1])
                    with label_col:
                        st.markdown(competition)
                    with spin_col:
                        st.number_input(
                            f"Peso de {competition}", min_value=0.0, max_value=10.0, step=0.1,
                            value=clamp_weight(season_comp_config.get(selected_position, {}).get(competition, 1.0)),
                            format="%.1f",
                            key=competition_weight_widget_key(selected_season, selected_position, competition),
                            label_visibility="collapsed",
                            help="Peso entre 0 y 10. Usa + y − para ajustarlo.",
                        )
        st.form_submit_button(
            f"✅ Aplicar pesos de {selected_position}", type="primary", use_container_width=True,
            on_click=apply_weight_inputs,
            args=(selected_season, selected_position, model_stats, competitions),
        )

    save_col, reset_col, _ = st.columns([1.5, 1.7, 3])
    with save_col:
        if st.button("💾 Guardar pesos de todas las temporadas", type="secondary", key=f"save_weights_{safe_filename(selected_season)}"):
            try:
                save_weight_configs(st.session_state["puntuador_weight_configs"])
                st.success(f"Configuraciones por temporada guardadas en `{WEIGHTS_PATH}`.")
            except OSError as exc:
                st.error(f"No se pudieron guardar los pesos: {exc}")
    with reset_col:
        st.button(
            f"↩ Restablecer {selected_season} a 1",
            key=f"reset_weights_{safe_filename(selected_season)}",
            on_click=reset_season_weights,
            args=(selected_season, positions, model_stats, competitions),
            help="Restablece únicamente la temporada seleccionada.",
            use_container_width=True,
        )
    st.caption(
        "Las configuraciones son independientes por temporada y se guardan en JSON. "
        "El ranking solo se recalcula cuando abres la sección Ranking o Comparador."
    )

else:
    ranking = get_cached_ranking(
        zdata, selected_season, model_stats, positions,
        stat_weights, competition_weights, data_signature,
    )
    comparison_tab(dataset, player_comp, selected_season, ranking)

with st.sidebar:
    st.header("Archivos")
    st.caption("Archivos locales del puntuador.")
    if DATASET_PATH.exists():
        try:
            st.download_button(
                "Descargar dataset_ml_base.csv",
                read_file_bytes(str(DATASET_PATH), file_stat.st_mtime_ns, file_stat.st_size),
                "dataset_ml_base.csv", "text/csv", key="download_dataset",
            )
        except OSError as exc:
            st.warning(f"No se pudo leer el dataset: {exc}")
    if WEIGHTS_PATH.exists():
        try:
            weight_stat = WEIGHTS_PATH.stat()
            st.download_button(
                "Descargar configuración de pesos JSON",
                read_file_bytes(str(WEIGHTS_PATH), weight_stat.st_mtime_ns, weight_stat.st_size),
                WEIGHTS_PATH.name, "application/json", key="download_weights_json",
            )
        except OSError as exc:
            st.warning(f"No se pudo leer el JSON de pesos: {exc}")
    st.markdown("---")
    st.markdown("**Fórmula utilizada**")
    st.latex(r"S_{p,c}=\frac{\sum_s d_s\,z_{p,c,s}\,w_{p,s}}{\sum_s w_{p,s}}\,W_{p,c}")
    st.latex(r"S_p=\sum_c S_{p,c}")
    st.caption("d = orientación (+1 o −1); z = z-score; w = peso estadístico; W = peso de la competición. Los pesos se guardan por temporada y posición.")

st.divider()
st.caption(
    "Notas: las competiciones no se agrupan ni se renombran. Los jugadores sin datos en una competición "
    "no reciben una contribución por ella. Un dato ausente se trata como contribución neutra (z-score 0) "
    "y no entra en la media/desviación. Si una variable no varía en una competición, su z-score es 0. "
    "Los puestos reales 1–30 de 2025/26 se sustituyen por la clasificación facilitada. Los pesos iniciales a 1 "
    "son valores de partida editables, no una afirmación sobre su importancia real."
)
