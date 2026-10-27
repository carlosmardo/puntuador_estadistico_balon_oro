from __future__ import annotations

import json
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

# Columnas que se muestran en las fichas y en el comparador, además de las
# variables exactas de puntuación. Se usan solo si existen en el dataset.
DISPLAY_EXTRA_STATS = [
    "goals", "assists", "appearances", "minutesPlayed",
    "expectedGoals", "expectedAssists",
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
def load_dataset(path_string: str, file_mtime: float) -> pd.DataFrame:
    del file_mtime  # forma parte de la clave de caché para detectar cambios del archivo
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

def default_weight_tables(positions: list[str], stats: list[str], competitions: list[str]):
    stat_tables: dict[str, pd.DataFrame] = {}
    competition_tables: dict[str, pd.DataFrame] = {}
    for position in positions:
        allowed = stats_allowed_for_position(position, stats)
        stat_tables[position] = pd.DataFrame({
            "Estadística": allowed,
            "Orientación": ["Negativa" if stat_sign(s) < 0 else "Positiva" for s in allowed],
            "Peso": [1.0] * len(allowed),
        })
        competition_tables[position] = pd.DataFrame({
            "Competición": competitions,
            "Peso": [1.0] * len(competitions),
        })
    return stat_tables, competition_tables


def load_saved_weights(positions: list[str], stats: list[str], competitions: list[str]):
    default_stats, default_comps = default_weight_tables(positions, stats, competitions)
    if not WEIGHTS_PATH.exists():
        return default_stats, default_comps
    try:
        saved = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.warning("No se pudo leer el archivo de pesos guardado. Se usarán pesos iniciales de 1.")
        return default_stats, default_comps

    saved_stats = saved.get("stat_weights", {})
    saved_comps = saved.get("competition_weights", {})
    for position in positions:
        if position in saved_stats:
            default_stats[position]["Peso"] = [
                max(0.0, float(saved_stats[position].get(stat, 1.0)))
                for stat in default_stats[position]["Estadística"]
            ]
        if position in saved_comps:
            default_comps[position]["Peso"] = [
                max(0.0, float(saved_comps[position].get(comp, 1.0)))
                for comp in default_comps[position]["Competición"]
            ]
    return default_stats, default_comps


def table_to_weight_map(table: pd.DataFrame, label_col: str) -> dict[str, float]:
    if table is None or table.empty:
        return {}
    weights = {}
    for row in table.to_dict("records"):
        label = str(row.get(label_col, ""))
        try:
            value = float(row.get("Peso", 1.0))
        except (TypeError, ValueError):
            value = 1.0
        weights[label] = float(np.clip(value if np.isfinite(value) else 1.0, 0.0, 10.0))
    return weights


def stat_editor_key(idx: int) -> str:
    """Clave interna del widget; nunca se asigna manualmente en session_state."""
    version = int(st.session_state.get("puntuador_editor_version", 0))
    return f"stat_weights_editor_{idx}_v{version}"


def competition_editor_key(idx: int) -> str:
    """Clave interna del widget; cambia al restablecer los pesos."""
    version = int(st.session_state.get("puntuador_editor_version", 0))
    return f"competition_weights_editor_{idx}_v{version}"


def apply_data_editor_changes(base_table: pd.DataFrame, editor_key: str) -> pd.DataFrame:
    """Reconstruye el DataFrame a partir de la tabla base y las celdas editadas.

    Streamlit guarda en session_state de st.data_editor un objeto DataEditorState
    (con edited_rows), no el DataFrame completo. Como el editor usa num_rows='fixed',
    basta con aplicar las ediciones de las filas existentes.
    """
    table = base_table.copy(deep=True)
    state = st.session_state.get(editor_key)
    if state is None:
        return table

    edits = getattr(state, "edited_rows", None)
    if edits is None and isinstance(state, dict):
        edits = state.get("edited_rows", {})
    if not isinstance(edits, dict):
        return table

    for row_position, changes in edits.items():
        try:
            row_position = int(row_position)
        except (TypeError, ValueError):
            continue
        if row_position < 0 or row_position >= len(table) or not isinstance(changes, dict):
            continue
        for column, value in changes.items():
            if column in table.columns:
                table.iloc[row_position, table.columns.get_loc(column)] = value
    return table


def current_weight_maps(positions: list[str], stats: list[str], competitions: list[str]):
    stat_weights: dict[str, dict[str, float]] = {}
    comp_weights: dict[str, dict[str, float]] = {}
    defaults_stats, defaults_comps = default_weight_tables(positions, stats, competitions)
    for idx, position in enumerate(positions):
        # Las tablas base están en claves independientes de las claves del widget.
        stat_base_key = f"stat_weights_base_{idx}"
        comp_base_key = f"competition_weights_base_{idx}"
        stat_base = st.session_state.get(stat_base_key, defaults_stats[position])
        comp_base = st.session_state.get(comp_base_key, defaults_comps[position])
        stat_table = apply_data_editor_changes(stat_base, stat_editor_key(idx))
        comp_table = apply_data_editor_changes(comp_base, competition_editor_key(idx))
        stat_weights[position] = table_to_weight_map(stat_table, "Estadística")
        comp_weights[position] = table_to_weight_map(comp_table, "Competición")
    return stat_weights, comp_weights


def save_weights(stat_weights: dict, comp_weights: dict):
    payload = {
        "version": 1,
        "description": "Pesos manuales del puntuador estadístico; no son coeficientes aprendidos.",
        "stat_weights": stat_weights,
        "competition_weights": comp_weights,
    }
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_widget_weights(positions: list[str], stats: list[str], competitions: list[str]):
    """Restablece las tablas base y cambia las claves para crear widgets limpios.

    No escribe nunca en las claves internas de st.data_editor, ya que Streamlit
    prohíbe asignarles valores mediante st.session_state.
    """
    default_stats, default_comps = default_weight_tables(positions, stats, competitions)
    for idx, position in enumerate(positions):
        st.session_state[f"stat_weights_base_{idx}"] = default_stats[position].copy(deep=True)
        st.session_state[f"competition_weights_base_{idx}"] = default_comps[position].copy(deep=True)
    st.session_state["puntuador_editor_version"] = int(
        st.session_state.get("puntuador_editor_version", 0)
    ) + 1


def calculate_ranking(
    zdata: pd.DataFrame,
    season: str,
    model_stats: list[str],
    positions: list[str],
    stat_weights: dict[str, dict[str, float]],
    comp_weights: dict[str, dict[str, float]],
):
    season_data = zdata[zdata["temporada"].eq(season)].copy()
    if season_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Una fila por jugador y temporada para la tabla de ranking.
    identity_cols = [c for c in ["temporada", "nombre", "club", "posicion", "nacionalidad", "posicion_balon_oro"] if c in season_data.columns]
    candidates = season_data[identity_cols].drop_duplicates(["temporada", "nombre"]).copy().reset_index(drop=True)
    candidates["score_puntuacion"] = 0.0
    details = []

    for candidate_idx, candidate in candidates.iterrows():
        name = str(candidate["nombre"])
        position = str(candidate.get("posicion", ""))
        allowed_stats = stats_allowed_for_position(position, model_stats)
        weights_stats = stat_weights.get(position, {})

        player_rows = season_data[season_data["nombre"].map(fold_text).eq(fold_text(name))]
        total_score = 0.0
        for competition, comp_group in player_rows.groupby("competicion", sort=False):
            comp_row = comp_group.iloc[0]
            comp_weight = max(0.0, comp_weights.get(position, {}).get(str(competition), 1.0))

            # Denominador fijo para cada pareja posición-competición: solo incluye
            # variables que tienen al menos un valor observado entre los jugadores
            # de esa posición en esa competición. No depende de los NaN de cada jugador.
            population = season_data[
                season_data["competicion"].eq(str(competition))
                & season_data["posicion"].astype(str).eq(position)
            ]
            active_stats = [
                stat for stat in allowed_stats
                if stat in population.columns
                and pd.to_numeric(population[stat], errors="coerce").notna().any()
            ]
            denominator = sum(max(0.0, weights_stats.get(stat, 1.0)) for stat in active_stats)

            stat_rows = []
            weighted_sum = 0.0
            for stat in allowed_stats:
                stat_weight = max(0.0, weights_stats.get(stat, 1.0))
                z_value = float(pd.to_numeric(pd.Series([comp_row.get(f"z__{stat}", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
                raw_value = comp_row.get(stat, np.nan)
                sign = stat_sign(stat)
                # Los signos están fijados: peso no negativo × orientación monotónica.
                contribution_before_comp = (
                    sign * z_value * stat_weight / denominator
                    if denominator > 1e-12 and stat in active_stats else 0.0
                )
                weighted_sum += contribution_before_comp
                stat_rows.append({
                    "temporada": season,
                    "nombre": name,
                    "posicion": position,
                    "competicion": str(competition),
                    "estadistica": stat,
                    "valor_agregado": raw_value,
                    "zscore": z_value,
                    "orientacion": "Negativa" if sign < 0 else "Positiva",
                    "peso_estadistica": stat_weight,
                    "contribucion_antes_peso_competicion": contribution_before_comp,
                })
            competition_score = weighted_sum
            competition_contribution = competition_score * comp_weight
            total_score += competition_contribution
            for stat_row in stat_rows:
                stat_row["peso_competicion"] = comp_weight
                stat_row["puntuacion_competicion"] = competition_score
                stat_row["contribucion_competicion"] = competition_contribution
                stat_row["contribucion_final"] = stat_row["contribucion_antes_peso_competicion"] * comp_weight
                details.append(stat_row)

        candidates.loc[candidate_idx, "score_puntuacion"] = total_score

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
    detail_df = pd.DataFrame(details)
    return candidates, detail_df


# ============================================================
# ESTADÍSTICAS DE FICHA Y COMPARADOR
# ============================================================

def find_player_rows(raw_df: pd.DataFrame, season: str, name: str) -> pd.DataFrame:
    mask = raw_df["temporada"].eq(season) & raw_df["nombre"].map(fold_text).eq(fold_text(name))
    return raw_df.loc[mask].copy()


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
        player_detail = detail_df[detail_df["nombre"].map(fold_text).eq(fold_text(name))] if not detail_df.empty else pd.DataFrame()
        if not player_detail.empty:
            comp_score = player_detail[["competicion", "peso_competicion", "puntuacion_competicion", "contribucion_competicion"]].drop_duplicates()
            st.markdown("**Contribución de cada competición a la puntuación**")
            st.dataframe(comp_score.sort_values("contribucion_competicion", ascending=False), use_container_width=True, hide_index=True)
    for idx, competition in enumerate(competitions, start=1):
        with tabs[idx]:
            subset = rows[rows["competicion"].astype(str).eq(competition)]
            display_stats_table(subset, f"Estadísticas · {competition}", f"{key_prefix}_comp_{idx}")
            player_comp_detail = detail_df[
                detail_df["nombre"].map(fold_text).eq(fold_text(name))
                & detail_df["competicion"].eq(competition)
            ] if not detail_df.empty else pd.DataFrame()
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


def show_ranking_cards(ranking: pd.DataFrame, raw_df: pd.DataFrame, detail_df: pd.DataFrame, season: str):
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
    display_count = st.selectbox(
        "Jugadores que mostrar", [25, 50, 100, "Todos"], index=1, key=f"count_{season}"
    )
    if display_count != "Todos":
        filtered = filtered.head(int(display_count))
    st.caption(f"Mostrando {len(filtered):,} jugadores · temporada {season}".replace(",", "."))

    for _, row in filtered.iterrows():
        rank = "—" if pd.isna(row.get("posicion_predicha")) else str(int(row["posicion_predicha"]))
        actual_value = row.get("posicion_real", np.nan)
        actual = "—" if pd.isna(actual_value) else str(int(actual_value))
        name = html_escape(str(row.get("nombre", "Jugador desconocido")))
        club = html_escape(str(row.get("club", "—") or "—"))
        position = html_escape(str(row.get("posicion", "—") or "—"))
        score = float(row.get("score_puntuacion", 0.0))
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
        with st.expander(f"📊 Ver estadísticas · {name}", expanded=False):
            show_player_detail(raw_df, season, name, detail_df, f"detail_{season}_{safe_filename(name)}")


def comparison_tab(raw_df: pd.DataFrame, season: str, ranking: pd.DataFrame, detail_df: pd.DataFrame):
    st.subheader("Comparador de jugadores")
    season_rows = raw_df[raw_df["temporada"].eq(season)]
    if season_rows.empty:
        st.info("No hay datos para esta temporada.")
        return
    players = season_rows[["nombre", "club", "posicion"]].drop_duplicates("nombre").copy()
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
    for label in chosen:
        name = player_map[label]
        rows = find_player_rows(raw_df, season, name)
        players_rows[label] = rows
        vals = aggregate_display_stats(rows)
        item = {"Jugador": label}
        item.update({stat: value for stat, value in vals.items() if pd.notna(value)})
        if "resultado_competicion" in rows.columns:
            results = rows[["competicion", "resultado_competicion"]].dropna().drop_duplicates()
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

    competitions = sorted({str(comp) for player_frame in players_rows.values() for comp in player_frame["competicion"].dropna().unique()})
    selected_competitions = st.multiselect(
        "Competiciones individuales", competitions, default=competitions, key=f"compare_comps_{season}"
    )
    for idx, comp in enumerate(selected_competitions):
        with st.expander(comp, expanded=False):
            comp_result = []
            for label, player_frame in players_rows.items():
                subset = player_frame[player_frame["competicion"].astype(str).eq(comp)]
                if subset.empty:
                    continue
                vals = aggregate_display_stats(subset)
                item = {"Jugador": label}
                for stat in chosen_stats:
                    if stat == "resultado_competicion":
                        result_values = subset["resultado_competicion"].dropna().astype(str).drop_duplicates().tolist() if "resultado_competicion" in subset else []
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


# ============================================================
# MAIN
# ============================================================

if not DATASET_PATH.exists():
    st.error(
        f"No se encuentra `{DATASET_PATH}`."
        "del proyecto y vuelve a ejecutar Streamlit."
    )
    st.stop()

try:
    dataset = load_dataset(str(DATASET_PATH), DATASET_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"No se pudo cargar el dataset: {exc}")
    st.stop()

seasons = sorted(dataset["temporada"].dropna().unique().tolist(), key=season_order)
positions = sorted(dataset["posicion"].dropna().astype(str).str.strip().unique().tolist())
competitions = sorted(dataset["competicion"].dropna().astype(str).str.strip().unique().tolist())
model_stats = available_model_stats(dataset)
if not seasons or not positions or not competitions or not model_stats:
    st.error("El dataset no contiene temporadas, posiciones, competiciones o estadísticas suficientes.")
    st.stop()

player_comp = build_player_competition_table(dataset, model_stats)
zdata = calculate_zscores(player_comp, model_stats)

# Inicializar las tablas base una sola vez. Las claves del data_editor se reservan
# exclusivamente para el estado interno del widget; nunca se les asignan DataFrames.
if "puntuador_weights_initialized" not in st.session_state:
    stat_defaults, comp_defaults = load_saved_weights(positions, model_stats, competitions)
    for idx, position in enumerate(positions):
        st.session_state[f"stat_weights_base_{idx}"] = stat_defaults[position].copy(deep=True)
        st.session_state[f"competition_weights_base_{idx}"] = comp_defaults[position].copy(deep=True)
    st.session_state["puntuador_editor_version"] = 0
    st.session_state["puntuador_weights_initialized"] = True

# Selector de temporada disponible en TODO el dataset.
season_default = "2025/26" if "2025/26" in seasons else seasons[-1]
selected_season = st.selectbox(
    "Temporada que quieres puntuar",
    seasons,
    index=seasons.index(season_default),
    key="selected_season",
    help="Puedes seleccionar cualquier temporada que aparezca en dataset_ml_base.csv.",
)

st.caption(
    f"Dataset: {len(dataset):,} filas · {len(seasons)} temporadas · {len(positions)} posiciones · "
    f"{len(competitions)} competiciones individuales · {len(model_stats)} variables de puntuación".replace(",", ".")
)

# Leer pesos actuales desde el estado de los editores; cambiar cualquier valor recalcula el ranking.
stat_weights, competition_weights = current_weight_maps(positions, model_stats, competitions)
ranking, score_detail = calculate_ranking(
    zdata, selected_season, model_stats, positions, stat_weights, competition_weights
)

ranking_csv = ranking.to_csv(index=False).encode("utf-8-sig")
detail_csv = score_detail.to_csv(index=False).encode("utf-8-sig") if not score_detail.empty else b""

rank_tab, weights_tab, compare_tab = st.tabs(["🏆 Ranking", "⚙️ Pesos por posición", "🔎 Comparador"])

with rank_tab:
    st.subheader(f"Ranking estadístico · {selected_season}")
    st.write(
        "La puntuación suma las contribuciones de cada competición. En cada competición, "
        "los z-scores se multiplican por los pesos de las estadísticas de la posición; "
        "el total se divide por la suma fija de los pesos estadísticos permitidos para esa posición. "
        "Después se multiplica por el peso de esa competición."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Jugadores", f"{len(ranking):,}".replace(",", "."))
    m2.metric("Competiciones", len(competitions))
    m3.metric("Estadísticas", len(model_stats))
    d1, d2 = st.columns(2)
    d1.download_button("Descargar ranking CSV", ranking_csv, f"ranking_puntuador_{selected_season.replace('/', '-')}.csv", "text/csv", key=f"dl_ranking_{selected_season}")
    d2.download_button("Descargar desglose de contribuciones CSV", detail_csv, f"desglose_puntuador_{selected_season.replace('/', '-')}.csv", "text/csv", key=f"dl_detail_{selected_season}")

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
            st.caption("Estas métricas solo se muestran cuando hay puestos reales en el dataset. No intervienen en el cálculo de la puntuación.")

    show_ranking_cards(ranking, dataset, score_detail, selected_season)

with weights_tab:
    st.subheader("Configurar los pesos")
    st.info(
        "Son pesos manuales y fijos, no se entrenan automáticamente. El peso de cada estadística "
        "solo puede ir de 0 a 10; su orientación positiva o negativa está bloqueada según las "
        "restricciones del código de ML. Cada competición conserva su nombre original y tiene un peso propio."
    )
    position_tabs = st.tabs(positions)
    for idx, position in enumerate(positions):
        with position_tabs[idx]:
            st.markdown(f"### {position}")
            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### Pesos de las estadísticas")
                st.caption("El denominador se mantiene fijo por posición: reducir el peso de una estadística no aumenta artificialmente las demás.")
                stat_table_default = default_weight_tables(positions, model_stats, competitions)[0][position]
                st.data_editor(
                    st.session_state[f"stat_weights_base_{idx}"],
                    key=stat_editor_key(idx),
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    height=min(650, 45 + len(stat_table_default) * 35),
                    column_config={
                        "Estadística": st.column_config.TextColumn("Estadística", disabled=True),
                        "Orientación": st.column_config.TextColumn("Orientación (fija)", disabled=True),
                        "Peso": st.column_config.NumberColumn("Peso (0–10)", min_value=0.0, max_value=10.0, step=0.1, format="%.2f"),
                    },
                )
            with right:
                st.markdown("#### Pesos de las competiciones")
                st.caption("Cada competición se pondera por separado; no se agrupa en categorías como ‘grandes ligas’.")
                comp_table_default = default_weight_tables(positions, model_stats, competitions)[1][position]
                st.data_editor(
                    st.session_state[f"competition_weights_base_{idx}"],
                    key=competition_editor_key(idx),
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    height=min(650, 45 + len(comp_table_default) * 35),
                    column_config={
                        "Competición": st.column_config.TextColumn("Competición individual", disabled=True),
                        "Peso": st.column_config.NumberColumn("Peso (0–10)", min_value=0.0, max_value=10.0, step=0.1, format="%.2f"),
                    },
                )
    save_col, reset_col, _ = st.columns([1.2, 1.2, 3])
    with save_col:
        if st.button("💾 Guardar pesos", type="primary", key="save_weights"):
            latest_stats, latest_comps = current_weight_maps(positions, model_stats, competitions)
            save_weights(latest_stats, latest_comps)
            st.success(f"Pesos guardados en `{WEIGHTS_PATH}`. Se conservarán en futuras ejecuciones.")
    with reset_col:
        st.button(
            "↩ Restablecer a 1",
            key="reset_weights",
            on_click=reset_widget_weights,
            args=(positions, model_stats, competitions),
            help="Vuelve a poner todos los pesos a 1. No modifica el dataset.",
        )
    st.caption("El ranking se actualiza al modificar un peso. Usa ‘Guardar pesos’ para conservar la configuración en el archivo JSON.")

with compare_tab:
    comparison_tab(dataset, selected_season, ranking, score_detail)

with st.sidebar:
    st.header("Archivos")
    st.caption("Archivos locales del puntuador.")
    if DATASET_PATH.exists():
        st.download_button(
            "Descargar dataset_ml_base.csv",
            DATASET_PATH.read_bytes(),
            "dataset_ml_base.csv",
            "text/csv",
            key="download_dataset",
        )
    if WEIGHTS_PATH.exists():
        st.download_button(
            "Descargar configuración de pesos JSON",
            WEIGHTS_PATH.read_bytes(),
            WEIGHTS_PATH.name,
            "application/json",
            key="download_weights_json",
        )
    st.markdown("---")
    st.markdown("**Fórmula utilizada**")
    st.latex(r"S_{p,c}=\frac{\sum_s d_s\,z_{p,c,s}\,w_{p,s}}{\sum_s w_{p,s}}\,W_{p,c}")
    st.latex(r"S_p=\sum_c S_{p,c}")
    st.caption("d = orientación (+1 o −1); z = z-score; w = peso estadístico; W = peso de la competición.")

st.divider()
st.caption(
    "Notas: las competiciones no se agrupan ni se renombran. Los jugadores sin datos en una competición "
    "no reciben una contribución por ella. Un dato estadístico ausente se trata como contribución neutra "
    "(z-score 0) y no entra en el cálculo de la media/desviación. Si una variable no varía en una competición, "
    "su z-score es 0. Los pesos iniciales a 1 son valores de partida editables, no una afirmación sobre la importancia real."
)
