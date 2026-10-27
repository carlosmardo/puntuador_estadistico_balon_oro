# Puntuador estadístico del Balón de Oro

Aplicación Streamlit que calcula un ranking a partir de las estadísticas de `data/dataset_ml_base.csv`. **No es un modelo de Machine Learning**: no entrena coeficientes, no utiliza Optuna y no aprende de la clasificación histórica. La puntuación se obtiene mediante z-scores y pesos manuales configurables.

## Archivos

```text
proyecto/
├── app_puntuador.py
└── data/
    └── dataset_ml_base.csv
```

Al ejecutar y guardar una configuración, la aplicación crea también:

```text
data/puntuador/pesos_puntuador.json
```

## Instalación y ejecución

Instala las dependencias:

```bash
pip install streamlit pandas numpy
```

Inicia la aplicación desde la raíz del proyecto:

```bash
streamlit run app_puntuador.py
```

La aplicación necesita `data/dataset_ml_base.csv`, con al menos las columnas `temporada`, `nombre`, `competicion` y `posicion`. El resto de columnas estadísticas se utiliza cuando está disponible.

## Cómo se calcula la puntuación

### 1. Cada competición es individual

No se aplica el mapa de categorías de competiciones del modelo de ML. `Premier League`, `LaLiga`, `UEFA Champions League`, `FIFA World Cup`, copas, supercopas y cualquier otra competición son valores separados tal como aparecen en la columna `competicion`. Cada competición tiene su propio peso configurable para cada posición.

### 2. Agregación

Los registros se agregan por `temporada`, `nombre` y `competicion`:

- Las estadísticas acumulables se suman.
- Los porcentajes se reconstruyen sumando numeradores y denominadores y calculando después el porcentaje total; no se promedian porcentajes simples.
- `goalsAssistsPerMins` se calcula como goles más asistencias dividido por los minutos, usando `goalsAssistsSum` y `minutesPlayed`.
- `resultado_competicion` se convierte al mismo valor ordinal del script ML (por ejemplo, `Ganador = 1`, `Final = 0.9`, `Semifinales = 0.8`, `Cuartos de final = 0.7`). Ese valor es una característica positiva adicional.

### 3. Z-scores

Cada estadística se normaliza independientemente por temporada y competición individual:

\[
z_{i,c,s}=\frac{x_{i,c,s}-\mu_{c,s}}{\sigma_{c,s}}
\]

La media y desviación estándar se calculan con los valores observados de los jugadores que tienen una fila en esa competición. Los datos ausentes se excluyen de ese cálculo y reciben z-score 0 (contribución neutral). Si la desviación estándar es cero o no es válida, el z-score de esa característica es 0 para todos los jugadores de ese grupo.

Para las cinco estadísticas exclusivas de portero (`saves`, `goalsPrevented`, `highClaims`, `punches` y `successfulRunsOut`), la referencia se calcula solo entre porteros observados. Los jugadores que no aparecen en una competición no reciben una fila artificial ni una contribución por ella.

### 4. Variables de puntuación

La aplicación utiliza las mismas variables de características del script ML, siempre que las columnas necesarias existan en el CSV:

**Estadísticas acumulables**

- `goalsAssistsSum`
- `bigChancesCreated`
- `bigChancesMissed`
- `keyPasses`
- `successfulDribbles`
- `accuratePasses`
- `totalAttemptAssist`
- `aerialDuelsWon`
- `totalDuelsWon`
- `tacklesWon`
- `interceptions`
- `outfielderBlocks`
- `ballRecovery`
- `dispossessed`
- `dribbledPast`
- `clearances`
- `errorLeadToGoal`
- `errorLeadToShot`
- `saves`
- `goalsPrevented`
- `goalsConceded`
- `highClaims`
- `punches`
- `successfulRunsOut`
- `cleanSheet`
- `ownGoals`
- `penaltyConceded`
- `possessionWonAttThird`

**Porcentajes**

- `accuratePassesPercentage`
- `successfulDribblesPercentage`
- `aerialDuelsWonPercentage`
- `totalDuelsWonPercentage`
- `tacklesWonPercentage`
- `goalConversionPercentage`

También se utilizan `goalsAssistsPerMins` y `resultado_competicion`. `goals`, `assists`, `minutesPlayed`, `appearances`, `expectedGoals` y `expectedAssists` se muestran en las fichas si existen, pero no se añaden como características independientes al score; esto conserva la selección de variables del script ML proporcionado.

### 5. Restricciones por posición y signos

Los pesos editables representan la magnitud (entre 0 y 10), no el signo. El signo lo fija el programa para conservar las restricciones monótonas del modelo ML:

- Signo negativo: `bigChancesMissed`, `dispossessed`, `dribbledPast`, `errorLeadToGoal`, `errorLeadToShot`, `goalsConceded`, `ownGoals` y `penaltyConceded`.
- Todas las demás características utilizan signo positivo.

Para `Goalkeeper`, solo se puntúan las estadísticas permitidas en el modelo ML: `saves`, `goalsPrevented`, `goalsConceded`, `highClaims`, `punches`, `successfulRunsOut`, `cleanSheet`, `ownGoals`, `penaltyConceded`, `errorLeadToGoal`, `errorLeadToShot`, `accuratePassesPercentage` y `resultado_competicion`.

Para jugadores de campo se excluyen las cinco estadísticas exclusivas de portero: `saves`, `goalsPrevented`, `highClaims`, `punches` y `successfulRunsOut`.

### 6. Pesos por estadística y por competición

En la pestaña **Pesos por posición** se configuran dos tablas para cada posición:

1. **Pesos de las estadísticas:** definen la importancia relativa de cada variable para esa posición.
2. **Pesos de las competiciones:** cada competición individual tiene un peso independiente para esa posición.

Los pesos iniciales son 1.0 para todas las variables y competiciones elegibles. Son valores neutrales de partida, no pesos aprendidos ni una afirmación de que todas las estadísticas tengan la misma importancia real. Un peso 0 desactiva ese elemento.

Para cada posición `p` y competición `c`, la app identifica las estadísticas permitidas que tienen al menos un dato observado entre los jugadores de esa posición en esa competición. La suma de pesos de ese conjunto forma un denominador fijo para todos los jugadores de esa pareja posición-competición. De este modo, los NaN individuales no provocan que se vuelva a normalizar el score de ese jugador.

La puntuación de un jugador `i` en una competición `c` se calcula como:

\[
S_{i,c}=W_{p,c}\;\frac{\sum_{s\in A_{p,c}} d_s\,z_{i,c,s}\,w_{p,s}}{\sum_{s\in A_{p,c}}w_{p,s}}
\]

Donde:

- `W` es el peso manual de esa competición para la posición del jugador.
- `w` es el peso manual de la estadística para la posición.
- `d` es +1 o −1, según la restricción monótona.
- `z` es el z-score de la estadística.
- `A` contiene las estadísticas permitidas con datos observados para esa posición y competición.

La puntuación total es la suma de las contribuciones de las competiciones en las que tiene datos:

\[
S_i=\sum_c S_{i,c}
\]

La puntuación puede ser negativa; no es una probabilidad. El orden descendente de `score_puntuacion` define el ranking.

### 7. Qué ocurre con porteros y datos ausentes

Los porteros no se penalizan por no tener estadísticas de jugadores de campo: usan una lista de estadísticas permitidas propia y sus pesos se normalizan con las variables disponibles para porteros en cada competición. Los datos ausentes reciben contribución neutral, no se interpretan como una mala actuación. Esto ayuda a que la puntuación no dependa simplemente de contar cuántas columnas tiene cada posición.

Este ajuste no garantiza por sí solo que el score sea perfectamente comparable entre posiciones: la distribución y variabilidad de las estadísticas de porteros y jugadores de campo puede seguir siendo diferente. La configuración de pesos debe revisarse con rankings históricos y criterio futbolístico.

## Funcionalidades de Streamlit

- Seleccionar cualquier temporada del dataset, no solo 2025/26.
- Ver el ranking con puesto predicho, puesto real (cuando existe) y score.
- Buscar por nombre, club o posición.
- Desplegar la ficha individual de cada jugador, con estadísticas globales y pestañas por competición.
- Consultar el desglose de z-score, peso y contribución por estadística y competición.
- Comparar dos o más jugadores en estadísticas globales o en una competición individual, incluyendo `resultado_competicion`.
- Descargar ranking, desglose de puntuaciones, fichas y comparación en CSV.
- Editar los pesos y guardarlos en `data/puntuador/pesos_puntuador.json` para conservarlos entre ejecuciones.

## Archivos descargables

La pestaña de ranking permite descargar:

- `ranking_puntuador_<temporada>.csv`: ranking actual según los pesos seleccionados.
- `desglose_puntuador_<temporada>.csv`: contribución de cada estadística por jugador y competición.

El JSON de pesos guarda la configuración manual por posición. El ranking cambia al modificar pesos; para conservarlos en próximas ejecuciones, pulsa **Guardar pesos**.

## Limitaciones a tener en cuenta

- El ranking es un puntuador transparente y configurable, no una predicción probabilística del ganador.
- Los pesos iniciales no están optimizados automáticamente.
- Las estadísticas dependen de la calidad y cobertura del dataset, que puede variar entre temporadas y competiciones.
- El z-score es relativo al grupo de jugadores observados en una competición concreta; no equivale al rendimiento por 90 minutos. Por ello, los conteos acumulados y las tasas por minuto aportan información distinta.
- Las puntuaciones de distintas posiciones conviene evaluarlas con cuidado, porque las distribuciones de sus estadísticas pueden ser diferentes.
