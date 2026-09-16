# Plan de experimentos (ARTÍCULO) — Revisión Major (Machine Learning, SI Discovery Science 2025)

**Paper:** *On the Minimization of Graph Counterfactual Explanations: Theory and a Local Bounded Search Algorithm*
**Estado:** Major Revision. Plazo de reenvío: **25 jun 2026**.
**Alcance:** estos experimentos son **solo para el artículo**. La tesis NO se re-corre.
**Respuestas a revisores:** `document/paper/response_to_reviewers(esp).tex` y `(eng).tex`.

> Convención: "LBS" = `local_search.py`. "OBS"/"DBS" = backward search oblivious / data-driven (Abrate & Bonchi). "OFS"/"DFS" = forward search oblivious / data-driven. "DDBS" = DFS+DBS. "RHC" = hill-climbing genérico.

---

## Diseño central: protocolo desacoplado
- Cada **generador se ejecuta UNA vez** por dataset (su **semilla se guarda** para reproducibilidad); sus contrafactuales iniciales válidos se **reutilizan** para todos los minimizadores.
  - *Honestidad/ingeniería:* hoy el código re-ejecuta el generador junto a cada minimizador, pero como la semilla del generador es fija el conjunto de CFs es **idéntico** y los números no cambian. El refactor "generar una vez y guardar/reutilizar" (separar generación de minimización a nivel de ingeniería) queda como **mejora futura**; no afecta tablas.
- Consecuencias: comparación entre minimizadores **justa** (mismas entradas); **correctness idéntica** entre minimizadores (propiedad del generador); la **única estabilidad estocástica a evaluar es la de LBS**.
- **Semillas múltiples SOLO para LBS** (estabilidad, E1b). El resto corre con su semilla fija guardada (reproducible).
- **4 datasets:** Tree-Cycles, ASD, Synthie, BBBP (con y sin atributos).
- **4 generadores:** OFS, DCE, RSGG, DFS.
- **4 minimizadores:** OBS, DBS, LBS, RHC.
- **Métricas:** GED, FED (solo atributados: Synthie, BBBP), Oracle Calls. **Correctness NO en barras** (una vez por generador, E4).

## Prerrequisitos (antes de correr)
1. **Guardar contrafactuales por instancia** (no solo promedios) → habilita el reuse desacoplado y la métrica de plausibilidad (E2) a posteriori.
2. **Fijar y guardar la semilla del generador** (reproducibilidad). Para LBS: 3–4 semillas distintas (E1b).
3. **Verificar que RSGG es determinista** en inferencia (nota para coautores; no parece muestrear).
   *(Las 2 estrategias de atributos faltantes ya fueron re-agregadas; las 8 del paper están activas.)*

## Definiciones clave
- **RHC (hill-climbing):** búsqueda genérica sobre las mismas operaciones (del/swap/add) **sin** la política de prioridad ni el sobrepaso de LBS, al **mismo presupuesto de oracle calls** que LBS. Aísla si la ventaja de LBS viene de la *estrategia* y no del acceso al oráculo (R1.3).
- **DDBS = DFS + DBS** (pipeline completo); además DBS se evalúa desacoplado con los otros generadores.

---

## Experimentos

### E1 — Matriz principal (corrida única) + barras  ·  (R1.3, R1.4, R2.8, R2.9)  ·  PRIORIDAD ALTA
- Matriz **4 datasets × 4 generadores × 4 minimizadores**, **una corrida** bajo el protocolo desacoplado (semilla del generador fija/guardada).
- **Regenerar los gráficos de barras** (GED, FED, OC):
  - **leyenda única compartida** al pie de cada figura (R2.8);
  - **FED solo en Synthie y BBBP** (R2.9); omitir FED en TCR/ASD;
  - **correctness fuera de las barras** (va en E4).
- **Encuadre:** comparación **within-generator** como principal; cross-generator con cautela (R1.4).

### E1b — Estabilidad estocástica de LBS  ·  (R1.5, R2.12)  ·  PRIORIDAD ALTA
- Tabla nueva: **LBS × DCE × 4 datasets × 3 semillas** (hasta 4 si el cómputo lo permite), con **media ± desviación estándar sobre las semillas**. Es lo único que se re-ejecuta multi-semilla.
- **NO hacemos test de significancia (Wilcoxon).** Decisión: la tabla de estabilidad (media ± std) basta para responder a R1.5/R2.12; añadir un test formal sería hacer overfit a la petición del editor y podría abrir preguntas nuevas (normalidad, corrección por comparaciones múltiples, etc.). Mantener el reporte simple.

### E2 — Plausibilidad / cercanía a instancias reales  ·  (R1.6)
- Métrica: **distancia (GED) de cada contrafactual al grafo real más cercano de la clase objetivo**, LBS vs baselines.
- **A posteriori** sobre los contrafactuales guardados (prerrequisito 1); NO calcular GED contra el dataset completo en cada corrida.

### E6 — Comprobación on-manifold / no-adversarial  ·  (R2.5)  ·  OPCIONAL / FUTURO (solo si hay tiempo)
**Estado:** el revisor pidió *discutir* la línea CF vs. adversarial; eso ya está cubierto por la **discusión teórica** añadida al paper (subsección "Relationship to adversarial examples", al final del Método, `document/paper_reviewed/sections/algorithms_gs.tex`). Este experimento E6 lo **demostraría** empíricamente, pero **NO está comprometido** para esta revisión y el paper no hace ninguna afirmación empírica sobre él. Correrlo solo si sobra tiempo.
**Objetivo (si se corre):** demostrar que los contrafactuales de LBS son genuinos (on-manifold / justificados), no adversariales. Fundamento teórico y citas en la subsección citada y en el `\Resp` de R2.5 (Freiesleben 2022; Wachter et al. 2018; Pawelczyk et al. 2022; Browne & Swift 2020; Laugel et al. 2019; Van Looveren & Klaise 2021).
- **Métricas (post-hoc, caja negra, sobre los cf dumps; prerrequisito 1):**
  1. **Distancia a la instancia real más cercana** de la clase objetivo (GED + distancia de atributos). Más baja = más on-manifold. (Reúsa E2.)
  2. **Tasa de justificación (Laugel et al. 2019):** fracción de contrafactuales que son **ε-justificados**, es decir ε-conectados (cadena finita de pasos ≤ ε, todos de la misma clase predicha por el oráculo) a una instancia real de la clase objetivo correctamente clasificada. Equivale a co-clustering DBSCAN con minPts=2; se calcula solo con instancias guardadas + etiquetas del oráculo. Barrer un rango de ε y reportar la curva o un ε fijo justificado.
  3. **(Opcional) IM1 (Van Looveren & Klaise 2021):** razón de error de reconstrucción de un autoencoder de la clase objetivo vs. el de la clase original (requiere entrenar autoencoders de grafo por clase; marcar opcional).
- **Baseline adversarial (clave):** implementar un minimizador **adversarial explícito** = perturbación estructural mínima de caja negra buscada **directamente desde el grafo original** (sin semilla del generador, sin anclaje a datos) hasta voltear la etiqueta (estilo Nettack/greedy-flip, pero de caja negra). Correr las MISMAS métricas 1-3 sobre él.
- **Resultado esperado (a confirmar):** LBS (sobre todo con DCE/DCEM in-distribution) cae **cerca** de instancias reales y con **alta** tasa de justificación; el baseline adversarial logra GED comparable pero **lejos** de la variedad y con **baja** justificación. Esa separación es la prueba empírica.
- **Reporte (si se corre):** por generador (OFS, DCE, RSGG, DCEM) + el baseline adversarial. Habría que crear la tabla (distancia a instancia real más cercana + tasa de justificación). *Nota:* se creó y luego se retiró una tabla plantilla del paper (`manifold_check.tex`) al decidir dejar solo la discusión teórica; recuperarla de git si se decide correr E6.
- **No requiere re-correr** las minimizaciones (usa los cf dumps de E1); faltaría implementar el baseline adversarial + el cálculo de las métricas. Solo `results/`, no `results-legacy`.

### E3 — Ablación del orden de estrategias en Synthie  ·  (R2.11)
- Sobre **Synthie**: orden propuesto (del > swap > add, 1a antes que 1b) vs **`var_3` y `var_4`** (permutan prioridades). Reportar GED/FED/OC. Distinto de la ablación de inclusión (PI6, var_1/var_2).

### E4 — Correctness como propiedad del generador  ·  (R1.4, R2.6)
- Reportar la **fracción de instancias válidas por generador** (= correctness) **una sola vez**, fuera de las barras (p. ej. junto a la descripción de cada generador, **no** en material suplementario). Indica el subconjunto sobre el que se promedian GED/FED.

### E5 — Diagnóstico de presupuesto, Tabla 4  ·  (R2.10)
- **Sin correr nada (re-presentación):** (a) separar columnas de oracle calls **generador** vs **minimizador**; (c) aclarar que el tope es **por instancia** y que OC/GED son promedios (terminación por convergencia o presupuesto).
- **Solo si hay tiempo (re-correr el ablation):** (b) % de instancias que topan el presupuesto + mediana de OC por instancia.

### Tabla 3 (DCE/DCEM, BLS vs OBS)
- Se deja **tal cual**, solo se **quita la columna de correctness**.

---

## Mapeo comentario → experimento
| Comentario | Experimento |
|---|---|
| R1.3 (baselines) | E1 (DBS, RHC a presupuesto igualado) |
| R1.4 (within/cross + instancias válidas) | E1 (within-generator) + E4 |
| R1.5 / R2.12 (estabilidad, semillas) | **E1b** (LBS × DCE × 4 datasets × 3 semillas; sin Wilcoxon) |
| R1.6 (plausibilidad / moderación) | E2 |
| R2.5 (contrafactual vs. adversarial) | **E6** (cercanía a instancias reales + tasa de justificación + baseline adversarial) |
| R2.6 (correctness) | E4 + Tabla 3 (quitar columna) |
| R2.8 (leyenda compartida) | E1 |
| R2.9 (FED solo atributados) | E1 |
| R2.10 (oracle calls Tabla 4) | E5 |
| R2.11 (orden de estrategias) | E3 |

## Orden de ejecución sugerido
1. Prerrequisitos (guardar CFs por instancia; semilla del generador guardada).
2. **E1** (matriz única; de aquí salen barras y los datos para E2/E4).
3. **E1b** (estabilidad de LBS, lo único multi-semilla).
4. **E3** (ablación de orden en Synthie).
5. **E2** (plausibilidad, a posteriori sobre E1).
6. **E4** y **E5(a,c)** (derivables sin nuevas corridas).
7. **E5(b)** solo si sobra tiempo.

## Descartado (NO se hace en el artículo)
- Matriz **multi-semilla** completa → corrida única + tabla de estabilidad **solo de LBS**.
- Semillas comunes/justificación de determinismo en generación → **protocolo desacoplado** (generador una vez, semilla guardada).
- Multi-semilla en PROTEINS y COLORS-3 (quedan solo en la Tabla 3, corrida única).
- Recomputar cross-generator sobre subconjunto común de instancias.
- Selector / GA como sección nueva del artículo (contribución de la tesis, no del paper).

---

## Pendiente para el próximo batch (no olvidar)

### Correcciones de código
- **DBS baja la correctness del generador** (discrepancia detectada). Un minimizador NO debería bajar correctness; DBS a veces devuelve un no-contrafactual. Hay que: (1) revisar la causa en `dbs.py`, (2) arreglarla, (3) **re-correr todos los DBS**. Mientras tanto, en `stats_visualizer` la correctness de DBS se iguala a la del generador (se excluye DBS del cálculo de correctness por generador) como referencia temporal; al re-correr, quitar ese parche y usar la correctness real de DBS.
- **Asimetría de presupuesto LBS vs RHC**: LBS chequea `max_oracle_calls` solo al inicio del loop externo y puede sobrepasarlo en cientos/miles de llamadas; RHC corta exacto. Igualar el enforcement (cortar dentro del loop de candidatos en LBS) o reportar el OC medido por corrida en la tabla, para que el "mismo presupuesto" sea verificable (R2.10 / R1.3).

### Experimentos (segundo batch)
- **E3 (ablación de orden)**: correr `var_3` (add antes que swap, inner) y `var_4` (1b antes que 1a, outer) sobre **Synthie + DCE**. Los configs ya existen (`synthie/dce/dce-lcls-var-3` y `-var-4`, single-run) pero NO están en `REVISION_EXECUTION_ORDER.md`; hay que añadirlos y correrlos.
- **Ablación de inclusión (PI6)**: `var_1` / `var_2` (dejan fuera una estrategia) si se decide incluirla en el artículo (hoy es material de tesis).
- **E2 (plausibilidad / R1.6, R2.5)**: implementar el post-proceso sobre los cf dumps por instancia (distancia GED del contrafactual al grafo real más cercano de la clase objetivo, LBS vs baselines). No requiere re-correr; falta el cálculo + tabla/figura. Solo usa `results/` (los cf dumps), no `results-legacy`.
- **E5(b) (solo si sobra tiempo)**: % de instancias que topan el presupuesto de oracle calls + mediana de OC por instancia. Requiere re-correr el ablation de presupuesto.

### Mejoras de ingeniería (no bloquean tablas)
- **Generar una vez por dataset y reutilizar** los contrafactuales iniciales entre minimizadores (hoy el generador se re-ejecuta con cada minimizador; con semilla fija da lo mismo, pero desperdicia cómputo). Separar generación de minimización a nivel de pipeline.

### Tablas / notebooks (al terminar la corrida actual)
- Refrescar Table A (matriz no-seed), Table B (estabilidad LBS+DCE) y el notebook global; regenerar las figuras del paper.
- La corrida actual (este batch) cubre: matriz no-seed `4 ds x 4 gen x 4 min` + `dce-lcls` semillas 1-3 (E1b). Pendientes ya encolados (ver `REVISION_EXECUTION_ORDER.md`): los lcls/obs que faltan, `tcr-tco-300_rsgg_dbs`, completar `asd_rsgg_lcls` (4/10), `tcr-tco-300_rsgg_rhc` (9/10), `bbbp_ofs_obs` (re-corre completa), y las semillas LBS de synthie/bbbp/tcr.
