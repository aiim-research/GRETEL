# Plan de experimentos y notas de implementación — Revisión Major (Machine Learning, SI Discovery Science 2025)

**Paper:** *On the Minimization of Graph Counterfactual Explanations: Theory and a Local Bounded Search Algorithm*
**Estado:** Major Revision. Plazo de reenvío: **25 jun 2026**.
**Propósito de este archivo:** listar los experimentos a correr para responder a los revisores (con el objetivo de cada uno y el comentario que atiende) y las notas de implementación previas. La redacción de respuestas vive en `document/paper/response_to_reviewers.tex`.

> Convención: "LBS" = `local_search.py` (Búsqueda Local Acotada). "obs" = Oblivious Backward Search (baseline de minimización actual).

---

## Mapeo comentario del revisor → experimento

| Comentario | Qué pide | Experimento |
|---|---|---|
| R1.5 / R2.12 | Estadística (std, IC, tests, semillas) | **Exp. 1** (multi-semilla) |
| R2.11 | Ablación del *orden* de estrategias | **Exp. 2** (ordenamiento) |
| R1.3 | Baselines más fuertes | **Exp. 3** (DDBS + random/hill-climbing igualado) |
| R2.10 | Oracle calls de Tabla 4 (obs>max; GED vs presupuesto) | **Exp. 4** (diagnósticos de OC) |
| R1.4 / R2.6 | Limpieza cross- vs within-generator; correctness | **Exp. 5** (conteo de instancias válidas) |
| R1.6 / R2.5 | Plausibilidad / contrafactual vs adversarial | **Exp. 6** (métrica B-lite — *opcional, decisión coautores*) |
| R1.4 (Opción B) | Comparación cross-generator más limpia | **Exp. 7** (subconjunto común — *opcional, decisión coautores*) |
| R1.2 | Método "principiado" / extensiones | **Exp. 8** (Selector/GA — *resultados ya en tesis; decisión de inclusión*) |

---

## Prerrequisitos de implementación (hacer ANTES de relanzar)

### Nota A — Guardar los contrafactuales por instancia
Hoy el pipeline guarda solo **métricas agregadas** (promedios sobre el test set), **no** los grafos contrafactuales `G''` de cada instancia.
- **Acción:** modificar el guardado de resultados para persistir, por instancia, el contrafactual obtenido (estructura + atributos) junto con su GED/FED/OC.
- **Por qué:** permite calcular métricas *a posteriori* (p. ej. la de plausibilidad del Exp. 6, o cualquier análisis nuevo que pida un revisor) **sin re-lanzar** todos los experimentos. Dado el costo de cómputo, esto es clave.

### Nota B — Re-agregar las 2 estrategias de atributos faltantes
`local_search.py` (`self.methods`) tiene actualmente **6** estrategias; el paper reporta **8**.
- **Faltan:** `No changes` y `Average smoothing with disconnection`.
- **Acción:** re-agregarlas a `self.methods` (o a la rama atribuida de `evaluate`) **antes** de relanzar, para que los resultados coincidan con las 8 estrategias descritas en el paper (y con la definición `m = |A| = 8`).
- **Por qué:** consistencia entre código, paper y la respuesta a R2.3/R2.4. Si se relanza con 6, los números no cuadrarán con el manuscrito.

### Nota C — Fijar semillas (necesario para Exp. 1 y 2)
`local_search.py` usa `random` y `numpy.random` **sin semilla fija**. Para reproducibilidad y multi-semilla:
- **Acción:** sembrar explícitamente `random.seed(s)`, `numpy.random.seed(s)` (y `torch.manual_seed(s)` si el generador/oráculo lo requiere) por corrida, exponiendo `s` como parámetro de configuración.

---

## Experimentos

### Exp. 1 — Reporte estadístico multi-semilla  ·  (R1.5, R2.12)  ·  PRIORIDAD ALTA
- **Objetivo:** demostrar la **estabilidad** de las mejoras de LBS, que hoy se reportan como corrida única. Es el *house style* de la revista y el comentario de mayor retorno.
- **Qué correr:** las configuraciones principales (LBS vs obs, los 3 generadores de las figuras) con **5 semillas independientes**.
- **Datasets:** **Tree-Cycles, Synthie, ASD, BBBP, ENZYMES, BZR, AIDS** (los 7 factibles). **Excluir PROTEINS y COLORS-3** (costo prohibitivo) — declarar la exclusión en el paper.
- **A reportar:** **media ± desviación estándar** e **IC al 95 %** en tablas; **barras de error** en las Figs. 2–4; **test de Wilcoxon de rangos con signo** (pareado, LBS vs obs) con su p-valor y tamaño de efecto.
- **Notas:** pareado porque LBS y obs parten de las **mismas** semillas iniciales por instancia. GED no es normal → no usar t-test.

### Exp. 2 — Ablación del ordenamiento de estrategias  ·  (R2.11)
- **Objetivo:** justificar **empíricamente** que el orden de prioridad fijo (1a → 1b → 2 → 3, es decir del > swap > add) es adecuado; hoy solo se justifica por el objetivo.
- **Qué correr:** el orden propuesto vs **órdenes alternativos representativos**, p. ej.: (i) swap antes que deletion; (ii) addition con mayor prioridad; (iii) 1b antes que 1a.
- **Datasets:** subconjunto representativo (atribuido vs no-atribuido), p. ej. BBBP, Synthie, TCR, ASD; generador DCE para aislar el efecto del orden.
- **A reportar:** GED/FED/OC por orden. Tabla nueva. **Distinto** de la ablación de *inclusión* (Var 1 / Var 2) que ya existe (`local_search_var_1.py`, `local_search_var_2.py`).

### Exp. 3 — Baselines más fuertes  ·  (R1.3)
- **Objetivo:** descartar que la ventaja de LBS venga de un baseline débil.
- **3a — DDBS completo:** completar la evaluación de **DDBS** (Data-driven Backward Search, Abrate & Bonchi) en **todos** los datasets (hoy incompleto en varios). Localizar/integrar el explainer DDBS.
- **3b — Búsqueda genérica a presupuesto igualado:** implementar un minimizador **random search / random-restart hill-climbing** que use las **mismas** operaciones (del/swap/add) **sin** la política de prioridad, bajo el **mismo presupuesto de oracle calls** que LBS.
  - **Objetivo específico:** si LBS gana a la búsqueda genérica **con idéntico presupuesto**, la ventaja es de la *estrategia*, no del acceso al oráculo. Responde directamente a "parte de la ganancia puede venir de la debilidad del baseline".
  - **Implementación:** reutilizar la maquinaria de vecindad de `local_search.py`; reemplazar la lógica de prioridad por aceptación aleatoria/greedy simple.
- **Nota:** NO implementamos SA/tabú como baseline (se argumenta en el paper por qué requerirían rediseño específico del dominio → serían métodos nuevos, no baselines).

### Exp. 4 — Diagnósticos de oracle calls y presupuesto (Tabla 4)  ·  (R2.10)
- **Objetivo:** explicar las dos "anomalías" que señaló el revisor.
- **4a — Separar generador vs minimizador:** en la tabla de presupuesto, reportar en **columnas distintas** las oracle calls del **generador** y del **minimizador**. (Aclara por qué el OC de obs "excede" el máximo: el máximo acota al minimizador; el total incluye el generador (~405 en DCE/Synthie); obs agota su presupuesto → total ≈ presupuesto + generador.)
- **4b — Distribución por instancia:** añadir, por presupuesto, la **fracción de instancias que alcanzan el tope** (`max_oracle_calls`) y la **mediana del OC por instancia**.
  - **Objetivo específico:** hacer visible la distribución **bimodal** (instancias fáciles convergen baratas y dominan el promedio; difíciles topan el presupuesto). Explica por qué el GED promedio baja al subir el tope mientras el OC promedio sigue por debajo del tope.
- **Implementación:** `max_oracle_calls` ya se chequea por instancia (`if self.k > self.max_oracle_calls: break`); basta registrar, por instancia, si terminó por convergencia o por tope, y su `self.k` final.

### Exp. 5 — Conteo de instancias válidas por generador  ·  (R1.4, R2.6)
- **Objetivo:** sustentar que el claim principal es **within-generator** y que la *correctness* es esencialmente una propiedad del **generador** (no del minimizador).
- **Qué reportar:** número y **fracción de contrafactuales válidos** por generador y dataset (el subconjunto sobre el que se calculan GED/FED). Mover *correctness* a material suplementario.
- **Costo:** bajo (se deriva de las corridas existentes / del Exp. 1).

### Exp. 6 — Métrica de plausibilidad "B-lite"  ·  (R1.6, R2.5)  ·  OPCIONAL (decisión coautores)
- **Objetivo:** dar evidencia **cuantitativa** de plausibilidad / in-distribution, no solo discusión. Responde a la vez a R1.6 y R2.5.
- **Métrica:** distancia (GED) del contrafactual `G''` al **grafo real más cercano de la clase objetivo**, comparando LBS vs obs vs generador. Mide "¿parece un ejemplo real de la nueva clase?".
- **Requisitos:** depende de la **Nota A** (contrafactuales guardados por instancia). Correr como experimento **específico** — NO calcular GED contra el dataset completo a priori en todas las corridas (costoso).
- **Conjetura asociada (interesante):** LBS rinde mejor con DCE/DCEM (únicos generadores que dan **instancias reales**); conjeturamos que las semillas in-distribution favorecen simultáneamente efectividad y plausibilidad.

### Exp. 7 — Recomputo cross-generator en subconjunto común  ·  (R1.4 Opción B)  ·  OPCIONAL (decisión coautores)
- **Objetivo:** comparación cross-generator más limpia, sobre la **intersección** de instancias que **todos** los generadores resuelven (elimina el sesgo de "distinto subconjunto exitoso").
- **Costo:** medio (re-agregación; idealmente en la misma tanda del Exp. 1).

### Exp. 8 — Selector / GA: resultados preliminares  ·  (R1.2)  ·  DECISIÓN DE INCLUSIÓN
- **Estado:** ya hay resultados en la tesis (el selector reduce OC 2×–10×; variantes Trainable/Ponderation). Código: `local_search_selection_net*.py`, `local_search_trainable*.py`.
- **Decisión de coautores:** (a) mención + resultados preliminares como extensión/mejora [opción sugerida]; (b) sección nueva (cubre el ≥30 % de material nuevo del SI); (c) solo trabajo futuro.
- **Acción si se incluye:** consolidar las tablas del selector/GA con el formato estadístico del Exp. 1.

---

## Orden sugerido de ejecución
1. **Prerrequisitos** A, B, C (formato de guardado, 8 estrategias, semillas).
2. **Exp. 1** (multi-semilla) — el de mayor impacto; del Exp. 1 salen también los datos para Exp. 4 y Exp. 5.
3. **Exp. 3a** (DDBS) y **Exp. 2** (ordenamiento).
4. **Exp. 3b** (random/hill-climbing igualado).
5. **Exp. 4** (diagnósticos OC) y **Exp. 5** (conteos) — derivables de lo anterior.
6. Opcionales tras discusión: **Exp. 6** (B-lite), **Exp. 7** (subconjunto común), **Exp. 8** (Selector/GA).
