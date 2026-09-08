# Estado de los experimentos del modelo selector (sesión 2026-09-03 a 2026-09-08)

Documento de continuidad. Resume qué se construyó, qué se midió, qué se concluyó y qué queda por hacer, para retomar el trabajo tras clonar la rama `dev-bls-selector` en una máquina nueva. Los resultados numéricos citados están copiados en `lab/selector_experiments/` (JSON de resultados por scope, CSV del monitor, logs comprimidos) porque `lab/output/` está ignorado por git.

## 1. Punto de partida

El selector de la tesis (`src/explainer/future/metaheuristic/Tagging/OnlineSelector.py` + `local_search/local_search_selection_net.py`, "v1") reduce las oracle calls de BLS entre 2 y 10 veces en los experimentos de la tesis. La auditoría de esta sesión encontró:

- La variante con selector cambia también el esqueleto de búsqueda (12 intentos por nivel frente a 64/16 de la base, tope local `k_local`, abandono tras la primera iteración externa sin mejora), así que la reducción de OC no era atribuible solo al modelo.
- El modelo de adición de v1 casi no entrena (13 a 300 pasos de Adam en los checkpoints de la tesis frente a miles del de eliminación).
- El muestreador del pool de adición de v1 agota 500 000 intentos por llamada en grafos pequeños (0.45 s por llamada, llamado en cada intento de swap) y solo cubre pares incidentes al foco.
- Contexto incorrecto en el positivo final de adición, exploración epsilon inexistente en las fases principales, y actualizaciones perdidas si varios procesos comparten el checkpoint.
- `lab/output/results/<ds>_dce_lcls` (LBS base) contiene números **calibrados** (Synthie: GED x1.128, OC x1.375; BBBP: factor no uniforme). Los originales están en `lab/output/_backup_precalibration/results/`. Toda comparación nueva debe usar el backup sin calibrar.
- LBS base no respeta `max_oracle_calls` dentro de una iteración externa (en PROTEINS-10 con tope 1500 gastó 12 882 llamadas de minimizador). Hay que corregirlo antes de cualquier comparación a presupuesto igual.

## 2. Código nuevo (todo en la rama, v1 intacta)

- `src/explainer/future/metaheuristic/Tagging/OnlineSelectorV2.py`: selector v2. Características vectorizadas por snapshot (A_t = A XOR S), 14 rasgos nuevos (Adamic-Adar, resource allocation, caminos de longitud 3, cubetas de distancia, cierre de triángulo), enumeración completa de candidatos de adición, ensemble de 4 cabezas con prior aleatorio y Thompson sampling, orden de prueba por Gumbel-top-k con logQ, etiquetas blandas desde `predict_proba`, pooling `sum_bias` / `noisy_and` / `mean`, pérdida listwise con corrección logQ, buffer de repetición reservoir persistido, AdamW, L2 hacia la inicialización, shrink-and-perturb al cargar, reciclado de unidades dormidas, veto opcional, `lr_add` y pasos extra de replay para el modelo de adición. Checkpoints en `models/edge_selector_<tag>_<dataset>.pt` (formato version 2).
- `src/explainer/future/metaheuristic/local_search/local_search_selection_net_v2.py`: BLS guiado por v2. Mismo esqueleto y orden de estrategias que v1, con todas las variantes v2b..v2g como flags de configuración (ver sección 4). `selector_mode: random` ejecuta el mismo esqueleto con propuestas uniformes (ablación). `BinaryModelV2` usa `predict_proba` (una llamada, igual coste que `predict`).
- `src/explainer/future/metaheuristic/Tagging/selector_monitor.py`: monitor precuencial sin coste en oracle calls. Línea `[monitor]` cada N llamadas y una fila por instancia en `lab/output/selector_logs/selector_<tag>_<dataset>.csv`: pasos, buffer, tasa de positivos, log-loss, skill frente a tasa base, AUC, intentos por éxito, ventaja de rango, EMAs de pérdidas, normas de gradiente y de actualización, desacuerdo entre cabezas, unidades dormidas, llamadas y éxitos por fase (eliminación / swap / adición), tamaño de bloques aceptados, reintentos, estadísticas por tamaño de bloque, cuotas por fase y éxitos por método de atributos. Ojo: la AUC y la ventaja de rango tienen sesgo de cascada (el aceptado es el último probado); la métrica limpia es intentos por éxito frente a la ablación aleatoria.
- `scripts/analyze_selector_runs.py`: `compare` (por instancia entre scopes, acepta rutas relativas a `lab/output` para la base sin calibrar), `logs` (curvas dentro del fold desde los logs), `monitor` (tendencias del CSV), `feedback` (tabla por instancia + desglose por fases).
- `src/evaluation/future/evaluator.py`: parámetro opcional `instance_ids` del evaluador para correr sobre un subconjunto fijo de instancias (protocolo PROTEINS-10).
- `src/evaluation/future/stages/fed.py`: `float()` en la FED (antes escribía un escalar numpy float32 que jsonpickle serializaba como diccionario).
- Configs: `lab/config/generate_minimize/{synthie,bbbp}/dce/dce-lcls-net-v2{,b,c,d,e,f,g}{,-random}/` (10 folds, semilla 0) y `lab/config/generate_minimize/proteins10/dce/*` (fold 0, 10 instancias fijas, `max_oracle_calls` 1500). `proteins15/` es el intento previo con 15 instancias y presupuesto completo (descartado por lento). Ids fijos en `lab/config/snippets/proteins_10_instances.json` y `proteins_15_instances.json`.
- `SELECTOR_V2_EXECUTION_ORDER.md`: cola para `scripts/run_revision_queue.py --list SELECTOR_V2_EXECUTION_ORDER.md --workers 1` (en serie, porque los folds comparten checkpoint).
- Checkpoints v1 de la tesis apartados como `models/edge_selector_<ds>-<hash>.pt.thesis-bak` para que v1 arranque en frío en las comparaciones.

## 3. Resultados

Todos los métodos arrancan en frío, generador DCE con semilla fija (mismos contrafactuales iniciales para todos), base sin calibrar.

### Synthie, folds 0 y 1 (80 instancias, presupuesto 10 000)

| Variante | GED media | GED máx | OC media | OC mediana | GED vs base (mejor/igual/peor) | OC menores que base |
|---|---|---|---|---|---|---|
| LBS base | 2.84 | 44 | 2847 | 753 | | |
| v1 | 5.84 | 155 | 1607 | 1136 | 9/53/18 | 26/80 |
| v2 | 2.42 | 16 | 1823 | 1354 | 9/58/13 | 26/80 |
| **v2b** | **1.95** | 11 | 2014 | 470 | 13/60/7 | 72/80 |
| v2b-random | 1.99 | 11 | 2188 | 456 | 13/61/6 | 74/80 |
| v2c | 2.11 | | 2004 | 475 | 14/58/8 | 74/80 |
| v2e | 13.9 | | 1574 | 755 | 5/55/20 | 40/80 |
| v2f | 4.83 | | 2098 | 712 | 12/57/11 | 45/80 |
| v2f-random | 4.94 | | 2100 | 730 | 13/59/8 | 49/80 |

v2b frente a v2b-random: GED igual en 67, 6 a favor de una y 7 de la otra; OC 34 a 31. En Synthie el modelo no aporta nada medible: el 75 % de las eliminaciones sueltas funcionan y el ranking no puede mejorar el azar.

### PROTEINS-10 (fold 0, ids 22, 122, 157, 186, 270, 504, 564, 701, 844, 1075, presupuesto 1500)

| Variante | GED media | OC minimizador (media) | GED vs base (mejor/igual/peor) |
|---|---|---|---|
| LBS base | 50.8 | 12 882 (sobrepasa el tope) | |
| v1 | 40.0 | 728 | 5/4/1 |
| v2 | 45.2 | 782 | 5/2/3 |
| v2b | 56.0 | 1351 | 2/3/5 |
| v2c | 61.8 | 1350 | 2/3/5 |
| v2d | 61.4 | 1354 | 2/2/6 |
| **v2e** | **37.1** | 1092 | 6/3/1 |
| v2e-random | 39.2 | 1226 | 4/4/2 |
| v2f | 39.2 | 1357 | 5/2/3 |
| v2f-random | 38.3 | 1357 | 5/3/2 |

Por instancia (GED final; inicial entre paréntesis): id 22 (109): base 18, v1 23, v2 23, v2e 22, v2e-random 18. id 122 (121): 114, 104, 82, 77, 79. id 157 (163): 154, 69, 69, 66, 77. id 186 (59): 5, 5, 59, 5, 5. id 270 (71): 30, 25, 22, 25, 28. id 504 (80): 41, 35, 31, 34, 29. id 564 (110): nadie la reduce. id 701 (87): todos llegan a 1 (v2b en 10 llamadas gracias a bloques de 32). id 844 (94): 22, 15, 19, 19, 30. id 1075 (114): 13, 13, 36, 12, 15.

En PROTEINS el ranker sí aporta, poco pero de forma consistente: 13.4 llamadas por eliminación conseguida (v2e) frente a 17.4 (v2e-random), AUC precuencial 0.74 a 0.80, 1.8 a 2.4 intentos por éxito. La base necesita unas 700 llamadas por mejora. Swap y adición no consiguen ningún éxito en PROTEINS con este presupuesto (0 éxitos en más de 14 000 llamadas acumuladas). Los métodos de atributos productivos son `random_walk_diffusion` (unos 390 de 560 éxitos) e `identity`; los otros cinco casi nunca.

### BBBP

Solo se completó v2 en el fold 0 (204 instancias); v2b quedó en 36/204 al cambiar a PROTEINS. No analizado. La base de BBBP en `results/` también está calibrada (usar el backup).

## 4. Variantes y por qué existen

Todas son flags de `local_search_selection_net_v2.LocalSearch` sobre el mismo código. `model_tag` separa los checkpoints.

- **v2**: rediseño del selector, esqueleto v1. Bien en Synthie y PROTEINS, pero abandona la instancia tras la primera iteración externa sin mejora (deja presupuesto sin usar) y elimina una arista por barrido.
- **v2b** (`block_removal`, `block_init 8`, `block_random_mix`, `retry_outer`, pooling `noisy_and`, `veto 0`, `swap_record_add false`, `lr_add 2e-3`, `extra_replay_steps_add 3`): escalera descendente de bloques desde 8 (x2 tras éxito), mitad de intentos rankeados y mitad al azar, reintentos mientras quede presupuesto (semántica de la base). Mejor en Synthie. En PROTEINS los bloques casi nunca funcionan y cada escalón fallido cuesta 84 llamadas (12 intentos x 7 métodos de atributos), así que el barrido (-) se come el tope local y swap/adición nunca corren.
- **v2c** (`block_init 1`, `block_shrink_on_fail`, `phase_budget` 0.5/0.25/0.25 fijas, `block_example_weight 0.5`, pooling `sum_bias`): cuotas fijas por fase. Peor en PROTEINS: gasta el 40 % en swap/adición improductivas y los escalones de bloque siguen costando antes de llegar al tamaño 1.
- **v2d** (`block_policy bandit` con priors uniformes, `adaptive_phase_budget`, `adaptive_method_order`): el bandit con priors uniformes prueba primero bloques grandes y quema el presupuesto. Peor en PROTEINS. Solo se completó PROTEINS-10.
- **v2e** (`block_policy unlock`: el tamaño 2k solo se prueba cuando el tamaño k tiene tasa >= 0.5 en >= 6 intentos, `fill_remove_share`, cuotas adaptativas con suelo 0.05, `adaptive_method_order`): mejor en PROTEINS (las cuotas convergen a 0.90/0.05/0.05). Se hunde en Synthie porque en la fase final swap y adición quedan con el 5 % del presupuesto y `fill_remove_share` gasta el 90 % del presupuesto local en barridos de eliminación que fallan.
- **v2f** (v2e sin `fill_remove_share`, cuota fija de eliminación 0.5 y swap/adición sin tope): empata con v2e en PROTEINS; en Synthie cae por la instancia 199 (232 aristas, ninguna eliminación suelta posible): la regla de desbloqueo excluía para siempre el tamaño 4 tras un primer fracaso, mientras que v2b la dejó en 8 con bloques al azar.
- **v2g** (`block_policy bandit` con prior Beta(1, tamaño), 6 intentos por escalón de bloque, hasta 2 tamaños de bloque más el tamaño 1 siempre, `bandit_min_value 0.3`, cuotas tipo v2f, `adaptive_method_order`): diseñada para unificar v2b y v2e. **No llegó a correr** (la máquina se apagó con 0/10 en PROTEINS y 5/40 en Synthie; los restos están en `lab/output/discarded/v2g_partial`). Es lo primero a lanzar al retomar.

## 5. Conclusiones de la ronda de calibración

1. El esqueleto de búsqueda explica casi toda la ganancia en OC y GED; el ranker aprendido aporta poco (nada medible en Synthie, un 20 a 25 % menos de llamadas por eliminación en PROTEINS). Para el paper hay que reportarlo así y cualquier propuesta de modelo debe batir a la ablación aleatoria del mismo esqueleto, no a la base.
2. El reparto del presupuesto entre fases y el tamaño de los bloques de eliminación son dependientes del dataset: PROTEINS solo produce en eliminación de una arista; Synthie necesita bloques grandes en el descenso y cientos de llamadas de adición en la fase final.
3. Ninguna variante gana en ambos datasets. La candidata para escalar debe igualar a v2b en Synthie y a v2e en PROTEINS; v2g es la propuesta pendiente.
4. El orden adaptativo de métodos de atributos es una mejora gratuita y debería pasar también a la base.
5. Con 10 instancias por dataset y una sola corrida las diferencias de 2 a 3 puntos de GED en PROTEINS no son concluyentes; la decisión de escalar debe basarse en las tendencias del monitor (llamadas por éxito, éxitos por fase) además de la GED.

## 6. Siguientes pasos

1. Clonar la rama, recrear el entorno (`environment.yml`, entorno conda `GRTL`), regenerar las cachés de datasets y oráculos (no viajan en git: `lab/data/cache/datasets`, `lab/data/cache/oracles`; los oráculos GCN se reentrenan con la config del snippet do-pair). Sin la misma caché del oráculo los números no serán idénticos a los de aquí.
2. Lanzar v2g y v2g-random en PROTEINS-10 y Synthie folds 0-1 (configs listas): `python tests/run_experiments.py --one lab/config/generate_minimize/proteins10/dce/dce-lcls-net-v2g/generate_minimize0.jsonc --run-number 1` y los equivalentes; analizar con `scripts/analyze_selector_runs.py feedback ... --monitor lab/output/selector_logs/selector_v2g_<ds>.csv`.
3. Corregir la asimetría de presupuesto de LBS base (tope dentro de la iteración) como opción `strict_budget` para no alterar los hashes de las configs del paper.
4. Si v2g funciona en ambos: escalar con `SELECTOR_V2_EXECUTION_ORDER.md` (añadir tier v2g) contra base sin calibrar y v2g-random, 10 folds de Synthie y BBBP, y PROTEINS con más instancias.
5. Ideas de modelado pendientes que sí podrían mover la aguja en PROTEINS: usar el ranking para decidir el orden de los métodos de atributos por candidato, pretraining offline del ranker con los logs ya guardados (parejas contexto/arista/etiqueta del buffer), y transferencia del checkpoint entre folds (ya ocurre dentro de un dataset).

## 7. Dónde está cada cosa

- Código: rutas en la sección 2.
- Resultados copiados para git: `lab/selector_experiments/results/` (JSON `results_<fold>_1.json` por scope, incluida la base sin calibrar de Synthie, BBBP y PROTEINS-10), `lab/selector_experiments/selector_logs/` (CSV del monitor por variante y dataset), `lab/selector_experiments/queue_logs/` (logs de las corridas relevantes, comprimidos).
- En la máquina original, además: `lab/output/discarded/` (intentos fallidos de v2b y restos de v2d, v2g), `lab/output/_backup_precalibration/` (base sin calibrar completa), `lab/output/results/` (todo).
- Memoria de la sesión de Claude: `~/.claude/projects/-home-rodrigo-projects-GRETEL/memory/` (no viaja en git; el contenido relevante está en este documento).
