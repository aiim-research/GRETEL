# Casos en que el selector empeora a LBS vanilla (PROTEINS, fold 0)

Documento de trabajo para una investigación en otra sesión. Reúne el contexto, los logs completos de las tres instancias en las que el selector termina peor que LBS vanilla, el diagnóstico que se desprende de ellos y un prompt para lanzar una investigación profunda. Fecha: 2026-10-05. Rama `dev-bls-selector` de GRETEL.

## 1. Contexto

**Problema.** Minimización de explicaciones contrafactuales en grafos (GC-min): dado un grafo G, un clasificador de caja negra (aquí una GCN entrenada sobre PROTEINS) y un contrafactual inicial G' producido por un generador (DCE), encontrar un contrafactual válido cuya distancia de edición a G (GED, número de aristas alteradas) sea mínima. La solución se representa como el conjunto S de pares (u, v) que difieren entre G y G'. El problema es NP-duro. Cada consulta al oráculo cuenta como una llamada; el presupuesto es de 10 000 llamadas por instancia (protocolo de la tesis). En PROTEINS los nodos tienen atributos, y cada candidato se evalúa con hasta 7 métodos de manipulación de atributos (identidad, suavizados, difusión por paseo aleatorio, etc.), así que un candidato que falla cuesta 7 llamadas.

**LBS vanilla** (`src/explainer/future/metaheuristic/local_search/local_search.py`): búsqueda local acotada con mejora a la primera. En cada iteración externa: (1) barrido de eliminación: para niveles i = 0, step, 2·step, ... hasta |S|, prueba `neigh_factor^3 = 64` veces quitar i aristas elegidas al azar de S (es decir, prueba bloques aleatorios de todos los tamaños, incluidos bloques grandes); (2) si nada funciona, reduce S al azar y alterna swaps (quitar i y añadir i aristas al azar) y adiciones compensatorias, expandiendo; (3) repite mientras queden iteraciones (`n`, que se reinicia con cada mejora) y presupuesto. Con `strict_budget` corta exactamente a las 10 000 llamadas.

**El selector** (antes llamado v3b-pre-frozen; `local_search_selection_net_v2.py` + `Tagging/OnlineSelectorV2.py`): el mismo tipo de búsqueda pero guiada por una red (MLP de 2 capas, 64 unidades, 4 cabezas con prior aleatorio) que puntúa cada arista candidata con características estructurales y de atributos (100 dimensiones). Fue preentrenada offline con unos 13 700 ejemplos (movimiento, ¿cambió la clase?) recogidos con propuestas aleatorias en 72 instancias de PROTEINS, y queda congelada durante la búsqueda. Diferencias de su esqueleto respecto a vanilla (vienen de la variante v2e):

- Barrido de eliminación ordenado por la red: puntuaciones estandarizadas dentro del conjunto de candidatos y muestreo Plackett-Luce con temperatura 0.25 (casi voraz), memoria tabú para no repetir aristas ya probadas en el mismo estado.
- Bloques de eliminación con política `unlock`: siempre se prueba el tamaño 1; el tamaño 2k solo se desbloquea cuando el tamaño k ha tenido una tasa de éxito >= 0.5 en >= 6 intentos. Si las eliminaciones sueltas nunca funcionan, nunca se prueba un bloque.
- Presupuesto por iteración externa: tope local `local_budget_frac` 0.2 del total, repartido entre fases (eliminación / swap / adición) con cuotas adaptativas (suelo 0.05).
- Swap y reducción muestreados con puntuaciones estandarizadas a temperatura 4 (aleatoriedad similar a v2e).
- `retry_outer`: tras una iteración externa sin mejora, reintenta mientras `k <= max_oracle_calls`, pero el contador de iteraciones `n` (inicial `min(max_runtime=20, runtime_factor·|S|)`) baja 2 por reintento y no se reinicia hasta la siguiente mejora, así que el selector abandona tras unos 10 reintentos fallidos aunque quede presupuesto.

Parámetros del minimizador del selector:

```json
{
 "neigh_factor": 4,
 "runtime_factor": 4,
 "max_runtime": 20,
 "max_neigh": 20,
 "attributed": true,
 "max_oracle_calls": 10000,
 "seed": 0,
 "selector_mode": "frozen",
 "dense_reward": true,
 "adaptive_trials": true,
 "lock_whole_minimize": true,
 "model_tag": "selector",
 "selector": {
  "hidden_dim": 64,
  "num_hidden_layers": 2,
  "n_heads": 4,
  "prior_scale": 1.0,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "l2_init": 0.01,
  "exploration_prob": 0.05,
  "acquisition": "thompson",
  "pooling": "sum_bias",
  "logq_correction": true,
  "replay_capacity": 2000,
  "replay_batch": 32,
  "veto_threshold": 0.0,
  "veto_min_examples": 500,
  "shrink": 0.97,
  "perturb_std": 0.003,
  "lr_add": 0.002,
  "extra_replay_steps_add": 3,
  "score_standardise": true,
  "temp_remove": 1.0,
  "target": "flip",
  "tabu": true,
  "sweep_temp": 0.25,
  "sample_temp": 4.0
 },
 "block_removal": true,
 "block_init": 1,
 "swap_record_add": false,
 "swap_record_remove": true,
 "flush_every_add": 4,
 "retry_outer": true,
 "block_random_mix": true,
 "monitor_log_every": 100,
 "block_shrink_on_fail": true,
 "block_example_weight": 0.5,
 "local_budget_frac": 0.2,
 "phase_budget": {
  "remove": 0.5,
  "swap": 0.25,
  "add": 0.25
 },
 "block_policy": "unlock",
 "bandit_max_rungs": 3,
 "adaptive_phase_budget": true,
 "phase_share_floor": 0.05,
 "adaptive_method_order": true,
 "unlock_min_trials": 6,
 "unlock_rate": 0.5,
 "fill_remove_share": true,
 "init_checkpoint": "models/pretrained/edge_selector_v3-pre_PROTEINS.pt"
}
```

Parámetros de LBS vanilla:

```json
{
 "neigh_factor": 4,
 "runtime_factor": 4,
 "max_runtime": 20,
 "max_neigh": 20,
 "attributed": true,
 "seed": 0,
 "max_oracle_calls": 10000,
 "strict_budget": true
}
```

**Resultados previos.** En 40 instancias de PROTEINS (fold 0, tope 2000, 3 semillas), el selector gasta un 9 % menos de llamadas que la mejor variante anterior y un 14 % menos que el mismo esqueleto con propuestas aleatorias (Wilcoxon pareado, Holm p = 0.002 y 0.009), sin diferencia significativa de GED. El entrenamiento importa: frente a la misma red sin entrenar ahorra 195 a 277 llamadas por instancia. En el experimento actual (10 folds, tope 10 000), con las primeras 29 instancias del fold 0, el selector termina con mejor GED que vanilla en 13, igual en 13 y peor en 3, y en las instancias grandes la diferencia es enorme (por ejemplo, id 179: vanilla se queda en 411 aristas, el selector baja a 91 y alcanza 411 en 70 llamadas; id 78: vanilla llega a 18 con 9819 llamadas, el selector a 17 con 54). Las tres instancias peores son 148, 159 y 165.

## 2. Resumen de las tres instancias

| id | inicial | vanilla final (llamadas de su última mejora / total) | selector final (llamadas de su última mejora / total) |
|---|---|---|---|
| 148 | 48 | 8 (1929 / 10 000) | 48 (ninguna mejora / 2436) |
| 159 | 75 | 1 (2493 / 10 000) | 75 (ninguna mejora / 2625) |
| 165 | 58 | 9 (319 / 10 000) | 15 (51 / 2256) |

## 3. Diagnóstico que se desprende de los logs

1. **El selector abandona con el 75 % del presupuesto sin usar.** En las tres instancias termina entre 2256 y 2625 llamadas de 10 000. La causa es el contador de iteraciones `n`: cada iteración externa fallida lo baja en 2 y solo se reinicia con una mejora, así que tras unos 10 reintentos la búsqueda se detiene aunque quede presupuesto. Vanilla, en cambio, sigue hasta agotar las 10 000 llamadas.
2. **En 148 y 159 la primera mejora de vanilla es un bloque grande.** Vanilla pasa de 48 a 33 (15 aristas a la vez) tras 1827 llamadas, y de 75 a 51 (24 aristas a la vez) tras 2136 llamadas. En ambas, ninguna eliminación suelta funciona. El selector nunca prueba bloques grandes: la política `unlock` solo desbloquea el tamaño 2 si el tamaño 1 tiene éxito, cosa que nunca ocurre, así que solo prueba eliminaciones sueltas, swaps y adiciones. Cada reintento del selector gasta unas 210 llamadas (unos 30 candidatos de 7 métodos), casi todas en sueltas que ya sabe que fallan.
3. **En 165 el selector llega muy rápido a un mínimo local peor.** Con eliminaciones sueltas baja de 58 a 15 en 51 llamadas (43 éxitos seguidos, uno por llamada), mucho más rápido que vanilla. Pero a partir de 15 ninguna eliminación suelta funciona y se queda ahí. Vanilla, que avanzó con bloques de 3 y de 2 aristas (58, 55, 52, 49, ...), sigue otra trayectoria y termina en 9 con 319 llamadas. Es decir, la búsqueda voraz por sueltas es más rápida pero puede caer en una cuenca de atracción peor, y luego el selector no explora lo suficiente (ni bloques ni presupuesto restante) para salir.
4. **La aleatoriedad no es la causa principal.** En 148 y 159, con otras semillas, el selector a veces encuentra la salida (en experimentos anteriores, semilla 2: 148 baja a 7 y 159 a 1), pero lo hace por azar, a través de swaps. El problema de fondo es estructural: el esqueleto del selector renuncia a dos cosas que vanilla sí hace siempre, probar bloques aleatorios de cualquier tamaño y usar todo el presupuesto.

## 4. Ajustes que propongo (para discutir)

1. **Usar todo el presupuesto.** Mientras quede presupuesto, un reintento no debe consumir `n` (o `n` se reinicia en cada reintento). Así el selector no abandona antes que vanilla.
2. **Respaldo con la política de vanilla (garantía de peor caso).** Cuando el selector falla k veces seguidas en el mismo estado, ejecutar un barrido de eliminación como el de vanilla (bloques aleatorios de todos los tamaños). Con eso, en el peor caso, la búsqueda contiene la de vanilla. Es la idea de las mejoras seguras de política y de los bandidos conservadores: mezclar la política aprendida con la de referencia y pasar a la de referencia cuando la aprendida deja de rendir.
3. **Explorar bloques aunque las sueltas fallen.** Añadir a la política `unlock` una pequeña probabilidad de probar bloques grandes al azar (por ejemplo, la mitad de S), o desbloquear tamaños por fallo de las sueltas y no solo por éxito.
4. **Evitar caer en cuencas peores (165).** Cuando las sueltas dejan de funcionar, volver a un estado anterior de la trayectoria (backtracking) y probar desde ahí movimientos de bloque, o combinar la eliminación voraz con un número pequeño de reinicios.

## 5. Logs completos

Formato de cada línea: hora y mensaje. `Found solution with size: N, calls: k` marca cada mejora con el tamaño de la nueva solución y las llamadas del minimizador gastadas hasta entonces. En vanilla, `actual ---> m` y `actual +++> m` son las reducciones y expansiones aleatorias de la fase de swap y adición. En el selector, `outer iteration without improvement (k, k_local)` marca un reintento fallido con las llamadas totales y las de esa iteración, y la línea `[monitor][instance]` resume la instancia (fases, éxitos, bloques probados en `rungs`).

### id 148, LBS vanilla

```
16:30:57 Generator output for instance 148 replayed from cache (1114 generator oracle calls added back)
16:30:57 Initial solution size: 48
16:32:12 ============> (-) Found solution with size: 33, calls: 1827
16:32:12 ============> (-) Found solution with size: 31, calls: 1830
16:32:12 ============> (-) Found solution with size: 29, calls: 1833
16:32:12 ============> (-) Found solution with size: 27, calls: 1836
16:32:12 ============> (-) Found solution with size: 25, calls: 1839
16:32:12 ============> (-) Found solution with size: 23, calls: 1842
16:32:12 ============> (-) Found solution with size: 21, calls: 1845
16:32:12 ============> (-) Found solution with size: 19, calls: 1848
16:32:12 ============> (-) Found solution with size: 18, calls: 1851
16:32:14 ============> (-) Found solution with size: 17, calls: 1896
16:32:14 ============> (-) Found solution with size: 16, calls: 1899
16:32:15 ============> (-) Found solution with size: 15, calls: 1902
16:32:15 ============> (-) Found solution with size: 14, calls: 1905
16:32:15 ============> (-) Found solution with size: 13, calls: 1908
16:32:15 ============> (-) Found solution with size: 12, calls: 1911
16:32:15 ============> (-) Found solution with size: 11, calls: 1914
16:32:15 ============> (-) Found solution with size: 10, calls: 1923
16:32:15 ============> (-) Found solution with size: 9, calls: 1926
16:32:16 ============> (-) Found solution with size: 8, calls: 1929
16:33:01 actual ---> 1
16:33:39 actual +++> 5
16:34:14 actual +++> 7
16:34:23 actual ---> 4
16:34:55 actual +++> 6
16:35:28 actual +++> 8
16:35:33 actual ---> 4
16:36:05 actual +++> 7
16:36:07 actual ---> 2
16:36:42 actual +++> 4
16:37:15 actual +++> 7
16:37:15 actual ---> 4
16:37:48 actual +++> 5
16:38:05 actual +++> 7
16:38:05 Oracle calls limit reached
16:38:05 evaluated instance with id 148
```

### id 148, selector

```
15:46:42 Generator output for instance 148 replayed from cache (1114 generator oracle calls added back)
15:46:43 Initial solution size: 48
15:47:09 outer iteration without improvement (k=448, k_local=448), retrying (n=18)
15:47:28 outer iteration without improvement (k=756, k_local=308), retrying (n=16)
15:47:42 outer iteration without improvement (k=966, k_local=210), retrying (n=14)
15:47:56 outer iteration without improvement (k=1176, k_local=210), retrying (n=12)
15:48:11 outer iteration without improvement (k=1386, k_local=210), retrying (n=10)
15:48:26 outer iteration without improvement (k=1596, k_local=210), retrying (n=8)
15:48:40 outer iteration without improvement (k=1806, k_local=210), retrying (n=6)
15:48:55 outer iteration without improvement (k=2016, k_local=210), retrying (n=4)
15:49:10 outer iteration without improvement (k=2226, k_local=210), retrying (n=2)
15:49:24 outer iteration without improvement (k=2436, k_local=210), retrying (n=0)
15:49:24 Oracle calls: 2436
15:49:24 original: 48, final: 48, removed: 0, added: 0
15:49:24 [monitor][instance] calls=2436 best=48 rem[n=198 pos=0.357 ll=0.805/0.652 skill=-0.235 auc=nan succ=0 tries=nan rankadv=+nan steps=1105 bce=nan grad=nan hstd=0.081] add[n=300 pos=0.369 ll=1.477/0.658 skill=-1.243 auc=nan succ=0 tries=nan rankadv=+nan steps=476 bce=nan grad=nan hstd=0.254] wall=161.4s init=48 final=48 phases(calls rem/swap/add=336/1050/1050 succ=0/0/0) block(mean=nan max=0
15:49:25 evaluated instance with id 148
```

### id 159, LBS vanilla

```
16:38:05 Generator output for instance 159 replayed from cache (1114 generator oracle calls added back)
16:38:05 Initial solution size: 75
16:39:25 ============> (-) Found solution with size: 51, calls: 2136
16:39:26 ============> (-) Found solution with size: 48, calls: 2154
16:39:26 ============> (-) Found solution with size: 45, calls: 2166
16:39:27 ============> (-) Found solution with size: 42, calls: 2178
16:39:27 ============> (-) Found solution with size: 39, calls: 2184
16:39:29 ============> (-) Found solution with size: 37, calls: 2244
16:39:30 ============> (-) Found solution with size: 35, calls: 2262
16:39:33 ============> (-) Found solution with size: 33, calls: 2346
16:39:33 ============> (-) Found solution with size: 31, calls: 2361
16:39:34 ============> (-) Found solution with size: 29, calls: 2373
16:39:35 ============> (-) Found solution with size: 27, calls: 2391
16:39:35 ============> (-) Found solution with size: 25, calls: 2394
16:39:35 ============> (-) Found solution with size: 23, calls: 2397
16:39:35 ============> (-) Found solution with size: 21, calls: 2400
16:39:35 ============> (-) Found solution with size: 19, calls: 2403
16:39:35 ============> (-) Found solution with size: 18, calls: 2406
16:39:35 ============> (-) Found solution with size: 17, calls: 2415
16:39:35 ============> (-) Found solution with size: 16, calls: 2424
16:39:36 ============> (-) Found solution with size: 15, calls: 2427
16:39:36 ============> (-) Found solution with size: 14, calls: 2430
16:39:36 ============> (-) Found solution with size: 13, calls: 2433
16:39:36 ============> (-) Found solution with size: 12, calls: 2436
16:39:36 ============> (-) Found solution with size: 11, calls: 2439
16:39:36 ============> (-) Found solution with size: 10, calls: 2454
16:39:37 ============> (-) Found solution with size: 9, calls: 2469
16:39:37 ============> (-) Found solution with size: 8, calls: 2472
16:39:37 ============> (-) Found solution with size: 7, calls: 2475
16:39:37 ============> (-) Found solution with size: 6, calls: 2478
16:39:37 ============> (-) Found solution with size: 5, calls: 2481
16:39:37 ============> (-) Found solution with size: 4, calls: 2484
16:39:37 ============> (-) Found solution with size: 3, calls: 2487
16:39:37 ============> (-) Found solution with size: 2, calls: 2490
16:39:37 ============> (-) Found solution with size: 1, calls: 2493
16:39:38 evaluated instance with id 159
```

### id 159, selector

```
15:49:25 Generator output for instance 159 replayed from cache (1114 generator oracle calls added back)
15:49:26 Initial solution size: 75
15:49:58 outer iteration without improvement (k=595, k_local=595), retrying (n=18)
15:50:20 outer iteration without improvement (k=945, k_local=350), retrying (n=16)
15:50:35 outer iteration without improvement (k=1155, k_local=210), retrying (n=14)
15:50:50 outer iteration without improvement (k=1365, k_local=210), retrying (n=12)
15:51:04 outer iteration without improvement (k=1575, k_local=210), retrying (n=10)
15:51:04 outer iteration without improvement (k=1575, k_local=0), retrying (n=9)
15:51:20 outer iteration without improvement (k=1785, k_local=210), retrying (n=7)
15:51:35 outer iteration without improvement (k=1995, k_local=210), retrying (n=5)
15:51:49 outer iteration without improvement (k=2205, k_local=210), retrying (n=3)
15:52:03 outer iteration without improvement (k=2415, k_local=210), retrying (n=1)
15:52:16 outer iteration without improvement (k=2625, k_local=210), retrying (n=-1)
15:52:16 Oracle calls: 2625
15:52:16 original: 75, final: 75, removed: 0, added: 0
15:52:16 [monitor][instance] calls=2625 best=75 rem[n=225 pos=0.286 ll=0.831/0.598 skill=-0.389 auc=nan succ=0 tries=nan rankadv=+nan steps=1105 bce=nan grad=nan hstd=0.092] add[n=300 pos=0.309 ll=1.692/0.618 skill=-1.737 auc=nan succ=0 tries=nan rankadv=+nan steps=476 bce=nan grad=nan hstd=0.271] wall=170.6s init=75 final=75 phases(calls rem/swap/add=525/1050/1050 succ=0/0/0) block(mean=nan max=0
15:52:17 evaluated instance with id 159
```

### id 165, LBS vanilla

```
16:39:38 Generator output for instance 165 replayed from cache (1114 generator oracle calls added back)
16:39:38 Initial solution size: 58
16:39:38 ============> (-) Found solution with size: 55, calls: 2
16:39:38 ============> (-) Found solution with size: 52, calls: 3
16:39:38 ============> (-) Found solution with size: 49, calls: 4
16:39:38 ============> (-) Found solution with size: 46, calls: 5
16:39:38 ============> (-) Found solution with size: 43, calls: 6
16:39:38 ============> (-) Found solution with size: 40, calls: 7
16:39:38 ============> (-) Found solution with size: 37, calls: 10
16:39:38 ============> (-) Found solution with size: 35, calls: 13
16:39:39 ============> (-) Found solution with size: 33, calls: 34
16:39:41 ============> (-) Found solution with size: 31, calls: 85
16:39:41 ============> (-) Found solution with size: 29, calls: 94
16:39:41 ============> (-) Found solution with size: 27, calls: 103
16:39:43 ============> (-) Found solution with size: 25, calls: 160
16:39:45 ============> (-) Found solution with size: 23, calls: 211
16:39:46 ============> (-) Found solution with size: 21, calls: 238
16:39:46 ============> (-) Found solution with size: 19, calls: 247
16:39:47 ============> (-) Found solution with size: 18, calls: 256
16:39:47 ============> (-) Found solution with size: 17, calls: 259
16:39:47 ============> (-) Found solution with size: 16, calls: 268
16:39:48 ============> (-) Found solution with size: 15, calls: 277
16:39:48 ============> (-) Found solution with size: 14, calls: 286
16:39:48 ============> (-) Found solution with size: 13, calls: 289
16:39:48 ============> (-) Found solution with size: 12, calls: 304
16:39:48 ============> (-) Found solution with size: 11, calls: 307
16:39:48 ============> (-) Found solution with size: 10, calls: 310
16:39:49 ============> (-) Found solution with size: 9, calls: 319
16:40:51 actual ---> 4
16:41:26 actual +++> 7
16:42:03 actual +++> 9
16:42:25 actual ---> 4
16:43:03 actual +++> 7
16:43:42 actual +++> 9
16:43:56 actual ---> 3
16:44:32 actual +++> 7
16:45:07 actual +++> 8
16:45:15 actual ---> 4
16:45:53 actual +++> 7
16:46:30 actual +++> 9
16:46:30 Oracle calls limit reached
16:46:30 evaluated instance with id 165
```

### id 165, selector

```
15:52:17 Generator output for instance 165 replayed from cache (1114 generator oracle calls added back)
15:52:18 Initial solution size: 58
15:52:18 ============> (-) Found solution with size: 57, calls: 1
15:52:18 ============> (-) Found solution with size: 56, calls: 2
15:52:18 ============> (-) Found solution with size: 55, calls: 3
15:52:18 ============> (-) Found solution with size: 54, calls: 4
15:52:18 ============> (-) Found solution with size: 53, calls: 5
15:52:19 ============> (-) Found solution with size: 52, calls: 6
15:52:19 ============> (-) Found solution with size: 51, calls: 7
15:52:19 ============> (-) Found solution with size: 50, calls: 8
15:52:19 ============> (-) Found solution with size: 49, calls: 9
15:52:19 ============> (-) Found solution with size: 48, calls: 10
15:52:19 ============> (-) Found solution with size: 47, calls: 11
15:52:19 ============> (-) Found solution with size: 46, calls: 12
15:52:19 ============> (-) Found solution with size: 45, calls: 13
15:52:19 ============> (-) Found solution with size: 44, calls: 14
15:52:19 ============> (-) Found solution with size: 43, calls: 15
15:52:20 ============> (-) Found solution with size: 42, calls: 16
15:52:20 ============> (-) Found solution with size: 41, calls: 17
15:52:20 ============> (-) Found solution with size: 40, calls: 18
15:52:20 ============> (-) Found solution with size: 39, calls: 19
15:52:20 ============> (-) Found solution with size: 38, calls: 20
15:52:20 ============> (-) Found solution with size: 37, calls: 21
15:52:20 ============> (-) Found solution with size: 36, calls: 22
15:52:20 ============> (-) Found solution with size: 35, calls: 23
15:52:20 ============> (-) Found solution with size: 34, calls: 24
15:52:20 ============> (-) Found solution with size: 33, calls: 25
15:52:20 ============> (-) Found solution with size: 32, calls: 26
15:52:21 ============> (-) Found solution with size: 31, calls: 27
15:52:21 ============> (-) Found solution with size: 30, calls: 28
15:52:21 ============> (-) Found solution with size: 29, calls: 29
15:52:21 ============> (-) Found solution with size: 28, calls: 30
15:52:21 ============> (-) Found solution with size: 27, calls: 31
15:52:21 ============> (-) Found solution with size: 26, calls: 32
15:52:21 ============> (-) Found solution with size: 25, calls: 33
15:52:21 ============> (-) Found solution with size: 24, calls: 34
15:52:21 ============> (-) Found solution with size: 23, calls: 35
15:52:21 ============> (-) Found solution with size: 22, calls: 36
15:52:21 ============> (-) Found solution with size: 21, calls: 37
15:52:22 ============> (-) Found solution with size: 20, calls: 38
15:52:22 ============> (-) Found solution with size: 19, calls: 39
15:52:22 ============> (-) Found solution with size: 18, calls: 40
15:52:22 ============> (-) Found solution with size: 17, calls: 41
15:52:22 ============> (-) Found solution with size: 16, calls: 42
15:52:23 ============> (-) Found solution with size: 15, calls: 51
15:52:41 outer iteration without improvement (k=366, k_local=315), retrying (n=18)
15:52:53 outer iteration without improvement (k=576, k_local=210), retrying (n=16)
15:53:06 outer iteration without improvement (k=786, k_local=210), retrying (n=14)
15:53:20 outer iteration without improvement (k=996, k_local=210), retrying (n=12)
15:53:34 outer iteration without improvement (k=1206, k_local=210), retrying (n=10)
15:53:47 outer iteration without improvement (k=1416, k_local=210), retrying (n=8)
15:54:01 outer iteration without improvement (k=1626, k_local=210), retrying (n=6)
15:54:13 outer iteration without improvement (k=1836, k_local=210), retrying (n=4)
15:54:27 outer iteration without improvement (k=2046, k_local=210), retrying (n=2)
15:54:41 outer iteration without improvement (k=2256, k_local=210), retrying (n=0)
15:54:41 Oracle calls: 2256
15:54:41 original: 58, final: 15, removed: 43, added: 0
15:54:41 [monitor][instance] calls=2256 best=15 rem[n=209 pos=0.505 ll=0.702/0.693 skill=-0.013 auc=0.884 succ=43 tries=1.0 rankadv=+0.389 steps=1105 bce=nan grad=nan hstd=0.152] add[n=300 pos=0.360 ll=1.519/0.653 skill=-1.325 auc=nan succ=0 tries=nan rankadv=+nan steps=476 bce=nan grad=nan hstd=0.241] wall=143.4s init=58 final=15 phases(calls rem/swap/add=156/1050/1050 succ=43/0/0) block(mean=1.0
15:54:42 evaluated instance with id 165
```

## 6. Prompt para la investigación profunda

Copiar en otra sesión (por ejemplo, con búsqueda web), adjuntando este documento:

```text
Contexto: trabajo en minimización de explicaciones contrafactuales en grafos con un clasificador de caja negra (GC-min, NP-duro). Uso una búsqueda local acotada (LBS) con mejora a la primera sobre el conjunto S de aristas alteradas: barridos de eliminación (quitar 1 o varias aristas), swaps (quitar i y añadir i) y adiciones compensatorias, con un presupuesto fijo de consultas al oráculo (10 000) y cada candidato evaluado con hasta 7 manipulaciones de atributos de nodo. Añadí un "selector": una red que puntúa aristas candidatas (preentrenada offline con unos 13 700 ejemplos de movimientos y su resultado, y congelada) y ordena los barridos de forma casi voraz (Plackett-Luce con temperatura 0.25 sobre puntuaciones estandarizadas, memoria tabú). En promedio ahorra un 9 a 14 % de llamadas sin perder GED, y en instancias grandes es muchísimo mejor que la búsqueda sin modelo. Pero en algunas instancias termina peor que la búsqueda vanilla (aleatoria): adjunto el documento con los logs y mi diagnóstico (abandona con el 75 % del presupuesto sin usar, nunca prueba bloques de eliminación grandes cuando las eliminaciones sueltas fallan, y su descenso voraz cae en cuencas de atracción peores).

Objetivo: que el selector sea, en el peor caso, tan bueno como la búsqueda vanilla, sin perder su ventaja media. No quiero cambiar los operadores de búsqueda (eliminación, swap, adición, métodos de atributos), solo cómo el selector propone, ordena, explora y reparte el presupuesto.

Investiga en profundidad y dame, con referencias:
1. Técnicas con garantías de "no empeorar respecto a una política de referencia": safe policy improvement, conservative bandits / conservative exploration, mezcla o portafolio con la política base, algoritmos de "learning-augmented" o "algorithms with predictions" (consistencia frente a robustez), y cómo aplicarlas a una búsqueda local guiada por un modelo de ranking.
2. Cómo combinar heurísticas aprendidas con búsqueda local en optimización combinatoria sin perder robustez: learned heuristics + fallback, adaptive large neighborhood search con selección adaptativa de operadores, hyper-heuristics, reinicios y backtracking, escape de mínimos locales (perturbaciones de tamaño adaptativo, como iterated local search o variable neighborhood search).
3. Exploración frente a explotación en el orden de los candidatos: cuándo pasar de voraz a aleatorio (por fallos consecutivos, incertidumbre del modelo, presupuesto restante), Thompson sampling o UCB sobre tamaños de bloque, y cómo calibrar la temperatura.
4. Asignación del presupuesto: criterios para no abandonar antes de tiempo, reparto entre fases, y reglas de parada basadas en el presupuesto restante.
5. Cualquier trabajo sobre minimización de contrafactuales en grafos o tareas parecidas (edit distance mínima con oráculo de caja negra, group testing adaptativo, búsqueda de subconjuntos mínimos o "minimal unsatisfiable subsets", delta debugging / ddmin) que sugiera estrategias de bloques. En particular, delta debugging prueba subconjuntos de tamaño decreciente, y eso parece muy relacionado con los casos donde una eliminación de 15 o 24 aristas a la vez funciona y ninguna suelta.

Para cada idea: explica la intuición, la garantía teórica si la hay, el coste en llamadas al oráculo, y cómo la implementarías concretamente en mi caso (qué cambiar en la política del selector). Termina con una lista priorizada de 3 a 5 cambios concretos y un diseño experimental para validarlos (brazos, métricas, prueba estadística pareada por instancia).
```
