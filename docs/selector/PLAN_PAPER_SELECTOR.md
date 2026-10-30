# Plan del paper del selector: versiones, experimentos y colas

Escrito el 2026-10-29, tras `docs/selector/ANALISIS_CONTROLES.md`. Es la lista completa de lo que hay que lanzar y en qué orden, y el mapa entre los nombres del paper y el código.

## 1. Versiones

| Nombre en el paper | Qué es | Código | Configs |
|---|---|---|---|
| bls_v0 | BLS del primer paper (aceptado) tal como su pseudocódigo lo describe: niveles de eliminación desde una arista (arreglo en el código, commit `3061c7c22`), parada tras la primera iteración externa sin mejora, las siete estrategias de atributos | `local_search.py` (y `local_search_var_1..4.py` para las variantes de la tesis y la revisión) | `<dataset>-sel/dce/dce-lcls-v0` y todo `docs/rerun/` |
| bls_v1 | El esqueleto de búsqueda del selector con propuestas uniformes y sin red: barrido exhaustivo de aristas sueltas, sobrepaso geométrico con retroceso a la profundidad de v0, near-miss, parada tras una iteración sin mejora, tope local y cuotas por fase | `local_search_selection_net_v2.py` con `selector_mode: random` | `<dataset>-sel/dce/dce-lcls-sel-random` |
| selector_v1 | bls_v1 guiado por el selector preentrenado y congelado (el "selector" de `ANALISIS_CONTROLES.md`) | `local_search_selection_net_v2.py` + `Tagging/OnlineSelectorV2.py`, `selector_mode: frozen` | `<dataset>-sel/dce/dce-lcls-sel` |
| control sin entrenar | selector_v1 con la red sin entrenar | ídem, sin `init_checkpoint` | `<dataset>-sel/dce/dce-lcls-sel-untrained` |
| selector_v1 con fallback uniforme | selector_v1 con bloques uniformes en el retroceso del sobrepaso | ídem, `overshoot_fallback_uniform` | `proteins-sel/dce/dce-lcls-sel-uniform-fallback` |
| selector_v2 | La arquitectura nueva, por diseñar cuando termine el estudio de un fold en los otros datasets (sección 6 de `ANALISIS_CONTROLES.md`) | por escribir | por generar |
| bls-selector_v0 | El selector de la tesis. Fuera del paper; el documento del doctorado lo menciona como versión preliminar | `local_search_selection_net.py` + `Tagging/OnlineSelector.py` | `*/dce-lcls-net*` (no se relanzan) |

Los nombres de módulo (`_v2` es selector_v1) se renombran al final con `tools/move_module.py`, porque están en los hashes de los configs. Los números del primer paper se citan como referencia, no entran en las tablas: salieron de otras cachés y otros folds.

## 2. Reglas

- Generador DCE en todo; `generator_cache` compartido por fold (DCE es determinista).
- Diez folds, el test completo de cada uno, semilla 0, presupuesto 10 000 llamadas del minimizador.
- Preentrenamiento sin fuga: dos checkpoints por dataset (`models/pretrained/edge_selector_sel-pre_<dataset>-f1.pt` para todos los folds menos el 1, `-f2.pt` para el fold 1), entrenados con las colecciones de los folds 1 y 2 (`dce-lcls-sel-collect`, uniforme, 2000 llamadas). PROTEINS conserva `edge_selector_v3-pre_PROTEINS.pt` (instancias del fold 0) en los folds 1 a 9 y usa `-f1` en el fold 0. `scripts/pretrain_selector_checkpoints.py` construye los que falten; los checkpoints se commitean.
- Nada se edita en el servidor: configs y código se commitean aquí, el servidor hace `launchers/server_sync.sh` y lanza colas (`docs/selector/SERVER_RUNS.md`); los resultados vuelven con `make server-import`.

## 3. Estado de lo que ya existe (PROTEINS, 10 folds)

| Brazo | Estado |
|---|---|
| selector_v1 | folds 1 a 9 hechos y commiteados; fold 0 repetido con checkpoint `-f1` (el anterior se preentrenó con sus propias instancias; movido a `../phd/legacy-gretel`) |
| control sin entrenar, bls_v1 | folds 0 a 4 hechos y commiteados, 5 a 9 corriendo en el servidor (array 542453) |
| fallback uniforme | sin lanzar; fold 0 repuntado al checkpoint `-f1` |
| bls_v0 | sin lanzar (configs nuevos) |
| LBS con código viejo (`proteins-sel_dce_lcls`) | fuera del repositorio (`../phd/legacy-gretel/lab-output`); su análisis queda en `ANALISIS_CONTROLES.md` |

## 4. Todo lo que hay que lanzar

Generado por `scripts/gen_selector_paper_matrix.py` (brazos del paper) y `scripts/gen_rerun_queues.py` (tablas de la tesis y de la revisión con bls_v0). Las colas de servidor se lanzan de una en una con `launchers/slurm_submit.sh -p normal -m 10 <cola>` y se encadenan con `-c`.

### 4.1 Local (esta máquina, un worker)

1. `docs/selector/PAPER_LOCAL_1.md` (30 configs): colecciones de los folds 1 y 2 de los seis datasets pequeños (asd, tcr-500-28, tcr-tco-300, synthie, bbbp, aids) y su preliminar del fold 0 (bls_v0, bls_v1, control). PROTEINS y colors-3 son demasiado grandes para esta máquina y van enteros al servidor. Lo ya hecho (bls_v1 y control de asd, tcr-500-28 y tcr-tco-300 en el fold 0, colecciones y bls_v0 de asd) se salta solo. `python scripts/run_revision_queue.py --list docs/selector/PAPER_LOCAL_1.md --workers 1 --timeout 86400`.
2. `python scripts/pretrain_selector_checkpoints.py` (minutos) y commit de `models/pretrained/`.
3. `docs/selector/PAPER_LOCAL_2.md` (6): selector_v1 en el fold 0 de esos seis datasets. Con esto, más lo que ya hay de PROTEINS, se repite el análisis de arquitectura (sección 6 de `ANALISIS_CONTROLES.md`) y se diseña selector_v2.
4. Las colecciones de PROTEINS y colors-3 corren en el servidor (fase S1) y las de BZR y ENZYMES en la fase S3, tras sus cachés (`docs/rerun/README.md`, sección 2). `make server-import` las trae (`lab/output/selector_pretrain/`); después, el paso 2 de nuevo, commit de los checkpoints y `server_sync` antes de sus brazos del selector.

### 4.2 Servidor, paper del selector (`docs/selector/PAPER_QUEUES.md`, 14 colas de hasta 30)

| Fase | Colas | Contenido | Necesita |
|---|---|---|---|
| S1 | `PAPER_Q01` a `PAPER_Q08` (215) | bls_v0, bls_v1 y control, folds 1 a 9 de los seis datasets pequeños; colors-3 entero menos el selector (colecciones, bls_v0, bls_v1, control, 10 folds); PROTEINS: colecciones, bls_v0 (10 folds) y fallback uniforme folds 1 a 9 | nada |
| S2 | `PAPER_Q09` a `PAPER_Q11` (66) | selector_v1 folds 1 a 9 de los seis pequeños y 10 folds de colors-3; PROTEINS fold 0 de selector_v1 y del fallback uniforme | checkpoints commiteados y sincronizados (paso 4.1.2; los de colors-3 y PROTEINS tras importar sus colecciones) |
| S3 | `PAPER_Q12` a `PAPER_Q14` (84) | BZR y ENZYMES: colecciones, bls_v0, bls_v1, control y selector_v1, 10 folds | cachés de BZR y ENZYMES y sus checkpoints |

Más las colas ya existentes: `SERVER_PROTEINS_CONTROLS.md` (en marcha) y `SERVER_PROTEINS_FALLBACK.md` (su fold 0 está también en S2; la cola completa sirve igual, `slurm_submit` salta lo ya lanzado). `SERVER_PROTEINS_LBS_ABLATIONS.md` (nivel de una arista y parada sobre el código viejo) queda obsoleta con el arreglo de v0 y no se lanza.

### 4.3 Servidor, tablas de la tesis y de la revisión con bls_v0 (`docs/rerun/`)

Sin cambios de estructura: brazo ligero en un comando (660 configs), colas `Q01` a `Q39` (1140), reglas en `docs/rerun/README.md`. Regenerado con `stop_after_failed_iteration` y las siete estrategias en todos los configs de BLS (plain, var-1 a var-4 y semillas); OBS, DBS, RHC y los generadores solos no cambian. Lo esperado frente a los números del primer paper: GED igual o mejor, llamadas menores en casi todo.

### 4.4 Cuando exista selector_v2

Un brazo `dce-lcls-sel-v2` por dataset, 10 folds, con el mismo generador de configs (añadir el brazo a `gen_selector_paper_matrix.py`), más su control sin entrenar si la arquitectura lo admite. Unos 200 configs.

## 5. Orden sugerido

1. Local: `PAPER_LOCAL_1.md` (en marcha), checkpoints, `PAPER_LOCAL_2.md`. Servidor, en paralelo: `server_sync`, `PAPER_Q01` en adelante (S1 no necesita nada), los tres jobs de cachés de BZR y ENZYMES, y el brazo ligero del rerun.
2. Checkpoints commiteados (los seis pequeños desde local; colors-3 y PROTEINS tras `make server-import` de sus colecciones de S1): `PAPER_Q09` a `PAPER_Q11`; cachés de BZR y ENZYMES traídas y commiteadas: `PAPER_Q12` a `PAPER_Q14`, con la importación de sus colecciones y sus checkpoints entre medias.
3. Análisis de arquitectura con todos los datasets, diseño de selector_v2, sus configs y colas.
4. Rerun de las tablas de la tesis y la revisión (`docs/rerun/Q01` a `Q39`), que no bloquea nada del paper.
5. Al cerrar: renombrar módulos (`move_module.py`), análisis final con `scripts/analyze_selector_arms.py` por dataset.
