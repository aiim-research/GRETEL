Artefactos de la sesión de calibración del selector (2026-09-03 a 2026-09-08). Ver `SELECTOR_EXPERIMENTS_STATUS.md` en la raíz.

- `results/<scope>/.../results_<fold>_1.json`: resultados por instancia de cada scope (sin los dumps `cf_*.json`). `results/_backup_precalibration/` es la base LBS sin calibrar de Synthie y BBBP; `results/proteins10_dce_lcls` es la base en PROTEINS-10.
- `selector_logs/selector_<tag>_<dataset>.csv`: monitor precuencial, una fila por instancia.
- `queue_logs/*.log.gz`: logs de las corridas relevantes (analizables con `scripts/analyze_selector_runs.py logs`).

Para usarlos con `scripts/analyze_selector_runs.py compare`, pasar scopes como rutas relativas a `lab/output`, por ejemplo copiando este directorio a `lab/output/results/` o usando `compare base=../selector_experiments/results/_backup_precalibration/synthie_dce_lcls ...`.
