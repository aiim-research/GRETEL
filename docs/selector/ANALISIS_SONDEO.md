# Análisis del sondeo (fold 0 de cada dataset y oráculo) y plan ajustado para selector_v2

Fecha: 2026-11-03. Datos: el sondeo del servidor (`PAPER_SONDEO_A.md` y `PAPER_SONDEO_B.md`, 125 configs, importados con `make server-import`), el sondeo local de margen (`PAPER_LOCAL_3/4.md`), los folds 5 a 9 de PROTEINS que llegaron con la misma importación, las colecciones del fold 1 y sus checkpoints. Script: `scripts/sondeo_summary.py` (una fila por celda, JSON con cada instancia) sobre `scripts/analyze_selector_arms.py`. Este documento sustituye, para la decisión de arquitectura, a la sección 6 de `ANALISIS_CONTROLES.md`, que salía de un solo dataset.

Brazos: bls_v0 (`dce-lcls-v0`, `local_search.py` con nivel desde una arista, parada tras iteración fallida y presupuesto estricto), bls_v1 (`dce-lcls-sel-random`, el esqueleto del selector con propuestas uniformes), selector_v1 (`dce-lcls-sel`, el mismo esqueleto con la red congelada preentrenada en la colección del fold 1). Presupuesto 10 000 llamadas, siete métodos de atributos, DCE como generador, 33 celdas: 7 datasets por 3 oráculos (original, margen τ=0.9 "m90", GCN robusta "rob") más 12 de tri (3 datasets por regla exacta, GCN, GCN-m90 y GCN-rob).

## 1. Resumen

1. **Tres regímenes, no uno.** El comportamiento de los tres brazos lo decide la pareja dataset-oráculo, no el dataset. (A) Descenso largo con ranking útil: las 12 celdas de tri, bbbp con el oráculo original, PROTEINS con el original y el robusto. Aquí selector_v1 ya gana con claridad (tri GED 4.5 frente a 6.9 de bls_v1, 48 de 50 instancias; bbbp 3.2 frente a 4.5, 40 contra 5). (B) Atascado: bbbp-m90, bbbp-rob, los tres aids, proteins-m90 y en menor medida PROTEINS y bzr-rob. Ninguna eliminación suelta funciona al inicio en el 45 % al 96 % de las instancias, y la GED se concentra en ellas. (C) Trivial: synthie, colors-3, enzymes y bzr con cualquiera de los tres oráculos. El contrafactual mínimo tiene 1 a 3 aristas, todos los brazos llegan, no hay sitio para ningún modelo y no lo habrá.

2. **El ranking de sueltas funciona y es la palanca del régimen A.** En tri la red gasta los mismos intentos por éxito que el azar (1.1 frente a 1.3) y termina en un óptimo local un 35 % más pequeño: lo que cambia es el orden de las reversiones, no cuántas fallan. PROTEINS en los folds 1 a 9 (1001 instancias, sin fuga) lo confirma contra la red sin entrenar: 19.4 frente a 22.7 en las instancias que avanzan (p 10⁻³⁶), un 17 % menos de llamadas y mitad de tiempo de pared.

3. **El régimen B es una lotería de bloques, y hoy la gana quien más presupuesto gasta a ciegas.** En las colecciones uniformes (2000 llamadas) los bloques al azar en instancias atascadas no aciertan nunca: 0 de 4433 en PROTEINS, 0 de 19 995 en bbbp-m90, 0 de 20 184 en aids. Con 10 000 llamadas, bls_v0 los encuentra tarde (mediana 7000 llamadas en bbbp-m90, bloques de 4 aristas) porque repite niveles ciegos hasta agotar el presupuesto, mientras el esqueleto del selector se detiene con 4000 a 7000 llamadas sin usar. Por eso bls_v1 pierde contra bls_v0 en bbbp-m90 (16 contra 57, p < 0.001), aids-m90 (10 contra 25, p 0.001) y bbbp-rob (15 contra 28). Donde hay ranking, los bloques compuestos con la red (reducir por ranking y volver a añadir por ranking) rescatan el doble que el azar: 92 instancias frente a 38 en bbbp-m90, 77 frente a 46 en bbbp.

4. **Dos ajustes de configuración, no de código, antes de la flota.** La escalera de bloques "unlock" con estadísticas de toda la corrida es un interruptor dependiente de la historia que se activa en unos brazos y no en otros de la misma celda (tri-80-gcn: selector 91 % de la GED por bloques, bls_v1 2 %) y, cuando se activa, cambia GED por llamadas y borra la ventaja del ranking (tri-80-gcn 20.0 frente a 20.8, cuando tri-80 con la regla exacta da 11.9 frente a 18.8). Propuesta: `block_removal false` en todos los brazos. Y el fallback de bloques debe poder gastar todo el presupuesto restante cuando nada ha funcionado (`overshoot_budget_frac 1.0`), que es exactamente lo que hace bls_v0 y lo que hoy le da sus victorias en el régimen B.

5. **selector_v2 se centra en el régimen B sin tocar lo que ya funciona.** Misma featurización y mismo tronco por arista, con dos cambios: una cabeza de valor (si la arista acaba fuera del contrafactual final, no solo si su reversión mantiene la clase) para el ranking de sueltas, y un puntuador de bloques (agregación de las puntuaciones por arista con la evidencia de la instancia) que sustituya al muestreo near-miss en el fallback. Antes de implementarlo en línea, un estudio offline sobre una colección nueva con identificadores (fold 1 de cada celda, presupuesto completo, selector_v1 congelado) con dos umbrales de decisión (sección 6). La propuesta A de `ANALISIS_CONTROLES.md` (gemelo del oráculo) se aparca: offline no tendría positivos en las celdas duras y exige otra tubería.

6. **Para el paper.** Las tablas de synthie, colors-3, enzymes y bzr son empates en los tres oráculos y aids casi (de 15 a 13). La señal del paper está en PROTEINS (pequeña), bbbp (clara con los tres oráculos) y tri (grande en 11 de 12). proteins-m90 es degenerado (108 de 112 atascadas, GED 82 a 87 frente a 89.5 del DCE). Decisiones pendientes del usuario en la sección 7.

## 2. Tabla maestra

Fold 0. GED media del contrafactual final. "gana/pierde" cuenta instancias con test de Wilcoxon pareado. "Atascadas" son las instancias que selector_v1 nunca mejoró y la parte de su GED total que concentran. "Intentos por éxito" es el número medio de eliminaciones sueltas probadas por cada una aceptada (monitor `rem_tries_mean`), bls_v1 y selector_v1. "AUC pre." es la AUC de validación del preentrenamiento en la colección del fold 1 (`models/pretrained/*.json`).

| Celda | n | DCE | bls_v0 | bls_v1 | sel_v1 | v1 vs v0 (gana/pierde, p) | sel vs v1 (gana/pierde, p) | Llamadas v0 / v1 / sel | Atascadas sel (GED %) | Intentos por éxito v1 / sel | AUC pre. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| proteins | 112 | 65.7 | 24.3 | 23.1 | 22.2 | 42/21, 0.024 | 40/26, 0.060 | 6240 / 3246 / 3118 | 20 (36 %) | 2.30 / 1.73 | 0.82 |
| proteins-m90 | 112 | 89.5 | 84.9 | 86.7 | 82.2 | 2/3, 0.42 | 7/2, 0.051 | 10004 / 7304 / 7304 | 103 (96 %) | 3.40 / 3.02 | sin positivos |
| proteins-rob | 112 | 64.1 | 36.0 | 34.1 | 33.6 | 31/27, 0.36 | 33/31, 0.32 | 6621 / 3450 / 3194 | 23 (58 %) | 2.72 / 1.85 | 0.88 |
| synthie | 40 | 189.4 | 2.7 | 2.2 | 2.5 | 7/2, 0.15 | 3/8, 0.15 | 2256 / 1291 / 1369 | 0 | 1.17 / 1.28 | 0.39 |
| synthie-m90 | 40 | 189.4 | 3.5 | 3.0 | 2.7 | 9/5, 0.19 | 7/7, 0.44 | 2535 / 1596 / 1867 | 0 | 1.33 / 1.38 | 0.46 |
| synthie-rob | 40 | 189.2 | 3.1 | 6.1 | 2.5 | 4/4, 0.78 | 5/4, 0.34 | 1658 / 1095 / 1177 | 0 | 1.44 / 1.41 | 0.61 |
| bbbp | 204 | 14.9 | (sin correr) | 4.5 | 3.2 | | 40/5, < 0.001 | . / 1478 / 1277 | 16 (40 %) | . | 0.93 |
| bbbp-m90 | 204 | 18.3 | 9.0 | 11.5 | 8.1 | 16/57, < 0.001 | 71/15, < 0.001 | 6274 / 4696 / 4550 | 74 (76 %) | 2.85 / 2.38 | 0.95 |
| bbbp-rob | 204 | 14.9 | 7.0 | 7.4 | 7.1 | 15/28, 0.068 | 29/14, 0.043 | 3774 / 2553 / 2574 | 92 (84 %) | 2.26 / 2.33 | 0.95 |
| aids | 200 | 15.2 | 12.8 | 13.1 | 13.1 | 16/19, 0.076 | 17/13, 0.38 | 5744 / 3954 / 3909 | 151 (78 %) | 4.47 / 4.41 | 0.93 |
| aids-m90 | 200 | 15.3 | 13.3 | 13.9 | 13.8 | 10/25, 0.001 | 16/12, 0.66 | 6151 / 4407 / 4404 | 159 (81 %) | 5.30 / 4.94 | 0.94 |
| aids-rob | 200 | 15.8 | 13.2 | 13.5 | 13.2 | 20/25, 0.16 | 35/22, 0.046 | 6614 / 4192 / 4334 | 133 (63 %) | 6.61 / 4.49 | 0.91 |
| bzr | 39 | 23.4 | 1.8 | 2.0 | 2.2 | 5/6, 0.52 | 7/3, 0.57 | 1348 / 1002 / 984 | 2 (31 %) | 1.67 / 1.56 | 0.83 |
| bzr-m90 | 39 | 23.9 | 3.1 | 2.7 | 3.0 | 9/4, 0.13 | 5/8, 0.39 | 2022 / 1571 / 1646 | 3 (33 %) | 2.62 / 2.10 | 0.88 |
| bzr-rob | 39 | 23.4 | 4.9 | 5.5 | 6.0 | 7/11, 0.51 | 6/6, 0.72 | 3276 / 2263 / 2272 | 8 (63 %) | 2.48 / 2.80 | 0.84 |
| enzymes | 60 | 57.9 | 3.6 | 3.6 | 4.2 | 8/13, 0.24 | 8/13, 0.087 | 1390 / 680 / 783 | 1 (19 %) | 1.50 / 1.54 | 0.54 |
| enzymes-m90 | 60 | 58.8 | 4.6 | 5.4 | 3.7 | 12/16, 0.51 | 13/12, 0.51 | 1753 / 1197 / 1482 | 0 | 1.91 / 2.03 | 0.79 |
| enzymes-rob | 60 | 57.9 | 4.0 | 3.8 | 3.7 | 18/13, 0.29 | 14/12, 0.83 | 1426 / 675 / 766 | 0 | 1.66 / 2.17 | 0.67 |
| colors-3-s20 | 18 | 99.4 | 1.0 | 1.1 | 1.1 | 0/1 | 0/0 | 441 / 21 / 24 | 0 | 1.37 / 1.48 | 0.77 |
| colors-3-m90-s20 | 18 | 99.4 | 1.3 | 1.6 | 1.4 | 2/2 | 1/1 | 493 / 219 / 181 | 0 | 2.20 / 2.10 | 0.75 |
| colors-3-rob-s20 | 18 | 99.4 | 1.0 | 1.0 | 1.0 | 0/0 | 0/0 | 446 / 20 / 18 | 0 | 1.42 / 1.27 | 0.82 |
| tri-40 | 50 | 146.4 | 6.9 | 6.9 | 4.5 | 20/17, 0.90 | 48/0, < 0.001 | 615 / 451 / 478 | 0 | 1.28 / 1.09 | 0.86 |
| tri-40-gcn | 50 | 146.4 | 7.3 | 7.8 | 4.5 | 17/29, 0.17 | 39/2, < 0.001 | 670 / 479 / 362 | 0 | 1.67 / 1.10 | 0.93 |
| tri-40-gcn-m90 | 50 | 146.4 | 9.8 | 10.4 | 6.1 | 21/26, 0.34 | 40/4, < 0.001 | 1110 / 693 / 643 | 0 | 1.46 / 1.14 | 0.91 |
| tri-40-gcn-rob | 50 | 146.4 | 6.8 | 7.0 | 4.3 | 19/22, 0.70 | 42/6, < 0.001 | 606 / 473 / 397 | 0 | 1.31 / 1.13 | 0.92 |
| tri-40n | 50 | 144.0 | 5.4 | 5.4 | 3.9 | 19/21, 0.71 | 36/5, < 0.001 | 425 / 310 / 235 | 0 | 1.62 / 1.34 | 0.91 |
| tri-40n-gcn | 50 | 144.0 | 5.4 | 5.6 | 3.4 | 16/24, 0.67 | 35/4, < 0.001 | 475 / 319 / 220 | 0 | 1.39 / 1.34 | 0.92 |
| tri-40n-gcn-m90 | 50 | 144.7 | 9.1 | 8.0 | 4.6 | 27/15, 0.018 | 41/2, < 0.001 | 896 / 392 / 346 | 0 | 1.99 / 1.51 | 0.89 |
| tri-40n-gcn-rob | 50 | 144.0 | 5.9 | 6.2 | 3.5 | 17/19, 0.60 | 38/3, < 0.001 | 669 / 440 / 349 | 0 | 1.46 / 1.31 | 0.93 |
| tri-80 | 50 | 360.7 | 18.4 | 18.8 | 11.9 | 18/24, 0.30 | 48/1, < 0.001 | 2127 / 1079 / 1591 | 0 | 1.34 / 1.08 | 0.92 |
| tri-80-gcn | 50 | 360.7 | 19.9 | 20.8 | 20.0 | 21/28, 0.45 | 26/20, 0.28 | 2824 / 1466 / 1065 | 0 | 1.38 / 1.50 | 0.92 |
| tri-80-gcn-m90 | 50 | 360.7 | 26.6 | 26.1 | 19.5 | 25/23, 0.55 | 38/11, < 0.001 | 2890 / 1584 / 1381 | 0 | 1.45 / 1.13 | 0.90 |
| tri-80-gcn-rob | 50 | 360.7 | 21.9 | 23.4 | 12.5 | 19/25, 0.24 | 48/1, < 0.001 | 2867 / 1555 / 1282 | 0 | 1.38 / 1.45 | 0.90 |

Notas. bbbp con el oráculo original no tiene bls_v0 (el sondeo del servidor lo dejó al rerun local y este solo corrió bls_v1 y selector_v1). colors-3 son los 20 instancias fijas del subconjunto, 18 con las tres corridas completas. proteins-m90: el preentrenamiento no tiene positivos en validación (tasa base de eliminaciones sueltas que funcionan: 2 %), el checkpoint es ruido y la ventaja de selector_v1 (7 contra 2) sale del fallback, no de la red.

## 3. Los tres regímenes

### 3.1 Régimen A: descenso largo donde el orden decide el óptimo (tri, bbbp, PROTEINS)

El DCE entrega un contrafactual muy lejano (146 a 361 ediciones en tri, 66 en PROTEINS, 15 en bbbp) y la búsqueda es un descenso de cientos de reversiones sueltas. En tri el 66 % al 77 % de las eliminaciones sueltas funcionan (tasa base de las colecciones), así que bls_v1 y selector_v1 gastan casi lo mismo por éxito (1.3 y 1.1 intentos). La diferencia está en el estado final: ambos terminan en óptimos locales (ninguna reversión suelta mantiene la clase), pero el de la red es un 35 % más pequeño en tri-40 (4.5 frente a 6.9, 48 de 50 instancias), un 37 % en tri-80 (11.9 frente a 18.8) y un 46 % en tri-80-gcn-rob. Es el hallazgo 3 de `ANALISIS_CONTROLES.md` (el orden cambia el óptimo) generalizado: una reversión que mantiene la clase pero cierra otras reversiones futuras es peor que otra que las deja abiertas, y la red, entrenada a predecir "mantiene la clase", captura parte de esa estructura porque en tri las ediciones que importan son las que tocan triángulos.

En bbbp (14.9 de DCE) el selector gana 40 contra 5 con la mitad del tiempo, y con los oráculos duros sigue ganando (bbbp-m90 71 contra 15, bbbp-rob 29 contra 14), aunque ahí la ganancia viene del régimen B (sección 3.2). En PROTEINS el fold 0 da 40 contra 26 (p 0.06) y los folds 1 a 9 lo zanjan (sección 4).

Qué lo hace posible: AUC de preentrenamiento de 0.86 a 0.95 en todas estas celdas, con tasas base entre 0.15 y 0.68.

### 3.2 Régimen B: atascado al inicio (bbbp-m90, bbbp-rob, aids, proteins-m90, parte de PROTEINS y bzr-rob)

"Atascada al inicio" es una instancia en la que ninguna eliminación suelta mantiene la clase en el primer barrido, tanto en bls_v1 como en selector_v1. Tabla con el número de atascadas, cuántas rescata cada brazo (llega a mejorar algo), por qué mecanismo lo hace selector_v1 la primera vez y en qué llamada (mediana), el presupuesto que cada brazo deja sin gastar en ellas, y cuántos bloques ciegos acertaron en la colección uniforme del fold 1 sobre instancias atascadas.

| Celda | Atascadas al inicio | Rescatadas sel / v1 / v0 | Primer rescate de sel: mecanismo, llamada | Sin gastar sel / v1 / v0 | Bloques ciegos en colección: éxitos / intentos |
|---|---|---|---|---|---|
| proteins | 43 de 112 | 23 / 20 / 19 | overshoot, 1549 | 4888 / 5128 / 2745 | 0 / 4433 |
| proteins-m90 | 108 de 112 | 5 / 0 / 1 | reducir y añadir, 7386 | 2582 / 2583 / 0 | 0 / 15 444 |
| proteins-rob | 46 de 112 | 23 / 20 / 11 | overshoot, 1130 | 4822 / 4773 / 1990 | 0 / 5577 |
| bbbp | 93 de 204 | 77 / 46 / . | overshoot y swap, 1550 | 7317 / 6910 / . | 53 / 6596 |
| bbbp-m90 | 166 de 204 | 92 / 38 / 78 | reducir y añadir, 5487 | 4446 / 4270 / 2366 | 0 / 19 995 |
| bbbp-rob | 140 de 204 | 48 / 40 / 50 | swap y overshoot, 1060 | 6319 / 6348 / 4636 | 0 / 14 784 |
| aids | 167 de 200 | 16 / 17 / 22 | overshoot y swap, 1108 | 5659 / 5593 / 4060 | 0 / 20 184 |
| aids-m90 | 173 de 200 | 14 / 14 / 22 | overshoot, 1003 | 5207 / 5208 / 3743 | 2 / 22 030 |
| aids-rob | 162 de 200 | 29 / 19 / 26 | reducir y añadir, 4257 | 5199 / 5392 / 3607 | 5 / 19 587 |
| bzr-rob | 26 de 39 | 18 / 19 / 21 | overshoot, 826 | 6714 / 6682 / 5216 | 0 / 1510 |

Lecturas:

- Los bloques existen pero son agujas en un pajar. A ciegas y con 2000 llamadas no aparece ninguno en PROTEINS, bbbp-m90, bbbp-rob ni aids. bls_v0 los encuentra con 10 000 llamadas porque su barrido de niveles (64 subconjuntos aleatorios por nivel, todos los niveles, iteración tras iteración) es un fallback ciego que no se rinde: en las 57 instancias de bbbp-m90 en que gana a bls_v1, su primera mejora llega en la llamada 7051 (mediana) con un bloque de 4.2 aristas, y bls_v1 nunca mejoró 50 de ellas. El fallback del esqueleto del selector es el mismo barrido pero tapado a la mitad del presupuesto (`overshoot_budget_frac 0.5`) y una sola vez por estado, y después gasta 1300 llamadas en swap y add que ahí no sirven y se para.
- El ranking también compone bloques. Los rescates de selector_v1 en bbbp-m90 (92 frente a 38 del azar y 78 de bls_v0) vienen de la fase de reducir y añadir: `reduce_random` quita las aristas peor puntuadas y `edge_add` devuelve las mejor puntuadas hasta recuperar la clase, con saldo neto de 4 a 7 aristas menos. En aids-rob pasa lo mismo (29 frente a 19). Con propuestas uniformes ese movimiento casi nunca acierta.
- El presupuesto sin gastar es la causa de que bls_v1 pierda contra bls_v0 en las celdas duras. No es que el esqueleto busque peor: es que deja de buscar. Con el fallback a presupuesto completo, bls_v1 contiene a bls_v0 (es el mismo barrido ciego) y la comparación vuelve a ser de igual presupuesto.
- Las instancias atascadas concentran la GED: 76 % de la de selector_v1 en bbbp-m90, 84 % en bbbp-rob, 63 % al 81 % en aids, 52 % en PROTEINS (folds 1 a 9). Cualquier ganancia de v2 que se vea en las tablas de oráculos duros tiene que salir de aquí.

### 3.3 Régimen C: sin sitio (synthie, colors-3, enzymes, bzr)

El DCE parte lejos (23 a 189 ediciones) y los tres brazos llegan a contrafactuales de 1 a 3 aristas con 20 a 2000 llamadas. Las diferencias son de décimas sin significación, en los tres oráculos. El margen y el reentrenamiento robusto no cambian el régimen (bzr-rob sube a 5 o 6, pero con 39 instancias y empates). El preentrenamiento aquí es malo (synthie 0.39 a 0.61, enzymes 0.54 a 0.79) porque la mitad o más de las sueltas funcionan y no hay nada que ordenar, y da igual. Ningún modelo va a mover estas filas.

## 4. PROTEINS en los folds 1 a 9: la confirmación limpia

`scripts/analyze_selector_arms.py --arms random=... sel=... untrained=... --folds 1 2 3 4 5 6 7 8 9`, 1001 instancias, checkpoint `-f1` (el fold 1 está en la colección, así que estrictamente limpios son los folds 2 a 9, y el resultado no cambia al quitar el 1).

- GED final: selector_v1 29.6, bls_v1 31.0, red sin entrenar 30.9. Curva anytime: a 250 llamadas 39.9 frente a 47.8 y 48.2, a 1000 llamadas 34.4 frente a 37.9 y 38.7.
- Instancias que avanzan (726): selector_v1 19.4 frente a 22.7 de la red sin entrenar (375 mejor, 218 empate, 133 peor, p 1.8·10⁻³⁶), llamadas 2382 frente a 2860 (p 7·10⁻⁴¹), tiempo de pared 96 s frente a 164 s.
- Intentos por eliminación conseguida: 1.77 (selector), 2.44 (uniforme), 2.66 (sin entrenar). AUC precuencial de la red: 0.84 frente a 0.68 sin entrenar.
- Nunca mejoradas por el selector: 275 (27 %), con el 52 % de su GED. En ellas el overshoot gasta 4677 llamadas por instancia con cero éxitos. Rescatadas por el fallback: 136, donde el selector termina en 7.0 y el azar en 8.4.
- La red sin entrenar es indistinguible del azar (30.85 frente a 30.98), como en los folds 0 a 4.

La conclusión de `ANALISIS_CONTROLES.md` se sostiene con nueve veces más datos: el modelo entrenado mejora el óptimo local y la velocidad, y el tercio atascado queda fuera de su alcance.

## 5. Dos confusores en la configuración de los brazos

### 5.1 La escalera de bloques

`block_removal true` con `block_policy unlock` y `rung_stats_scope run`: el tamaño de bloque 2k se desbloquea cuando el tamaño k ha tenido al menos 6 intentos con tasa de éxito ≥ 0.5, y las estadísticas se acumulan sobre toda la corrida. Una vez desbloqueada en una instancia fácil, la escalera ofrece bloques de 2, 4, 8 o más (ventanas de aristas consecutivas en el orden del ranking, mezcladas con muestras aleatorias) en todas las instancias siguientes. Parte de la GED total que cada brazo elimina con bloques de más de una arista, selector_v1 / bls_v1: tri-40-gcn 3 % / 86 %, tri-80-gcn 91 % / 2 %, tri-80-gcn-rob 84 % / 2 %, tri-40n 92 % / 91 %, tri-40n-gcn 93 % / 6 %, enzymes-m90 55 % / 67 %, enzymes-rob 66 % / 7 %, colors-3 99 % / 99 %. En el resto, por debajo del 25 %.

Efecto: ahorra llamadas (tri-80-gcn selector 1065 frente a 1591 en tri-80) y cuesta GED, porque un bloque de 4 aristas consecutivas del ranking revierte ediciones que importaban. Es la única celda de tri donde el selector no gana (20.0 frente a 20.8), con el mismo dataset y una GCN al 100 % de acierto. En tri-40n y tri-80-gcn-rob el selector gana a pesar de los bloques, así que el daño no es sistemático, pero el interruptor es aleatorio entre brazos de la misma celda y eso no se puede llevar a una tabla.

Propuesta: `block_removal false` en bls_v1, selector_v1 y v2. Los bloques quedan para el overshoot, el fallback y la fase de reducir y añadir, que es donde el régimen B los necesita.

### 5.2 El presupuesto del fallback

`overshoot_budget_frac 0.5`: cuando nada ha funcionado, el fallback ciego (los niveles de bls_v0 con 64 intentos cada uno) se corta a la mitad del presupuesto total. bls_v0 no se corta. Propuesta: `overshoot_budget_frac 1.0` en los tres brazos. Solo cambia las instancias sin ningún progreso (el overshoot tras una mejora tiene pocos tamaños y 12 intentos, y el tope no le llega), y en ellas el esqueleto pasa a hacer exactamente lo que hace bls_v0 hasta el final del presupuesto. Con `retry_outer` no hace falta tocar nada.

Ambos son parámetros de la configuración, leídos con `.get`, no cambios en el esqueleto. Cambian el hash de las configs de bls_v1 y selector_v1, que aún no se han lanzado en la flota, así que no se pierde nada corrido para el paper. Hay que repetir el fold 0 de unas pocas celdas para verificarlos antes de regenerar la matriz (sección 8).

## 6. selector_v2: diseño ajustado

Reglas que se mantienen: el esqueleto de búsqueda no cambia, el modelo es lo único que distingue selector_v2 de selector_v1, y los dos se comparan con la misma configuración del esqueleto (sección 5), de modo que ninguna ganancia de configuración se le atribuya al modelo.

### 6.1 Lo que los datos piden

1. Conservar el ranking de sueltas tal cual (featurización por arista, tronco MLP, acquisition Thompson, preentrenamiento offline en el fold 1). Es la palanca del régimen A y funciona.
2. Hacer que el ranking apunte al valor, no solo al flip. En el régimen A el orden decide el óptimo local y el objetivo actual ("la reversión mantiene la clase") es un sustituto. La etiqueta de valor existe gratis en cada corrida: si la arista acabó fuera del contrafactual final. Cabeza auxiliar entrenada con esa etiqueta y ranking por mezcla de ambas cabezas. Es una conjetura razonable, no un hecho medido, y por eso lleva comprobación offline (6.3).
3. Puntuar bloques en el régimen B. El fallback near-miss de hoy muestrea bloques con probabilidad proporcional a la probabilidad contrafactual de cada eliminación suelta: una señal específica de la instancia que la red no ve. v2 le pone un puntuador de conjuntos: logits por arista del mismo tronco, con los rasgos de evidencia de la instancia como entrada (veces probada, probabilidad media y máxima, que ya están en `evid` de la colección y en v4), agregados con el pooling `sum_bias` que ya existe, más el tamaño del bloque. Se entrena con los intentos de bloque de la colección nueva (overshoot, fallback, reducir y añadir, y el bloque final de cada instancia como positivo) y en línea compone bloques por búsqueda voraz sobre el puntuador en lugar de muestrearlos. Dato que lo justifica: la composición por ranking de sueltas ya rescata el doble que el azar, y el azar puro no acierta nunca a 2000 llamadas.
4. No modelar el oráculo (propuesta A). Offline no habría positivos de bloque en las celdas duras con las colecciones de 2000 llamadas, el formato de colección no guarda el conjunto S, y una GNN de fidelidad por instancia es otra tubería. Queda como trabajo futuro, con la colección nueva como material si algún día se retoma.

### 6.2 Colección nueva (requisito de todo lo demás)

El formato actual guarda rasgos por arista, modo, tamaño, etiqueta y paso, pero no los identificadores del contexto ni del movimiento, así que no se puede reconstruir el bloque final ni etiquetar valor. Añadir, opt-in y neutro para el hash (`collect_examples_v2`), el guardado de `(instancia, ids de S, ids del movimiento, modo, y, paso)` junto a los rasgos. Correrla en el fold 1 de cada celda (y el fold 2 para tener dos checkpoints, como en la matriz) con selector_v1 congelado, presupuesto completo y la configuración de la sección 5. Son las mismas 33 celdas por 2 folds, 66 configs, un paso de servidor como la cola A del sondeo. Con ella se entrenan las dos cabezas nuevas y se miden los umbrales de 6.3.

### 6.3 Umbrales de decisión (offline, antes de tocar la búsqueda)

Sobre instancias no vistas (fold 2 contra fold 1 de la misma celda), en las celdas de los regímenes A y B:

- Cabeza de valor: AUC de "la arista acaba fuera del contrafactual final" al inicio de la búsqueda, comparada con la AUC de la cabeza de flip para la misma etiqueta. Si la mezcla no supera a flip en al menos 0.03 de AUC en tri y PROTEINS, v2 no lleva cabeza de valor.
- Puntuador de bloques: en las instancias atascadas al inicio, AUC del puntuador sobre los intentos de bloque registrados (acertados contra fallidos) y, con 10 000 llamadas de presupuesto simulado por reposición, la fracción de atascadas en que la composición voraz encuentra un bloque válido antes que el muestreo near-miss. Si la AUC no llega a 0.75 o la composición no mejora al near-miss en bbbp-m90 y aids-rob, v2 no lleva puntuador de bloques y el régimen B se queda con la configuración de la sección 5.

Si ninguno de los dos pasa, selector_v2 no existe como arquitectura y el paper va con selector_v1 sobre la configuración corregida, que ya es significativo en 17 de las 33 celdas.

### 6.4 Prueba en línea

Con lo que pase los umbrales, un brazo `dce-lcls-sel-v2` en el fold 0 de las celdas con señal: tri-40, tri-80, tri-80-gcn, bbbp, bbbp-m90, bbbp-rob, proteins, proteins-rob, aids-rob (9 configs, un paso de servidor), contra selector_v1 con la misma configuración. Criterio para entrar en la flota: no perder en ninguna celda de régimen A y ganar con p < 0.05 en al menos dos del régimen B.

## 7. Decisiones del usuario

Tomadas el 2026-11-03: (1) la configuración corregida, (2) margen τ = 0.7 para toda la tabla, (3) los datasets del régimen C se quedan como filas de "no hace daño". Y, en una segunda ronda el mismo día: (4) el oráculo robusto sale de los experimentos (cambia poco salvo en bbbp y no aporta dificultad; su código, los árboles `<dataset>-rob` y los resultados del sondeo se quedan), y (5) no habrá selector_v2: con el entrenamiento demostrado sin fuga en PROTEINS (sección 4) y la flota extendiéndolo a las demás celdas, el paper va con selector_v1 sobre la configuración corregida. Las secciones 6 y 8 quedan como diseño de referencia por si se retoma. La flota va en tres batches de unos dos días (`PAPER_FLOTA.md`). Lo que sigue es el detalle de cada opción tal como se planteó.

Cómo quedan implementadas: `gen_selector_paper_matrix.py` genera la generación 2 de los brazos con nombres nuevos (`dce-lcls-v1`, `dce-lcls-sel-v1`, `dce-lcls-sel-v1-untrained`, `dce-lcls-collect2`), de modo que los resultados del primer sondeo quedan intactos bajo los nombres antiguos (`dce-lcls-sel-random`, `dce-lcls-sel`, `dce-lcls-sel-collect`, árboles `-m90`) y `slurm_submit.sh` no los confunde con resultados de la configuración nueva. Las celdas de margen pasan a `<dataset>-m70` y `<tri>-gcn-m70`. colors-3 va siempre como pseudo-dataset de 20 instancias por fold (40 en las colecciones). Cada brazo escribe su monitor con un `model_tag` por celda. Las colecciones nuevas (`<celda>_collect2-f<f>.pt`, con `collect_ids`) guardan los pares de aristas del contexto y del movimiento y un registro final por instancia, y sus checkpoints son `edge_selector_sel2-pre_<celda>-f<f>.pt`. `sondeo_summary.py --gen 1` reproduce la tabla de este documento, `--gen 2` lee la generación nueva.

1. Configuración de los brazos para la flota: `block_removal false` y `overshoot_budget_frac 1.0` en bls_v1, selector_v1 y v2 (sección 5). Recomendado.
2. proteins-m90: degenerado (108 de 112 atascadas, nadie se mueve del contrafactual del DCE). Opciones: dejarlo como fila límite, quitar la fila, o bajar el margen de la tabla a τ=0.7 para todos los datasets (el sondeo local de 20 instancias de PROTEINS con τ=0.7 dio bls_v0 66 y bls_v1 58, hay movimiento). Bajar τ cuesta repetir el fold 0 de 7 celdas (28 configs) y los cambios en bbbp-m90, que con 0.9 es la mejor celda del régimen B.
3. Datasets del régimen C en las tablas: synthie, colors-3, enzymes y bzr son empates con los tres oráculos. Dejarlos como filas de "no hace daño" (costo: 4 datasets por 3 oráculos por 4 brazos por 10 folds en la flota) o quitarlos y concentrar la flota en PROTEINS, bbbp, aids y tri.
4. aids: casi sin movimiento (de 15 a 13) y atascado al 80 %. Es la celda donde un puntuador de bloques tendría más que ganar y menos que perder. Mantener.
5. Si v2 no pasa los umbrales de 6.3: el paper con selector_v1 corregido (sección 6.3, último párrafo).

## 8. Orden de trabajo

1. Hecho el 2026-11-03: `collect_ids` en el esqueleto (solo registro, opt-in), la generación 2 de los brazos en el generador de la matriz con la configuración de la sección 5 y el margen 0.7, los scripts de preentrenamiento y de resumen adaptados.
2. Servidor, sondeo 2 (`PAPER_SONDEO.md`): fold 0 de las 33 celdas con los brazos nuevos, colecciones de los folds 1 y 2 (cola A, 132 configs, de los que los bls_v0 ya corridos se saltan), preentrenamiento (1 job), selector_v1 (cola B, 33). Hace a la vez de verificación de la configuración (se espera bls_v1 ≥ bls_v0 en bbbp y aids con los oráculos duros, y selector_v1 por encima de bls_v1 en tri-80-gcn), de sondeo de las celdas m70 y de colección para v2.
3. Local, offline: entrenamiento de las dos cabezas y umbrales de 6.3. Uno o dos días de cómputo de portátil.
4. Servidor, prueba en línea de v2 (6.4), 9 configs.
5. Añadir el brazo `dce-lcls-sel-v2` al generador si pasa, regenerar las colas y lanzar la flota (`PAPER_QUEUES.md`): bls_v0, bls_v1, selector_v1, control sin entrenar y selector_v2, 10 folds.

## 9. Pendientes menores encontrados en el sondeo

- bbbp con el oráculo original no tiene bls_v0 en el fold 0. Entra en la flota.
- Los monitores (`lab/output/selector_logs`) mezclan en un mismo CSV todas las corridas de un dataset, con las tres oráculos y los diez folds escribiendo a la vez en el servidor. `sondeo_summary.py` empareja por instancia (fold, id, tamaño inicial y final, llamadas). Para la flota conviene que el nombre del CSV lleve el oráculo (`model_tag` por celda en el generador), que es un cambio de configuración.
- El checkpoint de proteins-m90 no tiene positivos en validación. Si proteins-m90 se queda, su preentrenamiento necesita la colección a presupuesto completo (6.2).
- Los resultados de tcr (aparcado) y las cachés de IMDB y Mutagenicity (descartados) siguen sin commitear a propósito.
