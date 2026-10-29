# Controles del selector en PROTEINS: qué aporta el modelo, qué aporta la búsqueda y hacia dónde ir

Análisis del 2026-10-29 sobre los folds 0 a 4 de los cuatro brazos de `scripts/gen_selector_proteins_folds.py` (558 instancias, presupuesto 10 000 llamadas, DCE compartido por la caché del generador): LBS puro (`proteins-sel_dce_lcls`), el selector preentrenado y congelado (`-sel`), la misma red sin entrenar (`-sel-untrained`) y el mismo esqueleto con propuestas uniformes (`-sel-random`). Los folds 5 a 9 de los dos controles estaban aún en el servidor al escribir esto. Todos los números salen de `scripts/analyze_selector_arms.py --folds 0 1 2 3 4` (trayectorias de los logs, volcados por instancia y el monitor del selector). Las conclusiones de la sección 2 y la sección 5 no dependen de los folds que faltan.

## 1. Resumen

- El selector entrenado es mejor que LBS a cualquier presupuesto, no solo al final. Si LBS se cortara en el momento en que el selector se detiene, quedaría en GED 37.1, peor que con todo su presupuesto (35.0) y lejos del selector (26.5). Parar antes es una consecuencia de encontrar antes, no una causa de la mejora. La lógica de la discusión era correcta.
- La red sin entrenar no es mejor que no tener modelo: GED 27.31 frente a 27.63, p 0.73 en pareado, y gasta un 4 % más de llamadas (p 2e-6). "Sin modelo es algo peor" era ruido de la media.
- El modelo entrenado sí aporta, y más de lo que decía la media global: el 23 % de las instancias (128 de 558) no sale nunca de su contrafactual inicial en ningún brazo del selector y concentra el 45 % de la GED. En el 77 % restante, selector frente a red sin entrenar: GED 18.75 frente a 21.51 (−13 %, p 1e-18, 213/132/85), llamadas 2265 frente a 2742 (−17 %, p 1e-27). El modelo convierte 2.43 intentos por eliminación conseguida (uniforme) en 1.78, y el camino que sigue acaba en óptimos locales mejores. Además es más rápido en reloj: 85 s por instancia frente a 157 (sin entrenar) y 148 (uniforme), porque se detiene antes.
- La diferencia estructural entre LBS puro y el esqueleto del selector no es el modelo: es que LBS, con |S| ≥ 20, nunca prueba quitar una sola arista. Su rejilla de niveles usa un paso floor(|S|/20) + 1, así que quita exactamente 2 aristas por mejora entre 20 y 39, 3 entre 40 y 59, 4 o más por encima. Los brazos del selector consiguen 17 700 eliminaciones sueltas con |S| ≥ 20. Esto, más una verificación mucho más barata y la regla de parada, es lo que miden el brazo uniforme (GED −21 %, llamadas −48 % frente a LBS).
- Dónde va el presupuesto del selector: los barridos de eliminaciones sueltas producen el 96 % de la bajada de GED y cuestan alrededor del 11 % de las llamadas. El overshoot (48 %), el swap (21 %) y la adición (20 %) producen el 3.6 % restante. Todos los brazos, LBS incluido, gastan la mayor parte del presupuesto demostrando que han convergido (LBS el 91 % de sus llamadas después de su última mejora, el selector el 82 %).
- Fuga en el fold 0: las 72 instancias de preentrenamiento del selector son instancias de test del fold 0 (72 de 112). Los folds 1 a 9 están limpios. El efecto parece pequeño (la ventaja del selector sobre la red sin entrenar en el fold 0 no es mayor que en los folds 3 y 4), pero para el paper hay que reportar los folds 1 a 9 o preentrenar de nuevo con instancias del fold 1, como hace el estudio de los otros datasets (sección 7).
- Propuesta para el modelo (sección 6): dejar de puntuar aristas sueltas con rasgos fijos del dataset y aprender un gemelo del oráculo por instancia, que puntúe también bloques. Es lo único que puede entrar en el 23 % atascado, que es donde está casi la mitad de la GED.

## 2. ¿Parar antes es causa o consecuencia?

Curva anytime: tamaño medio de la mejor solución tras b llamadas del minimizador, 558 instancias. Un brazo que ya se detuvo conserva su tamaño final.

| Llamadas | LBS | Selector | Sin entrenar | Uniforme |
|---|---|---|---|---|
| 100 | 48.20 | 39.96 | 49.54 | 49.91 |
| 250 | 43.70 | 35.45 | 43.98 | 43.62 |
| 500 | 41.18 | 33.51 | 38.80 | 37.68 |
| 1000 | 38.20 | 30.32 | 33.96 | 32.79 |
| 2000 | 36.58 | 28.42 | 30.37 | 29.72 |
| 5000 | 35.46 | 27.36 | 28.59 | 28.48 |
| 10000 | 34.99 | 26.46 | 27.31 | 27.63 |

Tres cosas se leen aquí.

Primera, el selector entrenado está por debajo de LBS a todo presupuesto. Cortando LBS instancia a instancia en el número de llamadas donde el selector se detuvo, LBS queda en 37.09 frente a 26.46 del selector (300/200/58, p 1e-40). La mejora de GED no puede venir de parar antes. Parar antes solo ahorra llamadas, y las que ahorra son llamadas que LBS desperdicia: LBS gasta de media 7227 de sus 7955 llamadas (el 91 %) después de su última mejora, y seguir iterando tras la primera iteración externa sin mejora le gana 0.08 de GED por instancia (mejora en 24 de 558 instancias, 47 unidades de GED sobre una bajada total de 17 440).

Segunda, el matiz que conviene corregir en el argumento "LBS gasta el presupuesto porque no encuentra tan rápido". En la primera mitad de su propia bajada LBS es tan rápido como el selector (mediana 32 llamadas frente a 35, y su primera mejora llega en la llamada 2). Lo que agota su presupuesto es el coste de un solo barrido de verificación fallido: 64 intentos uniformes en cada nivel de la rejilla (entre 9 y 16 niveles), cada intento con hasta 6 llamadas por los métodos de atributos, es decir, entre 3456 y 6144 llamadas por barrido fallido. Una o dos iteraciones externas sin mejora y el presupuesto se acabó. El 71 % de las instancias de LBS llegan al tope de 10 000. En el esqueleto del selector una iteración externa fallida cuesta unas 2700 llamadas (barrido exhaustivo de aristas sueltas, overshoot geométrico, swap y adición).

Tercera, el modelo entrenado marca la diferencia sobre todo al principio. A 500 llamadas el selector va en 33.5 frente a 38.8 de la red sin entrenar y 37.7 del uniforme; a 1000, 30.3 frente a 34.0 y 32.8. Con todo el presupuesto los controles se acercan a su suelo, un punto de GED por encima. La ganancia del modelo es velocidad, y en la cuenta final se traduce en llamadas (el selector reduce el presupuesto que le queda por gastar) más que en GED. Nótese también que sin modelo entrenado los brazos del selector son más lentos que LBS en las primeras 250 llamadas: LBS quita de 2 en 2 (su primer nivel), ellos de 1 en 1.

## 3. Modelo entrenado, red sin entrenar y uniforme

Pareado, 558 instancias:

| Comparación | GED | Llamadas |
|---|---|---|
| Sin entrenar frente a uniforme | 27.31 frente a 27.63, 147/242/169, p 0.73 | 3730 frente a 3600, p 2e-6 (más) |
| Selector frente a sin entrenar | 26.46 frente a 27.31, 213/243/102, p 4e-11 | 4469 frente a 4844, p 2e-25 |
| Selector frente a uniforme | 26.46 frente a 27.63, 201/255/102, p 4e-12 | 4469 frente a 4714, p 9e-15 |

(Las llamadas de esta tabla incluyen las del generador, 1114 de media, como en los resultados. El resto del documento cuenta solo las del minimizador.)

La red sin entrenar no se distingue del uniforme en GED y gasta más llamadas, con más intentos por eliminación conseguida (2.72 frente a 2.43). Su AUC precuencial de 0.68 (frente a 0.50 del uniforme y 0.84 del entrenado) no debe leerse como poder predictivo: esa métrica tiene sesgo de cascada, y la métrica limpia es intentos por éxito. Conclusión: usar la red tal cual no es mejor que no usarla.

Lo que hace el modelo entrenado se ve mejor separando las instancias en las que nadie puede actuar. En 128 de 558 el selector no sale nunca de su contrafactual inicial (sección 4). En las otras 430:

| 430 instancias donde la búsqueda avanza | Selector | Sin entrenar | Uniforme |
|---|---|---|---|
| GED media | 18.75 | 21.51 | 21.35 |
| Llamadas del minimizador | 2265 | 2742 | 2600 |
| Selector mejor / igual / peor | | 213/132/85, p 1e-18 | 201/134/95, p 4e-15 |
| Llamadas, p | | 1e-27 | 9e-17 |

Mecanismo, del monitor (medias por instancia): intentos por eliminación conseguida 1.78 (selector), 2.43 (uniforme), 2.72 (sin entrenar); éxitos de eliminación por instancia 38.2, 37.2, 37.4. El modelo no consigue más eliminaciones, las consigue con menos intentos y en otro orden, y el orden importa: en una búsqueda de primera mejora el óptimo local depende del camino, y el camino del modelo entrenado acaba 2.8 unidades de GED más abajo que el de la red sin entrenar en estas 430 instancias. En tiempo de reloj del minimizador el selector tarda 85 s por instancia frente a 157 (sin entrenar) y 148 (uniforme), p 1e-80: el coste de inferencia de la red queda más que compensado por detenerse antes. (El tiempo de los volcados de resultados daba lo contrario porque el brazo del selector fue el que ejecutó el generador DCE en la mayoría de instancias y los controles lo leyeron de la caché.)

Por tamaño inicial, con los terciles de la sección anterior del análisis: en las instancias pequeñas (inicial 5 a 32) el selector empata con la red sin entrenar (p 0.64); en las medianas gana 78/69/33 (p 4e-5); en las grandes 99/48/38 (p 2e-9).

Fuga del preentrenamiento. Las 72 instancias de `v3-collect` con las que se preentrenó el selector son instancias de test del fold 0 (72 de las 112 evaluadas), y no aparecen en ningún otro fold. El fold 0 del brazo del selector evalúa por tanto instancias que la red vio en el preentrenamiento. Por folds, la ventaja del selector sobre la red sin entrenar es 0.8 en el fold 0, −0.7 en el 1, 0.5 en el 2, 1.9 en el 3 y 1.8 en el 4, así que la fuga no parece inflar el fold 0, pero el paper debe reportar los folds 1 a 9 o repetir el preentrenamiento con instancias disjuntas (el estudio de la sección 7 ya preentrena con el fold 1 y evalúa en el fold 0).

## 4. Dónde está la GED que falta

En 202 de las 558 instancias ninguna eliminación suelta mantiene el contrafactual al principio. 74 de ellas las rescata el fallback del overshoot (bloques ponderados por near-miss con la profundidad de LBS puro), con una primera bajada media de 7.1 aristas, y después descienden hasta una GED media de 6.8 (LBS llega a 12.5 en esas mismas, la red sin entrenar a 8.4, el uniforme a 8.5). Las otras 128 no mejoran nunca en el brazo del selector. LBS mejora 2 de ellas (fold 1 id 275 y fold 2 id 236, con bloques de 16 y 18 aristas), la red sin entrenar 17 y el uniforme 7: suerte en el muestreo de bloques, no un mecanismo.

Esas 128 instancias tienen un tamaño inicial medio de 52.4 (mediana 28), del orden del número de aristas del grafo (53 de media; la relación por instancia es 1.23 de media): el contrafactual de DCE casi no comparte aristas con el grafo original, restaurar cualquiera de ellas devuelve la clase original y, sin embargo, en algunos casos restaurar 16 a la vez no (con los siete métodos de atributos en juego, el paisaje del oráculo no es monótono). Concentran el 45 % de la GED del selector (6702 de 14 766) y le cuestan 7016 llamadas cada una (4784 en el fallback del overshoot, 989 en swap, 988 en adición) sin ningún éxito. El brazo `dce-lcls-sel-uniform-fallback`, con bloques uniformes en el fallback, mide si LBS tenía razón en esas dos.

En las 430 restantes la bajada viene casi entera de las eliminaciones sueltas:

| Tipo de mejora (558 instancias, brazo del selector) | Mejoras | GED quitada | Cuota |
|---|---|---|---|
| (-) barrido de eliminación | 21 227 | 21 363 | 96 % |
| (-o) overshoot | 79 (74 de ellas en el fallback) | 542 | 2 % |
| (=) swap | 21 | 219 | 1 % |
| (+) adición | 45 | 73 | 0.3 % |

El overshoot geométrico después de la primera mejora consigue 5 éxitos en 558 instancias y cuesta 661 llamadas por instancia (el 29 % del presupuesto de las 430 que avanzan). Swap y adición consiguen el 1.3 % de la GED con el 41 % de las llamadas. En LBS puro el 99 % de la bajada viene también de (-), con una media de 2.4 aristas por mejora porque sus niveles son bloques.

## 5. Qué cambia el esqueleto del selector respecto a BLS, y cómo defenderlo

Inventario completo, de `local_search.py` frente a `local_search_selection_net_v2.py` con la configuración final (`OVERSHOOT` en `gen_selector_proteins_folds.py`). La última columna dice si el cambio existe para usar un ranking (lo defendería como "entrada del selector") o si es una mejora de la búsqueda que no necesita modelo (lo defendería como mejora nativa de BLS, y es lo que el brazo uniforme mide).

| Mecanismo | BLS (tesis) | Esqueleto del selector | Efecto medido | Clasificación |
|---|---|---|---|---|
| Vecindario de eliminación | Niveles 0, paso, 2·paso, ..., con paso = floor(|S|/20) + 1 y 64 intentos uniformes por nivel. Con |S| ≥ 20 nunca prueba una arista suelta | Barrido exhaustivo de aristas sueltas en orden de ranking (peldaño 1, `fill_remove_share`), bloques de tamaño 2, 4, ... solo tras éxitos (`block_removal`, el tamaño se dobla) | 17 700 eliminaciones sueltas con |S| ≥ 20 por brazo; es la fuente del salto en instancias grandes (tercil alto 96 → 67) | Mixto. Un ranking exige barrer movimientos sueltos, pero LBS podría barrerlos también (basta `max_neigh` ≥ |S|). Es el cambio más importante y hay que declararlo |
| Intentos por nivel | 64 (eliminación), 16 (swap y adición) | 24 en los peldaños 1 y 2, 12 en los demás, con el doble en el nivel más cercano al tamaño medio de los éxitos (`adaptive_trials`) | Menos llamadas por barrido fallido | Entrada del selector en los barridos ordenados; en los brazos sin modelo es simplemente menos intentos uniformes |
| Overshoot | No existe como tal: sus niveles grandes están en el barrido normal | Tras un barrido de sueltas fallido, una sola vez por estado: bloques de tamaño 0.7·|S|, 0.49·|S|, ... (12 intentos cada uno); mientras la instancia no haya mejorado, la profundidad de LBS (tamaños 2, 2+paso, ..., 64 intentos) con bloques near-miss | 74 rescates y 5 éxitos posteriores; 1607 llamadas por instancia (48 % del presupuesto del selector) | Nativo: no usa el modelo |
| Near-miss | No existe (usa `predict`, descarta la probabilidad) | `predict_proba` al mismo coste (`dense_reward`); la probabilidad de la clase contrafactual de cada eliminación suelta fallida pondera los bloques del overshoot y la mitad de eliminación de los swaps | No aislado (los tres brazos lo llevan); el brazo `uniform-fallback` lo aísla en el fallback | Nativo: información que el oráculo ya da y LBS tira |
| Regla de parada | Hasta 20 iteraciones externas sin mejora o el tope | Se detiene tras la primera iteración externa sin mejora (`retry_outer` false), con tope local de 2000 llamadas por iteración (`local_budget_frac`) y cuotas por fase según productividad (suelo 5 %; en estado atascado 50/50 para swap y adición, `stuck_swap_add`) | LBS gana 0.08 de GED por instancia con sus reintentos y gasta en ellos el 91 % de sus llamadas | Nativo: política de presupuesto. Aplicada sola a LBS recortaría sus llamadas sin mover la GED, pero no produce el salto de GED (LBS cortado donde para el selector: 37.1) |
| Tabú, temperatura de barrido 0.25, mezcla 50 % uniforme (`block_random_mix`) | No existen | Memoria de movimientos ya probados en el estado, puntuaciones estandarizadas y mitad de los intentos sueltos uniformes como red de seguridad | Solo tienen sentido sobre un ranking | Entrada del selector |
| Reducción, swap y adición | Uniformes | La reducción y la mitad de adición de los swaps toman propuestas del modelo; la adición barre en orden de ranking | Swap y adición: 1.3 % de la GED en todos los brazos | Entrada del selector (sin modelo, uniformes) |
| Orden de los métodos de atributos | Fijo, 6 métodos | Por éxitos pasados (`adaptive_method_order`), 7 métodos (incluye `heat_kernel_diffusion`) | Un intento fallido cuesta hasta 7 llamadas frente a 6 en LBS: pequeña desventaja del selector que convendría igualar | Nativo (el orden); el séptimo método es un descuido de calibración |
| Verificación de la convergencia | Repite barridos enteros | Un barrido exhaustivo de sueltas (|S| × 7 llamadas) más overshoot, swap y adición | El 82 % de las llamadas del selector van después de su última mejora | Inherente a la búsqueda local de primera mejora con oráculo de caja negra |

Respuesta a la pregunta de fondo. Solo las filas de tabú y temperatura, las propuestas de reducción, swap y adición, y parte de los intentos por nivel son "entrada del selector" en sentido estricto. El vecindario de aristas sueltas, el overshoot, el near-miss, la regla de parada y el orden de los métodos no usan el modelo, y son exactamente lo que el brazo uniforme mide: GED −21 % y llamadas −48 % frente a LBS. Presentarlos como consecuencia del selector no resistiría la ablación que ya tenemos.

La forma honesta, y creo que más fuerte, de contarlo en el paper: BLS tal como está en la tesis gasta el presupuesto en una verificación cara con una rejilla que se vuelve gruesa al crecer |S|. Primera contribución, una política de vecindario y presupuesto para BLS (barrido exhaustivo de aristas sueltas, bloques solo cuando hace falta, parada tras una iteración sin mejora, near-miss): es BLS, no otro algoritmo, y el brazo uniforme es su ablación. Segunda contribución, un selector aprendido sobre esa política, que llega al mismo sitio en la mitad de llamadas y a mejores óptimos locales donde las eliminaciones sueltas funcionan (sección 3), con la ablación red sin entrenar frente a uniforme demostrando que lo que aporta es el entrenamiento y no la red. El argumento "parar antes porque converge antes" es válido para la segunda contribución (la curva anytime lo demuestra) y no para la primera.

Para una ablación limpia de la primera contribución faltaría un brazo: LBS puro con `max_neigh` igual a |S| (es decir, con el nivel de una arista) y con la regla de parada, para separar cuánto del −21 % es la rejilla y cuánto la parada. Es un config de LBS puro con dos parámetros, no un cambio de código, salvo la regla de parada, que LBS no tiene.

## 6. Propuesta para el modelo

Diagnóstico. El selector actual es un MLP sobre una arista suelta: 4k + 56 rasgos fijos (vectores de los dos extremos, similitudes estructurales antes y después del cambio, distancias), entrenado a predecir si quitarla mantiene el contrafactual. El banco de preentrenamiento mostró un techo de AUC 0.70 en instancias no vistas con estos rasgos, sea cual sea el modelo (red, GBM, logística), y en la búsqueda baja los intentos por éxito de 2.43 a 1.78. Su palanca está acotada por construcción: los barridos de sueltas son el 11 % de las llamadas, y el 23 % de instancias atascadas queda fuera de su alcance porque solo puntúa movimientos de una arista y ahí ninguna arista suelta funciona.

Lo que los datos piden al modelo:

1. Puntuar conjuntos, no aristas. El 45 % de la GED está en instancias donde solo funcionan bloques (LBS encontró bloques de 16 y 18; el fallback encontró bloques en 74 de 202), y el overshoot geométrico tras la primera mejora es casi inútil (5 éxitos en 558 instancias por 661 llamadas por instancia). Un proponente de bloques aprendido sustituiría a los dos.
2. Ser específico de la instancia. El techo de 0.70 es el techo de los priors del dataset. La búsqueda recibe miles de respuestas del oráculo por instancia (cada intento devuelve una probabilidad para hasta siete variantes de atributos) y el modelo actual las ignora. Los rasgos de evidencia de v4 fueron una forma débil de esto (AUC offline 0.76 → 0.83) y no se tradujeron a la búsqueda porque la política no cambiaba con ellos.
3. Predecir valor, no solo el siguiente flip. El orden cambia el óptimo local (18.75 frente a 21.51 en las 430 instancias que avanzan): una eliminación que mantiene el contrafactual pero cierra otras eliminaciones futuras es peor que otra que las deja abiertas.

Propuesta A, la principal: un gemelo del oráculo por instancia. Una GNN pequeña f que imita la probabilidad contrafactual del oráculo en los grafos que la búsqueda ya evaluó para esta instancia (cada intento aporta hasta siete grafos etiquetados, gratis), inicializada con un preentrenamiento offline sobre la partición de entrenamiento del dataset (consultar el oráculo offline en instancias de entrenamiento no cuenta en el protocolo, igual que el preentrenamiento actual). El selector pasa a ser un simulador barato: puntúa cualquier movimiento, suelto o bloque, por f(G ⊕ movimiento); ordena los barridos; compone bloques por búsqueda voraz o de haz sobre f (o por el gradiente de f respecto a la adyacencia, que es caja blanca porque f es nuestra); y puede saltarse swaps y adiciones que f da por perdidos. Entrada: adyacencia actual, atributos del método, indicador de qué aristas están en S. Coste: milisegundos por candidato en CPU frente a 20 a 40 ms por llamada al oráculo. Ajuste en línea: unos pocos pasos de gradiente por barrido sobre las respuestas acumuladas. Riesgos: fidelidad con los siete métodos de atributos (incluir el método como entrada o predecir el máximo sobre métodos), y que el paisaje no monótono de las instancias atascadas sea difícil también para f. Pero es lo único de la lista que puede ver lo que el modelo actual no ve.

Propuesta B, más barata y sobre el mismo esqueleto: un puntuador de conjuntos con evidencia. Sustituir el MLP por atención sobre las aristas de S y del bloque candidato, con contexto del grafo (embeddings de nodo de un codificador GNN congelado preentrenado en el dataset), la evidencia por arista acumulada en la instancia (veces evaluada, probabilidad media y máxima) y rasgos de estado (|S|, probabilidad del contrafactual actual, paso). Entrenado listwise sobre trayectorias recogidas a predecir el flip de un movimiento, suelto o bloque. Es incremental sobre v4 y no requiere modelar el oráculo.

Propuesta C, ortogonal y de política: un modelo de despacho que prediga desde el estado si swap, adición u overshoot pueden tener éxito y reparta el presupuesto. Toca el esqueleto, así que solo si se decide abrir esa puerta.

Esta propuesta es provisional: sale de un solo dataset. Cuando termine el estudio de un fold en los otros datasets (sección 7) hay que repetir este análisis con todos ellos antes de decidir la arquitectura, porque la respuesta puede cambiar. Lo que hay que volver a mirar por dataset: (a) la cuota de instancias atascadas y la GED que concentran, que es lo que justifica puntuar bloques; (b) intentos por eliminación conseguida del brazo uniforme, que dice si hay sitio para un ranking; (c) el reparto de llamadas y de GED entre barrido de sueltas, overshoot, swap y adición, que dice dónde está la palanca; (d) selector frente a red sin entrenar en las instancias que avanzan, que dice si los priors del dataset transfieren; (e) la AUC de validación del preentrenamiento de cada dataset (`models/pretrained/edge_selector_sel-pre_<dataset>.json`). Si en los demás datasets la GED no está en instancias atascadas, la propuesta A pierde su motivo principal y B basta; si el régimen de ASD (el 89 % de las eliminaciones sueltas funcionan, AUC 0.91 en validación) es el común, ningún ranking de sueltas aporta y el modelo debe apuntar a otra cosa (bloques, parada, o nada). El plan se reajusta con esa tabla.

Recomendación, a reserva de lo anterior: A, precedida de un estudio offline de uno o dos días de cómputo que no toca la búsqueda. (1) Una recolección nueva en PROTEINS que guarde, por intento, el conjunto S, el método de atributos y la probabilidad del oráculo (el grafo es G ⊕ S, no hace falta guardarlo), sobre las 72 instancias de `v3-collect` y las instancias atascadas de los folds 1 a 4. (2) Entrenar f y medir su fidelidad dentro de la instancia sobre estados no vistos y en instancias no vistas. (3) Dos métricas de decisión: AUC dentro de la instancia para eliminaciones sueltas (el modelo actual está en 0.70 offline) y precisión de los bloques que f propone en las instancias atascadas (la prueba real). Regla: si f supera 0.85 dentro de la instancia y encuentra bloques válidos en más del 30 % de las instancias atascadas offline, se implementa en línea. Si no, B.

## 7. Estudio de un fold en los otros datasets

`scripts/gen_selector_datasets_fold0.py` genera, para `asd`, `tcr-500-28`, `tcr-tco-300`, `synthie`, `bbbp`, `aids` y `colors-3`, los cuatro brazos en el test del fold 0 más un config de recolección en el fold 1 (uniforme, presupuesto 2000, `collect_examples`), cada dataset en su árbol `lab/config/generate_minimize/<dataset>-sel/dce/`. Los ajustes del minimizador son los de PROTEINS; del config de la tesis de cada dataset se conservan `attributed`, `recompute_features` y los tamaños de vecindario, de modo que el oráculo se consulta como en la tesis. El preentrenamiento usa el fold 1, disjunto del fold 0 evaluado, así que aquí no hay fuga.

Tamaños del test del fold 0: asd 11, synthie 40, tcr-tco-300 60, tcr-500-28 100, aids 200, bbbp 204, colors-3 1050. Colas: `docs/selector/DATASETS_FOLD0_1-collect.md` (28 configs: recolección y los tres brazos que no necesitan checkpoint) y `DATASETS_FOLD0_2-selector.md` (los 7 brazos del selector, después de preentrenar). Comandos en el docstring del generador. En esta máquina quedan unos 3 GB libres: un worker, no dos, y `--timeout 86400` (el del runner es de 2 h por config, y bbbp con LBS puro a 10 000 llamadas puede pasarlo).

Qué esperar, por el régimen de intentos por eliminación conseguida medido en la calibración: Synthie, donde el 75 % de las eliminaciones sueltas funcionan, y COLORS-3 (1.3 intentos por éxito) no dejan sitio al ranking; AIDS (221 intentos por éxito) tampoco, por el otro extremo; PROTEINS (14.6 en el esqueleto antiguo) es el régimen intermedio. La cifra que este estudio debe dar por dataset es esa: intentos por éxito del brazo uniforme, y el selector frente a la red sin entrenar en las instancias que avanzan.

## 8. Pendiente

- Folds 5 a 9 de los controles: confirmar las secciones 3 y 4 con las 1113 instancias.
- Brazo `uniform-fallback`: si rescata las dos instancias de LBS sin perder los 74 rescates del near-miss.
- Brazo de LBS puro con nivel de una arista y parada (sección 5), para la ablación de la primera contribución.
- Repetir el preentrenamiento de PROTEINS con instancias del fold 1 o reportar los folds 1 a 9.
- Igualar los métodos de atributos (6 o 7) entre LBS y el esqueleto del selector.
- Al terminar el estudio de los otros datasets: repetir el análisis de la sección 6 con todos ellos (las cinco medidas listadas allí) y reajustar el plan de arquitectura antes de implementar nada.
- Estudio offline del gemelo del oráculo (sección 6), si el reajuste lo confirma.
