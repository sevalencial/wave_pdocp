# Primera campaña FOM: amplitud espacial de la velocidad de propagación

Ejecución del 23 de septiembre de 2026, reproducible mediante
`configs/parameter_sweep.yaml` y `scripts/run_parameter_sweep.py`.

## Familia y alcance

Se usaron 100 valores uniformes de `mu in [0, 0.3]` en
$c_mu(x)=1+mu*sin(2\pi x/L)$. En todo el dominio se cumplen `0.7 <= c_mu <= 1.3`
y `||c_mu-c_nu||_infinity <= |mu-nu|`.

Se mantuvieron fijos `d(x)=0.1+0.05*cos(2*pi*x/L)`, la región de control `(0,L)`,
estado inicial cero y target `(sin(pi*x/L),0)`. Se usaron `L=1`, `T=2`, `beta=100`
y `N=100`. El punto `mu=0.2` reproduce el ejemplo individual anterior.

El tiempo sigue siendo continuo; semigrupo y Gramiano se calculan mediante
exponenciales matriciales. Los adjuntos usan las métricas `Q_h` y `R_h`, y los
coeficientes de difusión se evalúan directamente en semipuntos.

## Resultados de los 100 FOM

| Indicador | Resultado |
|---|---:|
| PASS / WARNING / FAIL | 100 / 0 / 0 |
| Máximo residual relativo | 4.61e-14 |
| Máximo error relativo de optimalidad | 4.28e-12 |
| Error terminal relativo H_h | 1.0653 % – 1.1329 % |
| Mínimo autovalor del Gramiano, entre todas las muestras | 0.536709 |
| Máximo autovalor del Gramiano, entre todas las muestras | 1.848941 |
| Máximo condicionamiento de W en H_h | 3.444962 |
| Máximo condicionamiento de I + beta W en H_h | 3.400240 |
| Máxima energía de control | 5.398323 |
| Reducción mínima del costo frente a control nulo | 98.8931 % |
| Máximo indicador local de sensibilidad del adjunto | 2.034264 |

Los campos y soluciones almacenados son finitos. Se verificaron la condición
terminal del adjunto, autoadjunción ponderada y positividad del Gramiano, espectro
disipativo de A y `J(u*) <= J(0)`. Al incrementar mu aumentaron suavemente el costo,
el error terminal y el condicionamiento. El indicador de sensibilidad no detectó
outliers con el criterio configurado; esto es evidencia numérica en la malla
paramétrica usada, no una demostración global de regularidad.

El error terminal no es el residual algebraico: permanece distinto de cero por
la penalización finita. Tampoco mide directamente el error de discretización.

## Refinamiento espacial

Se seleccionaron los identificadores `0,13,25,37,50,74,86,99`, correspondientes a
`mu=0,0.039394,0.075758,0.112121,0.151515,0.224242,0.260606,0.3`.
La selección incluye extremos, centro y máximos de condicionamiento, error,
energía y sensibilidad; los extremos duplicados se completan con puntos alejados
de los ya elegidos.

Se compararon `N=50,100,200`. El adjunto grueso se interpola linealmente a la
malla fina, con extremos homogéneos para estos snapshots suaves de Dirichlet;
la diferencia se mide con `Q_h` de la malla fina.

| Comparación | Rango de diferencias relativas H_h |
|---|---:|
| 50 → 100 | 0.5404 % – 1.1193 % |
| 100 → 200 | 0.2767 % – 0.5671 % |

Las diferencias decrecieron en los ocho parámetros, con factores de reducción
aproximadamente entre 0.507 y 0.512. Esta comparación incluye el error de
interpolación, especialmente en la componente H_0^1: no se atribuye ese factor
exclusivamente al orden del stencil ni se afirma una cota rigurosa respecto a
la solución continua. El dataset de esta campaña conserva la malla N=100.

## Clasificación, persistencia y reproducibilidad

Las tolerancias y umbrales están en YAML. Residual relativo, optimalidad relativa
y simetría ponderada deben ser menores que `1e-9`. La tolerancia de positividad
se escala con el espectro; valores negativos de redondeo no se convierten
automáticamente en `FAIL`. La condición infinita de un Gramiano singular es una
advertencia, mientras que arrays no finitos o diagnósticos NaN son fallos.
Los warnings de Python/SciPy se capturan, se imprimen y se guardan en los motivos.

Los indicadores entre muestras consecutivas se asignan al extremo derecho del
intervalo. La primera fila y los intervalos que tocan un fallo quedan vacíos en
CSV: esos vacíos no representan NaN en las soluciones ni sensibilidad cero.
No se conectan artificialmente intervalos a través de una muestra fallida.

El HDF5 de entrenamiento contiene exclusivamente muestras PASS, con semilla 42 y
splits disjuntos de 75/15/10. Se comprobó lectura y escritura de campos, adjuntos,
parámetros e índices, y que una repetición explícita conserva exactamente los
splits. Los WARNING se conservan en el archivo de auditoría, sin incorporación
automática al entrenamiento; los FAIL conservan sus motivos en CSV.

Se aprobaron 67 pruebas y Ruff. Las pruebas de integración usan cinco parámetros
con N=20 y refinamiento pequeño; no ejecutan la campaña de 100 FOM.

Las figuras están en `outputs/parameter_sweep/figures/`. Las tablas y el resumen
JSON guardados son la fuente de los valores anteriores. Este bloque no entrena
redes ni evalúa extrapolación: el split aleatorio permanece dentro del intervalo
paramétrico de la campaña.

## Comparación explícita con el target — 24 de septiembre de 2026

Se añadió `representative_final_states_vs_target.png`, con el desplazamiento y
la velocidad finales de los ocho parámetros representativos frente a los targets
almacenados en HDF5. El target común es `y^T(x)=sin(pi*x)`, `v^T(x)=0`; aparece
como línea negra discontinua.

La figura `representative_adjoints.png` muestra otra cantidad: el adjunto
`P_T=beta*(Y*(T)-Y^T)`. Por eso una primera componente negativa del adjunto es
compatible con un desplazamiento final positivo ligeramente inferior al target.

Las figuras pueden regenerarse con `scripts/plot_parameter_sweep.py` a partir de
los datos existentes. Esta ampliación es solo de visualización; conserva los
resultados numéricos y los splits de la campaña.
