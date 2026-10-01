# Wave-PDOCP

FOM de control óptimo para la ecuación de ondas 1D, semidiscreto en espacio y
continuo en tiempo. La formulación y las decisiones de la tesis están en
[`docs/WAVE_PDOCP_CONTEXT.md`](docs/WAVE_PDOCP_CONTEXT.md).

## Coeficientes constantes o funciones

`WaveParameter.wave_speed` y `WaveParameter.damping` aceptan números o funciones
vectorizadas de `x`. La función recibe un vector NumPy y devuelve otro de la misma
forma, o un escalar si representa un campo constante. Por ejemplo:

```python
import numpy as np
from wave_pdocp.domain.parameters import WaveParameter

mu = WaveParameter(
    wave_speed=lambda x: 1.0 + 0.2 * np.sin(2.0 * np.pi * x),
    damping=lambda x: 0.1 + 0.05 * np.cos(2.0 * np.pi * x),
    control_center=0.5,
    control_length=1.0,
    target_amplitude=1.0,
    target_mode=1,
)

# También se admiten constantes y combinaciones de constante y función:
mu_constante = WaveParameter(wave_speed=1.0, damping=0.0)
```

`c(x)` se evalúa en los N nodos y en los N+1 semipuntos; el operador conservativo
usa directamente `c(x_half)**2`. `d(x)` se evalúa en los nodos. Se comprueban forma,
valores finitos y positividad de `c` en los puntos evaluados. Las funciones deben
ser deterministas; no se evalúan mediante integración temporal.

El target es `target_amplitude * sin(target_mode * pi * x / L)`, con modo entero
positivo. El estado inicial y la velocidad objetivo siguen siendo cero. Un
`damping` negativo está permitido para estudios de antidamping, pero introduce
crecimiento en vez de disipación.

## Ejecutar un único caso

Desde la raíz del repositorio, con el entorno del proyecto:

```bash
.venv/bin/python scripts/run_single_mu.py
.venv/bin/python scripts/plot_control_reconstruction.py
```

Ambos scripts usan `SingleMuExperiment.variable_coefficient_example()`: un solo
mu con los campos del ejemplo anterior, `L=1`, `N=100`, `T=2`, `beta=100`, control
global y target `(sin(pi*x), 0)`. Los campos se definen en
`src/wave_pdocp/domain/examples.py`; malla, tiempo y penalización se configuran en
`src/wave_pdocp/orchestration/experiment.py`.

Los resultados quedan en `outputs/single_mu_variable/`:

- `metrics.json`: configuración, campos evaluados y diagnósticos;
- `final_state_vs_target.png`: desplazamiento y velocidad finales;
- `control_u.png`: control óptimo muestreado para visualización;
- `p_t.png`: componentes del adjunto terminal;
- `coefficients.png`: velocidad de propagación y amortiguamiento espaciales.

Para ejecutar sin interfaz gráfica:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/plot_control_reconstruction.py
```

El número de muestras del control solo afecta la figura: el FOM y su Gramiano
se calculan mediante exponenciales matriciales, sin pasos de tiempo.

## Interpretación y validación

Se resuelve `(I + beta W) P_T = beta (S(T) Y0 - YT)`. El objetivo penaliza el error
terminal y el esfuerzo del control; con beta finito no se exige alcanzar el target
exactamente. El residual del sistema y el error respecto al target son medidas
distintas.

Los diagnósticos usan el producto interno fijo `Q_h`, también con coeficientes
variables. El espectro y el condicionamiento se calculan en coordenadas
`C W C^{-1}`, donde `Q_h = C.T @ C`, usando solves y sin inversas explícitas.
La asimetría del Gramiano se reporta separadamente. El esfuerzo se calcula como
`P_T.T @ Q_h @ W @ P_T`, sin cuadratura temporal.

En el caso anterior se obtuvo un error terminal relativo en `H_h` de **1.096 %**,
un residual relativo de **3.18e-14** y `cond_H_h(I + beta W)` de **2.23**. Es una
validación individual del problema semidiscreto. La campaña descrita a
continuación añade comparaciones entre mallas para ocho parámetros representativos.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src scripts tests
```

Las pruebas incluyen una solución modal analítica con distintos beta y datos
iniciales, control reconstruido, flujo variable con solución manufacturada,
identidades de adjuntos, disipación de energía y optimalidad del caso variable.

## Primera campaña paramétrica

Se varía únicamente `mu` en `c_mu(x) = 1 + mu*sin(2*pi*x/L)`, con
`mu in [0, 0.3]`. Permanecen fijos `d(x)=0.1+0.05*cos(2*pi*x/L)`, control global,
datos iniciales cero, target `(sin(pi*x/L), 0)`, `L=1`, `T=2`, `beta=100` y `N=100`.
`ParameterSpace` representa los parámetros con un eje de dimensión explícito:
el arreglo de esta campaña tiene forma `(100, 1)`.

La configuración completa, incluidas tolerancias y rutas, está en
[`configs/parameter_sweep.yaml`](configs/parameter_sweep.yaml).

```bash
MPLBACKEND=Agg .venv/bin/python scripts/run_parameter_sweep.py --config configs/parameter_sweep.yaml
```

La ejecución resuelve 100 FOM, clasifica cada caso, analiza la sensibilidad local,
refina ocho parámetros con `N=50,100,200`, genera figuras y exporta los resultados.
Una ruta de salida existente detiene la ejecución; para reemplazar explícitamente
la campaña se añade `--overwrite`. La semilla 42 y los índices se guardan en HDF5;
no se recrean splits silenciosamente al leer el dataset.

Los archivos principales son:

- `data/datasets/wave_pdocp_fom.h5`: solo muestras `PASS` y splits `75/15/10`;
- `outputs/parameter_sweep/validated_snapshots.h5`: archivo de auditoría con
  snapshots finitos, incluidos `WARNING` y `FAIL` cuando existe solución;
- `outputs/metrics/fom_parameter_sweep.csv`: todas las muestras y motivos de estado,
  incluidas excepciones sin solución;
- `outputs/metrics/fom_mesh_refinement.csv`: resultados por parámetro y malla;
- `outputs/parameter_sweep/summary.json`: resumen y configuración;
- `outputs/parameter_sweep/figures/`: diagnósticos, mapas de adjuntos, curvas
  representativas, comparación de estados finales con el target y refinamiento.

`representative_adjoints.png` representa `P_T=(p1,p2)`.
`representative_final_states_vs_target.png` representa los estados finales bajo
control junto a sus targets guardados. En esta campaña el target es común:
`y^T(x)=sin(pi*x)` y `v^T(x)=0`, dibujado con línea negra discontinua.
Para regenerar las figuras usando los snapshots existentes, sin repetir solves
ni modificar el dataset o sus splits:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/plot_parameter_sweep.py
```

Los HDF5 almacenan `/metadata`, `/samples` y `/splits`. Cada muestra incluye
`sample_id`, `mu`, `status`, los campos físicos (también `wave_speed_half`),
`final_adjoint` y `final_state`. No se almacenan matrices FOM por muestra.
La metadata contiene fecha, configuración, fórmulas, cotas físicas, versión,
commit Git cuando existe y hashes de código/configuración.

Para auditar la persistencia sin repetir los solves:

```bash
.venv/bin/python scripts/validate_parameter_sweep.py
```

Para construir otro dataset a partir del archivo ya validado:

```bash
.venv/bin/python scripts/build_dataset.py --output data/datasets/wave_pdocp_fom_copy.h5
```

La primera campaña produjo **100 PASS, 0 WARNING, 0 FAIL** y splits de
**75/15/10**. El error terminal relativo quedó entre **1.065 % y 1.133 %**;
las diferencias entre mallas decrecieron en los ocho casos estudiados.
Consultar [`docs/PARAMETER_SWEEP_RESULTS.md`](docs/PARAMETER_SWEEP_RESULTS.md)
para resultados y criterios de interpretación.
