# Próximo bloque de desarrollo — Barrido paramétrico, validación FOM y construcción del dataset

## Propósito

Este documento define las tareas que deben implementarse a continuación en el proyecto **Wave-PDOCP**.

El objetivo de este bloque **NO** es entrenar todavía una red neuronal.

El objetivo es construir una campaña paramétrica pequeña y controlada que permita:

1. definir rigurosamente el espacio paramétrico \(\mathcal P\) y el mapeo \(\mu \mapsto\) campos físicos;
2. resolver el FOM para aproximadamente \(50\)-\(100\) valores de \(\mu\);
3. validar que cada solve es algebraica, física y numéricamente consistente;
4. verificar que la familia \(\mu\mapsto P_{\mu,h}^T\) se comporta regularmente;
5. realizar un estudio de refinamiento espacial sobre unos pocos parámetros representativos;
6. guardar los campos de entrada, soluciones FOM y diagnósticos;
7. separar el dataset en proporción \(75/15/10\) para `train/validation/test`.

Este bloque debe quedar terminado antes de implementar U-Net, FNO u otro surrogate.

---

# 1. Ecuación FOM que se debe preservar

Para cada parámetro \(\mu\), el solver semidiscreto resuelve

\[
\boxed{
\left(I+\beta W_{\mu,h}(T)\right)P_{\mu,h}^T
=
\beta
\left(
e^{TA_{\mu,h}}Y_{\mu,h}^0
-
Y_{\mu,h}^T
\right).
}
\]

Definir

\[
K_{\mu,h}
:=
I+\beta W_{\mu,h}(T)
\]

y

\[
b_{\mu,h}
:=
\beta
\left(
e^{TA_{\mu,h}}Y_{\mu,h}^0-Y_{\mu,h}^T
\right).
\]

Entonces:

\[
\boxed{
K_{\mu,h}P_{\mu,h}^T=b_{\mu,h}.
}
\]

El tiempo sigue siendo continuo en la formulación del FOM.

**No introducir una discretización temporal estructural en este bloque.**

---

# 2. Primer objetivo: formalizar el espacio paramétrico

Antes de generar datos se debe definir explícitamente:

\[
\boxed{
\mathcal P\subset\mathbb R^p.
}
\]

Para la primera campaña se recomienda comenzar con un único parámetro escalar:

\[
\boxed{
\mu\in[\mu_{\min},\mu_{\max}].
}
\]

La primera familia debe ser sencilla y estable.

No variar simultáneamente todos los ingredientes del problema.

En la primera campaña, variar **una sola característica física** y mantener las demás fijas.

Ejemplos posibles:

- amplitud de una perturbación suave en \(c_\mu(x)\);
- posición de un target suave;
- amplitud del damping;
- otro parámetro escalar previamente acordado.

La implementación no debe asumir que `mu` siempre será escalar: la arquitectura debe quedar preparada para

\[
\mu\in\mathbb R^p.
\]

---

# 3. Condiciones matemáticas sobre el mapeo paramétrico

El proyecto debe trabajar con un mapeo conceptual

\[
\boxed{
\mu
\longmapsto
\left(
c_\mu,
d_\mu,
\chi_{\omega_\mu},
Y_\mu^0,
Y_\mu^T
\right).
}
\]

Para la primera campaña se buscará una dependencia continua y, cuando sea posible, Lipschitz respecto de \(\mu\).

Idealmente:

\[
\|c_\mu-c_\nu\|_\infty
\le
L_c\|\mu-\nu\|,
\]

\[
\|d_\mu-d_\nu\|_\infty
\le
L_d\|\mu-\nu\|,
\]

\[
\|Y_\mu^0-Y_\nu^0\|_{H}
\le
L_0\|\mu-\nu\|,
\]

\[
\|Y_\mu^T-Y_\nu^T\|_{H}
\le
L_T\|\mu-\nu\|.
\]

Además, imponer cotas físicas uniformes:

\[
\boxed{
0<c_{\min}
\le
c_\mu(x)
\le
c_{\max}<\infty
}
\]

y

\[
\boxed{
0\le d_\mu(x)\le d_{\max}.
}
\]

Para esta primera campaña se recomienda mantener fija la región de control:

\[
\boxed{
\omega_\mu=\omega.
}
\]

Esto evita introducir desde el comienzo discontinuidades paramétricas asociadas al movimiento de una función característica.

---

# 4. Implementar una abstracción para el espacio paramétrico

Crear, por ejemplo:

```text
src/wave_pdocp/domain/
    parameter_space.py
```

Responsabilidades:

- definir límites de \(\mathcal P\);
- validar que un parámetro pertenece a \(\mathcal P\);
- generar parámetros uniformemente espaciados;
- permitir en el futuro muestreo aleatorio o multidimensional;
- no construir directamente matrices FOM.

Interfaz sugerida:

```python
class ParameterSpace:
    def contains(self, mu) -> bool:
        ...

    def sample_uniform_grid(self, n_samples: int):
        ...
```

Para una dimensión:

```python
mu_values = np.linspace(mu_min, mu_max, n_samples)
```

---

# 5. Mantener separada la parametrización física

`ParameterSpace` define dónde vive \(\mu\).

`parameterization.py` define cómo \(\mu\) genera los campos físicos.

Flujo:

```text
mu
 ↓
Parameterization
 ↓
c_mu
d_mu
control_mask
Y0_mu
YT_mu
```

No poner fórmulas paramétricas dentro de `FOMSolver`.

El solver solo debe recibir los campos/operadores resultantes.

---

# 6. Soporte correcto para \(c_\mu(x)\) variable

Para el operador

\[
\partial_x
\left(
c_\mu^2(x)y_x
\right)
\]

utilizar forma conservativa.

Sea

\[
a_\mu(x)=c_\mu^2(x).
\]

En los nodos interiores:

\[
\partial_x(a_\mu y_x)(x_i)
\approx
\frac{
a_{\mu,i+\frac12}(y_{i+1}-y_i)
-
a_{\mu,i-\frac12}(y_i-y_{i-1})
}{h^2}.
\]

Los valores

\[
a_{\mu,i+\frac12}
\]

deben obtenerse evaluando el campo en semipuntos o mediante una regla de promedio bien definida.

No usar una extensión artificial del coeficiente a \(N+2\) nodos como mecanismo principal.

---

# 7. Crear un experimento de barrido paramétrico

Crear:

```text
src/wave_pdocp/orchestration/
    parameter_sweep.py
```

o ampliar el orquestador existente con una clase dedicada:

```python
class ParameterSweepExperiment:
    ...
```

Entrada:

- `ParameterSpace`;
- `FOMSolver`;
- parametrización;
- número de muestras;
- configuración de validación;
- directorio de salida.

Salida:

- soluciones FOM;
- diagnósticos;
- tabla resumen;
- dataset validado.

Flujo:

```text
generate mu values
    ↓
for each mu:
    evaluate parameter fields
    ↓
    solve FOM
    ↓
    reconstruct final controlled state/control if required
    ↓
    compute diagnostics
    ↓
    classify PASS / WARNING / FAIL
    ↓
aggregate diagnostics
    ↓
global parametric validation
```

---

# 8. Número inicial de parámetros

Usar inicialmente:

```text
50 <= n_parameters <= 100
```

Valor recomendado para la primera ejecución completa:

```text
n_parameters = 100
```

Para \(\mu\in[\mu_{\min},\mu_{\max}]\):

\[
\mu_j
=
\mu_{\min}
+
\frac{j}{M-1}
(\mu_{\max}-\mu_{\min}),
\qquad
j=0,\dots,M-1.
\]

Usar muestreo uniformemente espaciado en la primera campaña.

---

# 9. Validaciones individuales para cada FOM

Para cada \(\mu_j\), calcular obligatoriamente los siguientes diagnósticos.

## 9.1 Residual algebraico

\[
r_{\mu,h}
=
K_{\mu,h}P_{\mu,h}^T
-
b_{\mu,h}.
\]

Norma absoluta:

\[
\boxed{
r_{\mathrm{abs}}
=
\|r_{\mu,h}\|_{H_h}
=
\sqrt{
r_{\mu,h}^TQ_hr_{\mu,h}
}.
}
\]

Residual relativo:

\[
\boxed{
r_{\mathrm{rel}}
=
\frac{
\|r_{\mu,h}\|_{H_h}
}{
\|b_{\mu,h}\|_{H_h}+\varepsilon_{\mathrm{safe}}
}.
}
\]

Criterio inicial recomendado:

```text
r_rel < 1e-9
```

No ocultar warnings numéricos.

---

# 10. Validación de la condición terminal del adjunto

Debe cumplirse:

\[
\boxed{
P_{\mu,h}^T
=
\beta
\left(
Y_{\mu,h}^*(T)-Y_{\mu,h}^T
\right).
}
\]

Definir:

\[
e_{\mathrm{opt}}
=
\left\|
P_{\mu,h}^T
-
\beta
\left(
Y_{\mu,h}^*(T)-Y_{\mu,h}^T
\right)
\right\|_{H_h}.
\]

Guardar:

```text
optimality_error
optimality_relative_error
```

Debe ser cercano a precisión numérica si la reconstrucción es consistente.

---

# 11. Validación del Gramiano

Para cada \(\mu\):

## 11.1 Autoadjunción ponderada

Comprobar:

\[
\boxed{
Q_hW_{\mu,h}
\approx
W_{\mu,h}^TQ_h.
}
\]

Definir error relativo:

\[
e_{\mathrm{sym}}
=
\frac{
\|Q_hW_{\mu,h}-W_{\mu,h}^TQ_h\|_F
}{
\|Q_hW_{\mu,h}\|_F+\varepsilon_{\mathrm{safe}}
}.
\]

Guardar:

```text
gramian_weighted_symmetry_error
```

---

# 12. Positividad del Gramiano

Transformar el operador a representación euclídea usando \(Q_h=C^TC\):

\[
\widetilde W
=
CW_{\mu,h}C^{-1}.
\]

Analizar:

\[
\lambda_{\min}(\widetilde W),
\qquad
\lambda_{\max}(\widetilde W).
\]

Se espera:

\[
\boxed{
\lambda_{\min}\ge0
}
\]

salvo pequeños errores negativos de redondeo.

No clasificar como `FAIL` un valor del orden de

```text
-1e-13
```

sin comparar primero con la escala espectral.

---

# 13. Condicionamiento

Guardar para cada parámetro:

\[
\boxed{
\kappa(W_{\mu,h})
}
\]

y, más importante,

\[
\boxed{
\kappa(K_{\mu,h})
=
\kappa(I+\beta W_{\mu,h}).
}
\]

La segunda cantidad corresponde a la matriz realmente resuelta.

No imponer inicialmente un umbral demasiado agresivo.

Un Gramiano mal condicionado puede ser una característica real del problema y no necesariamente un bug.

---

# 14. Estabilidad espectral de \(A_{\mu,h}\)

Guardar:

\[
\boxed{
\alpha_\mu
=
\max_j
\operatorname{Re}
\lambda_j(A_{\mu,h}).
}
\]

Interpretar según la parametrización.

Si

\[
d_\mu(x)>0,
\]

se espera comportamiento disipativo y partes reales no positivas.

Si

\[
d_\mu=0,
\]

se espera aproximadamente:

\[
\operatorname{Re}
\lambda_j(A_{\mu,h})
\approx0.
\]

---

# 15. Error terminal

Guardar:

\[
\boxed{
e_T(\mu)
=
\|Y_{\mu,h}^*(T)-Y_{\mu,h}^T\|_{H_h}.
}
\]

Y también:

\[
\boxed{
e_{T,\mathrm{rel}}(\mu)
=
\frac{
\|Y_{\mu,h}^*(T)-Y_{\mu,h}^T\|_{H_h}
}{
\|Y_{\mu,h}^T\|_{H_h}+\varepsilon_{\mathrm{safe}}
}.
}
\]

Separar además si es útil:

```text
max_displacement_error
max_velocity_error
```

---

# 16. Energía del control

Guardar:

\[
\boxed{
E_u(\mu)
=
\int_0^T
\|u_\mu^*(t)\|_U^2dt.
}
\]

Esta cantidad debe ser finita.

Si la reconstrucción temporal se realiza solo para validación, debe mantenerse separada de la formulación semidiscreta del FOM.

---

# 17. Verificación del funcional óptimo

Calcular:

\[
J_\mu(u_\mu^*)
\]

y comparar con el control nulo:

\[
J_\mu(0).
\]

Como \(u=0\) es admisible:

\[
\boxed{
J_\mu(u_\mu^*)\le J_\mu(0).
}
\]

Guardar:

```text
optimal_cost
zero_control_cost
cost_reduction
```

Definir:

\[
\text{cost reduction}
=
1-
\frac{
J_\mu(u_\mu^*)
}{
J_\mu(0)
}.
\]

Si

\[
J_\mu(u_\mu^*)>J_\mu(0)
\]

más allá de tolerancia numérica, marcar el caso como `FAIL`.

---

# 18. Verificación de números finitos

Para cada solve comprobar:

```python
np.all(np.isfinite(...))
```

en al menos:

- `A`;
- `B`;
- `A_adj`;
- `B_adj`;
- `W`;
- `S_T`;
- `rhs`;
- `P_T`;
- estado final reconstruido;
- control reconstruido si se calcula.

Cualquier `NaN` o `Inf` debe ser `FAIL`.

---

# 19. Clasificación PASS / WARNING / FAIL

Crear una clasificación automática por muestra.

## PASS

Caso algebraica y numéricamente consistente.

Ejemplo de condiciones:

```text
all arrays finite
relative_residual < 1e-9
optimality_relative_error < tolerance
Gramian weighted symmetry error < tolerance
J_opt <= J_zero + tolerance
```

## WARNING

El solve es consistente pero aparece alguna característica extrema, por ejemplo:

- Gramiano muy condicionado;
- sistema \(I+\beta W\) con condición elevada;
- energía de control muy grande;
- error terminal elevado;
- variación paramétrica local elevada;
- autovalor muy cercano a cero;
- runtime anormal.

Los `WARNING` NO deben eliminarse automáticamente.

Pueden ser los casos físicamente más interesantes.

## FAIL

Ejemplos:

- `NaN` / `Inf`;
- solver lineal falla;
- residual relativo grande;
- identidad terminal del adjunto falla fuertemente;
- Gramiano presenta una inconsistencia algebraica grande;
- `J_opt > J_zero` de forma significativa;
- excepciones no recuperables.

---

# 20. Validación global del mapa \(\mu\mapsto P_{\mu,h}^T\)

Una vez resueltos todos los parámetros, ordenar:

\[
\mu_1<\mu_2<\cdots<\mu_M.
\]

Calcular para parámetros consecutivos:

\[
\boxed{
D_j
=
\frac{
\|P_{\mu_{j+1},h}^T-P_{\mu_j,h}^T\|_{H_h}
}{
\|\mu_{j+1}-\mu_j\|
}.
}
\]

Interpretación:

`D_j` es una estimación discreta de la sensibilidad local del mapa parámetro-solución.

Guardar:

```text
local_parameter_lipschitz_indicator
```

No imponer que `D_j` sea constante.

Sí detectar outliers extremos.

---

# 21. Cambio relativo entre snapshots consecutivos

Calcular:

\[
\boxed{
\delta_j
=
\frac{
\|P_{\mu_{j+1},h}^T-P_{\mu_j,h}^T\|_{H_h}
}{
\|P_{\mu_j,h}^T\|_{H_h}+\varepsilon_{\mathrm{safe}}
}.
}
\]

Guardar:

```text
relative_snapshot_change
```

Esta métrica ayuda a detectar saltos inesperados en la familia paramétrica.

---

# 22. Métricas globales a graficar contra \(\mu\)

Generar gráficas:

\[
\mu
\mapsto
\|P_{\mu,h}^T\|_{H_h},
\]

\[
\mu
\mapsto
e_T(\mu),
\]

\[
\mu
\mapsto
E_u(\mu),
\]

\[
\mu
\mapsto
J_\mu(u_\mu^*),
\]

\[
\mu
\mapsto
\kappa(W_{\mu,h}),
\]

\[
\mu
\mapsto
\kappa(I+\beta W_{\mu,h}),
\]

\[
\mu
\mapsto
\lambda_{\min}(W_{\mu,h}),
\]

\[
\mu
\mapsto
\max\operatorname{Re}\lambda(A_{\mu,h}),
\]

\[
\mu_j
\mapsto
D_j,
\]

\[
\mu_j
\mapsto
\delta_j.
\]

No crear automáticamente una figura individual completa para cada uno de los 100 parámetros.

Usar resúmenes globales.

---

# 23. Visualización de la variedad de adjuntos

Separar:

\[
P_{\mu,h}^T
=
\begin{pmatrix}
p_{1,\mu,h}^T\\
p_{2,\mu,h}^T
\end{pmatrix}.
\]

Construir dos mapas 2D:

\[
\boxed{
(\mu,x)
\mapsto
p_{1,\mu,h}^T(x)
}
\]

y

\[
\boxed{
(\mu,x)
\mapsto
p_{2,\mu,h}^T(x).
}
\]

Esto permitirá inspeccionar visualmente la estructura de la variedad paramétrica antes del entrenamiento ML.

También generar curvas individuales para unos pocos parámetros representativos.

---

# 24. Selección de parámetros representativos

Después del barrido global seleccionar aproximadamente \(5\)-\(10\) parámetros.

Incluir:

1. \(\mu_{\min}\);
2. \(\mu_{\max}\);
3. un parámetro cercano al centro;
4. parámetro con mayor `system_condition_number`;
5. parámetro con mayor `terminal_error`;
6. parámetro con mayor `control_energy`;
7. parámetro asociado al mayor `D_j`;
8. otros casos `WARNING` relevantes.

Estos serán utilizados para el estudio de refinamiento espacial.

---

# 25. Estudio de refinamiento espacial

No repetir el estudio de malla para los 100 parámetros.

Solo para los parámetros representativos.

Usar inicialmente:

\[
N\in\{50,100,200\}.
\]

Opcionalmente:

\[
N=25
\]

si se quiere observar mejor la tasa.

Comparar:

\[
P_{\mu,h}^T.
\]

Como las soluciones viven en mallas diferentes, deben llevarse a una malla común antes de comparar.

Definir, por ejemplo:

\[
\boxed{
E_N(\mu)
=
\frac{
\|P_{\mu,2N}^T-\mathcal I(P_{\mu,N}^T)\|
}{
\|P_{\mu,2N}^T\|
},
}
\]

donde \(\mathcal I\) interpola/proyecta la solución gruesa a la malla fina.

Verificar que los errores decrezcan al refinar.

---

# 26. Diferencia entre validación algebraica y convergencia espacial

Mantener explícitamente esta distinción:

\[
\boxed{
\text{residual pequeño}
\Rightarrow
\text{el sistema discreto fue resuelto correctamente}
}
\]

pero

\[
\boxed{
\text{convergencia con }h
\Rightarrow
\text{la semidiscretización se estabiliza hacia la solución continua}.
}
\]

Un residual de \(10^{-14}\) no prueba por sí solo convergencia de la PDE.

---

# 27. Datos que deben almacenarse por parámetro

Para cada \(\mu_j\), guardar al menos:

## Parámetro

```text
mu
sample_id
```

## Campos físicos

```text
wave_speed
damping
control_mask
initial_displacement
initial_velocity
target_displacement
target_velocity
```

## Ground truth

```text
final_adjoint
```

con

\[
P_{\mu,h}^T\in\mathbb R^{2N}.
\]

Opcionalmente guardar también:

```text
final_state
control_energy
optimal_cost
```

No guardar matrices enormes como `W` para todas las muestras salvo que exista una razón clara.

El Gramiano puede ser costoso en almacenamiento.

Guardar principalmente sus diagnósticos escalares.

---

# 28. Formato del dataset

Usar preferiblemente HDF5:

```text
data/datasets/wave_pdocp_fom.h5
```

Estructura sugerida:

```text
/metadata
    grid_size
    domain_length
    final_time
    beta
    parameter_space
    parameterization_name

/samples
    mu
    wave_speed
    damping
    control_mask
    initial_displacement
    initial_velocity
    target_displacement
    target_velocity
    final_adjoint

/splits
    train_indices
    validation_indices
    test_indices
```

---

# 29. Tabla de diagnósticos

Crear además:

```text
outputs/metrics/fom_parameter_sweep.csv
```

Columnas sugeridas:

```text
sample_id
mu
status
runtime_seconds
absolute_residual
relative_residual
optimality_error
optimality_relative_error
terminal_error
relative_terminal_error
max_displacement_error
max_velocity_error
control_energy
optimal_cost
zero_control_cost
cost_reduction
gramian_min_eigenvalue
gramian_max_eigenvalue
gramian_condition_number
system_condition_number
gramian_weighted_symmetry_error
state_spectral_abscissa
adjoint_norm
local_parameter_lipschitz_indicator
relative_snapshot_change
```

---

# 30. Guardar metadata reproducible

Guardar junto al dataset:

- fecha de creación;
- commit Git si es posible;
- configuración utilizada;
- \(N\);
- \(L\);
- \(T\);
- \(\beta\);
- fórmula/identificador de parametrización;
- rango de \(\mu\);
- número de muestras;
- tolerancias de validación;
- versión del dataset.

---

# 31. Separación train / validation / test

La división se hace **después** de validar los FOM.

Solo muestras `PASS` deben entrar automáticamente al dataset de entrenamiento.

Los `WARNING` se revisan antes de decidir.

Los `FAIL` no entran al dataset hasta que se determine la causa.

Proporciones requeridas:

\[
\boxed{
75\% \text{ train},
\qquad
15\% \text{ validation},
\qquad
10\% \text{ test}.
}
\]

Para \(M=100\):

```text
75 train
15 validation
10 test
```

Usar un `random_seed` fijo y guardar los índices.

Ejemplo:

```python
seed = 42
```

No regenerar silenciosamente los splits en cada ejecución.

---

# 32. Consideración sobre el split

Para el primer benchmark, un split aleatorio reproducible de los parámetros validados es aceptable.

Sin embargo, dejar la implementación preparada para futuros esquemas:

```text
random
blocked
interpolation
extrapolation
```

Un split aleatorio evalúa principalmente interpolación dentro de \(\mathcal P\).

No afirmar extrapolación si el test permanece dentro del rango observado durante entrenamiento.

---

# 33. Crear un objeto de resultado del barrido

Sugerencia:

```python
@dataclass(frozen=True)
class ParameterSweepResult:
    parameters: np.ndarray
    statuses: list[str]
    diagnostics: pd.DataFrame
    dataset_path: Path
    metrics_path: Path
```

No devolver listas desestructuradas de decenas de objetos sin metadata.

---

# 34. Organización de archivos sugerida

Añadir o completar:

```text
src/wave_pdocp/
│
├── domain/
│   ├── parameter_space.py
│   └── parameterization.py
│
├── diagnostics/
│   ├── fom_validation.py
│   ├── parametric.py
│   └── spectral.py
│
├── orchestration/
│   └── parameter_sweep.py
│
├── data/
│   ├── generator.py
│   ├── splitting.py
│   └── schema.py
│
├── io/
│   ├── dataset_io.py
│   └── metrics_io.py
│
└── visualization/
    ├── parametric_plots.py
    └── fom_plots.py
```

Scripts:

```text
scripts/
├── run_parameter_sweep.py
├── validate_parameter_sweep.py
└── build_dataset.py
```

Configuraciones:

```text
configs/
├── single_mu.yaml
└── parameter_sweep.yaml
```

---

# 35. Configuración YAML sugerida

Ejemplo conceptual:

```yaml
experiment:
  name: first_parameter_sweep
  seed: 42

domain:
  length: 1.0
  final_time: 2.0

discretization:
  n_interior: 100

optimal_control:
  beta: 100.0

parameter_space:
  dimension: 1
  min: 0.0
  max: 1.0
  n_samples: 100
  sampling: uniform_grid

validation:
  relative_residual_tolerance: 1.0e-9
  weighted_symmetry_tolerance: 1.0e-9
  optimality_tolerance: 1.0e-9

mesh_refinement:
  enabled: true
  sizes: [50, 100, 200]
  representative_samples: 8

dataset:
  format: hdf5
  train_fraction: 0.75
  validation_fraction: 0.15
  test_fraction: 0.10
```

Los valores específicos de `parameter_space.min/max` deben definirse según la parametrización física escogida.

---

# 36. Tests unitarios nuevos

Crear tests para:

## Parameter space

- extremos incluidos;
- número correcto de muestras;
- orden correcto;
- validación de parámetros fuera de rango.

## Parametrización

- dimensiones correctas;
- todos los campos finitos;
- \(c_\mu(x)\ge c_{\min}\);
- \(d_\mu(x)\ge0\);
- variación coherente con \(\mu\).

## Validación FOM

- residual pequeño;
- condición terminal del adjunto;
- Gramiano ponderadamente autoadjunto;
- costo óptimo no mayor al costo sin control.

## Persistencia

- escribir dataset;
- volverlo a leer;
- comprobar igualdad de shapes, parámetros y snapshots.

## Split

- conjuntos disjuntos;
- unión igual al conjunto completo;
- proporciones correctas;
- reproducibilidad con la misma semilla.

---

# 37. Test de integración principal

Crear un test pequeño y barato, por ejemplo con:

```text
N = 20
n_parameters = 5
```

El test debe:

1. generar los parámetros;
2. resolver los 5 FOM;
3. validar cada solución;
4. guardar un dataset temporal;
5. recargarlo;
6. comprobar dimensiones;
7. generar el split;
8. comprobar que no hay solapamientos.

No usar \(100\) FOM en CI/tests unitarios.

---

# 38. Logging

Cada solve debe registrar como mínimo:

```text
sample i / M
mu
runtime
status
relative residual
terminal error
system condition number
```

Ejemplo:

```text
[023/100] mu=0.3141 | PASS | r_rel=2.1e-13 | kappa=3.8 | time=0.43s
```

No imprimir matrices completas durante el barrido.

---

# 39. Resumen final de la campaña

Al terminar, imprimir y guardar:

```text
number of samples
PASS count
WARNING count
FAIL count

max relative residual
median relative residual

max optimality error
max terminal error

min / max Gramian eigenvalue
max Gramian condition number
max system condition number

max local parameter sensitivity
max control energy

total runtime
mean FOM runtime
```

---

# 40. Criterios de aceptación de este bloque

Completado el 23 de septiembre de 2026 para la familia de amplitud de `c_mu`.
Ver `docs/PARAMETER_SWEEP_RESULTS.md` para la ejecución, resultados y límites.

Este bloque se considera terminado cuando:

- [x] \(\mathcal P\) está definido explícitamente;
- [x] existe una parametrización física clara \(\mu\mapsto\) campos;
- [x] se pueden generar \(50\)-\(100\) parámetros automáticamente;
- [x] todos los FOM se ejecutan desde un orquestador;
- [x] cada solve tiene diagnóstico individual;
- [x] existen estados `PASS/WARNING/FAIL`;
- [x] no aparecen `NaN`/`Inf` en campos y soluciones;
- [x] los residuos algebraicos de los casos `PASS` son pequeños;
- [x] se verifica la condición terminal del adjunto;
- [x] se verifica la autoadjunción ponderada del Gramiano;
- [x] se monitorea positividad y condicionamiento;
- [x] se compara \(J(u^*)\) contra \(J(0)\);
- [x] se estudia \(\mu\mapsto P_{\mu,h}^T\);
- [x] se generan gráficos paramétricos globales;
- [x] se hace refinamiento de malla para \(5\)-\(10\) parámetros;
- [x] se guardan campos y \(P_{\mu,h}^T\) en HDF5;
- [x] se guarda una tabla CSV de diagnósticos;
- [x] el split \(75/15/10\) es reproducible;
- [x] existen tests unitarios e integración;
- [x] toda la campaña puede reproducirse desde un YAML y un script.

---

# 41. Lo que NO se debe hacer todavía

No implementar todavía:

- U-Net;
- FNO;
- DeepONet;
- autoencoders;
- entrenamiento PyTorch;
- tuning de hiperparámetros;
- comparación entre redes;
- certificación del surrogate.

La prioridad actual es:

\[
\boxed{
\text{FOM confiable}
\rightarrow
\text{familia paramétrica validada}
\rightarrow
\text{dataset confiable}.
}
\]

Solo después:

\[
\boxed{
\text{dataset}
\rightarrow
\text{surrogate neuronal}.
}
\]

---

# 42. Orden recomendado de implementación

Seguir este orden:

```text
1. ParameterSpace
2. Parametrización física
3. ParameterSweepExperiment
4. Diagnósticos por FOM
5. Clasificación PASS/WARNING/FAIL
6. Diagnósticos globales en mu
7. Visualizaciones paramétricas
8. Selección de casos representativos
9. Refinamiento espacial
10. Persistencia HDF5 + CSV
11. Split 75/15/10
12. Tests completos
```

No empezar por la persistencia o el split antes de validar correctamente las soluciones.

---

# 43. Principio central

El dataset que luego usará la red debe considerarse **ground truth numérico**.

Por tanto, no basta con generar:

\[
\{(\mu_j,P_{\mu_j,h}^T)\}_{j=1}^M.
\]

Se debe generar:

\[
\boxed{
\left\{
\mu_j,
X_{\mu_j},
P_{\mu_j,h}^T,
\text{diagnósticos}_{\mu_j},
\text{status}_{\mu_j}
\right\}_{j=1}^M.
}
\]

La red neuronal solo debe entrenarse después de demostrar que estos snapshots son coherentes con el FOM y que la familia paramétrica está suficientemente bien comportada.
