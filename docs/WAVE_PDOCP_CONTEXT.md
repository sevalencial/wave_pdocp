# WAVE-PDOCP — Contexto matemático, numérico y de implementación

Este documento resume el contexto actual del proyecto de tesis **Wave-PDOCP** para que un agente de programación (por ejemplo, Codex dentro de VS Code) pueda continuar el desarrollo sin perder las decisiones matemáticas y de arquitectura ya fijadas.

Las actualizaciones de implementación están en las secciones 41 y 42; la campaña
paramétrica del 23 de septiembre de 2026 se resume en la sección 42.

---

# 1. Objetivo general de la tesis

El objetivo de la tesis es estudiar **problemas de control óptimo paramétrico para la ecuación de ondas** y construir modelos sustitutos no lineales, especialmente redes neuronales, capaces de aproximar eficientemente el **adjunto óptimo en tiempo final**.

La idea central no es aprender directamente el control \(u_\mu^*(t)\), sino aproximar el mapa

\[
\mu \longmapsto P_\mu^T,
\]

porque, una vez conocido el adjunto final \(P_\mu^T\), se pueden reconstruir:

- la trayectoria adjunta completa;
- el control óptimo;
- la trayectoria óptima del estado.

La ecuación central del proyecto es

\[
\boxed{
\left(I+\beta W_\mu(T)\right)P_\mu^T
=
\beta\left(S_\mu(T)Y_\mu^0-Y_\mu^T\right),
}
\]

donde:

- \(S_\mu(t)=e^{tA_\mu}\) es el semigrupo del sistema;
- \(W_\mu(T)\) es el Gramiano de controlabilidad;
- \(Y_\mu^0\) es el estado inicial;
- \(Y_\mu^T\) es el target;
- \(\beta>0\) es el parámetro de penalización terminal.

La tesis debe comparar modelos lineales como POD / Reduced Basis con aproximaciones no lineales como U-Net, y posteriormente incorporar certificación de error mediante residuos.

---

# 2. Modelo continuo

Se considera la ecuación de ondas paramétrica en una dimensión:

\[
\boxed{
\partial_{tt} y_\mu
-
\partial_x\left(c_\mu^2(x)\partial_x y_\mu\right)
+
d_\mu(x)\partial_t y_\mu
=
\chi_{\omega_\mu}(x)u_\mu(x,t),
}
\]

para

\[
(x,t)\in(0,L)\times(0,T).
\]

Condiciones de frontera de Dirichlet:

\[
y_\mu(0,t)=y_\mu(L,t)=0.
\]

Datos iniciales:

\[
y_\mu(x,0)=y_\mu^0(x),
\qquad
\partial_t y_\mu(x,0)=y_\mu^1(x).
\]

El parámetro \(\mu\) podrá afectar, entre otros:

- velocidad de propagación \(c_\mu(x)\);
- damping \(d_\mu(x)\);
- región de control \(\omega_\mu\);
- estado inicial;
- target final;
- posición, frecuencia y fase de targets oscilatorios.

---

# 3. Espacio de estado

Se define

\[
v_\mu=\partial_t y_\mu,
\]

y

\[
Y_\mu(t)
=
\begin{pmatrix}
y_\mu(t)\\
v_\mu(t)
\end{pmatrix}.
\]

El espacio de estado natural es

\[
\boxed{
H=H_0^1(0,L)\times L^2(0,L).
}
\]

La razón viene de la energía:

\[
E_\mu(t)
=
\frac12
\int_0^L
\left[
c_\mu^2(x)|y_x|^2+|y_t|^2
\right]dx.
\]

Para \(d_\mu\ge0\),

\[
\frac{dE_\mu}{dt}
=
-
\int_0^L d_\mu(x)|y_t|^2dx
\le0.
\]

Para mantener un único espacio de Hilbert independiente de \(\mu\), se trabaja con

\[
H_0^1\times L^2
\]

y no con una norma de energía dependiente de \(c_\mu\).

---

# 4. Formulación abstracta de primer orden

El sistema se escribe como

\[
\boxed{
\dot Y_\mu(t)
=
A_\mu Y_\mu(t)
+
B_\mu u_\mu(t).
}
\]

Formalmente,

\[
A_\mu
=
\begin{pmatrix}
0&I\\
\partial_x(c_\mu^2\partial_x\cdot)&-d_\mu
\end{pmatrix}.
\]

Para control distribuido interno,

\[
B_\mu u
=
\begin{pmatrix}
0\\
\chi_{\omega_\mu}u
\end{pmatrix}.
\]

---

# 5. Funcional de control óptimo

Se utiliza el funcional cuadrático

\[
\boxed{
J_\mu(u)
=
\frac12\int_0^T \|u(t)\|_U^2dt
+
\frac{\beta}{2}
\|Y_\mu(T)-Y_\mu^T\|_H^2.
}
\]

Interpretación:

1. El primer término penaliza el esfuerzo del control.
2. El segundo término penaliza no alcanzar el target.

Puede verse como una penalización del problema rígido

\[
Y_\mu(T)=Y_\mu^T.
\]

La elección cuadrática permite obtener un sistema de optimalidad lineal.

---

# 6. Solución mild

Para

\[
\dot Y=AY+Bu,
\qquad
Y(0)=Y^0,
\]

se tiene

\[
\boxed{
Y(t)
=
e^{tA}Y^0
+
\int_0^t e^{(t-s)A}Bu(s)\,ds.
}
\]

En tiempo final,

\[
\boxed{
Y(T)
=
e^{TA}Y^0
+
\int_0^T e^{(T-s)A}Bu(s)\,ds.
}
\]

---

# 7. Derivación del adjunto final y control óptimo

Tomando una perturbación \(u+\varepsilon h\), la variación del estado final es

\[
\delta Y(T)
=
\int_0^T e^{(T-s)A}Bh(s)\,ds.
\]

Se define el adjunto final como

\[
\boxed{
P^T
=
\beta\left(Y^*(T)-Y^T\right).
}
\]

La condición de optimalidad conduce a

\[
\boxed{
u^*(t)
=
-B^\dagger e^{(T-t)A^\dagger}P^T.
}
\]

La trayectoria adjunta es

\[
\boxed{
P(t)
=
e^{(T-t)A^\dagger}P^T.
}
\]

Equivalente a

\[
-\dot P=A^\dagger P.
\]

---

# 8. Gramiano de controlabilidad

Sustituyendo el control óptimo en la expresión del estado final:

\[
Y^*(T)
=
e^{TA}Y^0
-
W(T)P^T,
\]

donde

\[
\boxed{
W(T)
=
\int_0^T
e^{(T-s)A}
BB^\dagger
e^{(T-s)A^\dagger}
\,ds.
}
\]

Con el cambio de variable \(r=T-s\),

\[
\boxed{
W(T)
=
\int_0^T
e^{rA}
BB^\dagger
e^{rA^\dagger}
\,dr.
}
\]

El Gramiano también puede entenderse como

\[
W(T)
=
\mathcal C_T\mathcal C_T^\dagger,
\]

donde

\[
\mathcal C_Tu
=
\int_0^T e^{(T-s)A}Bu(s)\,ds.
\]

Por tanto \(W(T)\) es autoadjunto y positivo semidefinido en el producto interno correspondiente.

---

# 9. Ecuación final para el adjunto

Usando

\[
P^T=\beta(Y^*(T)-Y^T),
\]

y

\[
Y^*(T)=e^{TA}Y^0-W(T)P^T,
\]

se obtiene

\[
P^T
=
\beta
\left[
e^{TA}Y^0-W(T)P^T-Y^T
\right].
\]

Entonces

\[
\boxed{
(I+\beta W(T))P^T
=
\beta\left(e^{TA}Y^0-Y^T\right).
}
\]

IMPORTANTE:

El lado derecho correcto es

\[
\boxed{
\beta\left(e^{TA}Y^0-Y^T\right),
}
\]

NO

\[
\beta e^{TA}(Y^0-Y^T).
\]

El target \(Y^T\) ya vive en tiempo final y no debe propagarse con el semigrupo.

---

# 10. Discretización: solo espacio

Una decisión fundamental del proyecto:

\[
\boxed{
\text{NO se discretiza el tiempo en la formulación del FOM.}
}
\]

Se hace solamente una semidiscretización espacial mediante diferencias finitas.

El tiempo permanece continuo.

Por tanto:

\[
x,t
\longrightarrow
x_i,t.
\]

No se usan estructuralmente:

- \(t_n\);
- \(\Delta t\);
- Crank–Nicolson;
- matrices de evolución temporal por pasos.

El sistema semidiscreto sigue siendo una ODE matricial continua en tiempo.

---

# 11. Malla espacial

Se divide \([0,L]\) en \(N+1\) subintervalos interiores:

\[
h=\frac{L}{N+1}.
\]

Nodos interiores:

\[
x_i=ih,
\qquad i=1,\dots,N.
\]

Se define

\[
\mathbf y_\mu(t)
=
\begin{pmatrix}
y_1(t)\\
\vdots\\
y_N(t)
\end{pmatrix}
\in\mathbb R^N.
\]

La velocidad:

\[
\mathbf v_\mu(t)
=
\dot{\mathbf y}_\mu(t)
\in\mathbb R^N.
\]

El estado semidiscreto:

\[
\boxed{
Y_{\mu,h}(t)
=
\begin{pmatrix}
\mathbf y_\mu(t)\\
\mathbf v_\mu(t)
\end{pmatrix}
\in\mathbb R^{2N}.
}
\]

El control distribuido:

\[
\boxed{
u_{\mu,h}(t)\in\mathbb R^N.
}
\]

---

# 12. Discretización del operador espacial

Sea

\[
a_\mu(x)=c_\mu^2(x).
\]

La forma conservativa

\[
\partial_x(a_\mu y_x)
\]

se discretiza en el nodo interior \(x_i\) como

\[
\boxed{
\partial_x(a_\mu y_x)(x_i)
\approx
\frac1{h^2}
\left[
a_{\mu,i+\frac12}(y_{i+1}-y_i)
-
a_{\mu,i-\frac12}(y_i-y_{i-1})
\right].
}
\]

Se recomienda evaluar directamente

\[
a_{\mu,i+\frac12}
=
c_\mu^2(x_{i+\frac12}),
\]

en los semipuntos

\[
x_{i+\frac12}
=
\left(i+\frac12\right)h.
\]

Esto es preferible a extender artificialmente los valores nodales.

Para \(c_\mu(x)\equiv c\),

\[
L_{\mu,h}
=
\frac{c^2}{h^2}
\begin{pmatrix}
-2&1&&\\
1&-2&1&\\
&\ddots&\ddots&\ddots\\
&&1&-2
\end{pmatrix}.
\]

---

# 13. Damping y control discretos

Damping:

\[
\boxed{
D_{\mu,h}
=
\operatorname{diag}
\left(
d_\mu(x_1),\dots,d_\mu(x_N)
\right).
}
\]

Control interno:

\[
\boxed{
C_{\mu,h}
=
\operatorname{diag}
\left(
\chi_{\omega_\mu}(x_1),\dots,\chi_{\omega_\mu}(x_N)
\right).
}
\]

Se mantiene esta representación matricial diagonal del control.

---

# 14. Sistema semidiscreto

La ecuación semidiscreta de segundo orden es

\[
\boxed{
\ddot{\mathbf y}_\mu
=
L_{\mu,h}\mathbf y_\mu
-
D_{\mu,h}\dot{\mathbf y}_\mu
+
C_{\mu,h}u_\mu.
}
\]

En primer orden:

\[
\boxed{
\dot Y_{\mu,h}
=
A_{\mu,h}Y_{\mu,h}
+
B_{\mu,h}u_\mu.
}
\]

con

\[
\boxed{
A_{\mu,h}
=
\begin{pmatrix}
0&I_N\\
L_{\mu,h}&-D_{\mu,h}
\end{pmatrix}
\in\mathbb R^{2N\times2N},
}
\]

y

\[
\boxed{
B_{\mu,h}
=
\begin{pmatrix}
0\\
C_{\mu,h}
\end{pmatrix}
\in\mathbb R^{2N\times N}.
}
\]

---

# 15. Productos internos discretos

El producto interno continuo se aproxima por

\[
\left\langle
\begin{pmatrix}y\\v\end{pmatrix},
\begin{pmatrix}z\\w\end{pmatrix}
\right\rangle_H
=
\int_0^L y_xz_x\,dx
+
\int_0^L vw\,dx.
\]

La parte \(H_0^1\) produce

\[
K_h
=
\frac1h
\begin{pmatrix}
2&-1&&\\
-1&2&-1&\\
&\ddots&\ddots&\ddots\\
&&-1&2
\end{pmatrix}.
\]

La parte \(L^2\) se aproxima por

\[
hI_N.
\]

Por tanto

\[
\boxed{
Q_h
=
\begin{pmatrix}
K_h&0\\
0&hI_N
\end{pmatrix}
\in\mathbb R^{2N\times2N}.
}
\]

El producto interno del control es

\[
\boxed{
R_h=hI_N.
}
\]

Entonces:

\[
\langle Y,Z\rangle_{H_h}
=
Y^TQ_hZ,
\]

\[
\langle u,v\rangle_{U_h}
=
u^TR_hv.
\]

---

# 16. Adjuntos discretos ponderados

No se debe usar automáticamente la transpuesta.

El adjunto de \(A_{\mu,h}\) respecto a \(Q_h\) es

\[
\boxed{
A_{\mu,h}^\dagger
=
Q_h^{-1}
A_{\mu,h}^T
Q_h.
}
\]

El adjunto de \(B_{\mu,h}\) entre \(U_h\) y \(H_h\) es

\[
\boxed{
B_{\mu,h}^\dagger
=
R_h^{-1}
B_{\mu,h}^T
Q_h.
}
\]

En código se debe evitar formar inversas explícitas. Usar `scipy.linalg.solve`.

Para el caso de prueba

\[
c=1,\qquad d=0,\qquad C=I,
\]

se espera:

\[
\boxed{
A_h^\dagger=-A_h,
}
\]

y

\[
\boxed{
B_h^\dagger=
\begin{pmatrix}
0&I
\end{pmatrix}.
}
\]

Estas propiedades deben utilizarse como tests.

---

# 17. Semigrupo semidiscreto

Como \(A_{\mu,h}\) es una matriz finita:

\[
\boxed{
S_{\mu,h}(t)
=
e^{tA_{\mu,h}}.
}
\]

Se usa

```python
scipy.linalg.expm
```

para calcularlo.

---

# 18. Gramiano semidiscreto

El Gramiano semidiscreto es

\[
\boxed{
W_{\mu,h}(T)
=
\int_0^T
e^{tA_{\mu,h}}
B_{\mu,h}B_{\mu,h}^\dagger
e^{tA_{\mu,h}^\dagger}
\,dt.
}
\]

Tiene dimensión

\[
\boxed{
W_{\mu,h}(T)\in\mathbb R^{2N\times2N}.
}
\]

---

# 19. Ecuación diferencial de Lyapunov para el Gramiano

Definiendo

\[
W(t)
=
\int_0^t
e^{sA}
BB^\dagger
e^{sA^\dagger}
\,ds,
\]

se tiene

\[
\boxed{
\dot W(t)
=
AW(t)+W(t)A^\dagger+BB^\dagger,
\qquad
W(0)=0.
}
\]

Derivación:

\[
\dot W(t)
=
e^{tA}BB^\dagger e^{tA^\dagger}.
\]

Por otro lado,

\[
AW+WA^\dagger
=
\int_0^t
\frac{d}{ds}
\left(
e^{sA}BB^\dagger e^{sA^\dagger}
\right)ds,
\]

por tanto

\[
AW+WA^\dagger
=
e^{tA}BB^\dagger e^{tA^\dagger}
-
BB^\dagger.
\]

Así,

\[
\dot W=AW+WA^\dagger+BB^\dagger.
\]

---

# 20. Cálculo del Gramiano mediante exponencial de bloques

Para matrices de tamaño moderado se está usando una fórmula tipo Van Loan.

Definir

\[
G=BB^\dagger,
\]

y

\[
\mathcal M
=
\begin{pmatrix}
A&G\\
0&-A^\dagger
\end{pmatrix}.
\]

Si

\[
e^{T\mathcal M}
=
\begin{pmatrix}
F&H\\
0&J
\end{pmatrix},
\]

entonces

\[
F=e^{TA},
\]

y

\[
\boxed{
W(T)=H\,e^{TA^\dagger}.
}
\]

Esta estrategia evita discretizar el tiempo en la formulación estructural.

---

# 21. Sistema FOM que debe resolverse

Para cada \(\mu\):

\[
\boxed{
\left(I+\beta W_{\mu,h}(T)\right)
P_{\mu,h}^T
=
\beta
\left(
e^{TA_{\mu,h}}Y_{\mu,h}^0
-
Y_{\mu,h}^T
\right).
}
\]

El vector buscado tiene dimensión

\[
\boxed{
P_{\mu,h}^T\in\mathbb R^{2N}.
}
\]

Nunca calcular la inversa explícita.

Usar un solver lineal:

```python
scipy.linalg.solve
```

---

# 22. Residual del sistema

Definir

\[
b_{\mu,h}
=
\beta
\left(
e^{TA_{\mu,h}}Y_{\mu,h}^0-Y_{\mu,h}^T
\right).
\]

Entonces

\[
\boxed{
r_{\mu,h}
=
\left(I+\beta W_{\mu,h}\right)
P_{\mu,h}^T
-
b_{\mu,h}.
}
\]

Norma ponderada:

\[
\boxed{
\|r_{\mu,h}\|_{H_h}
=
\sqrt{
r_{\mu,h}^TQ_hr_{\mu,h}
}.
}
\]

También se recomienda calcular el residual relativo:

\[
\boxed{
\frac{
\|r_{\mu,h}\|_{H_h}
}{
\|b_{\mu,h}\|_{H_h}
}.
}
\]

En el primer experimento con \(N=50\) se obtuvo aproximadamente:

\[
\|r_{\mu,h}\|_{H_h}
\approx
2.53\times10^{-14},
\]

lo que indica que el sistema lineal discreto está siendo resuelto prácticamente a precisión de máquina.

Esto valida el solve algebraico, pero NO prueba todavía convergencia respecto al problema continuo.

---

# 23. Primer parámetro de prueba \(\mu_0\)

La etapa actual usa un solo parámetro extremadamente amigable para validar el FOM.

Se propone:

\[
L=1,
\]

\[
c_{\mu_0}(x)=1,
\]

\[
d_{\mu_0}(x)=0,
\]

control global:

\[
\omega_{\mu_0}=(0,1),
\]

por tanto

\[
C_{\mu_0,h}=I.
\]

Condiciones iniciales:

\[
y_{\mu_0}^0=0,
\qquad
y_{\mu_0}^1=0.
\]

Target:

\[
\boxed{
y_{\mu_0}^T(x)=\sin(\pi x),
}
\]

\[
v_{\mu_0}^T(x)=0.
\]

Parámetro de penalización inicial:

\[
\beta=1.
\]

Este caso se utiliza solo para validar el solver.

Un único \(\mu\) NO sirve para entrenar una red neuronal.

---

# 24. Parametrización general

Aunque se empieza con \(\mu_0\), el código debe quedar preparado desde el principio para múltiples parámetros.

El solver no debe tener valores físicos hardcodeados.

Debe existir una capa de parametrización:

```text
mu
  ↓
parameterization.evaluate(mu, grid)
  ↓
c_mu(x_i)
d_mu(x_i)
chi_omega_mu(x_i)
Y0_mu(x_i)
YT_mu(x_i)
```

Importante:

`evaluate_parameter` debe devolver siempre campos discretos, independientemente de si originalmente eran constantes o funciones.

No asumir:

```python
c = np.full(...)
```

como única posibilidad.

Para el caso general se debe permitir que

\[
c_\mu=c_\mu(x),
\qquad
d_\mu=d_\mu(x).
\]

Idealmente la parametrización debe producir tanto valores nodales como valores en semipuntos cuando sean necesarios para ensamblar \(L_{\mu,h}\).

---

# 25. Arquitectura de software

Se utiliza una arquitectura híbrida:

- OOP para objetos con estado/configuración/responsabilidad.
- Programación funcional para operaciones matemáticas.

Regla:

\[
\boxed{
\text{el orquestador decide qué hacer;}
}
\]

\[
\boxed{
\text{las clases representan quién lo hace;}
}
\]

\[
\boxed{
\text{las funciones implementan las matemáticas.}
}
\]

Estructura aproximada:

```text
wave_pdocp/
│
├── configs/
│
├── src/
│   └── wave_pdocp/
│       ├── domain/
│       ├── discretization/
│       ├── operators/
│       ├── fom/
│       ├── data/
│       ├── reduction/
│       ├── surrogates/
│       ├── certification/
│       ├── metrics/
│       ├── orchestration/
│       ├── io/
│       └── utils/
│
├── scripts/
├── tests/
├── notebooks/
├── data/
└── outputs/
```

---

# 26. Responsabilidad de carpetas

## `domain/`

Representa el problema físico:

- malla;
- parámetros;
- parametrización;
- estados iniciales;
- targets;
- regiones de control.

## `discretization/`

Contiene:

- diferencias finitas;
- productos internos discretos;
- matrices de stiffness / masa.

## `operators/`

Contiene:

- ensamblaje de \(A_{\mu,h}\);
- ensamblaje de \(B_{\mu,h}\);
- adjuntos ponderados;
- semigrupos;
- Gramianos.

## `fom/`

Contiene el Full Order Model:

- solver;
- objeto `FOMSolution`;
- reconstrucción posterior;
- validaciones.

## `orchestration/`

Coordina experimentos completos.

No debe implementar matemáticas.

## `io/`

Entrada/salida:

- guardar resultados;
- cargar YAML;
- serialización;
- logs;
- HDF5;
- checkpoints.

## `reduction/`

Métodos lineales:

- POD;
- SVD;
- Reduced Basis;
- análisis de valores singulares;
- errores de proyección.

## `certification/`

Certificación a posteriori:

\[
r_{\mu,h}(\widehat P)
=
(I+\beta W_{\mu,h})\widehat P-b_{\mu,h}.
\]

Más adelante incluirá:

- estimadores de error;
- cotas del adjunto;
- cotas del control;
- cotas del estado.

## `surrogates/`

Modelos ML:

- U-Net 1D;
- FNO;
- otros modelos;
- losses;
- training;
- inference.

---

# 27. Arquitectura neuronal propuesta

La primera arquitectura principal será una **U-Net 1D**.

Objetivo:

\[
\boxed{
X_\mu
\longmapsto
P_{\mu,h}^T.
}
\]

No se recomienda alimentar solo el vector \(\mu\).

Se recomienda alimentar directamente los campos físicos, por ejemplo:

\[
X_\mu(x_i)
=
\begin{pmatrix}
y_\mu^0(x_i)\\
y_\mu^1(x_i)\\
y_\mu^T(x_i)\\
v_\mu^T(x_i)\\
c_\mu(x_i)\\
d_\mu(x_i)\\
\chi_{\omega_\mu}(x_i)
\end{pmatrix}.
\]

Así:

\[
X_\mu\in\mathbb R^{C_{\text{in}}\times N}.
\]

La salida tiene dos canales:

\[
\boxed{
P_{\mu,h}^T
\in
\mathbb R^{2\times N}.
}
\]

Pero NO se debe comenzar todavía con la U-Net.

Primero debe validarse completamente el FOM.

---

# 28. Flujo completo previsto del proyecto

```text
mu
 ↓
parameterization
 ↓
physical fields
 ↓
spatial finite differences
 ↓
A_mu,h , B_mu,h
 ↓
weighted adjoints
 ↓
exp(T A_mu,h)
 ↓
W_mu,h(T)
 ↓
(I + beta W_mu,h) P_mu,h^T = b_mu,h
 ↓
P_mu,h^T
```

Después, offline:

```text
mu_1,...,mu_M
 ↓
FOM solves
 ↓
dataset {(X_mu_j, P_mu_j,h^T)}
 ↓
POD / SVD baseline
 ↓
U-Net 1D
 ↓
prediction P_hat
 ↓
residual certification
 ↓
control/state reconstruction
```

---

# 29. Diagnósticos que deben implementarse ahora

Antes de pasar a varios parámetros, añadir:

1. gráfica del target

\[
y_\mu^T(x);
\]

2. gráfica de la primera componente

\[
p_1^T(x);
\]

3. gráfica de la segunda componente

\[
p_2^T(x);
\]

4. espectro de

\[
A_{\mu,h};
\]

5. espectro del Gramiano

\[
W_{\mu,h};
\]

6. condicionamiento de

\[
W_{\mu,h};
\]

7. condicionamiento de

\[
I+\beta W_{\mu,h};
\]

8. residual absoluto;

9. residual relativo.

Para \(d=0\), se espera que los autovalores de \(A_h\) estén aproximadamente sobre el eje imaginario.

Para el Gramiano se espera positividad semidefinida en el producto interno discreto.

---

# 30. Espectro del Gramiano: cuidado con el producto interno

El Gramiano es autoadjunto respecto a

\[
\langle x,y\rangle_{H_h}=x^TQ_hy.
\]

Por tanto:

\[
Q_hW=W^TQ_h.
\]

No es necesario que \(W=W^T\) en coordenadas euclídeas.

Para analizar el espectro se puede transformar a una representación euclídea mediante Cholesky:

\[
Q_h=C^TC,
\]

y

\[
\boxed{
\widetilde W=CWC^{-1}.
}
\]

Entonces \(\widetilde W\) debe ser simétrica salvo errores de redondeo.

---

# 31. Validaciones/tests fundamentales

Tests mínimos:

## Dimensiones

\[
A_h:(2N,2N),
\]

\[
B_h:(2N,N),
\]

\[
W_h:(2N,2N),
\]

\[
P_h^T:(2N,).
\]

## Adjuntos

Para vectores aleatorios \(Y,Z\):

\[
\langle A_hY,Z\rangle_{H_h}
=
\langle Y,A_h^\dagger Z\rangle_{H_h}.
\]

Para \(u,Y\):

\[
\langle B_hu,Y\rangle_{H_h}
=
\langle u,B_h^\dagger Y\rangle_{U_h}.
\]

## Caso conservativo

Para \(c=1,d=0\):

\[
A_h^\dagger\approx -A_h.
\]

Para control global:

\[
B_h^\dagger
\approx
(0,I).
\]

## Residual

\[
\|(I+\beta W)P^T-b\|_{H_h}
\]

debe ser cercano a precisión de máquina.

---

# 32. Advertencia numérica importante

Las semidiscretizaciones espaciales estándar de la ecuación de ondas pueden producir **modos espurios de alta frecuencia** y pérdida de observabilidad uniforme al refinar la malla.

Por tanto, al aumentar \(N\), se debe monitorear:

- espectro de \(A_h\);
- espectro de \(W_h\);
- condicionamiento del Gramiano;
- oscilaciones de alta frecuencia;
- comportamiento al refinar \(h\).

No aplicar filtrado todavía salvo que los experimentos lo requieran, pero dejar esta posibilidad prevista.

---

# 33. Primer experimento paramétrico después de validar \(\mu_0\)

Una vez validado el FOM con \(\mu_0\), variar solo un parámetro.

Recomendación:

\[
x_\mu^T\in[0.2,0.8].
\]

Usar un target localizado suave que se desplace espacialmente.

Después añadir gradualmente:

\[
x_\mu^T
\rightarrow
(x_\mu^T,k_\mu)
\rightarrow
(x_\mu^T,k_\mu,\phi_\mu)
\rightarrow
d_\mu
\rightarrow
\omega_\mu
\rightarrow
c_\mu(x).
\]

No combinar todos los mecanismos al comienzo.

---

# 34. Dataset futuro

Para cada parámetro \(\mu_j\):

\[
\mu_j
\longrightarrow
X_{\mu_j}
\longrightarrow
P_{\mu_j,h}^T.
\]

El dataset deberá almacenar:

- parámetros;
- campos físicos;
- target;
- condiciones iniciales;
- máscara de control;
- \(c_\mu\);
- \(d_\mu\);
- adjunto final FOM.

Formato sugerido:

- HDF5 (`h5py`).

---

# 35. Stack Python recomendado

Dependencias base:

```text
numpy
scipy
matplotlib
pyyaml
pydantic
h5py
tqdm
pytest
```

Machine learning más adelante:

```text
torch
scikit-learn
pandas
tensorboard
```

Desarrollo:

```text
ruff
mypy
```

---

# 36. Principios que Codex debe respetar

1. No discretizar el tiempo salvo que se solicite explícitamente.

2. Mantener el modelo semidiscreto continuo en tiempo.

3. No sustituir los adjuntos ponderados por simples transpuestas sin justificarlo.

4. No calcular inversas explícitas cuando puede usarse `solve`.

5. Mantener separadas:
   - física;
   - discretización;
   - operadores;
   - solver;
   - visualización;
   - orquestación.

6. El `FOMSolver` no debe convertirse en una clase monolítica.

7. Las operaciones matemáticas deben ser funciones puras siempre que sea razonable.

8. El orquestador no debe contener fórmulas matemáticas.

9. Todo código nuevo debe soportar el caso actual de un único \(\mu_0\), pero quedar preparado para múltiples parámetros.

10. No hardcodear \(c=1\), \(d=0\), target seno, etc. dentro del solver.

11. `parameterization` es responsable de convertir \(\mu\) en campos discretos.

12. Para \(c_\mu(x)\) variable, construir el operador de difusión en forma conservativa usando valores en semipuntos.

13. Mantener tests algebraicos y dimensionales.

14. Añadir visualizaciones/diagnósticos fuera del solver.

15. Antes de implementar redes neuronales, completar:
    - validación del FOM;
    - refinamiento de malla;
    - diagnósticos espectrales;
    - primer barrido paramétrico;
    - POD/SVD baseline.

---

# 37. Estado actual del proyecto

Actualmente ya se ha implementado conceptualmente:

- malla espacial;
- parámetros;
- parametrización inicial;
- matriz de difusión;
- matriz de damping;
- matriz de control;
- \(A_{\mu,h}\);
- \(B_{\mu,h}\);
- \(Q_h\);
- \(R_h\);
- adjuntos ponderados;
- semigrupo;
- Gramiano;
- solve de \(P_{\mu,h}^T\);
- residual;
- primer experimento para un solo \(\mu\);
- tests básicos.

Se obtuvo correctamente para \(N=50\):

```text
A shape   = (100, 100)
B shape   = (100, 50)
W shape   = (100, 100)
P_T shape = (100,)
```

y un residual ponderado aproximadamente

```text
2.53e-14
```

lo cual indica una resolución algebraica muy precisa del sistema FOM.

---

# 38. Próximo bloque de trabajo recomendado

Implementar ahora:

```text
diagnostics/
    spectral.py

visualization/
    fom_plots.py
```

con:

- target;
- \(p_1^T\);
- \(p_2^T\);
- espectro de \(A_h\);
- espectro de \(W_h\);
- condición de \(W_h\);
- condición de \(I+\beta W_h\);
- residual relativo.

Después hacer estudio de malla:

\[
N=25,50,100,200,\dots
\]

y comparar la estabilidad/convergencia de \(P_{\mu,h}^T\).

Solo después iniciar el primer conjunto paramétrico.

---

# 39. Referencias conceptuales usadas en la tesis

El proyecto se basa principalmente en:

- `Wave_PDOCP_Proposal.pdf`
- `PDOCP_Kleikamp_Lazar_Munoz.pdf`
- literatura de controlabilidad y aproximación numérica de la ecuación de ondas;
- metodología de Gramianos de controlabilidad;
- modelos sustitutos no lineales, principalmente U-Net.

La definición central del Gramiano usada es

\[
W_\mu(T)
=
\int_0^T
S_\mu(T-s)
B_\mu B_\mu^*
S_\mu^*(T-s)
\,ds.
\]

---

# 40. Resumen de una línea

El proyecto busca construir un FOM semidiscreto en espacio para resolver

\[
\boxed{
(I+\beta W_{\mu,h}(T))P_{\mu,h}^T
=
\beta(e^{TA_{\mu,h}}Y_{\mu,h}^0-Y_{\mu,h}^T),
}
\]

generar con él una familia de adjuntos finales paramétricos y posteriormente aprender el operador

\[
\boxed{
\text{campos físicos paramétricos}
\longmapsto
P_{\mu,h}^T
}
\]

con una U-Net 1D, comparándola contra POD/RB y certificando sus predicciones mediante residuos.

---

# 41. Actualización de implementación — 20 de septiembre de 2026

Se corrigió el factor `beta` que faltaba en el lado derecho del FOM y el uso de
`target_amplitude` en lugar de `target_mode` dentro del seno del target. También
se corrigió la llamada incompleta a `build_damping_matrix` en los tests.

`WaveParameter.wave_speed` y `damping` aceptan constantes o funciones vectorizadas
de la posición. La parametrización produce `wave_speed` y `damping` nodales, y
`wave_speed_half` en los N+1 semipuntos de `SpatialGrid.half_points`. La firma de
`build_diffusion_matrix(wave_speed_half, grid)` ahora requiere valores en
semipuntos; no se deben pasar los N valores nodales. Se evalúa directamente
`c(x_half)**2`, incluyendo las caras adyacentes a las fronteras.

`FOMSolution` conserva malla, campos evaluados, tiempo final y beta para que la
reconstrucción use los mismos datos del solve. La reconstrucción está en
`fom/reconstruction.py` y los diagnósticos en `diagnostics/fom.py`, fuera del solver
y del orquestador. Se mantienen los adjuntos ponderados y el tiempo continuo.

Los dos scripts comparten un único caso con:

- `c(x) = 1 + 0.2 sin(2 pi x)`;
- `d(x) = 0.1 + 0.05 cos(2 pi x)`;
- `L=1`, `N=100`, `T=2`, `beta=100`;
- control global, estado inicial cero y target `(sin(pi x), 0)`.

Este caso variable complementa el caso base conservativo de la sección 23, que
se conserva como referencia matemática en las pruebas. No se ha iniciado un
barrido paramétrico ni entrenamiento de redes.

Resultados del caso variable: error terminal relativo `H_h` aproximadamente
`1.096 %`, residual relativo `3.18e-14`, condicionamiento de `I + beta W` en `H_h`
aproximadamente `2.23`, y costo `J(u*)=2.67434` frente a `J(0)=246.720`.
El error terminal es de seguimiento con penalización finita, no una estimación
del error de discretización espacial. El refinamiento de malla sigue pendiente.

La configuración evaluada y las métricas están en
`outputs/single_mu_variable/metrics.json`; las figuras están en esa misma carpeta.
Consultar `README.md` para comandos y ejemplos de uso.

---

# 42. Primera campaña paramétrica completada — 23 de septiembre de 2026

Se implementó `ParameterSpace` y la familia
`c_mu(x)=1+mu*sin(2*pi*x/L)`, con `mu in [0,0.3]`. La familia conserva el damping
variable, target, estado inicial y control global del ejemplo anterior. Los
parámetros se representan con dimensión explícita `(M,p)`, aunque el muestreo
uniforme implementado en este bloque es unidimensional.

`ParameterSweepExperiment` coordina solves, clasificación PASS/WARNING/FAIL,
diagnósticos locales de sensibilidad, selección de ocho parámetros y comparación
de mallas N=50,100,200. Las fórmulas y validaciones están fuera del orquestador.
La configuración se carga de `configs/parameter_sweep.yaml`.

Los 100 casos resultaron PASS. El residual relativo máximo fue `4.61e-14`, el
condicionamiento máximo del sistema en H_h fue `3.40024` y el error terminal
relativo estuvo entre `1.0653 %` y `1.1329 %`. Las diferencias entre mallas
decrecieron en los ocho representantes, sin constituir por sí solas una cota
certificada respecto al problema continuo.

Se generaron HDF5 de snapshots validados y entrenamiento, CSV de diagnósticos y
refinamiento, resumen JSON, mapas de adjuntos y gráficas globales. El dataset de
entrenamiento usa solo PASS y splits reproducibles 75/15/10 con semilla 42. Los
WARNING se conservan para revisión y los FAIL se registran sin entrar al split.
No se almacenan matrices FOM por muestra.

La ejecución completa y los scripts de auditoría/construcción del dataset están
documentados en `README.md`. El informe de resultados y sus límites está en
`docs/PARAMETER_SWEEP_RESULTS.md`. Se aprobaron 67 pruebas y Ruff. Todavía no se
han variado otros ingredientes físicos ni entrenado modelos neuronales.
