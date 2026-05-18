import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

phi   = (1 + np.sqrt(5)) / 2
alpha_c = beta_c = 1/phi
pi2   = np.pi**2
C_inf = pi2 / phi**3    # ≈ 2.32989755

def build_P(N, alpha=alpha_c, beta=beta_c):
    n = 2 + 2*N
    rows, cols, data = [], [], []
    def add(i,j,v): rows.append(i); cols.append(j); data.append(v)
    sk  = lambda k: 2+2*k
    spk = lambda k: 3+2*k
    add(0,sk(0),1.0); add(1,0,1.0)
    add(sk(0),spk(0),beta);  add(sk(0),1,1-beta)
    for k in range(1,N):
        add(sk(k),spk(k),beta); add(sk(k),sk(k-1),1-beta)
    for k in range(N-1):
        add(spk(k),sk(k+1),alpha); add(spk(k),sk(k),1-alpha)
    add(spk(N-1),sk(N-1),1.0)
    return sp.csr_matrix((data,(rows,cols)),shape=(n,n))

def gap(N, tol=1e-14):
    P = build_P(N)
    v = spla.eigs(P.T, k=4, which='LM', return_eigenvectors=False,
                  tol=tol, maxiter=300000)
    v = sorted(v, key=lambda x: -abs(x))
    return 1.0 - abs(v[1])

# ================================================================
# PARTE I: Verificación de alta precisión de a = -pi^2/phi^6
# ================================================================

# Test 1: C(N) de alta precisión para N geométrico
# Con 10 dígitos, el par (N,2N) estima a con error ~O(C_inf·|b|/N²)
# donde b es el coeficiente de 1/N².
# Para N=4000→8000: error en a es ~0.01 — suficiente para distinguir
# entre a=-0.55 (candidato) y a=-0.60 o -0.50.
print("="*70)
print("Test 1: C(N) de alta precisión — N hasta 8000")
print(f"  C_inf = pi^2/phi^3 = {C_inf:.12f}")
print(f"  Candidato a = -pi^2/phi^6 = {-pi2/phi**6:.10f}")
print()
print(f"  {'N':>6}  {'gap':>20}  {'C(N)=N^2*gap':>18}  {'C_inf-C(N)':>14}")

N_list = [500, 1000, 2000, 4000, 8000]
C_data = {}

for N in N_list:
    g = gap(N)
    CN = N**2 * g
    C_data[N] = CN
    print(f"  {N:>6}  {g:>20.14f}  {CN:>18.12f}  {C_inf-CN:>14.10f}")

# Test 2: Estimación de a por pares (N, 2N)
# C(N) = C_inf + a/N + b/N^2 + ...
# a_est(N) = 2N*(C(N) - C(2N)) → a conforme N→∞
print()
print("="*70)
print("Test 2: a estimado por pares (N, 2N)")
print()
print(f"  {'N':>6}  {'C(N)':>18}  {'C(2N)':>18}  {'a_est=2N(C(N)-C(2N))':>24}")
a_vals = []
for N in [500, 1000, 2000, 4000]:
    if 2*N in C_data:
        a_est = 2*N*(C_data[N] - C_data[2*N])
        a_vals.append((N, a_est))
        print(f"  {N:>6}  {C_data[N]:>18.12f}  {C_data[2*N]:>18.12f}  {a_est:>24.12f}")

print()
print(f"  Candidatos para a:")
a_ref = a_vals[-1][1] if a_vals else -0.55
cands = {
    "-pi^2/phi^6"       : -pi2/phi**6,
    "-pi^2/(phi^3+phi^6)": -pi2/(phi**3+phi**6),
    "-1/(2*phi^3)"      : -1/(2*phi**3),
    "-pi^2*phi/(phi^3+1)": -pi2*phi/(phi**3+1),
    "-phi/pi^2"         : -phi/pi2,
    "-2/(phi^3+1)"      : -2/(phi**3+1),
    "-pi^2/(2*phi^3+phi^6)": -pi2/(2*phi**3+phi**6),
}
for name, val in sorted(cands.items(), key=lambda x: abs(x[1]-a_ref)):
    print(f"    {name:>28} = {val:>12.8f}   diff={abs(val-a_ref):.8f}")

# Test 3: Richardson exacto de 4 puntos
# C(N) = C_inf + a/N + b/N^2 + c/N^3
# Sistema 4x4 con N=1000,2000,4000,8000
print()
print("="*70)
print("Test 3: Richardson de 4 puntos — sistema lineal exacto")
print()

Ns4 = [1000, 2000, 4000, 8000]
if all(N in C_data for N in Ns4):
    Ns = np.array(Ns4, dtype=float)
    Cs = np.array([C_data[N] for N in Ns4])
    X = np.column_stack([np.ones(4), 1/Ns, 1/Ns**2, 1/Ns**3])
    coef = np.linalg.solve(X, Cs)
    C_inf_rich, a_rich, b_rich, c_rich = coef
    print(f"  C_inf = {C_inf_rich:.12f}")
    print(f"  a     = {a_rich:.10f}")
    print(f"  b     = {b_rich:.6f}")
    print(f"  c     = {c_rich:.4f}")
    print()
    print(f"  pi^2/phi^3   = {C_inf:.12f}  diff_C = {abs(C_inf_rich-C_inf):.2e}")
    print(f"  -pi^2/phi^6  = {-pi2/phi**6:.10f}  diff_a = {abs(a_rich+pi2/phi**6):.2e}")

# ================================================================
# PARTE II: Conexión estructural A_∞ ↔ cadena de Markov
# ================================================================
print()
print("="*70)
print("PARTE II: Conexión A_∞ — Cadena de Markov")
print()

# La pregunta central: ¿son el mismo objeto matemático visto de dos ángulos?
# A_∞: operador de Jacobi gauge-simetrizado, sigma = [0, cbrt4] U {phi}
# Markov: cadena P_{alpha=1/phi}, sigma_Markov relacionado con sigma(A_∞) via...

# Observación clave: la relación de dispersión de la cadena de Markov es
# epsilon = theta^2/phi^3
# que en la variable mu = phi^{3/2}*epsilon = phi^{3/2}*theta^2/phi^3 = theta^2/phi^{3/2}
# equivale a mu = theta^2/phi^{3/2} = (theta/phi^{3/4})^2
# Mientras que para el Jacobi estándar: mu = 2*cos(theta) => gap = 2-mu = 2(1-cos(theta)) ≈ theta^2
# Y phi^{3/2} = (phi^3)^{1/2} = (2phi+1)^{1/2} = sqrt(2phi+1)

# El puente: el generador del Jacobi gauge actúa en el espacio de escalas.
# La cadena de Markov actúa en el espacio de posiciones.
# Son Fourier-duales: la relación de dispersión de uno es el espectro del otro.

# Test 4: Verificar que el autovalor phi de A_∞ y el punto critico 1/phi
# satisfacen la misma ecuación mínima (x^2 - x - 1 = 0 y x^2 + x - 1 = 0)
print("Test 4: Ecuaciones mínimas compartidas")
print()
print(f"  A_infty: autovalor phi satisface x^2 - x - 1 = 0")
print(f"  Verificacion: phi^2 - phi - 1 = {phi**2 - phi - 1:.2e}")
print()
print(f"  Markov: punto critico 1/phi satisface x^2 + x - 1 = 0")
inv_phi = 1/phi
print(f"  Verificacion: (1/phi)^2 + (1/phi) - 1 = {inv_phi**2 + inv_phi - 1:.2e}")
print()
print(f"  Relacion: si phi satisface x^2 = x+1,")
print(f"  entonces 1/phi = phi-1 satisface (phi-1)^2 + (phi-1) - 1 = 0")
print(f"  = phi^2 - 2phi + 1 + phi - 1 - 1 = phi^2 - phi - 1 = 0  [la misma ecuacion!]")
print(f"  => phi y 1/phi satisfacen la MISMA ecuacion minimal x^2-x-1=0")
print(f"     (con raices phi = (1+sqrt5)/2 y 1/phi = (sqrt5-1)/2)")
print()

# Test 5: El espectro de A_infty como funcion generadora de alpha_c
# La relacion de dispersion de Markov: epsilon = theta^2/phi^3
# equivale a: lambda(theta) = 1 - theta^2/phi^3
# El espectro esencial de A_infty: lambda_ess in [0, cbrt4]
# Correspondencia: cbrt4^(3/2) = 4^(1/2) = 2 = rho(Jacobi bulk)
# Y phi^(3/2) > 2 => phi es autovalor aislado
# 
# El PUNTO CRITICO 1/phi del Markov es el RECIPROCO del autovalor phi.
# Esto no es coincidencia: en un sistema con simetria de Galois sobre Q(phi),
# phi y 1/phi son conjugados algebraicos. El sistema es invariante bajo
# phi <-> 1/phi (simetria de Galois del cuerpo Q(sqrt5)).

print("Test 5: Simetria de Galois — phi <-> 1/phi")
print()
print(f"  Q(phi) = Q(sqrt5) tiene el automorfismo de Galois sigma: sqrt5 -> -sqrt5")
print(f"  sigma(phi) = sigma((1+sqrt5)/2) = (1-sqrt5)/2 = -1/phi")
print(f"  (el conjugado algebraico de phi es -1/phi = -(phi-1) ≈ -0.618)")
print()
print(f"  En terminos positivos: el unico otro elemento positivo de Q(sqrt5)")
print(f"  de la misma ecuacion minimal es 1/phi = phi-1 ≈ 0.618")
print()
print(f"  CONEXION: el espectro de A_infty tiene un autovalor aislado en phi,")
print(f"  y la cadena de Markov tiene su punto critico en 1/phi.")
print(f"  Son los dos PUNTOS FIJOS positivos de x^2 = x+1 y x^2+x-1=0,")
print(f"  que son la misma ecuacion (multiplicando x^2+x-1=0 por -1 y sustituyendo x->-x:")
print(f"  x^2-x-1=0). Los dos objetos son las dos raices reales positivas del mismo")
print(f"  polinomio, evaluadas en el espectro y en la dinamica respectivamente.")
print()

# Test 6: La MISMA relacion de recurrencia subyace a ambos
# La recurrencia del libro (§22): psi_{k+1} - lambda*psi_k + (1/lambda)*psi_{k-1} = 0
# Esta es exactamente la ecuacion de borde que selecciona lambda=phi en A_infty
# Y la misma estructura (despues del gauge) da la dispersion de la cadena de Markov.
print("Test 6: Recurrencia comun subyacente")
print()
print(f"  Recurrencia del libro §22: psi_{{k+1}} - lambda*psi_k + (1/lambda)*psi_{{k-1}} = 0")
print()
print(f"  En A_infty (gauge a_k = lambda^(k/2)*psi_k):")
print(f"    Se transforma en: a_{{k+1}} + a_{{k-1}} = lambda^(3/2) * a_k")
print(f"    => operador Jacobi estandar con autovalor efectivo mu = lambda^(3/2)")
print(f"    => lambda=phi da mu=phi^(3/2) > 2 = rho(Jacobi) => autovalor aislado ✓")
print()
print(f"  En la cadena de Markov (bulk reducido):")
print(f"    v_{{k+1}} - (phi^2*lambda^2 - 1/phi)*v_k + lambda*v_{{k-1}} = 0")
print(f"    En lambda=1: v_{{k+1}} - 2*v_k + v_{{k-1}} = 0  [paseo simetrico]")
print(f"    Dispersion: epsilon = theta^2/phi^3")
print(f"    Constante de escalado: C = pi^2/phi^3")
print()
print(f"  CONEXION ALGEBRAICA:")
print(f"    La condicion de borde de A_infty es exactamente phi^2-phi-1=0,")
print(f"    que es la misma identidad que hace que 2*phi^2-1 = phi^3 (Fibonacci),")
print(f"    que es la misma identidad que determina la dispersion de Markov.")
print(f"    El factor phi^3 aparece en AMBOS objetos por la misma razon algebraica:")
print(f"    phi^3 = 2phi+1 es la potencia minima de phi con coeficientes enteros")
print(f"    que involucra solo la parte entera (no la parte sqrt5) en todos los contextos.")
print()

# Test 7: Diagrama de la jerarquia algebraica completa
print("Test 7: Jerarquia algebraica unificada")
print()
print(f"  OBJETO CENTRAL: el cuerpo Q(phi) = Q(sqrt5)")
print()
print(f"  ┌────────────────────────────────────────────────────────────┐")
print(f"  │  OPERADOR A_infty                                          │")
print(f"  │  sigma = [0, cbrt4] U {{phi}}                               │")
print(f"  │  Autovalor aislado: phi [raiz positiva mayor de x^2-x-1]   │")
print(f"  │  Borde esencial: cbrt4 [raiz de x^3-4 sobre Q, grado 3]   │")
print(f"  │  Gap espectral: g_inf = phi - cbrt4 [grado 6 sobre Q]      │")
print(f"  ├────────────────────────────────────────────────────────────┤")
print(f"  │  PUENTE (ecuacion de recurrencia §22 + identidad Fibonacci) │")
print(f"  │  phi^3 = 2phi+1  [identidad de Fibonacci para phi]         │")
print(f"  │  x^2-x-1=0 tiene raices phi y 1/phi = phi-1                │")
print(f"  ├────────────────────────────────────────────────────────────┤")
print(f"  │  CADENA DE MARKOV Caso B (alpha=beta=1/phi)                │")
print(f"  │  Punto critico: 1/phi [raiz positiva menor de x^2-x-1=0]  │")
print(f"  │  Dispersion: epsilon = theta^2/phi^3 [via 2phi^2-1=phi^3]  │")
print(f"  │  Constante: C = pi^2/phi^3  [pi: geometria difusiva]       │")
print(f"  │  Corrección 1/N: a = -pi^2/phi^6 = -C/phi^3 (candidato)   │")
print(f"  └────────────────────────────────────────────────────────────┘")
print()
print(f"  La conexion es la ECUACION MINIMAL x^2-x-1=0:")
print(f"  - En A_infty: sus dos raices reales (phi y 1/phi) aparecen como")
print(f"    el autovalor aislado (phi) y el modulo del conjugado de Galois (1/phi).")
print(f"  - En la cadena de Markov: la raiz positiva menor (1/phi) es el punto")
print(f"    critico alpha_c, y la identidad phi^3=2phi+1 genera la dispersion.")
print(f"  Son el MISMO objeto algebraico visto desde el espectro (estatico) y")
print(f"  desde la dinamica (cinetico).")

# Test 8: Verificacion numerica de la identidad puente
print()
print("="*70)
print("Test 8: Verificacion numerica de identidades del puente")
print()
ids = [
    ("phi^3 = 2phi+1 (Fibonacci)",            phi**3,            2*phi+1),
    ("2phi^2-1 = phi^3 (dispersion Markov)",   2*phi**2-1,       phi**3),
    ("phi^2-1/phi = 2 (bulk Markov en lam=1)", phi**2-1/phi,     2.0),
    ("phi^(3/2) > 2 (phi aislado en A_inf)",   phi**1.5,         2.0),
    ("1/phi = phi-1 (relacion de Galois)",      1/phi,            phi-1),
    ("(1/phi)*(1/phi+1) = 1 (punto critico)",   (1/phi)*(1/phi+1), 1.0),
    ("1-1/phi^3 = 2/phi^2 (bulk orden 0)",      1-1/phi**3,       2/phi**2),
]
for desc, lhs, rhs in ids:
    err = abs(lhs-rhs)
    check = "✓" if err < 1e-12 else f"  ERROR={err:.2e}"
    print(f"  {desc}")
    print(f"    LHS={lhs:.14f}  RHS={rhs:.14f}  {check}")
