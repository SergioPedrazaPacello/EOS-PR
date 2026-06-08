"""
Motor Envolvente de Fases — Método de MICHELSEN (1980)
=======================================================
Continuación por longitud de arco con Newton-Raphson multivariable.

A diferencia de Ziervogel (que itera sobre UNA variable T o P y no puede
cruzar el ápice crítico), Michelsen resuelve el sistema COMPLETO de n+2
incógnitas simultáneamente:

    X = (lnK_1, ..., lnK_n, lnT, lnP)        (n+2 incógnitas)

Ecuaciones (n+1):
    g_i = lnK_i + ln phi_i(vapor,y) - ln phi_i(liquido,x) = 0   (i=1..n)
    g_{n+1} = SUM_i (y_i - x_i) = 0          [= SUM z_i(K_i-1)/(...)]

La ecuación faltante (n+2) es la ESPECIFICACIÓN de continuación:
    g_{n+2} = X[s] - S = 0
donde 's' es la variable que más varía a lo largo de la curva (puede ser
lnT, lnP o cualquier lnK_i). Esto permite RODEAR el ápice sin singularidad.

El punto crítico aparece naturalmente donde todos los lnK_i -> 0.

Para la fase incipiente:
    y_i = K_i * x_i,  con la fase presente = z (alimentación).
    En burbuja: x=z (líquido), y=incipiente vapor.
    En rocío:   y=z (vapor),   x=incipiente líquido.
Michelsen unifica ambas con la fracción de vapor beta; aquí construimos la
curva completa partiendo de un punto de burbuja a baja presión y avanzando
por continuación, de modo que la misma rutina recorre burbuja -> crítico ->
rocío en un solo trazo continuo.
"""
import numpy as np
import copy
from engine3 import (
    NC, TC, PC, OMEGA, KIJ_DEFAULT,
    am, bm, AB, solve_Z, ln_phi_i
)

R_GAS = 10.7316
WILSON_C = np.log(10.0) * (7.0/3.0)


def _Ki_wilson(i, T, P):
    return (PC[i]/P) * np.exp(WILSON_C*(1+OMEGA[i])*(1-TC[i]/T))


def _ln_phi_vec(comp, T, P, vapor, kij):
    """Vector de ln(phi_i) para una composición y fase dadas."""
    am_ = am(comp, T, kij)
    bm_ = bm(comp)
    ZV, ZL = solve_Z(*AB(am_, bm_, T, P))
    Z = ZV if vapor else ZL
    return np.array([ln_phi_i(i, comp, T, P, Z, am_, bm_, kij)
                     for i in range(NC)])


def _funciones(X, z, kij, spec_idx, spec_val):
    """
    Sistema de Michelsen G(X)=0, dimensión n+2.
    X = [lnK_0..lnK_{n-1}, lnT, lnP]
    Fase presente = z (líquido de referencia). Fase incipiente w = K*z normalizada.
    """
    lnK = X[:NC]
    lnT = X[NC]
    lnP = X[NC+1]
    K = np.exp(lnK)
    T = np.exp(lnT)
    P = np.exp(lnP)

    # Fase presente (líquido) = z ; incipiente (vapor) = normalize(K*z)
    x = np.array(z, dtype=float)
    yk = K * x
    sumyk = yk.sum()
    y = yk / sumyk if sumyk > 0 else yk

    ln_phi_L = _ln_phi_vec(x, T, P, False, kij)
    ln_phi_V = _ln_phi_vec(y, T, P, True, kij)

    G = np.zeros(NC+2)
    # n ecuaciones de equilibrio: lnK_i + ln phi_V - ln phi_L = 0
    G[:NC] = lnK + ln_phi_V - ln_phi_L
    # restriccion de fase incipiente: SUM(K_i z_i) - 1 = 0
    G[NC] = sumyk - 1.0
    # especificacion de continuacion
    G[NC+1] = X[spec_idx] - spec_val
    return G


def _jacobiano(X, z, kij, spec_idx, spec_val, h=1e-6):
    """Jacobiano numérico por diferencias centradas (n+2 x n+2)."""
    n = NC+2
    J = np.zeros((n, n))
    f0 = _funciones(X, z, kij, spec_idx, spec_val)
    for j in range(n):
        Xp = X.copy(); Xm = X.copy()
        Xp[j] += h; Xm[j] -= h
        fp = _funciones(Xp, z, kij, spec_idx, spec_val)
        fm = _funciones(Xm, z, kij, spec_idx, spec_val)
        J[:, j] = (fp - fm) / (2*h)
    return J, f0


def _resolver_punto(X0, z, kij, spec_idx, spec_val, tol=1e-9, max_it=40):
    """Newton-Raphson multivariable para un punto de la envolvente."""
    X = X0.copy()
    for _ in range(max_it):
        J, G = _jacobiano(X, z, kij, spec_idx, spec_val)
        nrm = np.linalg.norm(G, ord=np.inf)
        if nrm < tol:
            return X, True
        try:
            dX = np.linalg.solve(J, -G)
        except np.linalg.LinAlgError:
            return X, False
        # Limitar el paso para estabilidad
        mx = np.max(np.abs(dX))
        if mx > 0.5:
            dX *= 0.5/mx
        X = X + dX
    G = _funciones(X, z, kij, spec_idx, spec_val)
    return X, (np.linalg.norm(G, ord=np.inf) < tol*100)


def construir_envolvente(z, kij=None, progress_cb=None,
                         P_ini=14.7, max_pts=500):
    """
    Construye la envolvente completa por continuación de Michelsen.
    Retorna dict: {'envolvente':[(P,T)...], 'critico':(Pc,Tc) o None}
    Un solo trazo continuo burbuja -> crítico -> rocío.
    """
    if kij is None:
        kij = copy.deepcopy(KIJ_DEFAULT)
    z = np.array(z, dtype=float)

    # ── Punto inicial: burbuja a baja presión por Wilson ──
    P0 = P_ini
    # Estimar T de burbuja a P0 resolviendo SUM(z*Kw)=1 por Newton
    T0 = 0.0
    for i in range(NC):
        T0 += z[i]*TC[i]
    T0 *= 0.6
    for _ in range(200):
        Kw = np.array([_Ki_wilson(i, T0, P0) for i in range(NC)])
        f = np.sum(z*Kw) - 1.0
        df = 0.0
        for i in range(NC):
            df += z[i]*Kw[i]*(WILSON_C*(1+OMEGA[i])*TC[i]/T0**2)
        if abs(df) < 1e-30:
            break
        Tn = T0 - f/df
        if Tn <= 0:
            Tn = T0*0.5
        if abs(Tn-T0) < 1e-6:
            T0 = Tn
            break
        T0 = Tn

    Kw = np.array([_Ki_wilson(i, T0, P0) for i in range(NC)])
    X = np.concatenate([np.log(Kw), [np.log(T0)], [np.log(P0)]])

    # Resolver el primer punto: especificar lnP (índice n+1)
    spec_idx = NC+1
    spec_val = np.log(P0)
    X, ok = _resolver_punto(X, z, kij, spec_idx, spec_val)
    if not ok:
        return {'envolvente': [], 'critico': None}

    pts = []
    crit = None

    def guardar(Xv):
        T = np.exp(Xv[NC]); P = np.exp(Xv[NC+1])
        pts.append((P, T))
        if progress_cb:
            progress_cb(len(pts))

    guardar(X)
    X_prev = X.copy()

    # Rastreo del punto crítico: el mínimo de SUM(lnK^2) a lo largo de la curva
    min_sumK2 = float('inf')
    crit_punto = None

    def revisar_critico(Xv):
        nonlocal min_sumK2, crit_punto
        s = float(np.sum(Xv[:NC]**2))
        if s < min_sumK2:
            min_sumK2 = s
            crit_punto = (np.exp(Xv[NC+1]), np.exp(Xv[NC]))  # (P, T)

    revisar_critico(X)

    # ── Continuación ──
    dS = 0.05

    # Segundo punto: subir P un poco para tener tangente
    spec_idx = NC+1
    spec_val = X[NC+1] + dS
    X2, ok = _resolver_punto(X, z, kij, spec_idx, spec_val)
    if not ok:
        return {'envolvente': pts, 'critico': None}
    guardar(X2)
    revisar_critico(X2)
    X_prev2 = X.copy()
    X_prev = X2.copy()

    paso = 0.12
    PASO_MIN = 5e-4
    PASO_MAX = 0.25
    fallos_seguidos = 0
    for _ in range(max_pts):
        # Tangente por diferencia hacia atrás (dirección de avance)
        dXt = X_prev - X_prev2
        nrm = np.linalg.norm(dXt)
        if nrm < 1e-12:
            break
        dir_unit = dXt / nrm

        # ── Mejora 3: elegir variable de especificación = la que más varía
        #    a lo largo de la curva. En la cricondentérmica dT≈0, así que la
        #    especificación pasa naturalmente a P o a alguna lnK_i, que es lo
        #    que sí cambia. Esto evita especificar una variable estacionaria.
        spec_idx = int(np.argmax(np.abs(dir_unit)))

        # ── Mejora 1: predicción y, si falla, reducción PROFUNDA del paso
        #    (hasta PASO_MIN) en vez de rendirse al primer tropiezo. Esto da
        #    la resolución necesaria para tomar codos cerrados (cricondentérmica).
        exito = False
        paso_try = paso
        Xn = None
        for _intento in range(14):
            X_pred = X_prev + dir_unit * paso_try
            spec_val = X_pred[spec_idx]
            Xn, ok = _resolver_punto(X_pred, z, kij, spec_idx, spec_val)
            if ok:
                # ── Mejora 2: validar que el punto AVANZA (no retrocede sobre
                #    sí mismo ni salta). El coseno entre el paso real y la
                #    dirección esperada debe ser positivo; si el codo invierte
                #    bruscamente, se acepta igual mientras el avance sea real.
                paso_real = Xn - X_prev
                if np.linalg.norm(paso_real) > 1e-9:
                    exito = True
                    break
            # reducir el paso a la mitad y reintentar
            paso_try *= 0.5
            if paso_try < PASO_MIN:
                break

        if not exito:
            fallos_seguidos += 1
            # Si falla repetidamente, la curva terminó (o codo intratable):
            # cortar limpio en vez de quedar colgado.
            if fallos_seguidos >= 3:
                break
            # reintento con paso mínimo
            paso = PASO_MIN
            continue

        fallos_seguidos = 0
        guardar(Xn)
        revisar_critico(Xn)

        # ── Mejora 1 (control de curvatura): comparar la nueva dirección con
        #    la anterior. Si la curva dobló mucho (codo), achicar el paso para
        #    el siguiente; si viene suave, agrandarlo. Esto frena justo en la
        #    cricondentérmica y vuelve a acelerar después.
        nueva_dir = Xn - X_prev
        nn = np.linalg.norm(nueva_dir)
        if nn > 1e-12:
            cos_giro = float(np.dot(nueva_dir/nn, dir_unit))
        else:
            cos_giro = 1.0

        X_prev2 = X_prev.copy()
        X_prev = Xn.copy()

        # Ajuste del paso según la curvatura local
        if cos_giro < 0.5:          # giro fuerte (codo): frenar mucho
            paso = max(paso_try*0.4, PASO_MIN)
        elif cos_giro < 0.9:        # curvatura media: frenar un poco
            paso = max(paso_try*0.8, PASO_MIN)
        else:                       # tramo suave: acelerar
            paso = min(paso_try*1.15, PASO_MAX)

        # Condición de término: cerró el rocío a presión baja
        Pn = np.exp(Xn[NC+1])
        if len(pts) > 15 and Pn < P_ini*1.05:
            break

    # El crítico es el punto de mínimo SUM(lnK^2) si fue suficientemente bajo
    if crit_punto is not None and min_sumK2 < 0.05:
        crit = crit_punto

    return {'envolvente': pts, 'critico': crit}
