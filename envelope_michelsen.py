"""
Motor Envolvente de Fases — Metodo de MICHELSEN (1980)
Continuacion por pseudo-longitud de arco con Newton-Raphson multivariable
y tangente exacta (vector nulo del Jacobiano).

CLAVE DE ROBUSTEZ Y VELOCIDAD: el sistema se arma SOLO con los componentes
presentes (z_i > 0). Los ausentes no aportan a la fase incipiente ni a la
restriccion, y sus lnK serian direcciones fantasma que desvian la marcha.
Trabajar en el subespacio activo elimina esas direcciones (robustez) y
reduce el Jacobiano de (NC+2) a (m+2), m = nro de componentes activos
(gran aceleracion cuando la mezcla usa pocos componentes).

Variables:  X = (lnK_1..lnK_m, lnT, lnP)   (m+2)
  g_i = lnK_i + ln phi_i(vapor) - ln phi_i(liquido) = 0   (i=1..m)
  g_{m+1} = SUM(K_i z_i) - 1 = 0
  g_{m+2} = X[spec] - S = 0
La tangente es el vector nulo del Jacobiano fisico (filas g_1..g_{m+1}).
El punto critico aparece donde lnK_i -> 0.
Ref: Michelsen, M.L. (1980). Fluid Phase Equilib. 4: 1-10.
"""
import numpy as np
import copy
from engine3 import (
    NC, TC, PC, OMEGA, KIJ_DEFAULT,
    am, bm, AB, solve_Z, ln_phi_i
)

R_GAS = 10.7316
WILSON_C = np.log(10.0) * (7.0/3.0)
kij_g = None   # kij global, fijado en construir_envolvente

# ════════════════════════════════════════════════════════════════════
# EVALUADOR EOS VECTORIZADO (idéntico a engine3, pero rápido)
# - matriz aij cacheada por T (solo depende de T)
# - am y ln_phi vectorizados con numpy
# - solve_Z por Cardano analítico (sin np.roots)
# Mismas ecuaciones de Peng-Robinson -> misma precisión y mismo método.
# ════════════════════════════════════════════════════════════════════
from engine3 import ai as _ai_f, bi as _bi_f, mi as _mi_f

_AI = np.array([_ai_f(i) for i in range(NC)])
_BI = np.array([_bi_f(i) for i in range(NC)])
_MI = np.array([_mi_f(i) for i in range(NC)])
_TC = np.array(TC)
_SQRT2 = np.sqrt(2.0)
_aij_cache = {}
_one_minus_kij = None   # (1 - kij) como matriz, fijada por run

def _set_kij(kij):
    global _one_minus_kij, _aij_cache
    _one_minus_kij = 1.0 - np.array(kij)
    _aij_cache = {}

def _aij_T(T):
    """Matriz aij(T) cacheada (solo depende de T)."""
    key = round(float(T), 6)
    M = _aij_cache.get(key)
    if M is None:
        alpha = (1.0 + _MI*(1.0 - np.sqrt(T/_TC)))**2
        aa = _AI*alpha                       # ai_alpha vector
        s = np.sqrt(aa)
        M = np.outer(s, s) * _one_minus_kij  # aij = sqrt(aa_i aa_j)(1-kij)
        _aij_cache[key] = M
    return M

def _cardano_Z(A, B):
    """Raíces (ZV mayor, ZL menor) de la cúbica PR por Cardano."""
    a2 = -(1.0 - B)
    a1 = A - 3.0*B*B - 2.0*B
    a0 = -(A*B - B*B - B*B*B)
    # cúbica z^3 + a2 z^2 + a1 z + a0 = 0
    p = a1 - a2*a2/3.0
    q = 2.0*a2**3/27.0 - a2*a1/3.0 + a0
    disc = (q*q)/4.0 + (p*p*p)/27.0
    shift = a2/3.0
    if disc > 0:
        sq = np.sqrt(disc)
        u = np.cbrt(-q/2.0 + sq)
        v = np.cbrt(-q/2.0 - sq)
        z = u + v - shift
        roots = [z]
    else:
        r = np.sqrt(-(p**3)/27.0) if p < 0 else 0.0
        if r < 1e-300:
            roots = [-shift]
        else:
            phi = np.arccos(max(-1.0, min(1.0, -q/2.0/r)))
            m = 2.0*np.sqrt(-p/3.0)
            roots = [m*np.cos((phi+2.0*np.pi*k)/3.0) - shift for k in range(3)]
    real = [zz for zz in roots if zz > B]
    if not real:
        real = [max(roots)]
    real.sort()
    return real[-1], real[0]

def _lnphi_vec(comp, T, P, vapor):
    """Vector ln(phi_i) para todos los componentes (vectorizado)."""
    M = _aij_T(T)
    Mc = M @ comp                 # sum_j comp_j aij  (vector)
    am_v = float(comp @ Mc)       # am = comp·M·comp
    bm_v = float(_BI @ comp)
    A = am_v*P/(R_GAS*T)**2
    B = bm_v*P/(R_GAS*T)
    ZV, ZL = _cardano_Z(A, B)
    Z = ZV if vapor else ZL
    if Z <= B:
        Z = B + 1e-12
    t1 = (_BI/bm_v)*(Z-1.0)
    t2 = -np.log(Z-B)
    denom = Z + (1.0-_SQRT2)*B
    numer = Z + (1.0+_SQRT2)*B
    if denom <= 0: denom = 1e-12
    if numer <= 0: numer = 1e-12
    t3 = A/(2.0*_SQRT2*B)*np.log(numer/denom)
    t4 = (2.0*Mc/am_v - _BI/bm_v)*t3
    return t1 + t2 - t4, Z




def _Ki_wilson(i, T, P):
    return (PC[i]/P) * np.exp(WILSON_C*(1+OMEGA[i])*(1-TC[i]/T))


def _ln_phi_full(comp_full, T, P, vapor):
    am_ = am(comp_full, T, kij_g)
    bm_ = bm(comp_full)
    ZV, ZL = solve_Z(*AB(am_, bm_, T, P))
    Z = ZV if vapor else ZL
    return np.array([ln_phi_i(i, comp_full, T, P, Z, am_, bm_, kij_g)
                     for i in range(NC)])


def _funciones(X, z, act, spec):
    """
    Sistema reducido G(X)=0 (dimension m+2), m = len(act).
    spec define la ecuacion de continuacion (fila m+1):
      ('coord', idx, val)        -> X[idx] - val = 0       (fija coordenada)
      ('arc', t_arc, X_ref, ds)  -> t_arc·(X - X_ref) - ds = 0  (long. de arco)
    La restriccion de arco mantiene al corrector sobre la rama correcta
    (no puede saltar lejos sin violar la distancia ds a lo largo de la tangente).
    """
    m = len(act)
    lnK = X[:m]
    T = np.exp(X[m]); P = np.exp(X[m+1])
    K = np.exp(lnK)

    x_full = np.array(z, dtype=float)
    y_full = np.zeros(NC)
    Kz = K * x_full[act]
    sKz = Kz.sum()
    if sKz <= 0:
        sKz = 1e-300
    for idx, i in enumerate(act):
        y_full[i] = Kz[idx] / sKz

    ln_phi_L = _ln_phi_full(x_full, T, P, False)
    ln_phi_V = _ln_phi_full(y_full, T, P, True)

    G = np.zeros(m+2)
    for idx, i in enumerate(act):
        G[idx] = lnK[idx] + ln_phi_V[i] - ln_phi_L[i]
    G[m] = sKz - 1.0
    if spec[0] == 'coord':
        G[m+1] = X[spec[1]] - spec[2]
    else:  # 'arc'
        _, t_arc, X_ref, ds = spec
        G[m+1] = float(np.dot(t_arc, X - X_ref)) - ds
    return G


def _jacobiano(X, z, act, spec, h=1e-6):
    n = len(act)+2
    J = np.zeros((n, n))
    for j in range(n):
        Xp = X.copy(); Xm = X.copy()
        Xp[j] += h; Xm[j] -= h
        fp = _funciones(Xp, z, act, spec)
        fm = _funciones(Xm, z, act, spec)
        J[:, j] = (fp - fm) / (2*h)
    return J


def _resolver_punto(X0, z, act, spec, tol=1e-9, max_it=40):
    """
    Newton-Raphson MODIFICADO: calcula el Jacobiano una vez y lo reutiliza
    mientras la convergencia avance bien (metodo de la cuerda). Recalcula el
    Jacobiano solo si una iteracion no reduce el residual. Como el predictor
    por tangente da un arranque excelente, esto reduce mucho los calculos de
    Jacobiano SIN cambiar el punto al que converge (la solucion la define
    G(X)=0, no el Jacobiano) -> precision identica, metodo intacto.
    """
    X = X0.copy()
    G = _funciones(X, z, act, spec)
    res = np.linalg.norm(G, ord=np.inf)
    if res < tol:
        return X, True
    J = _jacobiano(X, z, act, spec)
    refrescar = False
    for _ in range(max_it):
        if refrescar:
            J = _jacobiano(X, z, act, spec)
            refrescar = False
        try:
            dX = np.linalg.solve(J, -G)
        except np.linalg.LinAlgError:
            return X, False
        mx = np.max(np.abs(dX))
        if mx > 0.5:
            dX *= 0.5/mx
        X = X + dX
        G = _funciones(X, z, act, spec)
        res_new = np.linalg.norm(G, ord=np.inf)
        if res_new < tol:
            return X, True
        # Si el paso no mejoro el residual, refrescar el Jacobiano la proxima vez
        if res_new > 0.9*res:
            refrescar = True
        res = res_new
    return X, (res < tol*100)


def _tangente(X, z, act, t_prev=None):
    """Tangente exacta = vector nulo del Jacobiano fisico (SVD)."""
    n = len(act)+2
    Jf = np.zeros((len(act)+1, n))
    h = 1e-6
    for j in range(n):
        Xp = X.copy(); Xm = X.copy()
        Xp[j] += h; Xm[j] -= h
        fp = _funciones(Xp, z, act, ('coord',0,X[0]))[:len(act)+1]
        fm = _funciones(Xm, z, act, ('coord',0,X[0]))[:len(act)+1]
        Jf[:, j] = (fp - fm) / (2*h)
    try:
        _, _, Vt = np.linalg.svd(Jf)
    except np.linalg.LinAlgError:
        return t_prev
    t = Vt[-1]
    nrm = np.linalg.norm(t)
    if nrm < 1e-30:
        return t_prev
    t = t / nrm
    if t_prev is not None and np.dot(t, t_prev) < 0:
        t = -t
    return t


def construir_envolvente(z, kij=None, progress_cb=None,
                         P_ini=14.7, max_pts=600):
    global kij_g
    if kij is None:
        kij = copy.deepcopy(KIJ_DEFAULT)
    kij_g = kij
    z = np.array(z, dtype=float)

    act = [i for i in range(NC) if z[i] > 1e-8]
    if len(act) < 2:
        return {'envolvente': [], 'critico': None}
    m = len(act)

    # Punto inicial: burbuja a baja presion por Wilson
    P0 = P_ini
    T0 = float(np.sum(z*np.array(TC))) * 0.6
    for _ in range(200):
        Kw = np.array([_Ki_wilson(i, T0, P0) for i in range(NC)])
        f = np.sum(z*Kw) - 1.0
        df = sum(z[i]*Kw[i]*(WILSON_C*(1+OMEGA[i])*TC[i]/T0**2) for i in range(NC))
        if abs(df) < 1e-30:
            break
        Tn = T0 - f/df
        if Tn <= 0:
            Tn = T0*0.5
        if abs(Tn-T0) < 1e-6:
            T0 = Tn; break
        T0 = Tn

    Kw = np.array([_Ki_wilson(i, T0, P0) for i in act])
    X = np.concatenate([np.log(Kw), [np.log(T0)], [np.log(P0)]])

    X, ok = _resolver_punto(X, z, act, ('coord', m+1, np.log(P0)))
    if not ok:
        return {'envolvente': [], 'critico': None}

    pts = []
    crit = None
    min_sumK2 = float('inf')
    crit_punto = None

    def guardar(Xv):
        pts.append((np.exp(Xv[m+1]), np.exp(Xv[m])))
        if progress_cb:
            progress_cb(len(pts))

    def revisar_critico(Xv):
        nonlocal min_sumK2, crit_punto
        s = float(np.sum(Xv[:m]**2))
        if s < min_sumK2:
            min_sumK2 = s
            crit_punto = (np.exp(Xv[m+1]), np.exp(Xv[m]))

    guardar(X)
    revisar_critico(X)

    t = _tangente(X, z, act)
    if t is None:
        return {'envolvente': pts, 'critico': None}
    if t[m+1] < 0:
        t = -t

    paso = 0.10
    PASO_MIN = 5e-4
    PASO_MAX = 0.18
    fallos = 0
    X_prev = X.copy()

    for _ in range(max_pts):
        exito = False
        paso_try = paso
        Xn = None
        for _intento in range(14):
            # Predicción por tangente; corrección con restricción de ARCO:
            # t·(X - X_prev) = paso_try. Esto mantiene al corrector sobre la
            # rama actual (no puede saltar a la otra rama sin violar el arco).
            X_pred = X_prev + t * paso_try
            spec = ('arc', t, X_prev.copy(), paso_try)
            Xn, ok = _resolver_punto(X_pred, z, act, spec)
            if ok:
                avance = np.linalg.norm(Xn - X_prev)
                # El paso real debe ser comparable al pedido (sin salto brusco)
                if 0.2*paso_try < avance < 4*paso_try:
                    exito = True
                    break
            paso_try *= 0.5
            if paso_try < PASO_MIN:
                break

        if not exito:
            fallos += 1
            if fallos >= 3:
                break
            paso = PASO_MIN
            continue
        fallos = 0

        guardar(Xn)
        revisar_critico(Xn)

        t_new = _tangente(Xn, z, act, t_prev=t)
        if t_new is None:
            break
        cosang = float(np.dot(t, t_new))
        X_prev = Xn.copy()
        t = t_new

        if cosang < 0.2:
            paso = max(paso_try*0.35, PASO_MIN)
        elif cosang < 0.7:
            paso = max(paso_try*0.7, PASO_MIN)
        else:
            paso = min(paso_try*1.25, PASO_MAX)

        Pn = np.exp(Xn[m+1])
        if len(pts) > 20 and Pn < P_ini*1.05:
            break

    if crit_punto is not None and min_sumK2 < 0.5:
        crit = crit_punto

    return {'envolvente': pts, 'critico': crit}
