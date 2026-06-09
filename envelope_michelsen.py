"""
Motor Envolvente de Fases — Metodo de MICHELSEN (1980)
=======================================================
Continuacion por pseudo-longitud de arco con Newton-Raphson multivariable
y tangente exacta (vector nulo del Jacobiano fisico via SVD).

ESTRATEGIA BIDIRECCIONAL (soluciona la cola de baja presion de rocio):
  Trazo 1: desde burbuja a baja P → sube por burbuja → rodea critico
           → baja por rama de rocio superior hasta que la continuacion
           se detenga (tipicamente ~100-150 psi en gases livianos).
  Trazo 2: desde ROCIO a baja P → sube por rama de rocio inferior hasta
           conectarse con el Trazo 1. Arranca en la rama correcta y nunca
           compite con la de burbuja → resuelve el problema de la cola.
  Combinados: envolvente completa burbuja + rocio hasta presiones bajas.

SUBESPACIO ACTIVO: el sistema se arma solo con los componentes presentes
  (z_i > 0), reduciendo el Jacobiano de (NC+2) a (m+2). Gran aceleracion
  para composiciones con componentes ausentes (tipico en produccion).

Variables:  X = (lnK_1..lnK_m, lnT, lnP)   (m+2)
  g_i = lnK_i + ln phi_i(vapor) - ln phi_i(liquido) = 0   (i=1..m)
  g_{m+1} = SUM(K_i z_i) - 1 = 0
  g_{m+2} = t·(X - X_ref) - ds = 0      (restriccion de arco)
Ref: Michelsen, M.L. (1980). Fluid Phase Equilib. 4: 1-10.
"""
import numpy as np
import copy
import math
from engine3 import (
    NC, TC, PC, OMEGA, KIJ_DEFAULT,
    am, bm, AB, solve_Z, ln_phi_i,
)

R_GAS     = 10.7316
WILSON_C  = np.log(10.0) * (7.0/3.0)
kij_g     = None   # kij global, fijado en construir_envolvente
_max = max  # alias para max escalar (legibilidad)


# ── Helpers de bajo nivel ────────────────────────────────────────────────────
def _set_kij(kij):
    global kij_g; kij_g = kij

def _cardano_Z(A, B):
    p2 = -(1.0 - B); p1 = A - 3*B*B - 2*B; p0 = -(A*B - B*B - B**3)
    sh = p2/3.0; p = p1 - p2*p2/3.0; q = 2*p2**3/27 - p2*p1/3 + p0
    disc = q*q/4 + p**3/27
    if disc > 1e-14:
        sq = math.sqrt(disc)
        u  = np.cbrt(-q/2+sq); v = np.cbrt(-q/2-sq)
        raices = [u+v-sh]
    else:
        if abs(p) < 1e-30:
            raices = [-sh]
        else:
            mf  = 2*math.sqrt(-p/3)
            arg = max(-1.0, min(1.0, 3*q/(p*mf)))
            th  = math.acos(arg)/3
            raices = [mf*math.cos(th - 2*math.pi*k/3)-sh for k in range(3)]
    real = [r for r in raices if r > B]
    if not real: real = [max(raices)]
    real = sorted(real)
    return real[-1], real[0]

def _Ki_wilson(i, T, P):
    return (PC[i]/P) * np.exp(WILSON_C*(1+OMEGA[i])*(1-TC[i]/T))

def _gibbs_dep(Z, A, B):
    """Salida de Gibbs adimensional de PR (sin constante). Menor = mas estable."""
    sqrt2 = math.sqrt(2.0)
    d = Z + (1 - sqrt2)*B; n = Z + (1 + sqrt2)*B
    if d <= 0: d = 1e-30
    if n <= 0: n = 1e-30
    if Z <= B: Z = B + 1e-12
    return Z - 1.0 - math.log(Z - B) - A/(2*sqrt2*B)*math.log(n/d)

def _ln_phi_full(comp_full, T, P, vapor=None):
    """
    Coeficientes de fugacidad con seleccion de Z por MINIMA energia de Gibbs.
    Funciona correctamente tanto para la rama de burbuja (incipiente vapor)
    como para la de rocio inferior (incipiente liquido, K<1 para livianos).
    El parametro vapor se ignora — la fisica elige la raiz correcta.
    """
    am_ = am(comp_full, T, kij_g); bm_ = bm(comp_full)
    A, B = AB(am_, bm_, T, P)
    ZV, ZL = solve_Z(A, B)
    # Elegir la raiz de menor Gibbs (fase fisicamente estable)
    gV = _gibbs_dep(ZV, A, B)
    gL = _gibbs_dep(ZL, A, B)
    Z  = ZV if gV <= gL else ZL
    return np.array([ln_phi_i(i, comp_full, T, P, Z, am_, bm_, kij_g)
                     for i in range(NC)])


# ── Sistema de Michelsen (subespacio activo) ─────────────────────────────────
def _funciones(X, z, act, spec):
    """
    G(X) = 0 (dimension m+2). act = indices de componentes activos.
    spec: ('coord', idx, val)       -> X[idx] - val = 0
          ('arc',   t_arc, Xref, ds)-> t_arc·(X-Xref) - ds = 0  [restriccion de arco]
    En la restriccion de arco, el corrector no puede saltar a otra rama
    sin violar la distancia ds a lo largo de la tangente.
    """
    m  = len(act)
    T  = np.exp(X[m]); P = np.exp(X[m+1])
    K  = np.exp(X[:m])
    x_full = np.array(z, dtype=float)
    y_full = np.zeros(NC)
    Kz = K * x_full[act]; sKz = Kz.sum()
    if sKz <= 0: sKz = 1e-300
    for k, i in enumerate(act): y_full[i] = Kz[k]/sKz

    lpL = _ln_phi_full(x_full, T, P, False)
    lpV = _ln_phi_full(y_full, T, P, True)

    G = np.zeros(m+2)
    for k, i in enumerate(act): G[k] = X[k] + lpV[i] - lpL[i]
    G[m] = sKz - 1.0
    if spec[0] == 'coord':
        G[m+1] = X[spec[1]] - spec[2]
    else:
        _, t_arc, Xref, ds = spec
        G[m+1] = float(np.dot(t_arc, X - Xref)) - ds
    return G


def _jacobiano(X, z, act, spec, h=1e-6):
    n = len(act)+2
    J = np.zeros((n, n))
    for j in range(n):
        Xp = X.copy(); Xm = X.copy()
        Xp[j] += h; Xm[j] -= h
        J[:, j] = (_funciones(Xp, z, act, spec)
                 - _funciones(Xm, z, act, spec)) / (2*h)
    return J


def _resolver_punto(X0, z, act, spec, tol=1e-9, max_it=40):
    """Newton modificado: reutiliza el Jacobiano; lo refresca si el residual
    no mejora. La solucion converge al mismo G(X)=0 que Newton completo."""
    X   = X0.copy()
    G   = _funciones(X, z, act, spec)
    res = np.linalg.norm(G, ord=np.inf)
    if res < tol: return X, True
    J = _jacobiano(X, z, act, spec)
    refrescar = False
    for _ in range(max_it):
        if refrescar:
            J = _jacobiano(X, z, act, spec); refrescar = False
        try:
            dX = np.linalg.solve(J, -G)
        except np.linalg.LinAlgError:
            return X, False
        mx = np.max(np.abs(dX))
        if mx > 0.5: dX *= 0.5/mx
        X  = X + dX
        G  = _funciones(X, z, act, spec)
        rn = np.linalg.norm(G, ord=np.inf)
        if rn < tol: return X, True
        if rn > 0.9*res: refrescar = True
        res = rn
    return X, (res < tol*100)


def _tangente(X, z, act, t_prev=None):
    """Tangente exacta = vector nulo del Jacobiano fisico (SVD).
    Valida incluso en pliegues verticales (cricondentermica, cola baja P)."""
    n  = len(act)+2
    Jf = np.zeros((len(act)+1, n))
    h  = 1e-6
    for j in range(n):
        Xp = X.copy(); Xm = X.copy()
        Xp[j] += h; Xm[j] -= h
        fp = _funciones(Xp, z, act, ('coord',0,X[0]))[:len(act)+1]
        fm = _funciones(Xm, z, act, ('coord',0,X[0]))[:len(act)+1]
        Jf[:, j] = (fp - fm)/(2*h)
    try:
        _, _, Vt = np.linalg.svd(Jf)
    except np.linalg.LinAlgError:
        return t_prev
    t = Vt[-1]; nrm = np.linalg.norm(t)
    if nrm < 1e-30: return t_prev
    t = t/nrm
    if t_prev is not None and np.dot(t, t_prev) < 0: t = -t
    return t


# ── Inicializacion Wilson (burbuja y rocio) ──────────────────────────────────
def _init_burbuja(z, act, P0):
    """Punto de burbuja a baja presion via Wilson. Retorna X o None."""
    m  = len(act)
    T0 = float(np.sum(z*np.array(TC))) * 0.6
    for _ in range(300):
        Kw = np.array([_Ki_wilson(i, T0, P0) for i in range(NC)])
        f  = np.sum(z*Kw) - 1.0
        df = sum(z[i]*Kw[i]*(WILSON_C*(1+OMEGA[i])*TC[i]/T0**2) for i in range(NC))
        if abs(df) < 1e-30: break
        Tn = T0 - f/df
        if Tn <= 0: Tn = T0*0.5
        if abs(Tn-T0) < 1e-8: T0 = Tn; break
        T0 = Tn
    Kw = np.array([_Ki_wilson(i, T0, P0) for i in act])
    X  = np.concatenate([np.log(Kw), [np.log(T0)], [np.log(P0)]])
    X, ok = _resolver_punto(X, z, act, ('coord', m+1, np.log(P0)))
    return X if ok else None


def _init_rocio(z, act):
    """
    Punto de ROCIO buscando la presion mas baja estable (EOS valida).
    Prueba desde 200 psi hacia abajo hasta encontrar convergencia.
    Devuelve (X, P_rocio) o (None, None).
    La continuacion desde este punto en ambas direcciones cubre toda la rama.
    """
    m  = len(act)
    # Probar presiones donde la EOS es estable (mas alto = mas estable)
    # Probar presiones BAJAS primero: encontrar la rama fria de rocio
    # (la que Trazo 1 no puede alcanzar), luego presiones altas.
    for P_try in [50, 60, 70, 80, 100, 120, 150, 200]:
        # Biseccion robusta para sum(z/K_wilson) = 1 en T fria
        Ta, Tb = 150.0, 440.0
        fa = sum(z[i]/_max(_Ki_wilson(i,Ta,P_try),1e-30) for i in range(NC)) - 1.0
        fb = sum(z[i]/_max(_Ki_wilson(i,Tb,P_try),1e-30) for i in range(NC)) - 1.0
        if fa*fb > 0: continue   # no hay cruce en este rango de T
        for _ in range(60):
            T0 = 0.5*(Ta+Tb)
            fm = sum(z[i]/_max(_Ki_wilson(i,T0,P_try),1e-30) for i in range(NC)) - 1.0
            if abs(fm)<1e-9 or (Tb-Ta)<0.01: break
            if fm*fa < 0: Tb = T0
            else: Ta = T0
        Kw_act  = np.array([_Ki_wilson(i, T0, P_try) for i in act])
        K_rocio = 1.0/np.where(Kw_act>1e-30, Kw_act, 1e-30)
        X = np.concatenate([np.log(K_rocio), [np.log(T0)], [np.log(P_try)]])
        X, ok = _resolver_punto(X, z, act, ('coord', m+1, np.log(P_try)),
                                tol=1e-7)
        if ok:
            return X, P_try
    return None, None


# ── Bucle de continuacion (reutilizable) ─────────────────────────────────────
def _trazar(X0, z, act, t0, max_pts, paso_ini=0.10,
            PASO_MIN=5e-4, PASO_MAX=0.10,
            p_stop_max=None, p_stop_min=None):
    """
    Traza la envolvente desde X0 en la direccion t0 hasta max_pts puntos.
    p_stop_max: detener cuando P > este valor (para trazo de rocio que no
                debe sobrepasar la zona ya cubierta por el trazo de burbuja).
    p_stop_min: detener cuando P < este valor (cierre por presion baja).
    Retorna (lista_de_pts, X_ultimo, min_sumK2, crit_punto).
    """
    m     = len(act)
    pts   = [(np.exp(X0[m+1]), np.exp(X0[m]))]
    X_prev = X0.copy()
    t      = t0.copy()

    min_sumK2  = float(np.sum(X0[:m]**2))
    crit_punto = (np.exp(X0[m+1]), np.exp(X0[m]))

    paso  = paso_ini
    fallos = 0

    for _ in range(max_pts):
        exito = False; paso_try = paso; Xn = None
        for _it in range(16):
            X_pred = X_prev + t * paso_try
            spec   = ('arc', t, X_prev.copy(), paso_try)
            Xn, ok = _resolver_punto(X_pred, z, act, spec)
            if ok:
                av = np.linalg.norm(Xn - X_prev)
                if 0.2*paso_try < av < 4*paso_try:
                    exito = True; break
            paso_try *= 0.5
            if paso_try < PASO_MIN: break

        if not exito:
            fallos += 1
            if fallos >= 3: break
            paso = PASO_MIN; continue
        fallos = 0

        pts.append((np.exp(Xn[m+1]), np.exp(Xn[m])))

        # Rastrear punto critico
        s = float(np.sum(Xn[:m]**2))
        if s < min_sumK2:
            min_sumK2  = s
            crit_punto = (np.exp(Xn[m+1]), np.exp(Xn[m]))

        t_new = _tangente(Xn, z, act, t_prev=t)
        if t_new is None: break
        cosang = float(np.dot(t, t_new))
        X_prev = Xn.copy(); t = t_new

        # Control adaptativo de paso segun curvatura local
        if cosang < 0.2:
            paso = max(paso_try*0.35, PASO_MIN)
        elif cosang < 0.7:
            paso = max(paso_try*0.7, PASO_MIN)
        else:
            paso = min(paso_try*1.25, PASO_MAX)

        Pn = np.exp(Xn[m+1])
        # Condicion de parada por presion
        if p_stop_max is not None and Pn > p_stop_max: break
        if p_stop_min is not None and Pn < p_stop_min: break

    return pts, X_prev, min_sumK2, crit_punto


# ── Punto de entrada principal ───────────────────────────────────────────────
def construir_envolvente(z, kij=None, progress_cb=None,
                         P_ini=14.7, max_pts=2000, paso_max=0.10):
    """
    Construye la envolvente completa por continuacion de Michelsen con
    inicializacion BIDIRECCIONAL:
      - Trazo 1 (desde burbuja): cubre burbuja + rama de rocio superior.
      - Trazo 2 (desde rocio):   cubre rama de rocio inferior (cola baja P).
    Ambos trazos se combinan en una sola curva continua.
    Retorna {'envolvente':[(P,T)...], 'critico':(Pc,Tc) o None}.
    """
    global kij_g
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    kij_g = kij
    z = np.array(z, dtype=float)

    act = [i for i in range(NC) if z[i] > 1e-8]
    if len(act) < 2: return {'envolvente': [], 'critico': None}
    m = len(act)

    # ── TRAZO 1: desde burbuja a baja P ─────────────────────────────────────
    X1 = _init_burbuja(z, act, P_ini)
    if X1 is None: return {'envolvente': [], 'critico': None}

    t1 = _tangente(X1, z, act)
    if t1 is None: return {'envolvente': [], 'critico': None}
    if t1[m+1] < 0: t1 = -t1   # apuntar hacia P creciente

    pts1, Xlast1, minK2_1, crit1 = _trazar(
        X1, z, act, t1,
        max_pts=max_pts,
        paso_ini=0.08, PASO_MAX=paso_max,
        p_stop_min=P_ini*0.95   # cierre cuando P vuelva a ser baja
    )
    if progress_cb:
        progress_cb(len(pts1))

    crit = crit1 if minK2_1 < 0.5 else None
    P_last1 = pts1[-1][0]   # presion del ultimo punto del trazo 1

    # ¿El Trazo 1 ya completó la envolvente (llegó a presión baja en rocío)?
    # Si si, no hace falta el Trazo 2 (evita trabajo redundante → más rápido).
    # Solo gases donde la continuación se atasca en el pliegue de rocío
    # (tipicamente ultralivianos) necesitan el Trazo 2.
    if P_last1 <= P_ini * 2.5:
        return {'envolvente': pts1, 'critico': crit}

    # ── TRAZO 2: bidireccional desde punto de rocio intermedio ─────────────
    # Inicializa en una presion intermedia (~200→20 psi) donde la EOS es
    # estable, luego traza en DOS direcciones:
    #   Trazo 2a (hacia arriba): conecta con el Trazo 1 (~P_last1)
    #   Trazo 2b (hacia abajo): extiende la cola de rocio hasta P baja
    # Esto evita el problema de cuencas de atraccion que ocurre al intentar
    # continuar la rama de rocio directamente desde P_last1 hacia abajo.
    X2, P_rocio = _init_rocio(z, act)
    pts2a = []; pts2b = []

    if X2 is not None:
        t2 = _tangente(X2, z, act)
        if t2 is not None:
            # Trazo 2a: subir hasta conectar con Trazo 1
            t2_up = t2 if t2[m+1] >= 0 else -t2
            pts2a, _, _, _ = _trazar(
                X2, z, act, t2_up,
                max_pts=max_pts//3,
                paso_ini=0.08, PASO_MAX=paso_max,
                p_stop_max=P_last1 * 1.15
            )
            # Trazo 2b: bajar hasta presion baja (cola de rocio)
            t2_dn = -t2_up   # dirección opuesta
            pts2b, _, _, _ = _trazar(
                X2, z, act, t2_dn,
                max_pts=max_pts//3,
                paso_ini=0.08, PASO_MAX=paso_max,
                p_stop_min=P_ini * 0.95
            )
            if progress_cb:
                progress_cb(len(pts1)+len(pts2a)+len(pts2b))

    # ── COMBINACION: Trazo1 + empalme + cola de rocio ────────────────────────
    # Orden de la curva resultante (continua):
    #   low-P burbuja → critico → rocio superior (Trazo1)
    #   → empalme via Trazo2a invertido (de P_last1 baja a P_rocio)
    #   → cola de rocio (Trazo2b de P_rocio baja a P baja)
    if pts2a or pts2b:
        # Trazo2a invertido: de su extremo alto (cerca P_last1) a P_rocio
        puente = list(reversed(pts2a)) if pts2a else []
        # Trazo2b directo: de P_rocio hacia presion baja
        cola   = pts2b if pts2b else []
        envolvente = pts1 + puente + cola
    else:
        envolvente = pts1

    return {'envolvente': envolvente, 'critico': crit}
