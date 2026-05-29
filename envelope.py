"""
Motor Envolvente de Fases — ThermoPhase
Réplica FIEL del método Excel (Ziervogel / continuación de curva)

CLAVE: durante el GoalSeek, la composición iterada (Yi para burbuja,
Xi para rocío) se mantiene FIJA. Solo después de converger el GoalSeek
se actualiza la composición. Esto es exactamente lo que hace el Excel
con O22:O34 (valores fijos) y la macro EncontrarPuntoSat*.

BURBUJA:
  Xi = zi (líquido = alimentación, fijo)
  Yi = normalize(zi*Ki) (vapor, iterado)
  Objetivo (R176): SUM(zi*Ki) = 1, donde Ki = phiL(Xi)/phiV(Yi)
  Wilson inicial: Yi = normalize(zi*Kw)

ROCÍO:
  Yi = zi (vapor = alimentación, fijo)
  Xi = normalize(zi/Ki) (líquido, iterado)
  Objetivo (R176): SUM(zi/Ki) = 1
  Wilson inicial: Xi = normalize(zi/Kw)
"""
import numpy as np
import copy
from engine3 import (
    NC, TC, PC, OMEGA, KIJ_DEFAULT,
    am, bm, AB, solve_Z, ln_phi_i
)

R_GAS = 10.7316
WILSON_C = np.log(10.0) * (7.0/3.0)   # ≈ 5.3727 (igual que el Excel)

# Límites físicos para el GoalSeek — evita divergencia cerca del punto crítico
_T_MIN = 20.0          # °R mínimo
_T_MAX_FACTOR = 1.8    # × Tc_max de la mezcla
_P_MIN = 1.0           # psia mínimo
_P_MAX_FACTOR = 1.5    # × Pc_max de la mezcla

def Ki_wilson(i, T, P):
    return (PC[i]/P) * np.exp(WILSON_C*(1+OMEGA[i])*(1-TC[i]/T))

# ── T inicial: Newton-Raphson sobre SUM(zi*Kw)=1 (columna D del Excel) ──
def T_inicial_wilson(z, P, T0=460.0, max_it=100):
    T = T0
    for _ in range(max_it):
        f = sum(z[i]*Ki_wilson(i, T, P) for i in range(NC)) - 1.0
        # derivada df/dT
        df = 0.0
        for i in range(NC):
            kw = Ki_wilson(i, T, P)
            df += z[i]*kw*(WILSON_C*(1+OMEGA[i])*TC[i]/T**2)
        if abs(df) < 1e-30: break
        T_new = T - f/df
        if T_new <= 0: T_new = T*0.5
        if abs(T_new - T) < 1e-6:
            return T_new
        T = T_new
    return T

# ── Coeficientes de fugacidad de una fase ────────────────────
def _phi(z_ph, T, P, vapor, kij):
    am_ = am(z_ph, T, kij); bm_ = bm(z_ph)
    ZV, ZL = solve_Z(*AB(am_, bm_, T, P))
    Z = ZV if vapor else ZL
    return [np.exp(np.clip(ln_phi_i(i,z_ph,T,P,Z,am_,bm_,kij),-500,500))
            for i in range(NC)]

def Ki_de_phi(x, y, T, P, kij):
    """Ki = phiL(x)/phiV(y)"""
    pv = _phi(y, T, P, True,  kij)
    pl = _phi(x, T, P, False, kij)
    return [pl[i]/pv[i] if pv[i]>1e-30 else 1.0 for i in range(NC)]

# ── Objetivos con composición FIJA ───────────────────────────
def obj_burbuja(T, P, z, y_fija, kij):
    """SUM(zi*Ki) - 1, con Xi=zi y Yi=y_fija (constante)."""
    Ki = Ki_de_phi(list(z), y_fija, T, P, kij)
    return sum(z[i]*Ki[i] for i in range(NC)) - 1.0, Ki

def obj_rocio(T, P, z, x_fija, kij):
    """SUM(zi/Ki) - 1, con Yi=zi y Xi=x_fija (constante)."""
    Ki = Ki_de_phi(x_fija, list(z), T, P, kij)
    return sum(z[i]/Ki[i] if Ki[i]>1e-30 else 1e30 for i in range(NC)) - 1.0, Ki

# ── GoalSeek tipo Excel: búsqueda LOCAL con método secante ───
def goalseek_secant(f, x0, tol=1e-7, max_it=120,
                    x_min=0.01, x_max=None):
    """
    Replica el GoalSeek de Excel: parte del valor actual x0 y busca
    la raíz MÁS CERCANA mediante secante. NO hace búsqueda global.
    x_min y x_max son límites físicos — rechaza resultados fuera de rango.
    """
    f0 = f(x0)
    if not np.isfinite(f0): return None
    if abs(f0) < tol: return x0

    dx = max(abs(x0)*0.02, 0.05)
    x_prev, f_prev = x0, f0
    x_cur = x0 + dx
    f_cur = f(x_cur)
    if not np.isfinite(f_cur):
        x_cur = x0 - dx
        f_cur = f(x_cur)
        if not np.isfinite(f_cur): return None

    for _ in range(max_it):
        if abs(f_cur) < tol:
            # Verificar límites físicos antes de aceptar
            if x_max is not None and x_cur > x_max: return None
            if x_cur < x_min: return None
            return x_cur
        denom = (f_cur - f_prev)
        if abs(denom) < 1e-30:
            break
        x_new = x_cur - f_cur*(x_cur - x_prev)/denom
        # Limitar saltos extremos (estabilidad)
        if x_new <= x_min:
            x_new = max(x_cur*0.5, x_min)
        if x_max is not None and x_new > x_max:
            x_new = min(x_cur*2.0, x_max)
        f_new = f(x_new)
        if not np.isfinite(f_new):
            x_new = (x_cur + x_prev)/2.0
            f_new = f(x_new)
            if not np.isfinite(f_new): return None
        x_prev, f_prev = x_cur, f_cur
        x_cur, f_cur = x_new, f_new

    # Verificar límites físicos en el resultado final
    if x_max is not None and x_cur > x_max: return None
    if x_cur < x_min: return None
    return x_cur if abs(f_cur) < tol*50 else None

# ── Encontrar punto de saturación (réplica EncontrarPuntoSat) ──
def encontrar_punto(tipo, var_it, T_in, P_in, comp_fija_in, z, kij,
                    tol_comp=1e-9, max_ext=1000):
    """
    tipo: 'B' burbuja, 'D' rocío
    var_it: 'P' (GoalSeek mueve P) o 'T' (GoalSeek mueve T)
    comp_fija_in: composición iterada inicial (Yi burbuja / Xi rocío)
    Retorna: (T, P, Ki, comp_fija, exito)
    Tolerancias iguales al Excel: tol_comp=1e-9, max_ext=1000
    """
    T, P = T_in, P_in
    comp = list(comp_fija_in)
    obj = obj_burbuja if tipo=='B' else obj_rocio
    comp_ant = None
    # Límites físicos basados en las propiedades críticas de la mezcla
    Tc_max = max(TC) * _T_MAX_FACTOR
    Pc_max = max(PC) * _P_MAX_FACTOR

    for _ in range(max_ext):
        # GoalSeek con composición FIJA y límites físicos
        if var_it == 'P':
            f = lambda p: obj(T, p, z, comp, kij)[0]
            P_new = goalseek_secant(f, P, x_min=_P_MIN, x_max=Pc_max)
            if P_new is None: return T, P, comp, comp, False
            P = P_new
        else:
            f = lambda t: obj(t, P, z, comp, kij)[0]
            T_new = goalseek_secant(f, T, x_min=_T_MIN, x_max=Tc_max)
            if T_new is None: return T, P, comp, comp, False
            T = T_new

        # Calcular Ki con el punto convergido
        _, Ki = obj(T, P, z, comp, kij)

        # Nueva composición iterada (O176:O188 del Excel)
        if tipo == 'B':
            # Yi = normalize(zi*Ki)
            s = sum(z[i]*Ki[i] for i in range(NC))
            if s <= 0: return T, P, comp, comp, False
            comp_new = [z[i]*Ki[i]/s for i in range(NC)]
        else:
            # Xi = normalize(zi/Ki)
            s = sum(z[i]/Ki[i] if Ki[i]>1e-30 else 1e30 for i in range(NC))
            if s <= 0: return T, P, comp, comp, False
            comp_new = [z[i]/Ki[i]/s if Ki[i]>1e-30 else 0 for i in range(NC)]

        # Convergencia en composición
        if comp_ant is not None:
            cambio = sum(abs(comp_new[i]-comp_ant[i]) for i in range(NC))
            if cambio < tol_comp:
                return T, P, Ki, comp_new, True

        comp_ant = list(comp_new)
        comp = comp_new

    return T, P, comp, comp, False

def calcular_beta(P1,T1,P2,T2):
    try:
        dLnT=np.log(T2/T1); dLnP=np.log(P2/P1)
    except: return 1.0
    if abs(dLnT)<1e-12: return 999.0
    if abs(dLnP)<1e-12: return 0.0
    return abs(dLnP/dLnT)

def crit_check(Ki, th=0.01):
    return sum((k-1)**2 for k in Ki) < th

# ══════════════════════════════════════════════════════════════
def _curva(tipo, z, kij, dT=5, dP=5, dT_min=2, dP_min=2,
           fac=0.5, tol=1e-9, max_it=1000,
           max_pts=500, crit_th=0.001, max_int=10, cb=None):

    P0 = 10.0
    # T inicial por Newton-Wilson (columna D del Excel)
    T0 = T_inicial_wilson(z, P0)
    if not np.isfinite(T0) or T0 <= 0:
        T0 = sum(z[i]*TC[i] for i in range(NC)) * (0.7 if tipo=='B' else 0.9)

    # Composición iterada inicial (Wilson)
    Kw = [Ki_wilson(i, T0, P0) for i in range(NC)]
    if tipo == 'B':
        s = sum(z[i]*Kw[i] for i in range(NC))
        comp0 = [z[i]*Kw[i]/s for i in range(NC)]   # Yi
    else:
        s = sum(z[i]/Kw[i] for i in range(NC))
        comp0 = [z[i]/Kw[i]/s for i in range(NC)]   # Xi

    # ── Punto 1: GoalSeek sobre T, P=10 ──────────────────────
    T1,P1,Ki1,comp1,ok = encontrar_punto(tipo,'T',T0,P0,comp0,z,kij,tol,max_it)
    if not ok: return [], False
    pts=[(P1,T1)]
    if cb: cb(len(pts))
    if crit_check(Ki1,crit_th): return pts,True

    # ── Punto 2: avanzar en T, GoalSeek sobre P ──────────────
    dT_a=dT; ok2=False; T2,P2,Ki2,comp2=T1,P1,Ki1,comp1
    for _ in range(max_int):
        r = encontrar_punto(tipo,'P',T1+dT_a,P1,list(comp1),z,kij,tol,max_it)
        T2,P2,Ki2,comp2,ok2 = r
        if ok2: break
        dT_a*=fac
        if dT_a<dT_min: return pts,True
    if not ok2: return pts,True

    beta=calcular_beta(P1,T1,P2,T2)
    vi='P' if beta>20 else 'T'
    vt='T' if vi=='P' else 'P'
    dT_d=1 if T2>=T1 else -1
    dP_d=1 if P2>=P1 else -1
    dT_a=dT; dP_a=dP

    pts.append((P2,T2))
    if cb: cb(len(pts))
    if crit_check(Ki2,crit_th): return pts,True
    Tp,Pp,Kp,compp=T2,P2,Ki2,comp2

    # ── Bucle principal ───────────────────────────────────────
    while len(pts)<max_pts:
        if crit_check(Kp,crit_th): return pts,True
        ok3=False; Tn,Pn,Kn,compn=Tp,Pp,Kp,compp
        for _ in range(max_int):
            Ta=Tp+dT_d*dT_a if vi=='T' else Tp
            Pa=Pp+dP_d*dP_a if vi=='P' else Pp
            r = encontrar_punto(tipo,vt,Ta,Pa,list(compp),z,kij,tol,max_it)
            Tn,Pn,Kn,compn,ok3 = r
            if ok3: break
            if vi=='T':
                dT_a*=fac
                if dT_a<dT_min: return pts,True
            else:
                dP_a*=fac
                if dP_a<dP_min: return pts,True
        if not ok3: return pts,True

        dT_a=dT; dP_a=dP
        dT_d=1 if Tn>=Tp else -1
        dP_d=1 if Pn>=Pp else -1
        beta2=calcular_beta(Pp,Tp,Pn,Tn)
        if beta2>20: vi='P'; vt='T'
        elif beta2<2: vi='T'; vt='P'

        pts.append((Pn,Tn))
        if cb: cb(len(pts))
        Tp,Pp,Kp,compp=Tn,Pn,Kn,compn

    return pts,False

def curva_burbuja(z,kij=None,progress_cb=None,**kw):
    if kij is None: kij=copy.deepcopy(KIJ_DEFAULT)
    return _curva('B',z,kij,cb=progress_cb,**kw)

def curva_rocio(z,kij=None,progress_cb=None,**kw):
    if kij is None: kij=copy.deepcopy(KIJ_DEFAULT)
    return _curva('D',z,kij,cb=progress_cb,**kw)

def curva_envolvente(z,kij=None,progress_cb=None):
    if kij is None: kij=copy.deepcopy(KIJ_DEFAULT)
    def cb_b(n):
        if progress_cb: progress_cb('burbuja',n)
    def cb_d(n):
        if progress_cb: progress_cb('rocio',n)
    pb,cb=curva_burbuja(z,kij,progress_cb=cb_b)
    pd,cd=curva_rocio(z,kij,progress_cb=cb_d)
    return {'burbuja':pb,'rocio':pd,'critico_burbuja':cb,'critico_rocio':cd}
