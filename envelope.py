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


# ── Factores de compresibilidad de las dos fases (para ϖ = |ZL-ZV|) ──
def _ZL_ZV(comp_x, comp_y, T, P, kij):
    """Devuelve (ZL, ZV) usando la composición líquida y vapor del punto."""
    amL = am(comp_x, T, kij); bmL = bm(comp_x)
    amV = am(comp_y, T, kij); bmV = bm(comp_y)
    ZVl, ZLl = solve_Z(*AB(amL, bmL, T, P))   # raíces con mezcla líquida
    ZVv, ZLv = solve_Z(*AB(amV, bmV, T, P))   # raíces con mezcla vapor
    return ZLl, ZVv

def _omega_crit(tipo, z, comp_fija, T, P, kij):
    """ϖ = |ZL - ZV| (criterio de Gallardo para cercanía al punto crítico)."""
    if tipo == 'B':
        x, y = list(z), comp_fija       # burbuja: líquido=z, vapor=iterado
    else:
        x, y = comp_fija, list(z)       # rocío: vapor=z, líquido=iterado
    try:
        ZL, ZV = _ZL_ZV(x, y, T, P, kij)
        return abs(ZL - ZV)
    except Exception:
        return 1.0   # si falla, asumir lejos del crítico

# ── GoalSeek por BISECCIÓN (sin derivada, robusto cerca del crítico) ──
def goalseek_biseccion(f, x0, x_min, x_max, tol=1e-5, max_it=80,
                       paso_grueso=None):
    """
    Encuentra la raíz de f por bisección. Primero busca un intervalo [a,b]
    con cambio de signo (búsqueda gruesa desde x0 hacia ambos lados),
    luego bisecta. No usa derivada: nunca diverge cerca del crítico.
    """
    f0 = f(x0)
    if not np.isfinite(f0): return None
    if abs(f0) < tol: return x0

    # Paso de búsqueda gruesa proporcional al valor actual
    if paso_grueso is None:
        paso_grueso = max(abs(x0)*0.05, 1.0)

    # Buscar intervalo con cambio de signo, expandiendo a ambos lados
    a, fa = x0, f0
    b, fb = None, None
    paso = paso_grueso
    for _ in range(35):
        # Probar hacia arriba
        xb = a + paso
        if xb <= x_max:
            fxb = f(xb)
            if np.isfinite(fxb) and fa*fxb < 0:
                b, fb = xb, fxb; break
        # Probar hacia abajo
        xb2 = a - paso
        if xb2 >= x_min:
            fxb2 = f(xb2)
            if np.isfinite(fxb2) and fa*fxb2 < 0:
                b, fb = a, fa
                a, fa = xb2, fxb2
                break
        paso *= 1.5
        if a+paso > x_max and a-paso < x_min:
            break
    if b is None:
        return None   # no se encontró cambio de signo

    # Asegurar a < b
    if a > b:
        a, b = b, a; fa, fb = fb, fa

    # Bisección
    for _ in range(max_it):
        m = 0.5*(a+b)
        fm = f(m)
        if not np.isfinite(fm): return None
        if abs(fm) < tol or (b-a) < 1e-6:
            return m
        if fa*fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

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
        # Detectar cercanía al punto crítico: ϖ = |ZL - ZV|
        omega_c = _omega_crit(tipo, z, comp, T, P, kij)
        cerca_critico = omega_c <= 0.2

        # GoalSeek con composición FIJA y límites físicos.
        # Cerca del crítico: BISECCIÓN (sin derivada, no diverge).
        # Lejos: SECANTE (rápido).
        if var_it == 'P':
            f = lambda p: obj(T, p, z, comp, kij)[0]
            if cerca_critico:
                P_new = goalseek_biseccion(f, P, _P_MIN, Pc_max)
            else:
                P_new = goalseek_secant(f, P, x_min=_P_MIN, x_max=Pc_max)
            if P_new is None: return T, P, comp, comp, False
            P = P_new
        else:
            f = lambda t: obj(t, P, z, comp, kij)[0]
            if cerca_critico:
                T_new = goalseek_biseccion(f, T, _T_MIN, Tc_max)
            else:
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
def _curva(tipo, z, kij, dT=5, dP=5, dT_min=0.3, dP_min=0.3,
           fac=0.5, tol=1e-9, max_it=400,
           max_pts=400, crit_th=0.0001, max_int=8, cb=None):

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


def _cerrar_envolvente(burb, rocio, n_interp=20):
    """
    Cierra visualmente la envolvente interpolando entre el último punto
    de burbuja y el último punto de rocío, pasando por un punto crítico
    estimado por extrapolación de las tendencias de ambas curvas.

    Los puntos interpolados se agregan a AMBAS curvas (mitad a cada una)
    para que la envolvente quede cerrada con puntos idénticos a los
    calculados. Retorna (burb_cerrada, rocio_cerrada).
    """
    if len(burb) < 3 or len(rocio) < 3:
        return burb, rocio

    # Extremos abiertos (últimos puntos calculados de cada rama)
    Pb_end, Tb_end = burb[-1]
    Pd_end, Td_end = rocio[-1]

    import math
    dist = math.sqrt((Pb_end-Pd_end)**2 + (Tb_end-Td_end)**2)
    # Si los extremos ya están casi pegados, no hace falta cerrar
    if dist < 1.0:
        return burb, rocio

    # Estimar punto crítico por extrapolación de las tangentes de cada rama.
    # Tangente de burbuja (dirección de los últimos 2-3 puntos)
    def tangente(pts, n=3):
        n = min(n, len(pts))
        p0 = pts[-n]; p1 = pts[-1]
        dP = p1[0]-p0[0]; dT = p1[1]-p0[1]
        norm = math.sqrt(dP*dP + dT*dT)
        if norm < 1e-9: return (0.0, 0.0)
        return (dP/norm, dT/norm)

    tb = tangente(burb)   # dirección saliente de burbuja
    td = tangente(rocio)  # dirección saliente de rocío

    # Punto crítico estimado: intersección aproximada de las dos tangentes
    # extendidas desde los extremos. Si no se cruzan limpiamente, usar
    # el punto medio elevado (el crítico suele estar por encima de ambos
    # extremos en una envolvente típica).
    Pc_est = None; Tc_est = None
    # Resolver: Pb_end + s*tb = Pd_end + u*td  (en plano P-T)
    denom = tb[0]*(-td[1]) - tb[1]*(-td[0])
    if abs(denom) > 1e-9:
        rhsP = Pd_end - Pb_end
        rhsT = Td_end - Tb_end
        s = (rhsP*(-td[1]) - rhsT*(-td[0])) / denom
        # Punto de intersección sobre la tangente de burbuja
        Pc_int = Pb_end + s*tb[0]
        Tc_int = Tb_end + s*tb[1]
        # Aceptar solo si la intersección está "adelante" y es razonable
        # (no demasiado lejos de los extremos)
        max_reach = dist * 2.5
        if (s > 0 and
            math.sqrt((Pc_int-Pb_end)**2+(Tc_int-Tb_end)**2) < max_reach):
            Pc_est, Tc_est = Pc_int, Tc_int

    if Pc_est is None:
        # Respaldo: punto medio entre extremos, ligeramente elevado en P
        Pc_est = (Pb_end + Pd_end)/2.0
        Tc_est = (Tb_end + Td_end)/2.0
        # Elevar un poco hacia donde apuntan las tangentes promedio
        Pc_est += (tb[0]+td[0])/2.0 * dist*0.4
        Tc_est += (tb[1]+td[1])/2.0 * dist*0.4

    # Generar curva suave: burbuja_end → crítico → rocío_end
    # mediante Bézier cuadrática con el crítico como punto de control.
    def bezier(p0, pc, p1, t):
        mt = 1-t
        x = mt*mt*p0[0] + 2*mt*t*pc[0] + t*t*p1[0]
        y = mt*mt*p0[1] + 2*mt*t*pc[1] + t*t*p1[1]
        return (x, y)

    p0 = (Pb_end, Tb_end)
    pc = (Pc_est, Tc_est)
    p1 = (Pd_end, Td_end)

    # Puntos interpolados (excluyendo extremos que ya existen)
    interp = []
    for k in range(1, n_interp):
        t = k/float(n_interp)
        P_i, T_i = bezier(p0, pc, p1, t)
        interp.append((P_i, T_i))

    # Repartir: primera mitad se agrega a burbuja, segunda mitad a rocío
    # (invertida para mantener continuidad), de modo que ambas curvas
    # avancen hacia el crítico con puntos idénticos.
    mid = len(interp)//2
    burb_extra  = interp[:mid]
    rocio_extra = list(reversed(interp[mid:]))

    burb_cerrada  = list(burb)  + burb_extra
    rocio_cerrada = list(rocio) + rocio_extra
    return burb_cerrada, rocio_cerrada



def propiedades_punto(T, P, x, y, kij=None, metodo_densidad='COSTALD'):
    """
    Calcula propiedades de las fases vapor y líquido en un punto (T,P) dado.
    Retorna dict con densidades, Z, PM y SG de cada fase.
    """
    from engine3 import (am as _am, bm as _bm, AB as _AB, solve_Z as _solveZ,
                          PM as _PM, costald_Vs, R_GAS as _R)
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    out = {}

    # Peso molecular de cada fase
    PM_v = sum(y[i]*_PM[i] for i in range(NC))
    PM_l = sum(x[i]*_PM[i] for i in range(NC))
    out['PM_v'] = PM_v; out['PM_l'] = PM_l

    # Vapor — siempre por EOS
    am_v = _am(y,T,kij); bm_v = _bm(y)
    ZV,_ = _solveZ(*_AB(am_v,bm_v,T,P))
    out['ZV'] = ZV
    out['rho_v'] = P*PM_v/(ZV*_R*T) if ZV>0 else None
    out['sg_v']  = PM_v/28.9625

    # Líquido — por COSTALD o EOS según método
    am_l = _am(x,T,kij); bm_l = _bm(x)
    _,ZL = _solveZ(*_AB(am_l,bm_l,T,P))
    if metodo_densidad == 'COSTALD':
        Vs = costald_Vs(x, T)
        if Vs and Vs > 0:
            out['rho_l'] = PM_l/Vs
            out['ZL'] = P*Vs/(_R*T)
        else:
            out['rho_l'] = P*PM_l/(ZL*_R*T) if ZL>0 else None
            out['ZL'] = ZL
    else:
        out['rho_l'] = P*PM_l/(ZL*_R*T) if ZL>0 else None
        out['ZL'] = ZL
    out['sg_l'] = (out['rho_l']/62.4) if out.get('rho_l') else None

    return out


def T_wilson_rocio(z, P, T0=500.0, max_it=200):
    """T inicial que resuelve SUM(zi/Kw)=1 (rocío) por Newton."""
    T = T0
    for _ in range(max_it):
        f = sum(z[i]/Ki_wilson(i,T,P) if z[i]>0 else 0 for i in range(NC)) - 1.0
        df = 0.0
        for i in range(NC):
            if z[i]==0: continue
            kw = Ki_wilson(i,T,P)
            # d(1/Kw)/dT = -(1/Kw)*(WILSON_C*(1+w)*Tc/T^2)
            df += -z[i]/kw*(WILSON_C*(1+OMEGA[i])*TC[i]/T**2)
        if abs(df) < 1e-30: break
        T_new = T - f/df
        if T_new <= 0: T_new = T*0.5
        if abs(T_new-T) < 1e-6: return T_new
        T = T_new
    return T

def P_wilson_sat(z, T, tipo, P0=100.0, max_it=200):
    """P inicial que resuelve la ecuación de Wilson de saturación por Newton.
    tipo 'B': SUM(zi*Kw)=1 ; tipo 'D': SUM(zi/Kw)=1."""
    P = P0
    for _ in range(max_it):
        if tipo=='B':
            f = sum(z[i]*Ki_wilson(i,T,P) for i in range(NC)) - 1.0
            # dKw/dP = -Kw/P  → d(SUM z*Kw)/dP = -SUM(z*Kw)/P
            df = -sum(z[i]*Ki_wilson(i,T,P) for i in range(NC))/P
        else:
            f = sum(z[i]/Ki_wilson(i,T,P) if z[i]>0 else 0 for i in range(NC)) - 1.0
            # d(1/Kw)/dP = (1/Kw)/P  → d(SUM z/Kw)/dP = SUM(z/Kw)/P
            df = sum(z[i]/Ki_wilson(i,T,P) if z[i]>0 else 0 for i in range(NC))/P
        if abs(df) < 1e-30: break
        P_new = P - f/df
        if P_new <= 0.1: P_new = P*0.5
        if abs(P_new-P) < 1e-4: return P_new
        P = P_new
    return max(P, 1.0)

def _comp_inicial(tipo, z, T, P):
    """Composición iterada inicial (Yi rocío / Xi rocío) según Wilson."""
    Kw = [Ki_wilson(i, max(T,50), P) for i in range(NC)]
    if tipo=='B':
        s = sum(z[i]*Kw[i] for i in range(NC))
        return [z[i]*Kw[i]/s for i in range(NC)]
    else:
        s = sum(z[i]/Kw[i] if Kw[i]>1e-30 else 1e30 for i in range(NC))
        return [z[i]/Kw[i]/s if Kw[i]>1e-30 else 0 for i in range(NC)]

def _es_trivial(Ki, umbral=0.5):
    """Detecta solución trivial (todos los Ki cercanos a 1)."""
    return sum((k-1)**2 for k in Ki) < umbral

def _resolver_con_reintentos(tipo, var_it, val_fijo, z, kij,
                             arranques, comp_func):
    """
    Intenta encontrar el punto de saturación desde varios valores de arranque.
    Acepta el primer resultado que converja y NO sea solución trivial.
    tipo: 'B'/'D'; var_it: 'T'/'P'; val_fijo: la condición fija (P si var_it='T').
    arranques: lista de valores iniciales para la variable a iterar.
    comp_func(T,P): retorna la composición inicial.
    Retorna (T, P, Ki, comp, exito).
    """
    mejor = None
    for v0 in arranques:
        if var_it == 'T':
            T_in, P_in = v0, val_fijo
        else:
            T_in, P_in = val_fijo, v0
        comp0 = comp_func(T_in, P_in)
        T,P,Ki,comp,ok = encontrar_punto(tipo, var_it, T_in, P_in, comp0,
                                         z, kij, tol_comp=1e-11, max_ext=2000)
        if ok and not _es_trivial(Ki):
            return T, P, Ki, comp, True
    # Si ningún arranque dio solución NO trivial, no hay punto de saturación
    return (val_fijo if var_it=='P' else 0.0,
            val_fijo if var_it=='T' else 0.0,
            [1.0]*NC, list(z), False)

def punto_saturacion(tipo_calc, valor, z, kij=None):
    """
    Calcula un punto de saturación individual.
    tipo_calc: 'T_rocio', 'T_burbuja', 'P_rocio', 'P_burbuja'
    valor: la condición fija (P en psi para T_*, T en °R para P_*)
    Retorna dict con: T (°R), P (psi), x (líquido), y (vapor), Ki, exito
    """
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)

    if tipo_calc == 'T_rocio':
        P = valor
        # Wilson correcto para rocío: SUM(z/Kw)=1
        T_w = T_wilson_rocio(z, P)
        # Arranques múltiples alrededor del estimado Wilson
        arranques = [T_w, T_w+40, T_w-40, T_w+80, T_w-80, 450, 550, 650]
        T,P,Ki,comp,ok = _resolver_con_reintentos(
            'D','T',P,z,kij,arranques,
            lambda Ti,Pi: _comp_inicial('D',z,Ti,Pi))
        props = propiedades_punto(T,P,comp,list(z),kij) if ok else {}
        return {'T':T,'P':P,'y':list(z),'x':comp,'Ki':Ki,'exito':ok,'props':props}

    elif tipo_calc == 'T_burbuja':
        P = valor
        # Wilson correcto para burbuja: SUM(z*Kw)=1
        T_w = T_inicial_wilson(z, P)
        arranques = [T_w, T_w+40, T_w-40, T_w+80, T_w-80, 350, 450, 550]
        T,P,Ki,comp,ok = _resolver_con_reintentos(
            'B','T',P,z,kij,arranques,
            lambda Ti,Pi: _comp_inicial('B',z,Ti,Pi))
        props = propiedades_punto(T,P,list(z),comp,kij) if ok else {}
        return {'T':T,'P':P,'x':list(z),'y':comp,'Ki':Ki,'exito':ok,'props':props}

    elif tipo_calc == 'P_rocio':
        T = valor
        # Wilson correcto para rocío sobre P: SUM(z/Kw)=1
        P_w = P_wilson_sat(z, T, 'D')
        arranques = [P_w, P_w*1.5, P_w*0.6, P_w*2.5, P_w*0.3, 100, 400, 800]
        T,P,Ki,comp,ok = _resolver_con_reintentos(
            'D','P',T,z,kij,arranques,
            lambda Ti,Pi: _comp_inicial('D',z,Ti,Pi))
        props = propiedades_punto(T,P,comp,list(z),kij) if ok else {}
        return {'T':T,'P':P,'y':list(z),'x':comp,'Ki':Ki,'exito':ok,'props':props}

    elif tipo_calc == 'P_burbuja':
        T = valor
        # Wilson correcto para burbuja sobre P: SUM(z*Kw)=1
        P_w = P_wilson_sat(z, T, 'B')
        arranques = [P_w, P_w*1.5, P_w*0.6, P_w*2.5, P_w*0.3, 100, 400, 800]
        T,P,Ki,comp,ok = _resolver_con_reintentos(
            'B','P',T,z,kij,arranques,
            lambda Ti,Pi: _comp_inicial('B',z,Ti,Pi))
        props = propiedades_punto(T,P,list(z),comp,kij) if ok else {}
        return {'T':T,'P':P,'x':list(z),'y':comp,'Ki':Ki,'exito':ok,'props':props}

    return None


def _cerrar_por_interseccion(burb, rocio):
    """
    Cierra la envolvente uniendo los extremos REALES de ambas ramas.
    Cuando ambas ramas avanzaron hasta cerca del punto crítico (gracias a
    la bisección), sus extremos quedan próximos y se conectan con un único
    segmento recto. Es el caso de 'intersección de las curvas de burbuja y
    rocío' que describen los papers — sin Bézier ni puntos inventados.
    El extremo de la rama de rocío se agrega al final de la de burbuja para
    que el trazado quede cerrado.
    """
    if len(burb) < 1 or len(rocio) < 1:
        return burb, rocio
    # Punto crítico aproximado = punto medio de los dos extremos reales
    Pb, Tb = burb[-1]
    Pd, Td = rocio[-1]
    Pc = 0.5*(Pb + Pd); Tc = 0.5*(Tb + Td)
    crit = (Pc, Tc)
    # Ambas ramas terminan en el mismo punto crítico común
    return burb + [crit], rocio + [crit]

def _cerrar_con_michelsen(burb, rocio, z, kij):
    """
    Rellena el hueco que Ziervogel deja en el apice con puntos REALES
    calculados por Michelsen (en vez del cierre geometrico por interseccion).
    Corre la envolvente de Michelsen, extrae el arco del apice comprendido
    entre las dos puntas de Ziervogel, y lo divide en lado-burbuja / lado-rocio
    por la cricondenbarica (presion maxima del arco) para colorearlo coherente.
    Si Michelsen falla o el hueco es despreciable, recurre al cierre geometrico.
    """
    import math
    if len(burb) < 2 or len(rocio) < 2:
        return burb, rocio
    Pb, Tb = burb[-1]; Pd, Td = rocio[-1]
    gap = math.hypot(Pb-Pd, Tb-Td)
    if gap < 3.0:
        # Ya practicamente cerrada: union directa de puntas
        return _cerrar_por_interseccion(burb, rocio)

    try:
        from envelope_michelsen import construir_envolvente
        # Paso grueso: solo necesitamos el arco del ápice para rellenar,
        # no el detalle fino. Esto acelera bastante el relleno.
        r = construir_envolvente(list(z), kij, paso_max=0.15)
        env = r.get('envolvente', [])
        if not env:
            return _cerrar_por_interseccion(burb, rocio)
    except Exception:
        return _cerrar_por_interseccion(burb, rocio)

    def closest(P, T):
        return min(range(len(env)),
                   key=lambda i: (env[i][0]-P)**2 + (env[i][1]-T)**2)
    i_b = closest(Pb, Tb); i_d = closest(Pd, Td)
    lo, hi = min(i_b, i_d), max(i_b, i_d)
    arco = env[lo:hi+1]
    if len(arco) < 2:
        return _cerrar_por_interseccion(burb, rocio)

    # Dividir el arco por su punto de presion maxima (cricondenbarica):
    # tramo ascendente -> lado burbuja, descendente -> lado rocio.
    imax = max(range(len(arco)), key=lambda i: arco[i][0])
    side1 = arco[:imax+1]
    side2 = arco[imax+1:]
    # Asignar el tramo que arranca cerca de la punta de burbuja a la burbuja
    d_b = math.hypot(arco[0][0]-Pb, arco[0][1]-Tb)
    d_d = math.hypot(arco[0][0]-Pd, arco[0][1]-Td)
    if d_b <= d_d:
        return burb + side1, rocio + side2
    else:
        return burb + side2, rocio + side1


def curva_envolvente(z,kij=None,progress_cb=None):
    if kij is None: kij=copy.deepcopy(KIJ_DEFAULT)
    def cb_b(n):
        if progress_cb: progress_cb('burbuja',n)
    def cb_d(n):
        if progress_cb: progress_cb('rocio',n)
    pb,cb=curva_burbuja(z,kij,progress_cb=cb_b)
    pd,cd=curva_rocio(z,kij,progress_cb=cb_d)
    # Rellenar el hueco del ápice con puntos REALES de Michelsen (no geométrico)
    pb,pd = _cerrar_con_michelsen(pb, pd, z, kij)
    return {'burbuja':pb,'rocio':pd,'critico_burbuja':cb,'critico_rocio':cd}
