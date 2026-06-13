"""
Modelo de formacion de HIDRATOS — van der Waals-Platteeuw (1959)
con constantes de Langmuir de Munck et al. (1988) y fugacidades de la
ecuacion de Peng-Robinson (motor engine3). Metodo de "agua libre"
(free water): asume agua presente, actividad del agua a_w = 1.

Equilibrio de formacion (3 fases V-Aq-H):
    Δμ_w^H = Δμ_w^L
  lado hidrato:  Δμ_w^H/RT = -Σ_m ν_m · ln(1 - Σ_j θ_mj)
  ocupacion:     θ_mj = C_mj·f_j / (1 + Σ_k C_mk·f_k)
  Langmuir:      C_mj(T) = (A_mj/T)·exp(B_mj/T)   [1/atm]
  lado agua:     Δμ_w^L/RT = Δμ0/(R·T0) - ∫(Δh/RT²)dT + Δv·(P-P0)/(RT)

Se evalua estructura I y II; la estable es la de mayor T de formacion.
Unidades del motor: P en psia, T en °R. Internamente se trabaja en K y atm.
"""
import math
from engine3 import NC, am, bm, AB, solve_Z, ln_phi_i, KIJ_DEFAULT
import copy

# Conversiones
PSIA_PER_ATM = 14.6959
R_J = 8.314462          # J/mol/K
T0  = 273.15            # K (referencia)

# Indices de formadores en el motor
C1, C2, C3, IC4, NC4, N2, CO2 = 2, 3, 4, 5, 6, 0, 1

# ── Constantes de Langmuir Munck et al. (1988): C=(A/T)exp(B/T) [1/atm] ──
# Formato: idx_comp -> (A_peq, B_peq, A_gra, B_gra)   (A en K/atm)
LANGMUIR_I = {
    C1:  (0.7228e-3, 3187.0, 23.35e-3, 2653.0),
    C2:  (0.0,       0.0,     3.039e-3, 3861.0),
    CO2: (0.2474e-3, 3410.0, 42.46e-3, 2813.0),
    N2:  (1.617e-3,  2905.0,  6.078e-3, 2431.0),
}
LANGMUIR_II = {
    C1:  (0.2207e-3, 3453.0, 100.0e-3, 1916.0),
    C2:  (0.0,       0.0,    240.0e-3, 2967.0),
    C3:  (0.0,       0.0,      5.455e-3, 4638.0),
    IC4: (0.0,       0.0,    189.3e-3, 3800.0),
    NC4: (0.0,       0.0,     30.51e-3, 3699.0),
    CO2: (0.0845e-3, 3615.0, 851.0e-3, 2025.0),
    N2:  (0.1742e-3, 3082.0,  18.00e-3, 1728.0),
}

# Numero de cavidades por molecula de agua
NU_I  = (2.0/46.0,  6.0/46.0)     # (peq, grande) estructura I
NU_II = (16.0/136.0, 8.0/136.0)   # (peq, grande) estructura II

# Propiedades de referencia del agua (reticulo vacio -> agua liquida)
# Munck/Sloan. Δμ0,Δh0 en J/mol ; Δv en cm³/mol
REF_I  = dict(dmu0=1483.5, dh0=1389.0, dv=4.6, dCp=-38.12,
               dh0_hielo=-4620.0, dv_hielo=3.0)  # calib. Area Sur
REF_II = dict(dmu0=1123.5, dh0=1025.0, dv=5.0, dCp=-38.12,
               dh0_hielo=-4983.0, dv_hielo=3.4)  # calib. Area Sur


def _fugacidades_atm(z, T_R, P_psia, kij):
    """Fugacidades (atm) de cada componente en la fase gas via PR (raiz vapor)."""
    am_ = am(z, T_R, kij); bm_ = bm(z)
    A, B = AB(am_, bm_, T_R, P_psia)
    ZV, _ = solve_Z(A, B)
    f = [0.0]*NC
    for i in range(NC):
        if z[i] > 0:
            lnphi = ln_phi_i(i, z, T_R, P_psia, ZV, am_, bm_, kij)
            f[i] = z[i]*math.exp(lnphi)*P_psia / PSIA_PER_ATM
    return f


def _dmu_hidrato_RT(T_K, f_atm, langmuir, nu):
    """Δμ_w^H/RT por van der Waals-Platteeuw para una estructura."""
    suma = 0.0
    for m, (nu_m) in enumerate(nu):       # m=0 peq, m=1 grande
        occ = 0.0          # Σ_j θ_mj
        denom = 1.0
        Cf = {}
        for j,(Ap,Bp,Ag,Bg) in langmuir.items():
            A = Ap if m==0 else Ag
            B = Bp if m==0 else Bg
            if A <= 0:
                continue
            C = (A/T_K)*math.exp(B/T_K)    # 1/atm
            Cf[j] = C*f_atm[j]
            denom += Cf[j]
        for j in Cf:
            occ += Cf[j]/denom
        if occ >= 1.0:
            occ = 1.0 - 1e-12
        suma += nu_m * math.log(1.0 - occ)
    return -suma


def _dmu_agua_RT(T_K, P_atm, ref):
    """
    Δμ_w^L/RT con free water (a_w=1). Usa referencia de AGUA LIQUIDA por
    encima de 273.15K y de HIELO por debajo (distinta entalpia y volumen),
    igual que el metodo Ng-Robinson de HYSYS.
    """
    dmu0 = ref['dmu0']; dCp = ref['dCp']
    if T_K >= T0:
        dh0 = ref['dh0']; dv_m3 = ref['dv']*1e-6
    else:
        dh0 = ref['dh0_hielo']; dv_m3 = ref['dv_hielo']*1e-6
    term0 = dmu0/(R_J*T0)
    n = 30; integ = 0.0
    dT = (T_K - T0)/n if n else 0
    for k in range(n):
        Tk = T0 + (k+0.5)*dT
        dh = dh0 + dCp*(Tk - T0)
        integ += dh/(R_J*Tk*Tk)*dT
    P_pa = P_atm*101325.0
    termP = dv_m3*P_pa/(R_J*T_K)
    return term0 - integ + termP


def _delta(T_R, P_psia, z, kij, struct):
    """f(T) = Δμ_w^L/RT - Δμ_w^H/RT para una estructura. =0 en formacion."""
    T_K = T_R/1.8
    P_atm = P_psia/PSIA_PER_ATM
    f_atm = _fugacidades_atm(z, T_R, P_psia, kij)
    if struct == 'I':
        dmuH = _dmu_hidrato_RT(T_K, f_atm, LANGMUIR_I, NU_I)
        dmuL = _dmu_agua_RT(T_K, P_atm, REF_I)
    else:
        dmuH = _dmu_hidrato_RT(T_K, f_atm, LANGMUIR_II, NU_II)
        dmuL = _dmu_agua_RT(T_K, P_atm, REF_II)
    return dmuL - dmuH


def temperatura_formacion(z, P_psia, kij=None):
    """
    Temperatura de formacion de hidratos (°R) a la presion dada (psia).
    Retorna dict: {'T_R','T_F','estructura','forma_aqui'(None)} o None si no forma.
    Evalua estructura I y II; la estable es la de MAYOR T de formacion.
    """
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    z = list(z)
    res = {}
    for struct in ('I','II'):
        # Buscar T donde _delta=0, por biseccion en rango fisico (°R)
        # Rango: ~ -40°F a 90°F  -> °R 419 a 549
        Ta, Tb = 400.0, 580.0
        fa = _delta(Ta, P_psia, z, kij, struct)
        fb = _delta(Tb, P_psia, z, kij, struct)
        if fa*fb > 0:
            res[struct] = None
            continue
        for _ in range(80):
            Tm = 0.5*(Ta+Tb)
            fm = _delta(Tm, P_psia, z, kij, struct)
            if abs(fm) < 1e-9 or (Tb-Ta) < 1e-4: break
            if fm*fa < 0: Tb, fb = Tm, fm
            else: Ta, fa = Tm, fm
        res[struct] = 0.5*(Ta+Tb)
    # Elegir estructura estable = mayor T de formacion
    cand = {k:v for k,v in res.items() if v is not None}
    if not cand:
        return None
    estr = max(cand, key=cand.get)
    T_R = cand[estr]
    return {'T_R':T_R, 'T_F':T_R-459.67, 'estructura':estr,
            'T_R_I':res.get('I'), 'T_R_II':res.get('II')}


if __name__ == '__main__':
    # VALIDACION: hidrato de metano puro (patron conocido)
    # Ref: a 273.15K(491.7°R) P_form~2.6MPa~377psia ; 280K~4.3MPa ; 285K~6.5MPa
    z_met = [0]*NC; z_met[C1] = 1.0
    print("=== Validacion hidrato de METANO puro ===")
    print("(Ref conocida: ~377psia@32°F, ~625psia@50°F, ~1050psia@68°F)")
    for P in [377, 625, 1050, 1500]:
        r = temperatura_formacion(z_met, P)
        if r:
            print(f"P={P}psia -> T_form={r['T_F']:.1f}°F  estructura {r['estructura']}")
        else:
            print(f"P={P}psia -> no forma")


def presion_formacion(z, T_R, kij=None):
    """Presion de formacion de hidratos (psia) a la temperatura dada (°R).
    Retorna dict {'P_psia','estructura'} o None si no forma en rango."""
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    z = list(z)
    best = None
    for struct in ('I', 'II'):
        Pa, Pb = 1.0, 8000.0
        # _delta a P baja y P alta (a P alta forma -> delta cambia signo)
        fa = _delta(T_R, Pa, z, kij, struct)
        fb = _delta(T_R, Pb, z, kij, struct)
        if fa*fb > 0:
            continue
        for _ in range(80):
            Pm = 0.5*(Pa+Pb)
            fm = _delta(T_R, Pm, z, kij, struct)
            if abs(fm) < 1e-9 or (Pb-Pa) < 1e-4: break
            if fm*fa < 0: Pb, fb = Pm, fm
            else: Pa, fa = Pm, fm
        Pf = 0.5*(Pa+Pb)
        # Estructura estable = la que forma a MENOR presion
        if best is None or Pf < best[1]:
            best = (struct, Pf)
    if best is None:
        return None
    return {'P_psia': best[1], 'estructura': best[0]}


def evaluar(z, T_R, P_psia, kij=None):
    """
    Evaluacion completa para la condicion (T_R, P_psia):
      - T de formacion a la P dada
      - P de formacion a la T dada
      - estructura del hidrato
      - si en la condicion actual FORMA o NO forma hidrato
      - etiqueta de equilibrio (V-Aq-H)
    """
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    rt = temperatura_formacion(z, P_psia, kij)
    rp = presion_formacion(z, T_R, kij)
    out = {'T_form_R': None, 'T_form_F': None, 'P_form_psia': None,
           'estructura': None, 'forma': False, 'equilibrio': None}
    if rt:
        out['T_form_R'] = rt['T_R']; out['T_form_F'] = rt['T_F']
        out['estructura'] = rt['estructura']
        # Forma si la T actual es <= T de formacion (mas frio que el umbral)
        out['forma'] = (T_R <= rt['T_R'])
    if rp:
        out['P_form_psia'] = rp['P_psia']
        if out['estructura'] is None:
            out['estructura'] = rp['estructura']
    out['equilibrio'] = 'V-Aq-H' if out['forma'] else 'V-Aq'
    return out
