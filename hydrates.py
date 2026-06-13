"""
Formacion de HIDRATOS — metodo Ng-Robinson RIGUROSO (free water).
Basado en los papers originales del autor:
  - Ng & Robinson (1976,1977,1980), Parrish & Prausnitz (1972),
    van der Waals & Platteeuw (1959), Holder/Saito.

Constantes de Langmuir por INTEGRACION del potencial de Kihara (no
correlaciones): C_ki = (4π/kT)∫₀^(Rc-a) exp(-W(r)/kT) r² dr.
Parametros de Kihara: Tabla 2 (Ng-Robinson 1977).
Lado del agua relativo al HIELO (Tabla 3.6) con switch hielo/liquido.
Modificacion Ng-Robinson con parametro de interaccion α (Tabla III).
Fugacidades de la fase gas por Peng-Robinson (engine3).

Unidades del motor: P en psia, T en °R. Internamente K, atm, cal/mol, CGS.
"""
import math, copy
from engine3 import NC, am, bm, AB, solve_Z, ln_phi_i, KIJ_DEFAULT

# ── Constantes fisicas / conversiones ──
KB = 1.380649e-16        # erg/K
ATM_ERG = 1.01325e6      # erg/cm³ por atm
A2CM = 1e-8              # Å -> cm
PSIA_PER_ATM = 14.6959
R_CAL = 1.987            # cal/mol/K
RP = 82.06              # atm·cm³/mol/K
T0 = 273.15             # K

# Indices de formadores en engine
N2, CO2, C1, C2, C3, IC4, NC4 = 0, 1, 2, 3, 4, 5, 6

# ── Kihara Tabla 2 (Ng-Robinson 1977): idx -> (a_core[Å], sigma[Å], eps/k[K]) ──
# a_core = (2a)/2 de la tabla
KIHARA = {
    C1:  (0.600/2, 3.2536, 152.68),
    C2:  (0.780/2, 3.3920, 174.04),
    C3:  (1.340/2, 3.2296, 213.58),
    IC4: (1.580/2, 2.9878, 252.54),
    NC4: (1.500/2, 3.4862, 169.41),
    N2:  (0.700/2, 3.2067, 128.51),
    CO2: (1.440/2, 2.9558, 169.52),
}
# Geometria de celda Parrish-Prausnitz: struct -> {cav: (Rc[Å], z)}
CELDA = {
    'I':  {'peq': (3.906, 20), 'gra': (4.326, 24)},
    'II': {'peq': (3.902, 20), 'gra': (4.683, 28)},
}
# Compatibilidad huesped-cavidad por tamaño (Sloan): que formador entra donde.
# C2 no entra en cavidades chicas (5¹²); C3/iC4/nC4 solo en sII grande (5¹²6⁴).
COMPAT = {
    'I':  {'peq': (N2, CO2, C1),        'gra': (N2, CO2, C1, C2)},
    'II': {'peq': (N2, CO2, C1),        'gra': (N2, CO2, C1, C2, C3, IC4, NC4)},
}
# Numero de cavidades por molecula de agua (vdW-P)
NU = {'I':  {'peq': 1.0/23.0, 'gra': 3.0/23.0},
      'II': {'peq': 2.0/17.0, 'gra': 1.0/17.0}}

# Propiedades de referencia del reticulo vacio relativas al HIELO (Tabla 3.6)
# dmu0,dh0 en cal/mol ; dv en cm³/mol
REF = {'I':  dict(dmu0=289.99, dh0_ice=275.0, dv=3.0),   # anclado a metano
       'II': dict(dmu0=206.04, dh0_ice=193.0, dv=3.4)}  # anclado a gas natural
DHF = 1435.94            # cal/mol calor de fusion hielo->liquido
DCP_LIQ = -9.11          # cal/mol/K (empty-liquido, ~ -38.12 J)

# Parametro de interaccion Ng-Robinson α (Tabla III) respecto al mas volatil
# (metano en gas natural). idx componente -> α
ALFA_NR = {C2: 1.02, C3: 1.00, IC4: 1.02, N2: 1.03, NC4: 1.00, CO2: 1.00}


def _W(r, Rc, z, eps_erg, sig, a):
    """Potencial de celda W(r) (PRO/II Eq 6-7), todo en cm/erg."""
    def d(N):
        return ((1 - r/Rc - a/Rc)**(-N) - (1 + r/Rc - a/Rc)**(-N))/N
    t1 = (sig**12)/(Rc**11 * r)*(d(10) + (a/Rc)*d(11))
    t2 = (sig**6)/(Rc**5 * r)*(d(4) + (a/Rc)*d(5))
    return 2*z*eps_erg*(t1 - t2)


def _langmuir(comp, struct, cav, T_K):
    """Constante de Langmuir C_ki por integracion de Kihara (1/atm)."""
    if comp not in KIHARA:
        return 0.0
    a_A, sig_A, eps_k = KIHARA[comp]
    Rc_A, z = CELDA[struct][cav]
    a = a_A*A2CM; sig = sig_A*A2CM; Rc = Rc_A*A2CM
    eps_erg = eps_k*KB; kT = KB*T_K
    Rmax = (Rc_A - a_A)*A2CM
    n = 60; integ = 0.0; dr = Rmax/n
    for i in range(n):
        r = (i+0.5)*dr
        integ += math.exp(-_W(r, Rc, z, eps_erg, sig, a)/kT)*r*r*dr
    return (4*math.pi/kT)*integ*ATM_ERG


def _fug_atm(z, T_R, P_psia, kij):
    """Fugacidades (atm) de la fase gas via PR (raiz vapor)."""
    am_ = am(z, T_R, kij); bm_ = bm(z)
    A, B = AB(am_, bm_, T_R, P_psia)
    ZV, _ = solve_Z(A, B)
    f = [0.0]*NC
    for i in range(NC):
        if z[i] > 0:
            f[i] = z[i]*math.exp(ln_phi_i(i, z, T_R, P_psia, ZV, am_, bm_, kij))*P_psia/PSIA_PER_ATM
    return f


def _dmu_H_RT(z, f_atm, T_K, struct):
    """Δμ_w^H/RT (lado hidrato, vdW-P) con modificacion Ng-Robinson."""
    suma = 0.0
    for cav in ('peq', 'gra'):
        sumCf = 0.0
        for j in COMPAT[struct][cav]:
            if f_atm[j] > 0:
                sumCf += _langmuir(j, struct, cav, T_K)*f_atm[j]
        suma += NU[struct][cav]*math.log(1.0 + sumCf)
    # Modificacion Ng-Robinson: producto sobre formadores (≠ referencia)
    prod = 1.0
    for j, a in ALFA_NR.items():
        if a != 1.0 and z[j] > 0:
            y = z[j]
            prod *= (1 + 3*(a-1)*y*y - 2*(a-1)*y*y*y)
    return prod*suma


def _dmu_L_RT(T_K, P_atm, struct):
    """Δμ_w^L/RT (lado agua, relativo al hielo, free water x_w=1)."""
    r = REF[struct]
    dmu0 = r['dmu0']; dv = r['dv']
    if T_K >= T0:
        dh0 = r['dh0_ice'] - DHF; dCp = DCP_LIQ
    else:
        dh0 = r['dh0_ice']; dCp = 0.0
    term0 = dmu0/(R_CAL*T0)
    n = 40; integ = 0.0; dT = (T_K - T0)/n
    for k in range(n):
        Tk = T0 + (k+0.5)*dT
        integ += (dh0 + dCp*(Tk - T0))/(R_CAL*Tk*Tk)*dT
    termP = dv*P_atm/(RP*T_K)
    return term0 - integ + termP


def _delta(z, T_R, P_psia, kij, struct):
    """Δμ_w^L/RT - Δμ_w^H/RT (=0 en formacion; <0 forma)."""
    T_K = T_R/1.8; P_atm = P_psia/PSIA_PER_ATM
    f = _fug_atm(z, T_R, P_psia, kij)
    return _dmu_L_RT(T_K, P_atm, struct) - _dmu_H_RT(z, f, T_K, struct)


def temperatura_formacion(z, P_psia, kij=None):
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    z = list(z); res = {}
    for struct in ('I', 'II'):
        Ta, Tb = 380.0, 600.0
        fa = _delta(z, Ta, P_psia, kij, struct)
        fb = _delta(z, Tb, P_psia, kij, struct)
        if fa*fb > 0:
            res[struct] = None; continue
        for _ in range(80):
            Tm = 0.5*(Ta+Tb); fm = _delta(z, Tm, P_psia, kij, struct)
            if abs(fm) < 1e-9 or (Tb-Ta) < 1e-4: break
            if fm*fa < 0: Tb = Tm
            else: Ta, fa = Tm, fm
        res[struct] = 0.5*(Ta+Tb)
    cand = {k: v for k, v in res.items() if v is not None}
    if not cand: return None
    estr = max(cand, key=cand.get)        # estable = mayor T de formacion
    T_R = cand[estr]
    return {'T_R': T_R, 'T_F': T_R-459.67, 'estructura': estr,
            'T_R_I': res.get('I'), 'T_R_II': res.get('II')}


def presion_formacion(z, T_R, kij=None):
    """Presion de formacion de hidratos (psia) a la T dada (°R)."""
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    z = list(z); best = None
    for struct in ('I', 'II'):
        Pa, Pb = 1.0, 8000.0
        fa = _delta(z, T_R, Pa, kij, struct)
        fb = _delta(z, T_R, Pb, kij, struct)
        if fa*fb > 0: continue
        for _ in range(80):
            Pm = 0.5*(Pa+Pb); fm = _delta(z, T_R, Pm, kij, struct)
            if abs(fm) < 1e-9 or (Pb-Pa) < 1e-4: break
            if fm*fa < 0: Pb = Pm
            else: Pa, fa = Pm, fm
        Pf = 0.5*(Pa+Pb)
        if best is None or Pf < best[1]:   # estable = forma a menor P
            best = (struct, Pf)
    if best is None: return None
    return {'P_psia': best[1], 'estructura': best[0]}


def evaluar(z, T_R, P_psia, kij=None):
    """Evaluacion completa: T y P de formacion, estructura, equilibrio,
    y si en la condicion (T_R,P_psia) forma o no hidrato."""
    if kij is None: kij = copy.deepcopy(KIJ_DEFAULT)
    rt = temperatura_formacion(z, P_psia, kij)
    rp = presion_formacion(z, T_R, kij)
    out = {'T_form_R': None, 'T_form_F': None, 'P_form_psia': None,
           'estructura': None, 'forma': False, 'equilibrio': None}
    if rt:
        out['T_form_R'] = rt['T_R']; out['T_form_F'] = rt['T_F']
        out['estructura'] = rt['estructura']
        out['forma'] = (T_R <= rt['T_R'])     # mas frio que el umbral
    if rp:
        out['P_form_psia'] = rp['P_psia']
        if out['estructura'] is None: out['estructura'] = rp['estructura']
    out['equilibrio'] = 'V-Aq-H' if out['forma'] else 'V-Aq'
    return out


if __name__ == '__main__':
    # Validacion metano puro (sin calibracion)
    z = [0.0]*NC; z[C1] = 1.0
    print("=== Metano puro (ref exp: ~384psia@32°F, ~625@50°F, ~1050@68°F) ===")
    for P in [384, 625, 1050]:
        r = temperatura_formacion(z, float(P))
        print(f"  P={P}: T_form={r['T_F']:.1f}°F  estr {r['estructura']}")
    # Gas Area Sur
    z = [0.005528,0.017681,0.926347,0.036675,0.008406,0.001498,0.001606,
         0.000745,0.000422,0.000531,0.000353,0.000162,0.000046]
    print("\n=== Gas Area Sur (HYSYS: 600psia -> 511.11°R / 51.44°F) ===")
    r = temperatura_formacion(z, 600.0)
    print(f"  P=600: T_form={r['T_R']:.2f}°R ({r['T_F']:.2f}°F) estr {r['estructura']}"
          f"  err {r['T_R']-511.1079:+.2f}°R")
    print(f"  (sI={r['T_R_I']:.1f}°R, sII={r['T_R_II']:.1f}°R)")
