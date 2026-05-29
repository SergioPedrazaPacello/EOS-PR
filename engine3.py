"""
Motor Peng-Robinson — réplica EXACTA del Excel
Algoritmo: Análisis de Estabilidad Michelsen + Flash Muskat-McDowell
Determinación de fase única por criterio ΣKi·zi vs Σzi/Ki (mismo que Excel)
"""
import numpy as np
import copy

R_GAS = 10.7316

COMPONENTES = ["N₂","CO₂","C1","C2","C3","iC4","nC4","iC5","nC5","C6","C7","C8","C9"]
NOMBRES = [
    "Nitrógeno [N₂]:",
    "Dióxido de carbono [CO₂]:",
    "Metano [C1]:",
    "Etano [C2]:",
    "Propano [C3]:",
    "Isobutano (2-metilpropano) [iC4]:",
    "n-Butano [nC4]:",
    "Isopentano (2-metilbutano) [iC5]:",
    "n-Pentano [nC5]:",
    "Hexano [C6]:",
    "Heptano [C7]:",
    "Octano [C8]:",
    "Nonano [C9]:"
]
PM = [28.013,44.0097,16.0429,30.0699,44.097,58.124,58.124,
      72.151,72.151,86.1779,100.205,114.232,128.259]
TC = [227.14920043945301,547.38001098632799,343.25820922851602,549.77041625976597,
      665.81641845703098,734.57281494140602,765.35820922851599,828.71641845703095,
      845.28001098632797,914.21641845703095,972.28443603515598,1023.47644042969,1070.27644042969]
PC = [492.31163474560498,1068.9278489999999,673.073579130908,708.34238530883795,
      617.37619874414099,529.04243227060499,550.65304957060505,483.49623909045403,
      489.51965902060499,439.69920907060498,396.93628085515098,362.10403957060498,333.59687255368698]
OMEGA = [3.9999801665544503e-2,0.23894000053405801,1.1498400010168599e-2,9.8600000143051106e-2,
         0.152400001883507,0.18479000031948101,0.20100000500678999,0.222240000963211,
         0.25389000773429898,0.30070000886917098,0.34979000687599199,0.401800006628037,0.445490002632141]
KIJ_DEFAULT = [
    [0,-1.9997e-2,3.5999e-2,5.0e-2,7.9998e-2,9.4999e-2,9.0e-2,9.4999e-2,0.1,0.149,0.1439,0.1,0.1],
    [-1.9997e-2,0,0.1,0.1298,0.135,0.1298,0.1298,0.125,0.125,0.125,0.1199,0.115,0.101],
    [3.5999e-2,0.1,0,2.2414e-3,6.8288e-3,1.3113e-2,1.2305e-2,1.7627e-2,1.7925e-2,2.3474e-2,2.8864e-2,3.4159e-2,3.8926e-2],
    [5.0e-2,0.1298,2.2414e-3,0,1.2580e-3,4.5735e-3,4.0964e-3,7.4133e-3,7.6094e-3,1.1414e-2,1.5324e-2,1.9319e-2,2.3017e-2],
    [7.9998e-2,0.135,6.8288e-3,1.2580e-3,0,1.0405e-3,8.1897e-4,2.5834e-3,2.7005e-3,5.1420e-3,7.8874e-3,1.0850e-2,1.3697e-2],
    [9.4999e-2,0.1298,1.3113e-2,4.5735e-3,1.0405e-3,0,1.3351e-5,3.4618e-4,3.9005e-4,1.5653e-3,3.2213e-3,5.2142e-3,7.2549e-3],
    [9.0e-2,0.1298,1.2305e-2,4.0964e-3,8.1897e-4,1.3351e-5,0,4.9514e-4,5.4723e-4,1.8663e-3,3.6464e-3,5.7502e-3,7.8831e-3],
    [9.4999e-2,0.125,1.7627e-2,7.4133e-3,2.5834e-3,3.4618e-4,4.9514e-4,0,1.2517e-6,4.3994e-4,1.4591e-3,2.8828e-3,4.4489e-3],
    [0.1,0.125,1.7925e-2,7.6094e-3,2.7005e-3,3.9005e-4,5.4723e-4,1.2517e-6,0,3.9345e-4,1.3733e-3,2.7618e-3,4.2986e-3],
    [0.149,0.125,2.3474e-2,1.1414e-2,5.1420e-3,1.5653e-3,1.8663e-3,4.3994e-4,3.9345e-4,0,2.9725e-4,1.0733e-3,2.0981e-3],
    [0.1439,0.1199,2.8864e-2,1.5324e-2,7.8874e-3,3.2213e-3,3.6464e-3,1.4591e-3,1.3733e-3,2.9725e-4,0,2.4128e-4,8.1754e-4],
    [0.1,0.115,3.4159e-2,1.9319e-2,1.0850e-2,5.2142e-3,5.7502e-3,2.8828e-3,2.7618e-3,1.0733e-3,2.4128e-4,0,1.7071e-4],
    [0.1,0.101,3.8926e-2,2.3017e-2,1.3697e-2,7.2549e-3,7.8831e-3,4.4489e-3,4.2986e-3,2.0981e-3,8.1754e-4,1.7071e-4,0]
]
NC = 13

# ── Parámetros individuales PR ──────────────────────────────
def ai(i):      return 0.45724*R_GAS**2*TC[i]**2/PC[i]
def bi(i):      return 0.07780*R_GAS*TC[i]/PC[i]
def mi(i):      w=OMEGA[i]; return 0.37464+1.54226*w-0.26992*w**2
def alpha(i,T): return (1+mi(i)*(1-np.sqrt(T/TC[i])))**2
def ai_alpha(i,T): return ai(i)*alpha(i,T)

def aij(i,j,T,kij):
    return np.sqrt(ai_alpha(i,T)*ai_alpha(j,T))*(1-kij[i][j])

def am(z,T,kij):
    s=0.0
    for i in range(NC):
        for j in range(NC):
            s+=z[i]*z[j]*aij(i,j,T,kij)
    return s

def bm(z):
    return sum(z[i]*bi(i) for i in range(NC))

def AB(am_val,bm_val,T,P):
    A=am_val*P/(R_GAS*T)**2
    B=bm_val*P/(R_GAS*T)
    return A,B

def solve_Z(A,B):
    roots=np.roots([1,-(1-B),A-3*B**2-2*B,-(A*B-B**2-B**3)])
    real=[r.real for r in roots if abs(r.imag)<1e-8 and r.real>B]
    if not real: real=[max(r.real for r in roots)]
    real=sorted(real)
    return real[-1], real[0]  # ZV (mayor), ZL (menor)

def ln_phi_i(i,z,T,P,Z,am_val,bm_val,kij):
    bi_=bi(i); A,B=AB(am_val,bm_val,T,P)
    sum_aij=sum(z[j]*aij(i,j,T,kij) for j in range(NC))
    if Z<=B: Z=B+1e-12
    t1=(bi_/bm_val)*(Z-1)
    t2=-np.log(Z-B)
    denom=Z+(1-np.sqrt(2))*B
    numer=Z+(1+np.sqrt(2))*B
    if denom<=0: denom=1e-12
    if numer<=0: numer=1e-12
    t3=A/(2*np.sqrt(2)*B)*np.log(numer/denom)
    t4=(2*sum_aij/am_val-bi_/bm_val)*t3
    return t1+t2-t4

def phi_i(i,z,T,P,Z,am_val,bm_val,kij):
    return np.exp(min(700,max(-700,ln_phi_i(i,z,T,P,Z,am_val,bm_val,kij))))

def Ki_wilson(i,T,P):
    return (PC[i]/P)*np.exp(5.373*(1+OMEGA[i])*(1-TC[i]/T))

# ══ Rachford-Rice ════════════════════════════════════════════
def RR(V,z,K):
    return sum(z[i]*(K[i]-1)/(1+V*(K[i]-1)) for i in range(NC))

def RR_deriv(V,z,K):
    return -sum(z[i]*(K[i]-1)**2/(1+V*(K[i]-1))**2 for i in range(NC))

def solve_V(z,K):
    """Newton para Rachford-Rice — busca raíz en intervalo válido"""
    Ks=[k for k in K if abs(k-1)>1e-12]
    if not Ks: return 0.5
    Vmin=max([-1.0/(k-1) for k in K if k>1.0+1e-10]+[0.0])
    Vmax=min([ 1.0/(1-k) for k in K if k<1.0-1e-10]+[1.0])
    Vmin=max(Vmin+1e-10,1e-10); Vmax=min(Vmax-1e-10,1-1e-10)
    if Vmin>=Vmax: return 0.5
    V=0.5*(Vmin+Vmax)
    for _ in range(500):
        f=RR(V,z,K); df=RR_deriv(V,z,K)
        if abs(df)<1e-30: break
        dV=-f/df
        V_new=V+dV
        if V_new<Vmin: V_new=0.5*(V+Vmin)
        if V_new>Vmax: V_new=0.5*(V+Vmax)
        if abs(V_new-V)<1e-14: V=V_new; break
        V=V_new
    return max(0.0,min(1.0,V))

# ══ Análisis de Estabilidad de Michelsen ═════════════════════
def analisis_estabilidad(z,T,P,kij,tol=1e-12,triv_tol=1e-4,tol_S=1e-4,max_iter=1000):
    """Replica exacta de la macro AnalisisEstabilidad del Excel"""
    Kv=[Ki_wilson(i,T,P) for i in range(NC)]
    Kl=[Ki_wilson(i,T,P) for i in range(NC)]

    am_z=am(z,T,kij); bm_z=bm(z)
    ZV_z,ZL_z=solve_Z(*AB(am_z,bm_z,T,P))

    convV=convL=1.0
    for it in range(max_iter):
        Yv_raw=[z[i]*Kv[i] for i in range(NC)]
        sv=sum(Yv_raw); 
        if sv<=0: sv=1
        Yv=[y/sv for y in Yv_raw]

        Yl_raw=[z[i]/Kl[i] if Kl[i]>1e-30 else z[i]*1e30 for i in range(NC)]
        sl=sum(Yl_raw)
        if sl<=0 or not np.isfinite(sl): sl=1
        Yl=[y/sl for y in Yl_raw]

        am_v=am(Yv,T,kij); bm_v=bm(Yv)
        am_l=am(Yl,T,kij); bm_l=bm(Yl)
        ZV_v,_=solve_Z(*AB(am_v,bm_v,T,P))
        _,ZL_l=solve_Z(*AB(am_l,bm_l,T,P))

        fug_v=[phi_i(i,Yv,T,P,ZV_v,am_v,bm_v,kij)*Yv[i]*P for i in range(NC)]
        fug_l=[phi_i(i,Yl,T,P,ZL_l,am_l,bm_l,kij)*Yl[i]*P for i in range(NC)]
        fug_zV=[phi_i(i,z,T,P,ZV_z,am_z,bm_z,kij)*z[i]*P for i in range(NC)]
        fug_zL=[phi_i(i,z,T,P,ZL_z,am_z,bm_z,kij)*z[i]*P for i in range(NC)]

        Riv=[np.log(fug_v[i]/fug_zV[i]) if (fug_v[i]>0 and fug_zV[i]>0) else 0 for i in range(NC)]
        Ril=[np.log(fug_l[i]/fug_zL[i]) if (fug_l[i]>0 and fug_zL[i]>0) else 0 for i in range(NC)]

        convV=sum(r**2 for r in Riv); convL=sum(r**2 for r in Ril)
        if convV<=tol and convL<=tol: break

        Kv=[max(1e-20,min(1e20, Kv[i]*np.exp(-min(200,max(-200,Riv[i]))))) for i in range(NC)]
        Kl=[max(1e-20,min(1e20, Kl[i]*np.exp(-min(200,max(-200,Ril[i]))))) for i in range(NC)]

    Sv=sum(z[i]*Kv[i] for i in range(NC))
    Sl=sum(z[i]/Kl[i] if Kl[i]>0 else 0 for i in range(NC))

    trivV=sum((Kv[i]-1)**2 for i in range(NC))
    trivL=sum((Kl[i]-1)**2 for i in range(NC))
    esTrivV=(trivV<triv_tol); esTrivL=(trivL<triv_tol)

    inestable=False
    if Sv>(1+tol_S) or Sl>(1+tol_S):
        resultado="INESTABLE - SE REQUIERE FLASH"
        inestable=True
        Ki_flash=[Kv[i]*Kl[i] for i in range(NC)]
    else:
        Ki_flash=[Ki_wilson(i,T,P) for i in range(NC)]
        if esTrivV and esTrivL:
            resultado="ESTABLE - LEJOS DE REGION BIFASICA"
        elif esTrivV:
            resultado="ESTABLE - TIENDE A VAPOR"
        elif esTrivL:
            resultado="ESTABLE - TIENDE A LIQUIDO"
        else:
            resultado="ESTABLE - FASE INDETERMINADA"

    return {"resultado":resultado,"inestable":inestable,"Ki_flash":Ki_flash,
            "Sv":Sv,"Sl":Sl,"Kv":Kv,"Kl":Kl,"iter":it+1}

# ══ Flash Muskat-McDowell ════════════════════════════════════
def flash_muskat(z,T,P,Ki_init,kij,tol=1e-16,max_iter=1000):
    """
    Réplica EXACTA de la macro CalculoFlash:
    - Si ΣKi·zi ≤ 1 → fase única LÍQUIDA (x=z, y=0)
    - Si Σzi/Ki ≤ 1 → fase única VAPOR  (x=0, y=z)
    - Resto: flash bifásico
    """
    K=list(Ki_init)
    
    for it in range(max_iter):
        # Criterio Excel (idéntico a las fórmulas I176:I188 y J176:J188)
        sumKz=sum(z[i]*K[i] for i in range(NC))      # K176:K188 → SUM
        sumZK=sum(z[i]/K[i] if K[i]>0 else 1e30 for i in range(NC))  # L176:L188 → SUM
        
        if sumKz<=1.0:
            # Toda fase LÍQUIDA
            x=list(z); y=[0.0]*NC
            V=0.0; L=1.0
            modo="liquido_unico"
        elif sumZK<=1.0:
            # Toda fase VAPOR
            x=[0.0]*NC; y=list(z)
            V=1.0; L=0.0
            modo="vapor_unico"
        else:
            # Bifásico
            V=solve_V(z,K)
            V=max(1e-10,min(1-1e-10,V))
            L=1-V
            x=[z[i]/(1+V*(K[i]-1)) for i in range(NC)]
            y=[K[i]*x[i]            for i in range(NC)]
            modo="bifasico"

        # EOS para ambas fases (siempre se calcula con composición disponible)
        z_v = y if sum(y)>1e-10 else z
        z_l = x if sum(x)>1e-10 else z
        sy=sum(z_v); sx=sum(z_l)
        if sy>0: z_v=[v/sy for v in z_v]
        if sx>0: z_l=[v/sx for v in z_l]

        am_v=am(z_v,T,kij); bm_v=bm(z_v)
        am_l=am(z_l,T,kij); bm_l=bm(z_l)
        ZV,_=solve_Z(*AB(am_v,bm_v,T,P))
        _,ZL=solve_Z(*AB(am_l,bm_l,T,P))

        fug_v=[phi_i(i,z_v,T,P,ZV,am_v,bm_v,kij)*y[i]*P for i in range(NC)]
        fug_l=[phi_i(i,z_l,T,P,ZL,am_l,bm_l,kij)*x[i]*P for i in range(NC)]

        # Restricción de igualdad de fugacidades
        restr=[]
        for i in range(NC):
            if fug_v[i]>1e-30 and fug_l[i]>1e-30:
                restr.append(np.log(fug_l[i]/fug_v[i]))
            else:
                restr.append(0)

        errorMax=max(abs(r) for r in restr) if restr else 0
        if errorMax<=tol and it>5:
            break

        # Actualizar Ki
        phi_l=[phi_i(i,z_l,T,P,ZL,am_l,bm_l,kij) for i in range(NC)]
        phi_v=[phi_i(i,z_v,T,P,ZV,am_v,bm_v,kij) for i in range(NC)]
        K_new=[]
        for i in range(NC):
            if phi_v[i]>1e-30:
                K_new.append(phi_l[i]/phi_v[i])
            else:
                K_new.append(K[i])
        # Limit K to avoid blowup
        K_new=[max(1e-20,min(1e20,k)) for k in K_new]
        K=K_new

    # ── Determinación final del modo (criterio del Excel) ───
    sumKz=sum(z[i]*K[i] for i in range(NC))
    sumZK=sum(z[i]/K[i] if K[i]>0 else 1e30 for i in range(NC))
    
    # Si ambos son ~1, la mezcla es monofásica → decidir vapor vs líquido por Gibbs
    if sumKz<=1.0+1e-6 and sumZK<=1.0+1e-6:
        # Monofásico — comparar G_V vs G_L con composición z
        am_z=am(z,T,kij); bm_z=bm(z)
        ZV_z,ZL_z=solve_Z(*AB(am_z,bm_z,T,P))
        if abs(ZV_z-ZL_z)<1e-8:
            # Una sola raíz real — usar criterio de B
            A_z,B_z=AB(am_z,bm_z,T,P)
            # Si B/Z < ~0.3 → vapor, si no → líquido (heurístico)
            if B_z/ZV_z < 0.2:
                x=[0.0]*NC; y=list(z); V=1.0; L=0.0; modo="vapor_unico"
            else:
                x=list(z); y=[0.0]*NC; V=0.0; L=1.0; modo="liquido_unico"
        else:
            # Dos raíces — calcular G de cada una y elegir la menor
            phi_V=[phi_i(i,z,T,P,ZV_z,am_z,bm_z,kij) for i in range(NC)]
            phi_L=[phi_i(i,z,T,P,ZL_z,am_z,bm_z,kij) for i in range(NC)]
            G_V=sum(z[i]*np.log(max(phi_V[i]*z[i],1e-300)) for i in range(NC) if z[i]>0)
            G_L=sum(z[i]*np.log(max(phi_L[i]*z[i],1e-300)) for i in range(NC) if z[i]>0)
            if G_V<=G_L:
                x=[0.0]*NC; y=list(z); V=1.0; L=0.0; modo="vapor_unico"
            else:
                x=list(z); y=[0.0]*NC; V=0.0; L=1.0; modo="liquido_unico"
    elif sumKz<=1.0:
        # Toda LÍQUIDA (zKi todos < 1 ⇒ no pueden formar vapor)
        x=list(z); y=[0.0]*NC
        V=0.0; L=1.0
        modo="liquido_unico"
    elif sumZK<=1.0:
        # Toda VAPOR
        x=[0.0]*NC; y=list(z)
        V=1.0; L=0.0
        modo="vapor_unico"
    else:
        V=solve_V(z,K)
        V=max(0.0,min(1.0,V))
        L=1-V
        x=[z[i]/(1+V*(K[i]-1)) for i in range(NC)]
        y=[K[i]*x[i]            for i in range(NC)]
        sx=sum(x); sy=sum(y)
        if sx>0: x=[xi/sx for xi in x]
        if sy>0: y=[yi/sy for yi in y]
        modo="bifasico"

    # ── Propiedades finales ─────────────────────────────────
    PM_v = sum(y[i]*PM[i] for i in range(NC)) if V>0 else 0.0
    PM_l = sum(x[i]*PM[i] for i in range(NC)) if L>0 else 0.0
    PM_z = sum(z[i]*PM[i] for i in range(NC))
    
    # Si el flash bifásico devolvió fases invertidas (vapor con PM mayor que líquido),
    # intercambiar etiquetas para mantener convención: vapor=fase liviana
    if modo=="bifasico" and V>0 and L>0 and PM_v>PM_l:
        x,y = y,x
        V,L = L,V
        PM_v,PM_l = PM_l,PM_v
        K = [1.0/k if k>1e-30 else 1e30 for k in K]
    # Calcular fracciones másicas con valores definitivos
    den_m = V*PM_v+L*PM_l if (V*PM_v+L*PM_l)>0 else 1
    Vm = V*PM_v/den_m if V>0 else 0.0
    Lm = L*PM_l/den_m if L>0 else 0.0

    ZV_fin = None; ZL_fin = None
    rho_v = None; rho_l = None
    sg_v = None; sg_l = None

    if V>0 and PM_v>0:
        am_v=am(y,T,kij); bm_v=bm(y)
        ZV_fin,_=solve_Z(*AB(am_v,bm_v,T,P))
        rho_v=P*PM_v/(ZV_fin*R_GAS*T)
        sg_v=PM_v/28.9625
    if L>0 and PM_l>0:
        am_l=am(x,T,kij); bm_l=bm(x)
        _,ZL_fin=solve_Z(*AB(am_l,bm_l,T,P))
        rho_l=P*PM_l/(ZL_fin*R_GAS*T)
        sg_l=rho_l/62.4  # SG líquido respecto al agua

    den_m = V*PM_v+L*PM_l if (V*PM_v+L*PM_l)>0 else 1
    Vm = V*PM_v/den_m if V>0 else 0.0
    Lm = L*PM_l/den_m if L>0 else 0.0

    return {
        "V":V,"L":L,"Vm":Vm,"Lm":Lm,
        "x":x,"y":y,"z":list(z),"K":K,
        "ZV":ZV_fin,"ZL":ZL_fin,
        "PM_v":PM_v if V>0 else None,
        "PM_l":PM_l if L>0 else None,
        "PM_z":PM_z,
        "rho_v":rho_v,"rho_l":rho_l,
        "sg_v":sg_v,"sg_l":sg_l,
        "modo":modo,
        "iter":it+1,
        "sumKz":sumKz,"sumZK":sumZK
    }

# ══ Punto de entrada ═════════════════════════════════════════
def calcular(z,T,P,kij=None):
    """
    Réplica de Sub AnalisisYFlash():
    1. Análisis de estabilidad de Michelsen
    2. SIEMPRE se corre el flash con los Ki del análisis
       (sea estable o inestable, como hace el usuario en el Excel)
    """
    if kij is None: kij=copy.deepcopy(KIJ_DEFAULT)
    estab=analisis_estabilidad(z,T,P,kij)
    flash=flash_muskat(z,T,P,estab["Ki_flash"],kij)
    flash["estabilidad"]=estab["resultado"]
    flash["iter_estab"]=estab["iter"]
    flash["inestable"]=estab["inestable"]
    return flash
