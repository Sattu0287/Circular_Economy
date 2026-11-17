import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Fuzzy to crisp (Jimenez 2007) ────────────────────────────────────────────
def cr(t): return (t[0]+2*t[1]+t[2])/4
def cap(t, a=0.7): return a*(t[0]+t[1])/2 + (1-a)*(t[2]+t[1])/2

# ── Case data (Sapphire Chemicals, Tables 7-15) ───────────────────────────────
J,K,C = 2,3,5

D1 = {(0,0):cr((300,400,500)),(0,1):cr((300,400,500)),(0,2):0,(0,3):0,(0,4):0,
      (1,0):cr((300,400,500)),(1,1):cr((250,350,450)),(1,2):0,(1,3):0,(1,4):0}
D2 = {(0,0):0,(0,1):0,(0,2):cr((350,450,550)),(0,3):cr((300,400,500)),(0,4):cr((250,350,450)),
      (1,0):0,(1,1):0,(1,2):cr((250,350,450)),(1,3):cr((225,325,425)),(1,4):cr((275,375,475))}

CC = {(j,c): cr((0.15,0.16,0.17)) for j in range(J) for c in range(C)}
RF = {(0,c): cr((0.30,0.35,0.40)) for c in range(C)}
RF.update({(1,c): cr((0.19,0.20,0.21)) for c in range(C)})

RP  = {0:cr((0.73,0.83,0.93)), 1:cr((0.50,0.60,0.70))}
CFr = {j:cr((0.34,0.40,0.46)) for j in range(J)}
CD  = {0:cr((0.04,0.05,0.06)),  1:cr((0.075,0.087,0.099))}
RC  = {j:cr((0.04,0.05,0.06))  for j in range(J)}
MP  = {0:cap((2400,2500,2700)), 1:cap((2100,2200,2300))}
MC  = {0:cap((1700,1800,1900)), 1:cap((1400,1500,1600))}
MR  = {j:cap((1400,1500,1600)) for j in range(J)}
MF  = {j:cap((1400,1500,1600)) for j in range(J)}
PC  = {(0,k):cr((2.34+0.04*k, 2.44+0.04*k, 2.54+0.04*k)) for k in range(K)}
PC.update({(1,k):cr((152+2*k,162+2*k,172+2*k)) for k in range(K)})
MXS = {k:cap((3500+k*500,4000+k*500,4500+k*500)) for k in range(K)}
MNS = {k:cap((90,100,110)) for k in range(K)}

# Transport costs (representative values)
TCSP={(j,k):cr((0.001+0.001*k,0.002+0.001*k,0.003+0.001*k)) for j in range(J) for k in range(K)}
TCDC={(j,c):cr((0.003+0.001*c,0.004+0.001*c,0.005+0.001*c)) for j in range(J) for c in range(C)}
TCPC={(j,c):cr((0.001+0.001*c,0.002+0.001*c,0.003+0.001*c)) for j in range(J) for c in range(C)}
TCCC={(j,c):cr((0.004,0.005,0.006)) for j in range(J) for c in range(C)}
TCCR={j:cr((0.006,0.007,0.008)) for j in range(J)}
TCRP={j:cr((0.004,0.005,0.006)) for j in range(J)}
TCRS={(j,k):cr((0.003+0.001*k,0.004+0.001*k,0.005+0.001*k)) for j in range(J) for k in range(K)}
TCCD={j:cr((0.005,0.006,0.007)) for j in range(J)}
TCRD={j:cr((0.0,0.001,0.002))   for j in range(J)}

# Carbon emissions (Table 3)
CU={0:cr((0.48,0.58,0.68)), 1:cr((0.5,0.6,0.7))}
CFe={0:cr((2.58,3.58,4.58)),1:cr((2.58,3.58,4.58))}
CRe={0:cr((2.31,3.31,4.31)),1:cr((4.57,5.57,6.57))}
CDe={0:cr((3.66,4.66,5.66)),1:cr((6.36,7.36,8.36))}

# Jobs (Table 15)
JVC={0:cr((0.075,0.175,0.275)),1:cr((0.20,0.21,0.22))}
JVF={0:cr((0.145,0.245,0.345)),1:0}
JVR={0:cr((0.20,0.21,0.22)),   1:cr((0.128,0.228,0.328))}
JVD={0:cr((0.0,0.035,0.07)),   1:cr((0.0,0.041,0.082))}
FIXED_JOBS = 8  # 2+1+1 per product type

# Collection/refurb/recycle percentages
eta={j:cap((0.85,0.90,0.95)) for j in range(J)}
lam={0:cap((0.55,0.60,0.65)),1:0}
dlt={j:cap((0.85,0.90,0.95)) for j in range(J)}

# ── Variable index map (30 variables) ────────────────────────────────────────
IDX,n = {},0
for j in range(J):
    for c in range(C): IDX[('C',j,c)]=n; n+=1
for j in range(J): IDX[('F',j)]=n; n+=1
for j in range(J):
    for k in range(K): IDX[('Y',j,k)]=n; n+=1
for j in range(J):
    for k in range(K): IDX[('S',j,k)]=n; n+=1
for j in range(J): IDX[('R',j)]=n; n+=1
for j in range(J): IDX[('RD',j)]=n; n+=1
for j in range(J): IDX[('W',j)]=n; n+=1
NV = n  # = 30

# ── Build objective coefficients ─────────────────────────────────────────────
def make_obj(obj='z1'):
    c = np.zeros(NV)
    for j in range(J):
        for k in range(K):
            if obj=='z1':
                c[IDX['S',j,k]] += PC[j,k]+TCSP[j,k]
                c[IDX['Y',j,k]] += RC[j]+TCRS[j,k]-RP[j]
            elif obj=='z2':
                c[IDX['S',j,k]] += TCSP[j,k]
                c[IDX['Y',j,k]] += TCRS[j,k]+CRe[j]
            else: # z3 negated
                c[IDX['Y',j,k]] -= JVR[j]
        for cz in range(C):
            if obj=='z1': c[IDX['C',j,cz]] += CC[j,cz]+RF[j,cz]+TCCC[j,cz]
            elif obj=='z2': c[IDX['C',j,cz]] += CU[j]+TCCC[j,cz]
            else: c[IDX['C',j,cz]] -= JVC[j]
        if obj=='z1':
            c[IDX['F',j]]  += CFr[j]+TCRP[j]
            c[IDX['R',j]]  += TCCR[j]
            c[IDX['RD',j]] += CD[j]+TCRD[j]
            c[IDX['W',j]]  += CD[j]+TCCD[j]
        elif obj=='z2':
            c[IDX['F',j]]  += CFe[j]+TCRP[j]
            c[IDX['R',j]]  += TCCR[j]
            c[IDX['RD',j]] += CDe[j]+TCRD[j]
            c[IDX['W',j]]  += CDe[j]+TCCD[j]
        else:
            c[IDX['F',j]]  -= JVF[j]
            c[IDX['RD',j]] -= JVD[j]
            c[IDX['W',j]]  -= JVD[j]
    return c

Z1,Z2,Z3n = make_obj('z1'), make_obj('z2'), make_obj('z3')

# ── Build constraints ─────────────────────────────────────────────────────────
Aeq,beq,Aub,bub = [],[],[],[]
for j in range(J):  # Eq(4): demand balance
    r=np.zeros(NV)
    for k in range(K): r[IDX['S',j,k]]=1
    r[IDX['F',j]]=1
    Aeq.append(r); beq.append(sum(D1[j,c]+D2[j,c] for c in range(C)))

for j in range(J):  # Eq(5): collection flow
    r=np.zeros(NV)
    for c in range(C): r[IDX['C',j,c]]=1
    r[IDX['R',j]]=-1; r[IDX['W',j]]=-1
    Aeq.append(r); beq.append(0)

for j in range(J):  # Eq(6): refurb/recycle flow
    r=np.zeros(NV)
    r[IDX['R',j]]=1; r[IDX['F',j]]=-1
    for k in range(K): r[IDX['Y',j,k]]=-1
    r[IDX['RD',j]]=-1
    Aeq.append(r); beq.append(0)

for j in range(J):  # production capacity
    r=np.zeros(NV)
    for k in range(K): r[IDX['S',j,k]]=1
    Aub.append(r); bub.append(MP[j])

for k in range(K):  # supplier limits
    r=np.zeros(NV)
    for j in range(J): r[IDX['S',j,k]]=1
    Aub.append(r); bub.append(MXS[k])
    Aub.append(-r); bub.append(-MNS[k])

for j in range(J):  # center capacities
    r=np.zeros(NV)
    for c in range(C): r[IDX['C',j,c]]=1
    Aub.append(r); bub.append(MC[j])
    r2=np.zeros(NV)
    for k in range(K): r2[IDX['Y',j,k]]=1
    Aub.append(r2); bub.append(MR[j])
    r3=np.zeros(NV); r3[IDX['F',j]]=1
    Aub.append(r3); bub.append(MF[j])

for j in range(J):  # percentage limits
    for c in range(C):
        r=np.zeros(NV); r[IDX['C',j,c]]=1
        Aub.append(r); bub.append(eta[j]*(D1[j,c]+D2[j,c]))
    r=np.zeros(NV)
    for k in range(K): r[IDX['Y',j,k]]=1
    r[IDX['R',j]]=-dlt[j]; Aub.append(r); bub.append(0)
    r2=np.zeros(NV); r2[IDX['F',j]]=1; r2[IDX['R',j]]=-lam[j]
    Aub.append(r2); bub.append(0)

Aeq=np.array(Aeq); beq=np.array(beq)
Aub=np.array(Aub); bub=np.array(bub)
BND=[(0,None)]*NV

# ── Solver ────────────────────────────────────────────────────────────────────
def solve(c, eA=None, eb=None):
    Au = np.vstack([Aub,eA]) if eA is not None else Aub
    bu = np.concatenate([bub,eb]) if eb is not None else bub
    r = linprog(c,A_ub=Au,b_ub=bu,A_eq=Aeq,b_eq=beq,bounds=BND,method='highs')
    return (np.maximum(r.x,0), r.fun) if r.success else (None,np.inf)

def obj(x): return Z1@x, Z2@x, -(Z3n@x)+FIXED_JOBS

# ── Ideal/Nadir ───────────────────────────────────────────────────────────────
x1,_ = solve(Z1); x2,_ = solve(Z2); x3,_ = solve(Z3n)
z1i,z2i,z3i = obj(x1)[0], obj(x2)[1], obj(x3)[2]  # ideal
z1n = max(obj(x)[0] for x in [x1,x2,x3])
z2n = max(obj(x)[1] for x in [x1,x2,x3])
z3n_val = min(obj(x)[2] for x in [x1,x2,x3])

print(f"Ideal  → Z1=${z1i:.0f}  Z2={z2i:.1f} kg  Z3={z3i:.1f} jobs")
print(f"Nadir  → Z1=${z1n:.0f}  Z2={z2n:.1f} kg  Z3={z3n_val:.1f} jobs")

# ── ε-Constraint ──────────────────────────────────────────────────────────────
pareto_e = []
for e2 in np.linspace(z2i, z2n, 6):
    for e3 in np.linspace(z3n_val, z3i, 6):
        eA = np.vstack([Z2, Z3n]); eb = np.array([e2, FIXED_JOBS-e3])
        x,_ = solve(Z1, eA, eb)
        if x is not None: pareto_e.append(dict(zip(['Z1','Z2','Z3','x'], [*obj(x),x])))

# ── LP-Metric ─────────────────────────────────────────────────────────────────
pareto_l = []
r1,r2,r3 = z1n-z1i+1e-9, z2n-z2i+1e-9, z3i-z3n_val+1e-9
for w1 in np.linspace(0.1,0.8,5):
    for w2 in np.linspace(0.1,0.8,5):
        w3=1-w1-w2
        if w3<=0: continue
        x,_ = solve((w1/r1)*Z1+(w2/r2)*Z2+(w3/r3)*Z3n)
        if x is not None: pareto_l.append(dict(zip(['Z1','Z2','Z3','x'], [*obj(x),x])))

# ── GCM ───────────────────────────────────────────────────────────────────────
x_g,_ = solve((1/3/r1)*Z1+(1/3/r2)*Z2+(1/3/r3)*Z3n)
g1,g2,g3 = obj(x_g)

# ── TOPSIS ────────────────────────────────────────────────────────────────────
def topsis(sols):
    if not sols: return []
    M = np.array([[s['Z1'],s['Z2'],s['Z3']] for s in sols])
    R = M / (np.sqrt((M**2).sum(0))+1e-9)
    V = R * (1/3)
    A = np.array([V[:,0].min(),V[:,1].min(),V[:,2].max()])
    B = np.array([V[:,0].max(),V[:,1].max(),V[:,2].min()])
    dp = np.sqrt(((V-A)**2).sum(1)); dn = np.sqrt(((V-B)**2).sum(1))
    return dn/(dp+dn+1e-9)

se,sl = topsis(pareto_e), topsis(pareto_l)
print(f"\nTOPSIS avg → ε-constraint:{np.mean(se):.4f}  LP-metric:{np.mean(sl):.4f}")
print(f"GCM solution → Cost=${g1:,.0f}  CO2={g2:,.0f} kg  Jobs={g3:.1f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15,10))
fig.suptitle("Multi-Objective Circular Supply Chain — EPR Paint Packaging",
             fontsize=13, fontweight='bold')

# 3D Pareto fronts
for i,(sols,sc,title,cm) in enumerate([(pareto_e,se,'ε-Constraint','viridis'),
                                        (pareto_l,sl,'LP-Metric','plasma')],1):
    ax = fig.add_subplot(2,3,i,projection='3d')
    if sols:
        xs=[s['Z1']/1000 for s in sols]; ys=[s['Z2']/1000 for s in sols]; zs=[s['Z3'] for s in sols]
        ax.scatter(xs,ys,zs,c=sc,cmap=cm,s=60,alpha=0.8)
        bi=np.argmax(sc)
        ax.scatter([xs[bi]],[ys[bi]],[zs[bi]],c='red',s=150,marker='*',label='Best')
    ax.set_xlabel('Z1 Cost (k$)'); ax.set_ylabel('Z2 CO2 (k kg)'); ax.set_zlabel('Z3 Jobs')
    ax.set_title(f'{title}\nAvg TOPSIS={np.mean(sc):.4f}',fontsize=9); ax.legend(fontsize=7)

# 2D trade-off panels
pairs = [('Z1','Z2','Cost(k$)','CO2(k kg)'),
         ('Z1','Z3','Cost(k$)','Jobs'),
         ('Z2','Z3','CO2(k kg)','Jobs')]
for i,(xa,ya,xl,yl) in enumerate(pairs,4):
    ax=fig.add_subplot(2,3,i)
    scale = lambda k,s: [s[k]/1000 if 'Z1' in k or 'Z2' in k else s[k] for s in (pareto_e or [])]
    xe=[s[xa]/(1000 if xa in('Z1','Z2') else 1) for s in pareto_e]
    ye=[s[ya]/(1000 if ya in('Z1','Z2') else 1) for s in pareto_e]
    xl2=[s[xa]/(1000 if xa in('Z1','Z2') else 1) for s in pareto_l]
    yl2=[s[ya]/(1000 if ya in('Z1','Z2') else 1) for s in pareto_l]
    ax.scatter(xe,ye,c='steelblue',s=50,alpha=0.7,label='ε-c')
    ax.scatter(xl2,yl2,c='orange',s=50,marker='^',alpha=0.7,label='LP')
    gvx=g1/1000 if xa=='Z1' else (g2/1000 if xa=='Z2' else g3)
    gvy=g1/1000 if ya=='Z1' else (g2/1000 if ya=='Z2' else g3)
    ax.scatter([gvx],[gvy],c='red',s=180,marker='*',zorder=5,label='GCM')
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=7)
    ax.set_title(f'{xl} vs {yl}',fontsize=9); ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pareto_results.png', dpi=150, bbox_inches='tight')
plt.close(); print("Plot saved.")
