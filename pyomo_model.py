

from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────────────────
# 1. FUZZY TO CRISP  (Jimenez 2007, alpha = 0.7)
# ─────────────────────────────────────────────────────────

def cr(t):
    """Expected value of triangular fuzzy number (p, m, o)."""
    return (t[0] + 2*t[1] + t[2]) / 4

def cap(t, a=0.7):
    """Crisp capacity bound from triangular fuzzy number."""
    return a*(t[0]+t[1])/2 + (1-a)*(t[2]+t[1])/2

# ─────────────────────────────────────────────────────────
# 2. SETS
# ─────────────────────────────────────────────────────────

J_set = [0, 1]     # products: 0=metal drum, 1=plastic drum
K_set = [0, 1, 2]  # suppliers
C_set = [0, 1, 2, 3, 4]  # customer zones

# ─────────────────────────────────────────────────────────
# 3. PARAMETERS  (Tables 7–15, Annexure)
# ─────────────────────────────────────────────────────────

# Demand via distributor and direct (Table 7)
D1 = {
    (0,0):cr((300,400,500)), (0,1):cr((300,400,500)), (0,2):0, (0,3):0, (0,4):0,
    (1,0):cr((300,400,500)), (1,1):cr((250,350,450)), (1,2):0, (1,3):0, (1,4):0
}
D2 = {
    (0,0):0, (0,1):0, (0,2):cr((350,450,550)), (0,3):cr((300,400,500)), (0,4):cr((250,350,450)),
    (1,0):0, (1,1):0, (1,2):cr((250,350,450)), (1,3):cr((225,325,425)), (1,4):cr((275,375,475))
}

# Collection & refund cost (Table 7)
CC = {(j,c): cr((0.15,0.16,0.17)) for j in J_set for c in C_set}
RF = {**{(0,c): cr((0.30,0.35,0.40)) for c in C_set},
      **{(1,c): cr((0.19,0.20,0.21)) for c in C_set}}

# Product costs and capacities (Table 8)
RP  = {0: cr((0.73,0.83,0.93)), 1: cr((0.50,0.60,0.70))}   # recycling profit
CFr = {j: cr((0.34,0.40,0.46)) for j in J_set}              # refurbishing cost
CD  = {0: cr((0.04,0.05,0.06)), 1: cr((0.075,0.087,0.099))} # disposal cost
RC  = {j: cr((0.04,0.05,0.06)) for j in J_set}              # recycling cost
MP  = {0: cap((2400,2500,2700)), 1: cap((2100,2200,2300))}  # max production
MC  = {0: cap((1700,1800,1900)), 1: cap((1400,1500,1600))}  # max collection
MR  = {j: cap((1400,1500,1600)) for j in J_set}             # max recycling
MF  = {j: cap((1400,1500,1600)) for j in J_set}             # max refurbishing

# Purchasing cost and supplier limits (Table 9)
PC = {(0,0):cr((2.34,2.44,2.54)), (0,1):cr((2.28,2.38,2.48)), (0,2):cr((2.30,2.40,2.50)),
      (1,0):cr((152,162,172)),     (1,1):cr((155,165,175)),     (1,2):cr((154,164,174))}
MXS = {0:cap((3500,4000,4500)), 1:cap((4500,5000,5500)), 2:cap((4500,5000,5500))}
MNS = {0:cap((90,100,110)),     1:cap((90,100,110)),     2:cap((90,100,110))}

# Transportation costs (Tables 10–11)
TCSP = {(j,k): cr((0.001+0.001*k, 0.002+0.001*k, 0.003+0.001*k)) for j in J_set for k in K_set}
TCDC = {(j,c): cr((0.003+0.001*c, 0.004+0.001*c, 0.005+0.001*c)) for j in J_set for c in C_set}
TCPC = {(j,c): cr((0.001+0.001*c, 0.002+0.001*c, 0.003+0.001*c)) for j in J_set for c in C_set}
TCCC = {(j,c): cr((0.004,0.005,0.006)) for j in J_set for c in C_set}
TCCR = {j: cr((0.006,0.007,0.008)) for j in J_set}
TCRP = {j: cr((0.004,0.005,0.006)) for j in J_set}
TCRS = {(j,k): cr((0.003+0.001*k, 0.004+0.001*k, 0.005+0.001*k)) for j in J_set for k in K_set}
TCCD = {j: cr((0.005,0.006,0.007)) for j in J_set}
TCRD = {j: cr((0.0,0.001,0.002))   for j in J_set}

# Carbon emissions at centers (Table 3)
CU  = {0: cr((0.48,0.58,0.68)), 1: cr((0.50,0.60,0.70))}
CFe = {0: cr((2.58,3.58,4.58)), 1: cr((2.58,3.58,4.58))}
CRe = {0: cr((2.31,3.31,4.31)), 1: cr((4.57,5.57,6.57))}
CDe = {0: cr((3.66,4.66,5.66)), 1: cr((6.36,7.36,8.36))}

# Jobs (Table 15 + paper assumptions)
JVC = {0: cr((0.075,0.175,0.275)), 1: cr((0.20,0.21,0.22))}
JVF = {0: cr((0.145,0.245,0.345)), 1: 0.0}
JVR = {0: cr((0.20,0.21,0.22)),    1: cr((0.128,0.228,0.328))}
JVD = {0: cr((0.0,0.035,0.070)),   1: cr((0.0,0.041,0.082))}
FIXED_JOBS = 8   # 2 collection + 1 refurb + 1 disposal per product type

# Collection / refurb / recycle percentage limits
eta = {j: cap((0.85,0.90,0.95)) for j in J_set}
lam = {0: cap((0.55,0.60,0.65)), 1: 0.0}
dlt = {j: cap((0.85,0.90,0.95)) for j in J_set}

# ─────────────────────────────────────────────────────────
# 4. BUILD PYOMO MODEL
# ─────────────────────────────────────────────────────────

def build_model():
    m = ConcreteModel()

    # Sets
    m.J = Set(initialize=J_set)
    m.K = Set(initialize=K_set)
    m.C = Set(initialize=C_set)

    # Decision Variables (all non-negative continuous)
    m.S  = Var(m.J, m.K, domain=NonNegativeReals)  # purchased from supplier
    m.Cv = Var(m.J, m.C, domain=NonNegativeReals)  # collected from customer
    m.F  = Var(m.J,       domain=NonNegativeReals)  # refurbished
    m.Y  = Var(m.J, m.K, domain=NonNegativeReals)  # recycled → supplier
    m.R  = Var(m.J,       domain=NonNegativeReals)  # sent to refurb/recycle
    m.RD = Var(m.J,       domain=NonNegativeReals)  # disposed from refurb center
    m.W  = Var(m.J,       domain=NonNegativeReals)  # disposed from collection

    # ── Objective Z1: Minimize Total Cost (Eq. 1) ──────────────────
    m.Z1 = Objective(sense=minimize, expr=(
        sum(PC[j,k]   * m.S[j,k]  for j in m.J for k in m.K) +   # purchase
        sum(TCSP[j,k] * m.S[j,k]  for j in m.J for k in m.K) +   # transport: supplier→prod
        sum(CC[j,c]   * m.Cv[j,c] for j in m.J for c in m.C) +   # collection cost
        sum(RF[j,c]   * m.Cv[j,c] for j in m.J for c in m.C) +   # refund to customer
        sum(TCCC[j,c] * m.Cv[j,c] for j in m.J for c in m.C) +   # transport: cust→collect
        sum(CFr[j]    * m.F[j]    for j in m.J) +                  # refurbishing
        sum(TCRP[j]   * m.F[j]    for j in m.J) +                  # transport: refurb→prod
        sum(TCCR[j]   * m.R[j]    for j in m.J) +                  # transport: collect→refurb
        sum(RC[j]     * m.Y[j,k]  for j in m.J for k in m.K) +   # recycling
        sum(TCRS[j,k] * m.Y[j,k]  for j in m.J for k in m.K) -   # transport: recycle→supplier
        sum(RP[j]     * m.Y[j,k]  for j in m.J for k in m.K) +   # profit from recycling
        sum(CD[j]     * (m.W[j] + m.RD[j]) for j in m.J) +        # disposal cost
        sum(TCCD[j]   * m.W[j]    for j in m.J) +                  # transport: collect→disposal
        sum(TCRD[j]   * m.RD[j]   for j in m.J)                    # transport: refurb→disposal
    ))

    # ── Objective Z2: Minimize Carbon Emissions (Eq. 2) ────────────
    m.Z2 = Objective(sense=minimize, expr=(
        sum(CU[j]     * m.Cv[j,c] for j in m.J for c in m.C) +   # emissions: collection
        sum(TCCC[j,c] * m.Cv[j,c] for j in m.J for c in m.C) +   # transport emissions
        sum(CFe[j]    * m.F[j]    for j in m.J) +                  # emissions: refurbishing
        sum(TCRP[j]   * m.F[j]    for j in m.J) +
        sum(TCCR[j]   * m.R[j]    for j in m.J) +
        sum(CRe[j]    * m.Y[j,k]  for j in m.J for k in m.K) +   # emissions: recycling
        sum(TCRS[j,k] * m.Y[j,k]  for j in m.J for k in m.K) +
        sum(TCSP[j,k] * m.S[j,k]  for j in m.J for k in m.K) +   # forward logistics
        sum(CDe[j]    * (m.W[j] + m.RD[j]) for j in m.J) +        # emissions: disposal
        sum(TCCD[j]   * m.W[j]    for j in m.J) +
        sum(TCRD[j]   * m.RD[j]   for j in m.J)
    ))

    # ── Objective Z3: Maximize Jobs (Eq. 3) ────────────────────────
    m.Z3 = Objective(sense=maximize, expr=(
        FIXED_JOBS +
        sum(JVC[j] * m.Cv[j,c] for j in m.J for c in m.C) +      # variable: collection
        sum(JVF[j] * m.F[j]    for j in m.J) +                     # variable: refurbishing
        sum(JVR[j] * m.Y[j,k]  for j in m.J for k in m.K) +      # variable: recycling
        sum(JVD[j] * (m.W[j] + m.RD[j]) for j in m.J)             # variable: disposal
    ))

    # ── Constraints ─────────────────────────────────────────────────

    # Eq(4): Demand balance — purchased + refurbished = total demand
    def demand_balance(m, j):
        return sum(m.S[j,k] for k in m.K) + m.F[j] == \
               sum(D1[j,c] + D2[j,c] for c in m.C)
    m.demand_balance = Constraint(m.J, rule=demand_balance)

    # Eq(5): Flow balance at collection center
    def collection_flow(m, j):
        return sum(m.Cv[j,c] for c in m.C) == m.R[j] + m.W[j]
    m.collection_flow = Constraint(m.J, rule=collection_flow)

    # Eq(6): Flow balance at refurb/recycle center
    def refurb_flow(m, j):
        return m.R[j] == m.F[j] + sum(m.Y[j,k] for k in m.K) + m.RD[j]
    m.refurb_flow = Constraint(m.J, rule=refurb_flow)

    # Eq(7): Production capacity
    def prod_capacity(m, j):
        return sum(m.S[j,k] for k in m.K) <= MP[j]
    m.prod_capacity = Constraint(m.J, rule=prod_capacity)

    # Eq(8): Supplier min and max limits
    def supplier_max(m, k):
        return sum(m.S[j,k] for j in m.J) <= MXS[k]
    m.supplier_max = Constraint(m.K, rule=supplier_max)

    def supplier_min(m, k):
        return sum(m.S[j,k] for j in m.J) >= MNS[k]
    m.supplier_min = Constraint(m.K, rule=supplier_min)

    # Eq(9): Collection center capacity
    def collect_capacity(m, j):
        return sum(m.Cv[j,c] for c in m.C) <= MC[j]
    m.collect_capacity = Constraint(m.J, rule=collect_capacity)

    # Eq(10): Recycling center capacity
    def recycle_capacity(m, j):
        return sum(m.Y[j,k] for k in m.K) <= MR[j]
    m.recycle_capacity = Constraint(m.J, rule=recycle_capacity)

    # Eq(11): Refurbishing center capacity
    def refurb_capacity(m, j):
        return m.F[j] <= MF[j]
    m.refurb_capacity = Constraint(m.J, rule=refurb_capacity)

    # Eq(12): Max collection percentage
    def max_collection(m, j, c):
        return m.Cv[j,c] <= eta[j] * (D1[j,c] + D2[j,c])
    m.max_collection = Constraint(m.J, m.C, rule=max_collection)

    # Eq(13): Max recycling percentage
    def max_recycle(m, j):
        return sum(m.Y[j,k] for k in m.K) <= dlt[j] * m.R[j]
    m.max_recycle = Constraint(m.J, rule=max_recycle)

    # Eq(14): Max refurbishing percentage
    def max_refurb(m, j):
        return m.F[j] <= lam[j] * m.R[j]
    m.max_refurb = Constraint(m.J, rule=max_refurb)

    return m

# ─────────────────────────────────────────────────────────
# 5. SOLVE SINGLE OBJECTIVE
# ─────────────────────────────────────────────────────────

def solve_single(active_obj, eps2=None, eps3=None, w=None):
    """
    Solve model with one active objective.
    eps2: upper bound on Z2 (for e-constraint)
    eps3: lower bound on Z3 (for e-constraint)
    w:    (w1,w2,w3) weights for LP-metric / GCM
    """
    m = build_model()

    # Deactivate all objectives, then activate chosen one
    m.Z1.deactivate()
    m.Z2.deactivate()
    m.Z3.deactivate()

    if w is not None:
        # LP-metric / GCM: minimise weighted normalised distance
        # Objectives already built; we rebuild as single weighted obj
        r1 = z1_nadir - z1_ideal + 1e-9
        r2 = z2_nadir - z2_ideal + 1e-9
        r3 = z3_ideal - z3_nadir + 1e-9
        m.weighted_obj = Objective(sense=minimize, expr=(
            w[0]/r1 * m.Z1.expr +
            w[1]/r2 * m.Z2.expr +
            w[2]/r3 * (-m.Z3.expr)   # negate since Z3 is max
        ))
    else:
        if   active_obj == 'z1': m.Z1.activate()
        elif active_obj == 'z2': m.Z2.activate()
        else:                    m.Z3.activate()

    # e-constraint bounds
    if eps2 is not None:
        m.eps2_con = Constraint(expr=m.Z2.expr <= eps2)
    if eps3 is not None:
        m.eps3_con = Constraint(expr=m.Z3.expr >= eps3)

    solver = SolverFactory('glpk')
    result = solver.solve(m, tee=False)

    if result.solver.termination_condition == TerminationCondition.optimal:
        z1_val = value(m.Z1.expr)
        z2_val = value(m.Z2.expr)
        z3_val = value(m.Z3.expr)
        return z1_val, z2_val, z3_val, m
    return None

# ─────────────────────────────────────────────────────────
# 6. FIND IDEAL AND NADIR POINTS
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  STEP 1: Individual Optimal (Ideal/Utopia) Solutions")
print("=" * 55)

r1_z1 = solve_single('z1')
r2_z2 = solve_single('z2')
r3_z3 = solve_single('z3')

z1_ideal = r1_z1[0]
z2_ideal = r2_z2[1]
z3_ideal = r3_z3[2]

z1_nadir = max(r1_z1[0], r2_z2[0], r3_z3[0])
z2_nadir = max(r1_z1[1], r2_z2[1], r3_z3[1])
z3_nadir = min(r1_z1[2], r2_z2[2], r3_z3[2])

print(f"  Ideal  Z1 = ${z1_ideal:>12,.2f}  (min cost)")
print(f"  Ideal  Z2 = {z2_ideal:>14,.2f}  kg CO2-eq")
print(f"  Ideal  Z3 = {z3_ideal:>14,.4f}  jobs")
print(f"  Nadir  Z1 = ${z1_nadir:>12,.2f}")
print(f"  Nadir  Z2 = {z2_nadir:>14,.2f}  kg CO2-eq")
print(f"  Nadir  Z3 = {z3_nadir:>14,.4f}  jobs")

# ─────────────────────────────────────────────────────────
# 7. ε-CONSTRAINT METHOD
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 2: ε-Constraint Method")
print("=" * 55)

pareto_eps = []
N = 6
for eps2 in np.linspace(z2_ideal, z2_nadir, N):
    for eps3 in np.linspace(z3_nadir, z3_ideal, N):
        res = solve_single('z1', eps2=eps2, eps3=eps3)
        if res:
            pareto_eps.append({'Z1': res[0], 'Z2': res[1], 'Z3': res[2]})

print(f"  Generated {len(pareto_eps)} Pareto solutions")

# ─────────────────────────────────────────────────────────
# 8. LP-METRIC METHOD  (p = 2)
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 3: LP-Metric Method  (p = 2)")
print("=" * 55)

pareto_lp = []
for w1 in np.linspace(0.1, 0.8, 5):
    for w2 in np.linspace(0.1, 0.8, 5):
        w3 = 1 - w1 - w2
        if w3 <= 0:
            continue
        res = solve_single(None, w=(w1, w2, w3))
        if res:
            pareto_lp.append({'Z1': res[0], 'Z2': res[1], 'Z3': res[2]})

print(f"  Generated {len(pareto_lp)} LP-metric solutions")

# ─────────────────────────────────────────────────────────
# 9. GLOBAL CRITERION METHOD  (equal weights, p = 2)
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 4: Global Criterion Method (GCM)")
print("=" * 55)

res_gcm = solve_single(None, w=(1/3, 1/3, 1/3))
g1, g2, g3 = res_gcm[0], res_gcm[1], res_gcm[2]

print(f"  Z1 (Cost)  = ${g1:>12,.2f}")
print(f"  Z2 (CO2)   = {g2:>14,.2f}  kg CO2-eq")
print(f"  Z3 (Jobs)  = {g3:>14,.4f}")

# ─────────────────────────────────────────────────────────
# 10. TOPSIS
# ─────────────────────────────────────────────────────────

def topsis(solutions):
    if not solutions:
        return []
    M  = np.array([[s['Z1'], s['Z2'], s['Z3']] for s in solutions])
    R  = M / (np.sqrt((M**2).sum(axis=0)) + 1e-9)
    V  = R * (1/3)
    # Z1,Z2 minimise → ideal = min; Z3 maximise → ideal = max
    A_pos = np.array([V[:,0].min(), V[:,1].min(), V[:,2].max()])
    A_neg = np.array([V[:,0].max(), V[:,1].max(), V[:,2].min()])
    d_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
    d_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
    return d_neg / (d_pos + d_neg + 1e-9)

scores_eps = topsis(pareto_eps)
scores_lp  = topsis(pareto_lp)

avg_eps = np.mean(scores_eps) if len(scores_eps) > 0 else 0
avg_lp  = np.mean(scores_lp)  if len(scores_lp)  > 0 else 0

print("\n" + "=" * 55)
print("  STEP 5: TOPSIS Comparison")
print("=" * 55)
print(f"  ε-Constraint avg TOPSIS : {avg_eps:.4f}")
print(f"  LP-Metric    avg TOPSIS : {avg_lp:.4f}")
winner = "LP-Metric" if avg_lp > avg_eps else "ε-Constraint"
print(f"  Superior method         : {winner}")

# Best solutions
if len(scores_eps) > 0:
    bi = np.argmax(scores_eps)
    be = pareto_eps[bi]
    print(f"\n  Best ε-Constraint → Z1=${be['Z1']:,.2f}  Z2={be['Z2']:,.2f}  Z3={be['Z3']:.4f}  TOPSIS={scores_eps[bi]:.4f}")
if len(scores_lp) > 0:
    bi2 = np.argmax(scores_lp)
    bl  = pareto_lp[bi2]
    print(f"  Best LP-Metric    → Z1=${bl['Z1']:,.2f}  Z2={bl['Z2']:,.2f}  Z3={bl['Z3']:.4f}  TOPSIS={scores_lp[bi2]:.4f}")

# ─────────────────────────────────────────────────────────
# 11. OPTIMAL FLOW (GCM solution)
# ─────────────────────────────────────────────────────────

m_gcm = res_gcm[3]
print("\n" + "=" * 55)
print("  STEP 6: Optimal Product Flow (GCM Solution)")
print("=" * 55)

for j in range(2):
    name = "Metal drum" if j == 0 else "Plastic drum"
    coll = sum(value(m_gcm.Cv[j,c]) for c in C_set)
    recy = sum(value(m_gcm.Y[j,k])  for k in K_set)
    purch= sum(value(m_gcm.S[j,k])  for k in K_set)
    print(f"  {name}: purchased={purch:.0f}  collected={coll:.0f}  "
          f"refurb={value(m_gcm.F[j]):.0f}  recycled={recy:.0f}  "
          f"disposed={value(m_gcm.W[j])+value(m_gcm.RD[j]):.0f}")

# ─────────────────────────────────────────────────────────
# 12. PLOTS
# ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Multi-Objective Circular Supply Chain — EPR Paint Packaging\n"
             "Poonia et al. (2025) | Pyomo + GLPK", fontsize=13, fontweight='bold')

# 3D Pareto front: ε-constraint
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
if pareto_eps:
    xs = [s['Z1']/1000 for s in pareto_eps]
    ys = [s['Z2']/1000 for s in pareto_eps]
    zs = [s['Z3']      for s in pareto_eps]
    sc = ax1.scatter(xs, ys, zs, c=scores_eps, cmap='viridis', s=60, alpha=0.8)
    bi = np.argmax(scores_eps)
    ax1.scatter([xs[bi]], [ys[bi]], [zs[bi]], c='red', s=150, marker='*', label='Best')
ax1.set_xlabel('Z1 Cost (k$)'); ax1.set_ylabel('Z2 CO2 (k kg)'); ax1.set_zlabel('Z3 Jobs')
ax1.set_title(f'ε-Constraint\nAvg TOPSIS = {avg_eps:.4f}', fontsize=9)
ax1.legend(fontsize=7)

# 3D Pareto front: LP-metric
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
if pareto_lp:
    xs2 = [s['Z1']/1000 for s in pareto_lp]
    ys2 = [s['Z2']/1000 for s in pareto_lp]
    zs2 = [s['Z3']      for s in pareto_lp]
    sc2 = ax2.scatter(xs2, ys2, zs2, c=scores_lp, cmap='plasma', s=60, alpha=0.8)
    bi2 = np.argmax(scores_lp)
    ax2.scatter([xs2[bi2]], [ys2[bi2]], [zs2[bi2]], c='red', s=150, marker='*', label='Best')
ax2.set_xlabel('Z1 Cost (k$)'); ax2.set_ylabel('Z2 CO2 (k kg)'); ax2.set_zlabel('Z3 Jobs')
ax2.set_title(f'LP-Metric\nAvg TOPSIS = {avg_lp:.4f}', fontsize=9)
ax2.legend(fontsize=7)

# TOPSIS bar chart
ax3 = fig.add_subplot(2, 3, 3)
ax3.bar(range(len(scores_eps)), sorted(scores_eps, reverse=True),
        color='steelblue', alpha=0.8, label=f'ε-c (avg={avg_eps:.3f})')
ax3.bar(range(len(scores_lp)),  sorted(scores_lp,  reverse=True),
        color='orange', alpha=0.6, label=f'LP (avg={avg_lp:.3f})')
ax3.axhline(avg_eps, color='steelblue', linestyle='--', linewidth=1.5)
ax3.axhline(avg_lp,  color='orange',    linestyle='--', linewidth=1.5)
ax3.set_xlabel('Solution Rank'); ax3.set_ylabel('TOPSIS Score')
ax3.set_title('TOPSIS Score Distribution'); ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 2D trade-off plots
pairs = [('Z1','Z2','Cost (k$)','CO2 (k kg)',4),
         ('Z1','Z3','Cost (k$)','Jobs',5),
         ('Z2','Z3','CO2 (k kg)','Jobs',6)]

for xa, ya, xl, yl, pos in pairs:
    ax = fig.add_subplot(2, 3, pos)
    scale = lambda v, k: v/1000 if k in ('Z1','Z2') else v
    ax.scatter([scale(s[xa],xa) for s in pareto_eps],
               [scale(s[ya],ya) for s in pareto_eps],
               c='steelblue', s=50, alpha=0.7, label='ε-c')
    ax.scatter([scale(s[xa],xa) for s in pareto_lp],
               [scale(s[ya],ya) for s in pareto_lp],
               c='orange', marker='^', s=50, alpha=0.7, label='LP')
    ax.scatter([scale(g1 if xa=='Z1' else g2 if xa=='Z2' else g3, xa)],
               [scale(g1 if ya=='Z1' else g2 if ya=='Z2' else g3, ya)],
               c='red', marker='*', s=180, zorder=5, label='GCM')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f'{xl} vs {yl}', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pareto_results_pyomo.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as pareto_results_pyomo.png")
