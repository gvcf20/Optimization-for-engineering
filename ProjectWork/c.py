import casadi as ca
import numpy as np
import math
import time
import os


# To perform global optimization I used chatgpt with the following prompt:
# Transform this code (b.py) from a local optimization solution problem into one
#  with a global solution.  
# ---------------- user params ----------------
n = 11
h = 0.5
eps_abs = 1e-8
alpha = 30.0           # LSE smoothing parameter
ipopt_maxiter = 2000
N_restarts = 80        # increase for stronger search
seed = 12345
b_bounds = (1.0, 10.0)
out_best = "best_pack11.npz"

np.random.seed(seed)

# ---------------- small helpers ----------------
def smooth_abs(x):
    return ca.sqrt(x**2 + eps_abs)

def build_stageA_opti():
    """Build Stage A (circle approx) and return (opti, b, xs, ys, thetas)."""
    opti = ca.Opti()
    b = opti.variable()                 # scalar
    xs = opti.variable(n)               # vector
    ys = opti.variable(n)
    thetas = opti.variable(n)

    # corner offsets CCW
    corners = np.array([[ h,  h],
                        [-h,  h],
                        [-h, -h],
                        [ h, -h]])
    # containment constraints (exact corners)
    for i in range(n):
        xi = xs[i]; yi = ys[i]; ti = thetas[i]
        c = ca.cos(ti); s = ca.sin(ti)
        for (dx, dy) in corners:
            px = xi + c*dx - s*dy
            py = yi + s*dx + c*dy
            opti.subject_to(px >= 0); opti.subject_to(px <= b)
            opti.subject_to(py >= 0); opti.subject_to(py <= b)

    # conservative circle non-overlap (distance between centers >= sqrt(2))
    min_center_sq = 2.0
    for i in range(n):
        for j in range(i):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            opti.subject_to(dx**2 + dy**2 >= min_center_sq - 1e-9)

    opti.subject_to(b >= b_bounds[0]); opti.subject_to(b <= b_bounds[1])
    opti.minimize(b)
    return opti, b, xs, ys, thetas

def build_stageB_opti():
    """Build Stage B (SAT-smooth) and return (opti, b, xs, ys, thetas)."""
    opti = ca.Opti()
    b = opti.variable()
    xs = opti.variable(n)
    ys = opti.variable(n)
    thetas = opti.variable(n)

    corner_offsets = np.array([[ h,  h],
                               [-h,  h],
                               [-h, -h],
                               [ h, -h]])

    # containment (exact)
    for i in range(n):
        xi = xs[i]; yi = ys[i]; ti = thetas[i]
        c = ca.cos(ti); s = ca.sin(ti)
        for (dx, dy) in corner_offsets:
            px = xi + c*dx - s*dy
            py = yi + s*dx + c*dy
            opti.subject_to(px >= 0); opti.subject_to(px <= b)
            opti.subject_to(py >= 0); opti.subject_to(py <= b)

    # SAT-smooth non-overlap
    tiny = 1e-9
    def edge_dirs(t):
        return ca.vertcat(ca.cos(t), ca.sin(t)), ca.vertcat(-ca.sin(t), ca.cos(t))

    for i in range(n):
        for j in range(i):
            xi = xs[i]; yi = ys[i]; ti = thetas[i]
            xj = xs[j]; yj = ys[j]; tj = thetas[j]
            e_i1, e_i2 = edge_dirs(ti)
            e_j1, e_j2 = edge_dirs(tj)
            axes = [e_i1, e_i2, e_j1, e_j2]

            f_list = []
            for u in axes:
                ux = u[0]; uy = u[1]
                proj = smooth_abs((xj - xi)*ux + (yj - yi)*uy)
                dot1_i = ux*e_i1[0] + uy*e_i1[1]
                dot2_i = ux*e_i2[0] + uy*e_i2[1]
                rho_i = h*(smooth_abs(dot1_i) + smooth_abs(dot2_i))
                dot1_j = ux*e_j1[0] + uy*e_j1[1]
                dot2_j = ux*e_j2[0] + uy*e_j2[1]
                rho_j = h*(smooth_abs(dot1_j) + smooth_abs(dot2_j))
                f_k = proj - (rho_i + rho_j)
                f_list.append(f_k)

            # smooth approximate max -> LSE (stable via pairwise smooth_max for m)
            m = f_list[0]
            for fk in f_list[1:]:
                m = 0.5*(m + fk) + ca.sqrt(0.25*(m - fk)**2 + tiny)
            s = 0
            for fk in f_list:
                s = s + ca.exp(alpha*(fk - m))
            lse = m + (1.0/alpha) * ca.log(s + 1e-16)
            opti.subject_to(lse >= -1e-9)

    opti.subject_to(b >= b_bounds[0]); opti.subject_to(b <= b_bounds[1])
    opti.minimize(b)
    return opti, b, xs, ys, thetas

# ---------------- solve helpers ----------------
def solve_opti(opti, b_var, xs_var, ys_var, thetas_var, init):
    """Set initials from dict init and solve; returns dict with solution or failure."""
    # set initials
    opti.set_initial(b_var, float(init["b"]))
    opti.set_initial(xs_var, init["xs"])
    opti.set_initial(ys_var, init["ys"])
    opti.set_initial(thetas_var, init["thetas"])

    p_opts = {}
    s_opts = {"max_iter": ipopt_maxiter, "tol": 1e-6, "print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except Exception as e:
        # try a single retry with relaxed initial guesses
        try:
            opti.set_initial(b_var, max( init["b"], 4.0 ))
            opti.set_initial(xs_var, 0.5 + np.random.rand(n)*3.5)
            opti.set_initial(ys_var, 0.5 + np.random.rand(n)*3.5)
            opti.set_initial(thetas_var, np.random.rand(n)*math.pi/2)
            sol = opti.solve()
        except Exception as e2:
            return {"ok": False, "error": str(e2)}
    # extract solution using the variable handles directly
    try:
        b_val = float(sol.value(b_var))
        xs_val = np.array(sol.value(xs_var)).flatten()
        ys_val = np.array(sol.value(ys_var)).flatten()
        thetas_val = np.array(sol.value(thetas_var)).flatten()
    except Exception as e:
        return {"ok": False, "error": f"extract failed: {e}"}
    return {"ok": True, "b": b_val, "xs": xs_val, "ys": ys_val, "thetas": thetas_val}

# ---------------- Stage 1 ----------------
print("Stage A: solving circle-approx for warm-start...")
optiA, bA, xsA, ysA, thetasA = build_stageA_opti()
# set initial guesses for Stage A
ix = []; iy = []
grid_y = [0.7, 1.7, 2.7, 3.7]
grid_x = [0.7, 1.7, 2.7]
count = 0
for yy in grid_y:
    for xx in grid_x:
        if count < n:
            ix.append(xx); iy.append(yy); count += 1
if len(ix) < n:
    pad = n - len(ix)
    ix += list(0.5 + 0.8*np.random.rand(pad))
    iy += list(0.5 + 0.8*np.random.rand(pad))
initA = {"b": 4.0, "xs": np.array(ix), "ys": np.array(iy), "thetas": np.zeros(n)}
resA = solve_opti(optiA, bA, xsA, ysA, thetasA, initA)
if not resA["ok"]:
    raise RuntimeError("Stage A failed: " + resA.get("error","unknown"))
warm = {"b": resA["b"], "xs": resA["xs"], "ys": resA["ys"], "thetas": resA["thetas"]}
print(f"Warm-start b = {warm['b']:.9f}")

# ---------------- multi-start Stage 2 ----------------
best = {"b": warm["b"], "xs": warm["xs"], "ys": warm["ys"], "thetas": warm["thetas"], "source": "warm"}
print("Starting multi-start Stage B with", N_restarts, "restarts...")
t_all = time.time()
for k in range(N_restarts):
    optiB, bB, xsB, ysB, thetasB = build_stageB_opti()

    # pick initialization strategy
    r = np.random.rand()
    if k == 0:
        init = warm
        tag = "warm"
    elif r < 0.4:
        # small perturbation of warm
        init = {
            "b": max(1.0, warm["b"] + np.random.randn()*0.05),
            "xs": warm["xs"] + 0.05*np.random.randn(n),
            "ys": warm["ys"] + 0.05*np.random.randn(n),
            "thetas": warm["thetas"] + 0.1*np.random.randn(n)
        }
        tag = "perturb_warm"
    elif r < 0.8:
        # random
        init = {
            "b": 4.0 + (np.random.rand()-0.5)*1.0,
            "xs": 0.5 + np.random.rand(n)*3.5,
            "ys": 0.5 + np.random.rand(n)*3.5,
            "thetas": np.random.rand(n)*math.pi/2
        }
        tag = "random"
    else:
        # structured grid + random rotations
        grid_x = np.linspace(0.7, 3.3, 4)
        grid_y = np.linspace(0.7, 3.3, 3)
        coords = [(xx,yy) for yy in grid_y for xx in grid_x][:n]
        xs0 = np.array([c[0] for c in coords]) + 0.05*np.random.randn(n)
        ys0 = np.array([c[1] for c in coords]) + 0.05*np.random.randn(n)
        init = {"b": 4.0, "xs": xs0, "ys": ys0, "thetas": np.random.rand(n)*math.pi/2}
        tag = "grid"

    t0 = time.time()
    res = solve_opti(optiB, bB, xsB, ysB, thetasB, init)
    t1 = time.time()
    if not res["ok"]:
        print(f"[{k+1}/{N_restarts}] tag={tag:12s} solve FAILED ({res.get('error','')}) time={t1-t0:.1f}s")
        continue

    b_val = res["b"]
    print(f"[{k+1}/{N_restarts}] tag={tag:12s} b={b_val:.6f} time={t1-t0:.1f}s")
    if b_val + 1e-9 < best["b"]:
        best = {"b": b_val, "xs": res["xs"], "ys": res["ys"], "thetas": res["thetas"], "source": tag}
        print("  -> NEW BEST b =", best["b"])
        np.savez(out_best, b=best["b"], xs=best["xs"], ys=best["ys"], thetas=best["thetas"], source=best["source"])

t_all_end = time.time()
print("Multi-start finished in %.1f s" % (t_all_end - t_all))
print("Best b = %.9f (source=%s)" % (best["b"], best["source"]))
# save final best (if not saved yet)
if not os.path.exists(out_best):
    np.savez(out_best, b=best["b"], xs=best["xs"], ys=best["ys"], thetas=best["thetas"], source=best["source"])
else:
    # overwrite with final best
    np.savez(out_best, b=best["b"], xs=best["xs"], ys=best["ys"], thetas=best["thetas"], source=best["source"])

print("Saved best to", out_best)
