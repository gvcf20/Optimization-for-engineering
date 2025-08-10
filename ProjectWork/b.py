import casadi as ca
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Chatgpt was used during the entire code for bug fixes and comment generation

# ---------- user params ----------
n = 11
h = 0.5
eps_abs = 1e-6          # smoothing for absolute: sqrt(x^2 + eps)
alpha = 1000           # LSE smoothing (larger -> closer to max, slower numerics)
ipopt_maxiter = 5000

# ---------- helpers ----------
def smooth_abs(x, eps=eps_abs):
    return ca.sqrt(x**2 + eps)

def stable_lse(f_list, alpha):
    # compute lse = (1/alpha) * log(sum(exp(alpha * f_k))) in a numerically stable way
    # use a shifted exponent trick using a simple max-approx via softmax-like small alpha_m
    # We compute a crude max estimate by arithmetic max of two then iterate (safe but not fmax)
    # For stability we compute m = max(f) approximately via pairwise comparisons using smooth max:
    # smooth_max(a,b) = (a+b)/2 + sqrt(((a-b)/2)^2 + tiny)
    tiny = 1e-8
    m = f_list[0]
    for fk in f_list[1:]:
        # smooth max between m and fk (differentiable)
        m = 0.5*(m + fk) + ca.sqrt(0.25*(m - fk)**2 + tiny)
    s = 0
    for fk in f_list:
        s = s + ca.exp(alpha * (fk - m))
    return m + (1.0/alpha) * ca.log(s + 1e-16)

# ---------- Stage 1 ----------
def solve_circle_approx():
    opti = ca.Opti()
    b = opti.variable()
    xs = opti.variable(n)
    ys = opti.variable(n)
    thetas = opti.variable(n)

    # containment (corners)
    corners = np.array([[ h,  h],
                        [-h,  h],
                        [-h, -h],
                        [ h, -h]])  # CCW

    for i in range(n):
        xi = xs[i]; yi = ys[i]; th = thetas[i]
        c = ca.cos(th); s = ca.sin(th)
        for (dx, dy) in corners:
            cx = xi + c*dx - s*dy
            cy = yi + s*dx + c*dy
            opti.subject_to(cx >= 0); opti.subject_to(cx <= b)
            opti.subject_to(cy >= 0); opti.subject_to(cy <= b)

    # conservative non-overlap (circumcircle radius r = sqrt(2)/2)
    min_center_sq = 2.0  # (sqrt(2))^2
    for i in range(n):
        for j in range(i):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            opti.subject_to(dx**2 + dy**2 >= min_center_sq - 1e-9)

    # bounds & initial guesses
    opti.subject_to(b >= 1.0); opti.subject_to(b <= 10.0)
    opti.set_initial(b, 4.0)

    # grid initial
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
    opti.set_initial(xs, np.array(ix))
    opti.set_initial(ys, np.array(iy))
    opti.set_initial(thetas, np.zeros(n))

    opti.minimize(b)
    p_opts = {}
    s_opts = {"max_iter": 2000, "tol": 1e-6, "print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol = opti.solve()
    except Exception as e:
        # try relaxed settings / random restart
        opti.set_initial(xs, 0.5 + 3.0*np.random.rand(n))
        opti.set_initial(ys, 0.5 + 3.0*np.random.rand(n))
        opti.set_initial(thetas, np.random.rand(n)*math.pi/2)
        s_opts["max_iter"] = 4000
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

    return {
        "b": float(sol.value(b)),
        "xs": np.array(sol.value(xs)).flatten(),
        "ys": np.array(sol.value(ys)).flatten(),
        "thetas": np.array(sol.value(thetas)).flatten()
    }

# ---------- Stage 2 ----------
def solve_sat_smooth(init=None, alpha=alpha):
    opti = ca.Opti()
    b = opti.variable()
    xs = opti.variable(n)
    ys = opti.variable(n)
    thetas = opti.variable(n)

    corner_offsets = np.array([[ h,  h],
                               [-h,  h],
                               [-h, -h],
                               [ h, -h]])

    # containment exact (corners)
    for i in range(n):
        xi = xs[i]; yi = ys[i]; th = thetas[i]
        c = ca.cos(th); s = ca.sin(th)
        for (dx, dy) in corner_offsets:
            cx = xi + c*dx - s*dy
            cy = yi + s*dx + c*dy
            opti.subject_to(cx >= 0); opti.subject_to(cx <= b)
            opti.subject_to(cy >= 0); opti.subject_to(cy <= b)

    # SAT smooth non-overlap for each pair
    def edge_dirs(th):
        return ca.vertcat(ca.cos(th), ca.sin(th)), ca.vertcat(-ca.sin(th), ca.cos(th))

    for i in range(n):
        for j in range(i):
            xi, yi, ti = xs[i], ys[i], thetas[i]
            xj, yj, tj = xs[j], ys[j], thetas[j]
            e_i1, e_i2 = edge_dirs(ti)
            e_j1, e_j2 = edge_dirs(tj)
            axes = [e_i1, e_i2, e_j1, e_j2]

            f_list = []
            for u in axes:
                ux = u[0]; uy = u[1]
                # smooth absolute projection of center difference
                proj = smooth_abs((xj - xi)*ux + (yj - yi)*uy)
                # projection radii (rho)
                dot1_i = ux*e_i1[0] + uy*e_i1[1]
                dot2_i = ux*e_i2[0] + uy*e_i2[1]
                rho_i = h*(smooth_abs(dot1_i) + smooth_abs(dot2_i))
                dot1_j = ux*e_j1[0] + uy*e_j1[1]
                dot2_j = ux*e_j2[0] + uy*e_j2[1]
                rho_j = h*(smooth_abs(dot1_j) + smooth_abs(dot2_j))
                f_k = proj - (rho_i + rho_j)
                f_list.append(f_k)

            # lse >= 0  (stable_lse uses a smooth approximate max)
            lse = stable_lse(f_list, alpha)
            opti.subject_to(lse >= -1e-8)

    # bounds & init
    opti.subject_to(b >= 1.0); opti.subject_to(b <= 10.0)
    if init is None:
        opti.set_initial(b, 4.0)
        opti.set_initial(xs, 0.5 + 3.0*np.random.rand(n))
        opti.set_initial(ys, 0.5 + 3.0*np.random.rand(n))
        opti.set_initial(thetas, np.random.rand(n)*math.pi/2)
    else:
        opti.set_initial(b, init["b"])
        opti.set_initial(xs, init["xs"])
        opti.set_initial(ys, init["ys"])
        opti.set_initial(thetas, init["thetas"])

    opti.minimize(b)
    p_opts = {}
    s_opts = {"max_iter": ipopt_maxiter, "tol": 1e-10, "print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)

    sol = opti.solve()
    return {
        "b": float(sol.value(b)),
        "xs": np.array(sol.value(xs)).flatten(),
        "ys": np.array(sol.value(ys)).flatten(),
        "thetas": np.array(sol.value(thetas)).flatten()
    }

# ---------- run 2-stage solve ----------
print("Stage A: circle approximation (fast) ...")
solA = solve_circle_approx()
print("Stage A done. b =", solA["b"])

print("Stage B: SAT smooth warm-started from Stage A ...")
solB = solve_sat_smooth(init=solA, alpha=alpha)
print("Stage B done. b =", solB["b"])

# ---------- plot Stage 2 solution ----------
b_val = solB["b"]; xs_val = solB["xs"]; ys_val = solB["ys"]; thetas_val = solB["thetas"]
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, b_val); ax.set_ylim(0, b_val)
for i in range(n):
    th = thetas_val[i]; xi = xs_val[i]; yi = ys_val[i]
    c = math.cos(th); s = math.sin(th)
    corners_xy = []
    for (dx, dy) in [[ h, h], [-h, h], [-h, -h], [h, -h]]:
        cx = xi + c*dx - s*dy
        cy = yi + s*dx + c*dy
        corners_xy.append((cx, cy))
    ax.add_patch(patches.Polygon(corners_xy, closed=True, edgecolor='k', alpha=0.7, facecolor='C0'))
    ax.plot(xi, yi, 'k.', ms=4)
ax.add_patch(patches.Rectangle((0,0), b_val, b_val, fill=False, lw=2))
ax.set_aspect('equal', adjustable='box')
ax.set_title(f"Stage B packing (SAT-smooth) b = {b_val:.4f}")
plt.show()
