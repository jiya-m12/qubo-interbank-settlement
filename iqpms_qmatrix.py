"""
IQPMS Q-Matrix for Interbank Payment Settlement (Gridlock Scenario)
====================================================================

Implements the QUBO formulation from De Santis et al. (2026) using the
Iterative Quadratic Polynomial (IQP) and Master-Satellite (MS) methods.

There is ONE Q-matrix. IQP and MS are construction techniques for the
penalty polynomials within that single matrix:

    Q(G, x, s) = W(x) + sum_u lam_u [ lam^IO_u * P^IO_u(x,s) + P^LIQ_u(x,s) ]

where:
    W(x)       = objective (total settled value)
    P^IO_u     = IN/OUT master penalty (IQP method, paper Eqs. 53-55)
    P^LIQ_u    = liquidity satellite penalty (MS + IQP, instance-specific)
    lam_u      = node penalty multiplier (Eq. 67, local tuning)
    lam^IO_u   = IN/OUT relative multiplier (Eq. 66)

References:
    De Santis et al., "Optimized QUBO formulation methods for quantum
    computing," arXiv:2406.07681v2 (2024).
"""

import numpy as np
from itertools import product as cartesian_product
from math import ceil, log2
from dataclasses import dataclass, field


# ===================================================================
# 1. NETWORK DATA STRUCTURES
# ===================================================================

@dataclass
class Arc:
    src: int
    dst: int
    weight: float
    index: int

@dataclass
class Bank:
    idx: int
    liquidity: float
    outgoing: list = field(default_factory=list)
    incoming: list = field(default_factory=list)

@dataclass
class Network:
    banks: list
    arcs: list

    @property
    def N_banks(self):
        return len(self.banks)

    @property
    def N_arcs(self):
        return len(self.arcs)


def generate_gridlock_network(seed=42):
    """
    6 banks, 9 arcs forming two interlocking cycles + cross-links.
    Liquidity at 25% of outflow => genuine gridlock.
    """
    arc_defs = [
        (0, 1, 8),  (1, 2, 7),  (2, 0, 9),   # cycle 1
        (2, 3, 6),  (3, 4, 5),  (4, 5, 7),    # cycle 2
        (5, 2, 8),  (1, 3, 4),  (4, 0, 3),    # cycle 2 close + cross
    ]

    banks = [Bank(idx=i, liquidity=0.0) for i in range(6)]
    arcs = []

    for idx, (s, d, w) in enumerate(arc_defs):
        arc = Arc(src=s, dst=d, weight=w, index=idx)
        arcs.append(arc)
        banks[s].outgoing.append(idx)
        banks[d].incoming.append(idx)

    for b in banks:
        total_out = sum(arcs[a].weight for a in b.outgoing)
        b.liquidity = ceil(0.25 * total_out)

    return Network(banks=banks, arcs=arcs)


# ===================================================================
# 2. CONSTRAINT CHECKS
# ===================================================================

def check_inout(x_local, n_in, n_out):
    """IN/OUT: either all zero, or >=1 incoming AND >=1 outgoing active."""
    inc = x_local[:n_in]
    out = x_local[n_in:]
    any_in = sum(inc) > 0
    any_out = sum(out) > 0
    if not any_in and not any_out:
        return True
    return any_in and any_out


def check_liquidity(x_local, w_local, n_in, n_out, capacity):
    """net_outflow = outflow - inflow <= capacity"""
    inflow = sum(w * x for w, x in zip(w_local[:n_in], x_local[:n_in]))
    outflow = sum(w * x for w, x in zip(w_local[n_in:], x_local[n_in:]))
    return (outflow - inflow) <= capacity + 1e-9


# ===================================================================
# 3. QUADRATIC POLYNOMIAL TOOLS
# ===================================================================

def all_assignments(M):
    """All 2^M binary assignments."""
    return [list(b) for b in cartesian_product([0, 1], repeat=M)]


def monomial_features(x, M):
    """
    Feature vector for quadratic polynomial evaluation.
    Order: [1, x0, x1, ..., x_{M-1}, x0*x1, x0*x2, ..., x_{M-2}*x_{M-1}]
    """
    v = [1.0]
    for i in range(M):
        v.append(float(x[i]))
    for i in range(M):
        for j in range(i+1, M):
            v.append(float(x[i] * x[j]))
    return v


def n_params(M):
    """Number of parameters in a quadratic polynomial over M variables."""
    return 1 + M + M*(M-1)//2


def param_index_to_pair(M):
    """Map parameter index to (i,j) pair. (-1,-1)=constant, (i,i)=linear."""
    pairs = [(-1, -1)]
    for i in range(M):
        pairs.append((i, i))
    for i in range(M):
        for j in range(i+1, M):
            pairs.append((i, j))
    return pairs


def coeffs_to_dict(coeffs_vec, M):
    """Convert coefficient vector to {(i,j): value} dict."""
    pairs = param_index_to_pair(M)
    return {p: float(c) for p, c in zip(pairs, coeffs_vec) if abs(c) > 1e-10}


def eval_poly(coeffs_dict, x):
    """Evaluate P(x) from coefficient dict."""
    M = len(x)
    val = coeffs_dict.get((-1, -1), 0.0)
    for i in range(M):
        val += coeffs_dict.get((i, i), 0.0) * x[i]
        for j in range(i+1, M):
            val += coeffs_dict.get((i, j), 0.0) * x[i] * x[j]
    return val


# ===================================================================
# 4. IQP SOLVER (Constrained: satisfy=equality, violate=inequality)
# ===================================================================

def solve_iqp_no_slack(M, satisfy, violate):
    """
    Find quadratic polynomial P(x) with minimal coefficient magnitude.
    P(x) = 0 for satisfy, P(x) <= -1 for violate.
    Minimizes max|coefficient| via LP for well-conditioned Q-matrix.
    """
    from scipy.optimize import linprog
    np_par = n_params(M)

    if not satisfy and not violate:
        return coeffs_to_dict(np.zeros(np_par), M)

    A_eq = np.array([monomial_features(x, M) for x in satisfy]) if satisfy else np.zeros((0, np_par))
    A_viol = np.array([monomial_features(x, M) for x in violate]) if violate else np.zeros((0, np_par))

    if len(satisfy) == 0:
        b_viol = -np.ones(len(violate))
        result, _, _, _ = np.linalg.lstsq(A_viol, b_viol, rcond=None)
        pred = A_viol @ result
        if all(p <= -1.0 + 1e-6 for p in pred):
            return coeffs_to_dict(result, M)
        return None

    # Null-space approach: solve satisfy exactly, optimize within DOF
    U, S, Vt = np.linalg.svd(A_eq, full_matrices=True)
    rank = np.sum(S > 1e-10)
    particular = np.zeros(np_par)
    null_basis = Vt[rank:].T
    n_null = null_basis.shape[1] if null_basis.ndim > 1 else 0

    if n_null == 0:
        pred = A_viol @ particular
        if all(p <= -1.0 + 1e-6 for p in pred):
            return coeffs_to_dict(particular, M)
        return None

    C = A_viol @ null_basis
    d = -1.0 - A_viol @ particular

    # Minimize max|coefficient|: variables = [alpha_1..n_null, t]
    # min t s.t. C @ alpha <= d, null_basis @ alpha <= t, -null_basis @ alpha <= t
    n_vars = n_null + 1
    obj = np.zeros(n_vars)
    obj[-1] = 1.0  # minimize t

    # Violate constraints: C @ alpha <= d → [C | 0] @ [alpha,t] <= d
    A1 = np.hstack([C, np.zeros((C.shape[0], 1))])
    b1 = d

    # Coeff bounds: null_basis @ alpha - t <= 0 → [NB | -1] @ [alpha,t] <= 0
    A2 = np.hstack([null_basis, -np.ones((np_par, 1))])
    b2 = np.zeros(np_par)
    # -null_basis @ alpha - t <= 0 → [-NB | -1] @ [alpha,t] <= 0
    A3 = np.hstack([-null_basis, -np.ones((np_par, 1))])
    b3 = np.zeros(np_par)

    A_ub = np.vstack([A1, A2, A3])
    b_ub = np.concatenate([b1, b2, b3])

    bounds = [(None, None)] * n_null + [(0, None)]  # t >= 0
    res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        alpha = res.x[:n_null]
        coeffs = particular + null_basis @ alpha
        p_eq = A_eq @ coeffs
        p_vi = A_viol @ coeffs
        if (all(abs(p) < 1e-6 for p in p_eq) and
            all(p <= -1.0 + 1e-6 for p in p_vi)):
            return coeffs_to_dict(coeffs, M)

    return None


def solve_iqp_with_slack(M, satisfy, violate, n_slack):
    """
    IQP with slack variables. For satisfying x:
      ∃ s: P(x,s) = 0  (witness — can differ per assignment)
      ∀ s: P(x,s) ≤ 0
    For violating x:
      ∀ s: P(x,s) ≤ -1

    Searches over witness patterns (which s is the witness for each
    satisfying assignment). Tractable for M ≤ 6.
    """
    from scipy.optimize import linprog

    total = M + n_slack
    np_par = n_params(total)
    slack_assigns = all_assignments(n_slack)
    n_sat = len(satisfy)
    n_sc = len(slack_assigns)  # 2^n_slack

    n_patterns = n_sc ** n_sat
    if n_patterns > 50000:
        pattern_iter = _heuristic_patterns(n_sat, n_sc)
    else:
        pattern_iter = range(n_patterns)

    for pidx in pattern_iter:
        # Decode: witness index for each satisfy assignment
        witnesses = []
        p = pidx
        for _ in range(n_sat):
            witnesses.append(p % n_sc)
            p //= n_sc

        eq_rows, eq_rhs = [], []
        ub_rows, ub_rhs = [], []

        for i, x in enumerate(satisfy):
            # Equality: P(x, witness_s) = 0
            x_wit = list(x) + list(slack_assigns[witnesses[i]])
            eq_rows.append(monomial_features(x_wit, total))
            eq_rhs.append(0.0)
            # Inequality: P(x, other_s) <= 0
            for j in range(n_sc):
                if j != witnesses[i]:
                    x_o = list(x) + list(slack_assigns[j])
                    ub_rows.append(monomial_features(x_o, total))
                    ub_rhs.append(0.0)

        for x in violate:
            for s in slack_assigns:
                x_ext = list(x) + list(s)
                ub_rows.append(monomial_features(x_ext, total))
                ub_rhs.append(-1.0)

        A_eq_orig = np.array(eq_rows)
        b_eq_orig = np.array(eq_rhs)
        A_ub_orig = np.array(ub_rows) if ub_rows else np.zeros((0, np_par))
        b_ub_orig = np.array(ub_rhs) if ub_rhs else np.zeros(0)

        # Minimize max|coefficient|: add auxiliary variable t
        # Variables: [c_0..c_{np-1}, t]
        n_vars = np_par + 1
        obj = np.zeros(n_vars); obj[-1] = 1.0  # min t

        # Original inequality: [A_ub | 0] @ [c, t] <= b_ub
        A_ub_ext = np.hstack([A_ub_orig, np.zeros((A_ub_orig.shape[0], 1))])
        b_ub_ext = b_ub_orig.copy()

        # Coeff bounds: c_i <= t → c_i - t <= 0
        I_pos = np.hstack([np.eye(np_par), -np.ones((np_par, 1))])
        I_neg = np.hstack([-np.eye(np_par), -np.ones((np_par, 1))])
        A_ub_full = np.vstack([A_ub_ext, I_pos, I_neg])
        b_ub_full = np.concatenate([b_ub_ext, np.zeros(2 * np_par)])

        # Equality: [A_eq | 0] @ [c, t] = b_eq
        A_eq_ext = np.hstack([A_eq_orig, np.zeros((A_eq_orig.shape[0], 1))])
        b_eq_ext = b_eq_orig.copy()

        bounds = [(None, None)] * np_par + [(0, None)]
        res = linprog(
            c=obj, A_ub=A_ub_full, b_ub=b_ub_full,
            A_eq=A_eq_ext if len(eq_rows) else None,
            b_eq=b_eq_ext if len(eq_rows) else None,
            bounds=bounds, method='highs', options={'presolve': True}
        )
        if not res.success:
            continue

        # Full verification
        coeffs = res.x[:np_par]
        ok = True
        for x in satisfy:
            has_zero = False
            for s in slack_assigns:
                xs = list(x) + list(s)
                v = sum(c * f for c, f in zip(coeffs, monomial_features(xs, total)))
                if v > 1e-6:
                    ok = False; break
                if abs(v) < 1e-5:
                    has_zero = True
            if not has_zero or not ok:
                ok = False; break

        if ok:
            for x in violate:
                for s in slack_assigns:
                    xs = list(x) + list(s)
                    v = sum(c * f for c, f in zip(coeffs, monomial_features(xs, total)))
                    if v > -1.0 + 1e-6:
                        ok = False; break
                if not ok:
                    break

        if ok:
            return coeffs_to_dict(coeffs, total)

    return None


def _heuristic_patterns(n_sat, n_sc):
    """Generate heuristic witness patterns when exhaustive is too large."""
    patterns = set()
    # Each uniform witness
    for w in range(n_sc):
        patterns.add(sum(w * (n_sc ** i) for i in range(n_sat)))
    # First assignment uses each witness, rest use 0
    for w in range(n_sc):
        patterns.add(w)
    # First assignment uses each witness, rest use 1
    for w in range(n_sc):
        p = w + sum(1 * (n_sc ** i) for i in range(1, n_sat))
        patterns.add(p)
    return sorted(patterns)


def solve_iqp(M, satisfy, violate, max_slack=4):
    """Try IQP with 0, 1, 2, ... slack variables until success."""
    # Try no slack first
    coeffs = solve_iqp_no_slack(M, satisfy, violate)
    if coeffs is not None:
        return coeffs, 0

    # Try with slack
    for ns in range(1, max_slack + 1):
        coeffs = solve_iqp_with_slack(M, satisfy, violate, ns)
        if coeffs is not None:
            return coeffs, ns

    return None, -1


# ===================================================================
# 5. IN/OUT MASTER PENALTY
# ===================================================================

def build_inout_master(n_in, n_out):
    """
    Build IN/OUT penalty polynomial.

    M=2,3: Paper's closed-form (Eqs. 52-53), verified correct.
    M>=4: Numerical IQP solver (paper's Eqs. 54-55 are buggy).

    Local ordering: [incoming_0,...,incoming_{n_in-1}, outgoing_0,...,outgoing_{n_out-1}]

    Returns (coeffs_dict, n_slack, reorder_map)
    """
    M = n_in + n_out

    if M == 0:
        return {}, 0, []
    if M == 1:
        return {(0, 0): -1.0}, 0, [0]

    reorder = list(range(M))

    if M == 2:
        # 1vs1 (Eq. 52): P = -x0 - x1 + 2*x0*x1
        coeffs = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
        return coeffs, 0, reorder

    if M == 3:
        # Paper Eq. 53 in paper ordering (x0=minority, x1,x2=majority):
        # P = -x0 - x1 - x2 + 2*x0*x1 + 2*x0*x2 - x1*x2
        #
        # Map to our local ordering [in_0,...,in_{n_in-1}, out_0,...,out_{n_out-1}]:
        minority = min(n_in, n_out)
        if n_in <= n_out:
            # 1vs2: minority=incoming(x0), majority=outgoing(x1,x2)
            # local matches paper ordering directly
            coeffs = {
                (0, 0): -1.0, (1, 1): -1.0, (2, 2): -1.0,
                (0, 1): 2.0, (0, 2): 2.0, (1, 2): -1.0,
            }
        else:
            # 2vs1: minority=outgoing(local x2), majority=incoming(local x0,x1)
            # Paper x0 -> local x2, paper x1 -> local x0, paper x2 -> local x1
            coeffs = {
                (0, 0): -1.0, (1, 1): -1.0, (2, 2): -1.0,
                (0, 1): -1.0, (0, 2): 2.0, (1, 2): 2.0,
            }
        return coeffs, 0, reorder

    # M >= 4: use numerical IQP solver with coefficient minimization
    assigns = all_assignments(M)
    satisfy = [x for x in assigns if check_inout(x, n_in, n_out)]
    violate = [x for x in assigns if not check_inout(x, n_in, n_out)]

    coeffs, ns = solve_iqp(M, satisfy, violate)
    if coeffs is None:
        raise RuntimeError(f"IQP failed for IN/OUT with n_in={n_in}, n_out={n_out}, M={M}")

    return coeffs, ns, reorder


# ===================================================================
# 6. LIQUIDITY SATELLITE PENALTY (MS + IQP)
# ===================================================================

def build_liquidity_satellite(n_in, n_out, w_in, w_out, capacity):
    """
    Build satellite penalty for liquidity under IN/OUT master.

    Only enforce for IN/OUT-satisfying assignments:
      - P = 0 if both IO and LIQ satisfied
      - P <= -1 if IO satisfied but LIQ violated
      - P = 0 for IO-violated (keeps P non-positive everywhere,
        minimizing lam^IO inflation)

    Returns (coeffs_dict, n_slack)
    """
    M = n_in + n_out
    w_local = list(w_in) + list(w_out)

    assigns = all_assignments(M)
    satisfy = []
    violate = []

    for x in assigns:
        if check_inout(x, n_in, n_out):
            if check_liquidity(x, w_local, n_in, n_out, capacity):
                satisfy.append(x)
            else:
                violate.append(x)
        else:
            # IO-violating: satellite is unconstrained, but we add to
            # satisfy (P=0) to keep polynomial non-positive everywhere.
            # This reduces lam^IO.
            satisfy.append(x)

    if not violate:
        # Satellite trivially satisfied for all IO-satisfying assignments
        return {}, 0

    coeffs, ns = solve_iqp(M, satisfy, violate)
    if coeffs is None:
        # If adding IO-violating as satisfy makes it infeasible,
        # fall back to unconstrained
        satisfy_core = [x for x in assigns
                        if check_inout(x, n_in, n_out)
                        and check_liquidity(x, w_local, n_in, n_out, capacity)]
        violate_core = [x for x in assigns
                        if check_inout(x, n_in, n_out)
                        and not check_liquidity(x, w_local, n_in, n_out, capacity)]
        coeffs, ns = solve_iqp(M, satisfy_core, violate_core)
        if coeffs is None:
            raise RuntimeError(
                f"IQP failed for satellite LIQ: M={M}, "
                f"{len(satisfy_core)} satisfy, {len(violate_core)} violate"
            )
    return coeffs, ns


# ===================================================================
# 7. PENALTY MULTIPLIERS (Paper Eqs. 66-67)
# ===================================================================

def compute_lambda_io(sat_coeffs, M, n_sat_slack, gamma=2.0):
    """
    lam^IO_u = 1 + gamma * max(0, max_{x,s} P^SAT(x,s))
    Amplifies master to suppress satellite accidental incentives.
    """
    if not sat_coeffs:
        return 1.0

    total = M + n_sat_slack
    max_val = -1e18
    for x in all_assignments(total):
        v = eval_poly(sat_coeffs, x)
        if v > max_val:
            max_val = v
    return 1.0 + gamma * max(0.0, max_val)


def compute_lambda_u(bank, arcs, gamma=2.0):
    """lam_u = gamma * sum of all weights on arcs incident to u. (Eq. 67)"""
    return gamma * sum(arcs[a].weight for a in bank.incoming + bank.outgoing)


# ===================================================================
# 8. Q-MATRIX ASSEMBLY: IQPMS
# ===================================================================

def build_iqpms_qubo(net, gamma=2.0):
    """
    Build the full IQPMS Q-matrix.

    Q(G, x, s) = W(x) + sum_u lam_u * [lam^IO_u * P^IO_u + P^LIQ_u]

    Returns (Q, n_logical, n_total, var_info)
    """
    N = net.N_arcs
    node_info = []
    total_slack = 0

    for bank in net.banks:
        n_in = len(bank.incoming)
        n_out = len(bank.outgoing)
        M = n_in + n_out

        if M == 0:
            node_info.append(None)
            continue

        local_arcs = bank.incoming + bank.outgoing
        w_in = [net.arcs[a].weight for a in bank.incoming]
        w_out = [net.arcs[a].weight for a in bank.outgoing]

        # Master
        io_coeffs, io_ns, io_reorder = build_inout_master(n_in, n_out)

        # Satellite
        liq_coeffs, liq_ns = build_liquidity_satellite(
            n_in, n_out, w_in, w_out, bank.liquidity
        )

        # Multipliers
        lam_io = compute_lambda_io(liq_coeffs, M, liq_ns, gamma)
        lam_u = compute_lambda_u(bank, net.arcs, gamma)

        slack_start = N + total_slack
        node_slack = io_ns + liq_ns

        node_info.append({
            'bank': bank.idx,
            'n_in': n_in, 'n_out': n_out, 'M': M,
            'local_arcs': local_arcs,
            'io_coeffs': io_coeffs, 'io_ns': io_ns, 'io_reorder': io_reorder,
            'liq_coeffs': liq_coeffs, 'liq_ns': liq_ns,
            'lam_io': lam_io, 'lam_u': lam_u,
            'slack_start': slack_start, 'node_slack': node_slack,
        })
        total_slack += node_slack

    n_total = N + total_slack
    Q = np.zeros((n_total, n_total))

    # Objective: W(x) = sum w_i * x_i
    for arc in net.arcs:
        Q[arc.index, arc.index] += arc.weight

    # Add penalties per node
    for info in node_info:
        if info is None:
            continue

        M = info['M']
        local_arcs = info['local_arcs']
        reorder = info['io_reorder']
        lam_u = info['lam_u']
        lam_io = info['lam_io']
        slack_start = info['slack_start']
        io_ns = info['io_ns']

        # Global index mapper for master penalty
        # Since we use identity reorder (numerical IQP in local ordering),
        # paper_idx == local_idx for logical variables.
        def io_global(var_idx):
            """Map polynomial variable index to global Q index."""
            if var_idx < M:
                return net.arcs[local_arcs[var_idx]].index
            else:
                # IO slack variable
                return slack_start + (var_idx - M)

        # Add lam_u * lam_io * P^IO
        for (pi, pj), c in info['io_coeffs'].items():
            if pi == -1:  # constant
                continue
            gi = io_global(pi)
            gj = io_global(pj)
            val = lam_u * lam_io * c
            if gi == gj:
                Q[gi, gi] += val
            else:
                i, j = min(gi, gj), max(gi, gj)
                Q[i, j] += val

        # Global index mapper for satellite penalty
        def liq_global(local_idx):
            """Map satellite's local index to global Q index."""
            if local_idx < M:
                return net.arcs[local_arcs[local_idx]].index
            else:
                # LIQ slack, starts after IO slack
                return slack_start + io_ns + (local_idx - M)

        # Add lam_u * P^LIQ
        for (li, lj), c in info['liq_coeffs'].items():
            if li == -1:
                continue
            gi = liq_global(li)
            gj = liq_global(lj)
            val = lam_u * c
            if gi == gj:
                Q[gi, gi] += val
            else:
                i, j = min(gi, gj), max(gi, gj)
                Q[i, j] += val

    return Q, N, n_total, {'n_slack': total_slack, 'node_info': node_info}


# ===================================================================
# 9. STANDARD Q-MATRIX (COMPARISON)
# ===================================================================

def build_standard_qubo(net, gamma=2.0):
    """Standard squared-penalty QUBO with binary-encoded slack."""
    N = net.N_arcs
    slack_info = []
    total_slack = 0

    for bank in net.banks:
        if not bank.outgoing and not bank.incoming:
            slack_info.append({'n_bits': 0, 'start': N + total_slack, 'max_s': 0})
            continue
        max_in = sum(net.arcs[a].weight for a in bank.incoming)
        max_s = int(bank.liquidity + max_in)
        n_bits = max(1, ceil(log2(max_s + 1))) if max_s > 0 else 1
        slack_info.append({'n_bits': n_bits, 'start': N + total_slack, 'max_s': max_s})
        total_slack += n_bits

    n_total = N + total_slack
    Q = np.zeros((n_total, n_total))

    for arc in net.arcs:
        Q[arc.index, arc.index] += arc.weight

    for bank in net.banks:
        si = slack_info[bank.idx]
        if not bank.outgoing and not bank.incoming:
            continue

        terms = []
        for a in bank.outgoing:
            terms.append((net.arcs[a].index, net.arcs[a].weight))
        for a in bank.incoming:
            terms.append((net.arcs[a].index, -net.arcs[a].weight))
        for k in range(si['n_bits']):
            terms.append((si['start'] + k, 2**k))

        const = -bank.liquidity
        lam = gamma * sum(net.arcs[a].weight for a in bank.incoming + bank.outgoing)

        # Expand penalty: -lam * (sum c_i x_i + const)^2
        for ti in range(len(terms)):
            gi, ci = terms[ti]
            Q[gi, gi] -= lam * (ci*ci + 2*const*ci)
            for tj in range(ti+1, len(terms)):
                gj, cj = terms[tj]
                i, j = min(gi, gj), max(gi, gj)
                Q[i, j] -= lam * 2 * ci * cj

    return Q, N, n_total, {'n_slack': total_slack, 'slack_info': slack_info}


# ===================================================================
# 10. VERIFICATION & SOLVERS
# ===================================================================

def check_feasibility(x_log, net):
    """Check feasibility and return settled value."""
    for bank in net.banks:
        n_in = len(bank.incoming)
        w_in = [net.arcs[a].weight for a in bank.incoming]
        w_out = [net.arcs[a].weight for a in bank.outgoing]
        x_in = [x_log[a] for a in bank.incoming]
        x_out = [x_log[a] for a in bank.outgoing]
        outflow = sum(w*x for w, x in zip(w_out, x_out))
        inflow = sum(w*x for w, x in zip(w_in, x_in))
        if outflow - inflow > bank.liquidity + 1e-9:
            return False, None
    val = sum(net.arcs[a].weight * x_log[a] for a in range(net.N_arcs))
    return True, val


def exhaustive_verify(Q, n_log, n_tot, net):
    """Exhaustive QUBO verification."""
    assert n_tot <= 28, f"n_tot={n_tot} too large for exhaustive"
    best_q = -1e18; best_qx = None
    best_t = -1e18; best_tx = None
    n_feas = 0

    for bits in range(2**n_tot):
        x = np.array([(bits >> k) & 1 for k in range(n_tot)], dtype=float)
        qv = float(x @ Q @ x)
        if qv > best_q:
            best_q = qv
            best_qx = x.copy()

        xl = [int(x[k]) for k in range(n_log)]
        f, tv = check_feasibility(xl, net)
        if f:
            n_feas += 1
            if tv > best_t:
                best_t = tv
                best_tx = xl[:]

    qxl = [int(best_qx[k]) for k in range(n_log)]
    qf, qtv = check_feasibility(qxl, net)
    match = qf and abs(qtv - best_t) < 0.5

    return {
        'match': match,
        'qubo_val': best_q, 'qubo_x': best_qx,
        'qubo_feas': qf, 'qubo_settled': qtv,
        'true_val': best_t, 'true_x': best_tx,
        'n_feas': n_feas, 'n_combos': 2**n_tot,
    }


def simulated_annealing(Q, n_tot, n_sweeps=20000, T_init=None, T_final=None,
                        n_restarts=10, seed=42):
    """SA with vectorized delta, multiple restarts, auto-tuned temperature."""
    rng = np.random.RandomState(seed)

    if T_init is None:
        max_entry = max(abs(Q.max()), abs(Q.min()))
        T_init = max(10.0, max_entry * 2.0)
    if T_final is None:
        T_final = T_init * 1e-4

    # Precompute symmetric matrix for fast delta
    Q_sym = Q + Q.T  # Q_sym[i,j] = Q[i,j] + Q[j,i]
    Q_diag = np.diag(Q)

    best_x, best_e = None, -1e18

    for restart in range(n_restarts):
        if restart == 0:
            x = np.zeros(n_tot)
        elif restart == 1:
            x = np.ones(n_tot)
        else:
            x = rng.randint(0, 2, n_tot).astype(float)

        e = float(x @ Q @ x)
        bx, be = x.copy(), e
        temps = np.geomspace(T_init, T_final, n_sweeps)

        for T in temps:
            for _ in range(n_tot):
                i = rng.randint(n_tot)
                flip = 1.0 - 2.0 * x[i]
                # delta = flip * (sum_{j!=i} Q_sym[i,j]*x[j] + Q[i,i])
                d = flip * (Q_sym[i] @ x - Q_sym[i, i] * x[i] + Q_diag[i])
                if d > 0 or rng.rand() < np.exp(d / max(T, 1e-12)):
                    x[i] = 1.0 - x[i]; e += d
                if e > be:
                    be = e; bx = x.copy()

        if be > best_e:
            best_e = be; best_x = bx.copy()

    return best_x, best_e


def greedy_sequential(net):
    bal = [b.liquidity for b in net.banks]
    arcs_sorted = sorted(range(net.N_arcs), key=lambda a: net.arcs[a].weight, reverse=True)
    settled = [0]*net.N_arcs; total = 0.0
    for a in arcs_sorted:
        arc = net.arcs[a]
        if bal[arc.src] >= arc.weight:
            settled[a] = 1; bal[arc.src] -= arc.weight; bal[arc.dst] += arc.weight
            total += arc.weight
    return total


def greedy_netting(net, rounds=20):
    bal = [b.liquidity for b in net.banks]
    settled = [0]*net.N_arcs; total = 0.0
    for _ in range(rounds):
        prog = False
        for a in sorted([i for i in range(net.N_arcs) if settled[i] == 0],
                        key=lambda a: net.arcs[a].weight, reverse=True):
            arc = net.arcs[a]
            if bal[arc.src] >= arc.weight:
                settled[a] = 1; bal[arc.src] -= arc.weight; bal[arc.dst] += arc.weight
                total += arc.weight; prog = True
        if not prog:
            break
    return total


# ===================================================================
# 11. POLYNOMIAL VERIFICATION
# ===================================================================

def verify_polynomials(net, node_info):
    """Independently verify every penalty polynomial is correct."""
    print("\nPENALTY POLYNOMIAL VERIFICATION")
    print("-" * 65)
    all_ok = True

    for info in node_info:
        if info is None:
            continue
        bank = net.banks[info['bank']]
        n_in, n_out, M = info['n_in'], info['n_out'], info['M']
        w_in = [net.arcs[a].weight for a in bank.incoming]
        w_out = [net.arcs[a].weight for a in bank.outgoing]
        w_local = w_in + w_out
        reorder = info['io_reorder']

        # --- Verify IN/OUT master ---
        io_ok = True
        io_total = M + info['io_ns']

        for x_log in all_assignments(M):
            sat = check_inout(x_log, n_in, n_out)

            # With identity reorder, x_log IS the polynomial's input
            if info['io_ns'] == 0:
                p = eval_poly(info['io_coeffs'], x_log)
                if sat and abs(p) > 1e-6:
                    io_ok = False
                if not sat and p > -1.0 + 1e-6:
                    io_ok = False
            else:
                if sat:
                    found0 = False
                    for s in all_assignments(info['io_ns']):
                        xs = x_log + list(s)
                        p = eval_poly(info['io_coeffs'], xs)
                        if abs(p) < 1e-6:
                            found0 = True
                        if p > 1e-6:
                            io_ok = False
                    if not found0:
                        io_ok = False
                else:
                    for s in all_assignments(info['io_ns']):
                        xs = x_log + list(s)
                        p = eval_poly(info['io_coeffs'], xs)
                        if p > -1.0 + 1e-6:
                            io_ok = False

        # --- Verify liquidity satellite ---
        liq_ok = True
        for x_log in all_assignments(M):
            sat_io = check_inout(x_log, n_in, n_out)
            if not sat_io:
                continue  # unconstrained
            sat_liq = check_liquidity(x_log, w_local, n_in, n_out, bank.liquidity)

            if info['liq_ns'] == 0:
                p = eval_poly(info['liq_coeffs'], x_log)
                if sat_liq and abs(p) > 1e-6:
                    liq_ok = False
                if not sat_liq and p > -1.0 + 1e-6:
                    liq_ok = False
            else:
                if sat_liq:
                    found0 = False
                    for s in all_assignments(info['liq_ns']):
                        xs = x_log + list(s)
                        p = eval_poly(info['liq_coeffs'], xs)
                        if abs(p) < 1e-6:
                            found0 = True
                        if p > 1e-6:
                            liq_ok = False
                    if not found0:
                        liq_ok = False
                else:
                    for s in all_assignments(info['liq_ns']):
                        xs = x_log + list(s)
                        p = eval_poly(info['liq_coeffs'], xs)
                        if p > -1.0 + 1e-6:
                            liq_ok = False

        s_io = "PASS" if io_ok else "FAIL"
        s_liq = "PASS" if liq_ok else "FAIL"
        if not io_ok or not liq_ok:
            all_ok = False
        print(f"  Bank {bank.idx}: N={M} ({n_in}vs{n_out})  "
              f"IO=[{s_io}, {info['io_ns']} slack]  "
              f"LIQ=[{s_liq}, {info['liq_ns']} slack]  "
              f"lam^IO={info['lam_io']:.2f}  lam_u={info['lam_u']:.1f}")

    print(f"\n  Overall: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


# ===================================================================
# 12. MAIN
# ===================================================================

def main():
    print("=" * 70)
    print("IQPMS Q-MATRIX: INTERBANK GRIDLOCK SETTLEMENT")
    print("=" * 70)

    net = generate_gridlock_network()

    # Print network
    total_val = sum(a.weight for a in net.arcs)
    print(f"\nNetwork: {net.N_banks} banks, {net.N_arcs} arcs, total value = {total_val}")
    for b in net.banks:
        tout = sum(net.arcs[a].weight for a in b.outgoing)
        tin = sum(net.arcs[a].weight for a in b.incoming)
        print(f"  Bank {b.idx}: C={b.liquidity:.0f}  out={tout:.0f}({len(b.outgoing)})  "
              f"in={tin:.0f}({len(b.incoming)})  N(u)={len(b.incoming)+len(b.outgoing)}")
    print("\nArcs:")
    for a in net.arcs:
        print(f"  x_{a.index}: {a.src}->{a.dst}  w={a.weight}")

    # ── IQPMS ──
    print(f"\n{'='*70}")
    print("IQPMS Q-MATRIX CONSTRUCTION")
    print("=" * 70)

    Q_iq, nl_iq, nt_iq, info_iq = build_iqpms_qubo(net, gamma=2.0)
    print(f"\n  Logical vars:   {nl_iq}")
    print(f"  Slack vars:     {info_iq['n_slack']}")
    print(f"  Total vars:     {nt_iq}")
    print(f"  Vars/arc:       {nt_iq/nl_iq:.2f}")

    # Verify polynomials
    polys_ok = verify_polynomials(net, info_iq['node_info'])

    # ── STANDARD ──
    print(f"\n{'='*70}")
    print("STANDARD Q-MATRIX (comparison)")
    print("=" * 70)

    Q_st, nl_st, nt_st, info_st = build_standard_qubo(net, gamma=2.0)
    print(f"\n  Logical vars:   {nl_st}")
    print(f"  Slack vars:     {info_st['n_slack']}")
    print(f"  Total vars:     {nt_st}")
    print(f"  Vars/arc:       {nt_st/nl_st:.2f}")

    # ── EXHAUSTIVE VERIFICATION ──
    print(f"\n{'='*70}")
    print("EXHAUSTIVE VERIFICATION")
    print("=" * 70)

    if nt_iq <= 25:
        print(f"\nIQPMS: 2^{nt_iq} = {2**nt_iq:,} configurations...")
        ev_iq = exhaustive_verify(Q_iq, nl_iq, nt_iq, net)
        print(f"  QUBO optimal settles: {ev_iq['qubo_settled']}")
        print(f"  True optimal:         {ev_iq['true_val']}")
        print(f"  Feasible:             {ev_iq['qubo_feas']}")
        print(f"  MATCH:                {'PASS' if ev_iq['match'] else '*** FAIL ***'}")
        print(f"  Feasible configs:     {ev_iq['n_feas']} / {ev_iq['n_combos']}")
    else:
        print(f"\nIQPMS: {nt_iq} vars — using SA only")
        ev_iq = None

    if nt_st <= 25:
        print(f"\nStandard: 2^{nt_st} = {2**nt_st:,} configurations...")
        ev_st = exhaustive_verify(Q_st, nl_st, nt_st, net)
        print(f"  QUBO optimal settles: {ev_st['qubo_settled']}")
        print(f"  True optimal:         {ev_st['true_val']}")
        print(f"  Feasible:             {ev_st['qubo_feas']}")
        print(f"  MATCH:                {'PASS' if ev_st['match'] else '*** FAIL ***'}")
        print(f"  Feasible configs:     {ev_st['n_feas']} / {ev_st['n_combos']}")
    else:
        print(f"\nStandard: {nt_st} vars — using SA only")
        ev_st = None

    # ── SA ──
    print(f"\n{'='*70}")
    print("SIMULATED ANNEALING")
    print("=" * 70)

    sa_iq, _ = simulated_annealing(Q_iq, nt_iq)
    xl_iq = [int(sa_iq[k]) for k in range(nl_iq)]
    f_iq, v_iq = check_feasibility(xl_iq, net)
    print(f"\n  IQPMS SA:    settled={v_iq}  feasible={f_iq}")

    sa_st, _ = simulated_annealing(Q_st, nt_st)
    xl_st = [int(sa_st[k]) for k in range(nl_st)]
    f_st, v_st = check_feasibility(xl_st, net)
    print(f"  Standard SA: settled={v_st}  feasible={f_st}")

    g1 = greedy_sequential(net)
    g2 = greedy_netting(net)

    # ── RESULTS ──
    true_val = ev_iq['true_val'] if ev_iq else (ev_st['true_val'] if ev_st else max(v_iq or 0, v_st or 0))

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Method':<35} {'Settled':>8} {'Feas':>6} {'Vars':>6}")
    print(f"  {'-'*58}")
    print(f"  {'True Optimum':<35} {true_val:>8.0f} {'--':>6} {'--':>6}")
    if ev_iq:
        m = 'PASS' if ev_iq['match'] else 'FAIL'
        print(f"  {'IQPMS exhaustive ['+m+']':<35} {ev_iq['qubo_settled']:>8.0f} "
              f"{'y' if ev_iq['qubo_feas'] else 'n':>6} {nt_iq:>6}")
    if ev_st:
        m = 'PASS' if ev_st['match'] else 'FAIL'
        print(f"  {'Standard exhaustive ['+m+']':<35} {ev_st['qubo_settled']:>8.0f} "
              f"{'y' if ev_st['qubo_feas'] else 'n':>6} {nt_st:>6}")
    print(f"  {'IQPMS SA':<35} {v_iq or 0:>8.0f} {'y' if f_iq else 'n':>6} {nt_iq:>6}")
    print(f"  {'Standard SA':<35} {v_st or 0:>8.0f} {'y' if f_st else 'n':>6} {nt_st:>6}")
    print(f"  {'Greedy sequential':<35} {g1:>8.0f} {'y':>6} {'--':>6}")
    print(f"  {'Greedy with netting':<35} {g2:>8.0f} {'y':>6} {'--':>6}")

    print(f"\n{'='*70}")
    print("VARIABLE REDUCTION")
    print("=" * 70)
    print(f"\n  Standard: {nt_st} vars = {nl_st} logical + {info_st['n_slack']} slack")
    print(f"  IQPMS:    {nt_iq} vars = {nl_iq} logical + {info_iq['n_slack']} slack")
    if info_st['n_slack'] > 0:
        red = (1 - info_iq['n_slack']/info_st['n_slack']) * 100
        print(f"  Slack reduction: {red:.1f}%")

    print(f"\n  Per-node detail:")
    print(f"  {'Bank':>6} {'N(u)':>5} {'Type':>6} {'Std':>6} {'IQPMS':>6} {'IO_s':>5} {'LIQ_s':>6}")
    print(f"  {'-'*42}")
    for i, bank in enumerate(net.banks):
        ni = len(bank.incoming); no = len(bank.outgoing); M = ni + no
        std_s = info_st['slack_info'][i]['n_bits']
        p = info_iq['node_info'][i]
        iq_s = p['node_slack'] if p else 0
        io_s = p['io_ns'] if p else 0
        liq_s = p['liq_ns'] if p else 0
        print(f"  {i:>6} {M:>5} {f'{ni}v{no}':>6} {std_s:>6} {iq_s:>6} {io_s:>5} {liq_s:>6}")

    # ── Q-MATRIX PRINTOUT ──
    print(f"\n{'='*70}")
    print("IQPMS Q-MATRIX (non-zero entries)")
    print("=" * 70)
    print(f"\n  Shape: {Q_iq.shape}")
    print(f"  Non-zeros: {np.count_nonzero(Q_iq)}")
    print(f"  Upper-triangular: {np.allclose(np.tril(Q_iq, -1), 0)}")

    print(f"\n  Diagonal (linear terms):")
    for i in range(nt_iq):
        if abs(Q_iq[i,i]) > 1e-10:
            lbl = f"x_{i}" if i < nl_iq else f"s_{i-nl_iq}"
            print(f"    Q[{i},{i}] = {Q_iq[i,i]:+10.4f}  ({lbl})")

    print(f"\n  Off-diagonal (couplings):")
    ct = 0
    for i in range(nt_iq):
        for j in range(i+1, nt_iq):
            if abs(Q_iq[i,j]) > 1e-10:
                li = f"x_{i}" if i < nl_iq else f"s_{i-nl_iq}"
                lj = f"x_{j}" if j < nl_iq else f"s_{j-nl_iq}"
                print(f"    Q[{i},{j}] = {Q_iq[i,j]:+10.4f}  ({li} * {lj})")
                ct += 1
                if ct >= 40:
                    rem = sum(1 for ii in range(nt_iq) for jj in range(ii+1,nt_iq)
                              if abs(Q_iq[ii,jj]) > 1e-10) - ct
                    if rem > 0:
                        print(f"    ... and {rem} more")
                    break
        if ct >= 40:
            break

    print(f"\n{'='*70}")
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
