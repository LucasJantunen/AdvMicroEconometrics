
import numpy as np
from numpy import linalg as la
from tabulate import tabulate
import pandas as pd

# ==============================
# Core estimators (API-compatible)
# ==============================

def estimate(y: np.ndarray, x: np.ndarray, transform: str = '', T=None, robust_se: bool = False) -> dict:
    """
    Regression of y on x with OLS; variance depends on 'transform' flag.
    If robust_se=True, returns cluster-robust (by firm) or HC if cross-sectional.
    Args:
        y: (NT,1)
        x: (NT,K)
        transform: '', 'fd', 'be', 'fe', 're' (affects df in variance())
        T: Either an int (#periods) for balanced panels OR
           a 1D array-like of group ids (cluster by firm) for robust SE.
    Returns dict with: 'b_hat','se','sigma2','t_values','R2','cov'
    """
    assert y.ndim == 2 and x.ndim == 2 and y.shape[1] == 1 and y.shape[0] == x.shape[0], "Bad shapes for y,x"
    b_hat = est_ols(y, x)
    residual = y - x @ b_hat
    SSR = float(residual.T @ residual)
    ybar = np.mean(y)
    SST = float((y - ybar).T @ (y - ybar))
    R2 = 1 - SSR / SST if SST > 0 else np.nan

    sigma2, cov, se = variance(transform, SSR, x, T)
    if robust_se:
        cov, se = robust(x, residual, T)

    t_values = np.divide(b_hat, se, out=np.full_like(b_hat, np.nan), where=se != 0)
    return {'b_hat': b_hat, 'se': se, 'sigma2': np.array([[sigma2]]), 't_values': t_values, 'R2': np.array([[R2]]), 'cov': cov}


def est_ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS closed-form beta = (X'X)^(-1) X'y"""
    return la.inv(x.T @ x) @ (x.T @ y)


def variance(transform: str, SSR: float, x: np.ndarray, T=None):
    """
    Compute sigma^2 and classical covariance under the assumed transform.
    For panels, 'T' can be an int (#periods) or array-like of firm ids (we use the median T).
    """
    K = x.shape[1]
    Nobs = x.shape[0]

    # infer N and T for panel df adjustment
    if transform in ('', 'fd', 'be'):
        N = Nobs
        df = N - K
    elif transform.lower() == 'fe':
        Ti = _infer_T_from_T_or_ids(T, Nobs)
        # df for within FE with entity dummies removed: N * (T - 1) - K ~= Nobs - N - K
        df = Nobs - _num_groups_from_T_or_ids(T, Nobs) - K
    elif transform.lower() == 're':
        df = Nobs - K
    else:
        raise Exception("Invalid transform provided.")

    df = max(int(df), 1)
    sigma2 = SSR / df
    cov = sigma2 * la.inv(x.T @ x)
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return sigma2, cov, se


def robust(x: np.ndarray, residual: np.ndarray, T=None):
    """
    Heteroskedasticity- and cluster-robust covariance.
    - If T is None or 1 -> HC (White) robust.
    - If T is int -> assume balanced panel with that T (cluster by blocks of length T).
    - If T is array-like -> treat as firm ids and cluster by unique ids (unbalanced OK).
    Returns (cov, se).
    """
    Ainv = la.inv(x.T @ x)
    nobs, K = x.shape

    # Case: Classical HC if T is None or 1
    if (T is None) or (isinstance(T, (int, np.integer)) and int(T) == 1):
        u2x = (residual ** 2) * x
        cov = Ainv @ (x.T @ u2x) @ Ainv
        se = np.sqrt(np.diag(cov)).reshape(-1, 1)
        return cov, se

    # Case: balanced panel if T is int >= 2
    if isinstance(T, (int, np.integer)) and int(T) >= 2:
        T = int(T)
        N = nobs // T
        B = np.zeros((K, K))
        for i in range(N):
            sl = slice(i * T, (i + 1) * T)
            ui = residual[sl]  # (T,1)
            Xi = x[sl, :]      # (T,K)
            B += Xi.T @ (ui @ ui.T) @ Xi
        cov = Ainv @ B @ Ainv
        se = np.sqrt(np.diag(cov)).reshape(-1, 1)
        return cov, se

    # Case: cluster by ids array-like
    ids = np.asarray(T).ravel()
    if ids.shape[0] != nobs:
        raise ValueError("Length of ids passed to robust() must match number of rows in x.")
    B = np.zeros((K, K))
    for gid in np.unique(ids):
        ii = (ids == gid)
        Xi = x[ii, :]
        ui = residual[ii]
        B += Xi.T @ (ui @ ui.T) @ Xi
    cov = Ainv @ B @ Ainv
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return cov, se


def print_table(labels: tuple, results: dict, headers=None, title="Results", _lambda: float = None, **kwargs) -> None:
    """Pretty table printout (LaTeX-friendly via tabulate if needed)."""
    if headers is None:
        headers = ["", "Beta", "Se", "t-values"]
    label_y, label_x = labels
    table = []
    b = results['b_hat'].reshape(-1, 1)
    se = results['se'].reshape(-1, 1)
    t = results['t_values'].reshape(-1, 1)
    for i, name in enumerate(label_x):
        row = [name, float(b[i]), float(se[i]), float(t[i])]
        table.append(row)
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    R2 = float(results['R2'])
    sigma2 = float(results['sigma2'])
    print(f"R\u00b2 = {R2:.3f}")
    print(f"\u03C3\u00b2 = {sigma2:.3f}")
    if _lambda is not None:
        print(f"\u03bb = {float(_lambda):.3f}")


# ==============================
# Firm-data specific helpers
# ==============================

def load_firm_data(csv_path: str, infer_cols: bool = True):
    """
    Load firms.csv and build (y, X, ids, year, labels).
    - Expects log variables: output, capital, labor.
    - Tries common column names: ldsa->y, lcap->k, lemp/lemp/l->l.
    Returns:
       y: (NT,1) np.ndarray
       X: (NT,K) with first column = 1 (constant), then k, l
       ids: (NT,) firm ids (int)
       year: (NT,) years (int)
       labels: (label_y, label_x)
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Guess columns
    y_col = _first_in(df.columns, ['y', 'ldsa', 'lny', 'log_output', 'output_log'])
    k_col = _first_in(df.columns, ['k', 'lcap', 'lnk', 'log_capital', 'capital_log'])
    l_col = _first_in(df.columns, ['l', 'lempl', 'lemp', 'lnl', 'log_labor', 'labor_log'])
    id_col = _first_in(df.columns, ['firm', 'id', 'firm_id', 'company', 'comp_id'])
    t_col = _first_in(df.columns, ['year', 't', 'time'])

    if y_col is None or k_col is None or l_col is None:
        raise ValueError("Could not infer y/k/l columns. Make sure CSV has ldsa (y), lcap (k), and lemp/lemp/l (l).")

    # If id/year missing, create them
    if t_col is None:
        T_guess = 12 if len(df) % 12 == 0 else None
        if T_guess:
            n = len(df) // 12
            df['year'] = np.tile(np.arange(1968, 1980), n)
            t_col = 'year'
        else:
            df['year'] = np.arange(len(df))
            t_col = 'year'
    if id_col is None:
        T_i = df.groupby(t_col).size().median()
        try:
            T_i = int(T_i)
        except:
            T_i = 12
        n = int(len(df) / T_i)
        ids = np.repeat(np.arange(n), T_i)
        if len(ids) != len(df):
            ids = np.arange(len(df))
        df['firm'] = ids
        id_col = 'firm'

    # Sort for panel order (by firm then year)
    df = df.sort_values([id_col, t_col]).reset_index(drop=True)

    y = df[[y_col]].to_numpy().astype(float)
    X = np.column_stack([np.ones((len(df), 1)), df[[k_col, l_col]].to_numpy().astype(float)])
    ids = df[id_col].to_numpy()
    year = df[t_col].to_numpy()

    labels = ('Log output', ['Constant', 'Log capital', 'Log labor'])
    return y, X, ids, year, labels


def within_transform(Z: np.ndarray, ids) -> np.ndarray:
    """Within-transform any (NT,K) matrix by demeaning per firm id (unbalanced OK)."""
    ids = np.asarray(ids)
    Zt = np.zeros_like(Z, dtype=float)
    for gid in np.unique(ids):
        ii = (ids == gid)
        Zt[ii, :] = Z[ii, :] - Z[ii, :].mean(axis=0, keepdims=True)
    return Zt


def between_transform(Z: np.ndarray, ids) -> np.ndarray:
    """Between-transform any (NT,K) matrix as firm means replicated per obs."""
    ids = np.asarray(ids)
    Zb = np.zeros_like(Z, dtype=float)
    for gid in np.unique(ids):
        ii = (ids == gid)
        Zb[ii, :] = Z[ii, :].mean(axis=0, keepdims=True)
    return Zb


def swamy_arora_lambda(y: np.ndarray, X: np.ndarray, ids) -> float:
    """
    Compute theta (lambda) for RE quasi-demeaning using Swamy–Arora component estimates.
    Returns scalar lambda in [0,1]. For unbalanced data, uses average T.
    """
    ids = np.asarray(ids)
    # Pooled OLS residuals
    b_ols = est_ols(y, X)
    u = (y - X @ b_ols).ravel()

    # Firm means of residuals
    firm_means = []
    T_list = []
    for gid in np.unique(ids):
        ii = (ids == gid)
        firm_means.append(u[ii].mean())
        T_list.append(ii.sum())
    firm_means = np.array(firm_means)
    T_bar = np.mean(T_list)

    # Variance components
    sigma_e2 = np.mean((u - between_transform(u.reshape(-1,1), ids).ravel())**2)
    sigma_u2 = max(0.0, np.var(firm_means, ddof=1) - sigma_e2 / np.mean(T_list))

    if sigma_e2 + T_bar * sigma_u2 == 0:
        return 0.0
    lam = 1.0 - np.sqrt(sigma_e2 / (sigma_e2 + T_bar * sigma_u2))
    return float(np.clip(lam, 0.0, 1.0))


def re_quasi_demean(Z: np.ndarray, ids, lam: float) -> np.ndarray:
    """Random-effects quasi-demeaning: Z* = Z - lam * Z_bar_i"""
    return Z - lam * between_transform(Z, ids)


def _first_in(iterable, options):
    for o in options:
        if o in iterable:
            return o
    return None


def _infer_T_from_T_or_ids(T, Nobs):
    if isinstance(T, (int, np.integer)):
        return int(T)
    if T is None:
        return 1
    # array-like ids -> average T
    ids = np.asarray(T)
    _, counts = np.unique(ids, return_counts=True)
    return int(round(counts.mean()))


def _num_groups_from_T_or_ids(T, Nobs):
    if isinstance(T, (int, np.integer)) and T >= 1:
        return int(Nobs // int(T))
    if T is None:
        return Nobs
    ids = np.asarray(T)
    return int(np.unique(ids).size)


# ==============================
# High-level convenience runners
# ==============================

def run_pooled(csv_path='/mnt/data/firms.csv', robust=True):
    y, X, ids, year, labels = load_firm_data(csv_path)
    res = estimate(y, X, transform='', T=ids if robust else 1, robust_se=robust)
    return res, labels

def run_fe(csv_path='/mnt/data/firms.csv', robust=True):
    y, X, ids, year, labels = load_firm_data(csv_path)
    yw = within_transform(y, ids)
    Xw = within_transform(X, ids)
    # Drop the constant column after within-transform (it's zero)
    Xw = Xw[:, 1:]
    res = estimate(yw, Xw, transform='fe', T=ids if robust else 1, robust_se=robust)
    # Relabel without constant
    labels_fe = (labels[0], labels[1][1:])
    return res, labels_fe

def run_re(csv_path='/mnt/data/firms.csv', robust=True):
    y, X, ids, year, labels = load_firm_data(csv_path)
    lam = swamy_arora_lambda(y, X, ids)
    ystar = re_quasi_demean(y, ids, lam)
    Xstar = re_quasi_demean(X, ids, lam)
    res = estimate(ystar, Xstar, transform='re', T=ids if robust else 1, robust_se=robust)
    return res, labels, lam

# ==============================
# Nice printing wrappers
# ==============================

def print_pooled(csv_path='/mnt/data/firms.csv', tablefmt='github'):
    res, labels = run_pooled(csv_path=csv_path, robust=True)
    print_table(labels, res, title='Pooled OLS (cluster-robust by firm)', headers=["", "Beta", "Se", "t"], tablefmt=tablefmt)

def print_fe(csv_path='/mnt/data/firms.csv', tablefmt='github'):
    res, labels = run_fe(csv_path=csv_path, robust=True)
    print_table(labels, res, title='Fixed Effects (within, cluster-robust by firm)', headers=["", "Beta", "Se", "t"], tablefmt=tablefmt)

def print_re(csv_path='/mnt/data/firms.csv', tablefmt='github'):
    res, labels, lam = run_re(csv_path=csv_path, robust=True)
    print_table(labels, res, title='Random Effects (GLS quasi-demeaned, cluster-robust by firm)', headers=["", "Beta", "Se", "t"], tablefmt=tablefmt, _lambda=lam)

# ==============================
# Example usage (uncomment to run as script)
# ==============================
# if __name__ == "__main__":
#     print_pooled()
#     print_fe()
#     print_re()
