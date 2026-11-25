"""
Kesirli power-law attention: Deney 1 (kritik üs, Teorem C ve Not C.1),
Deney 2 (attention seyrelmesi, Önerme C') ve Teorem A sağlama kontrolü.
Ayrık model: n = 1, token'lar tam sayı konumlarda, sorgu 0 konumunda.
"""
import json
import numpy as np

rng = np.random.default_rng(2026)
OUT = "/home/claude/deneyler"
LAM = 1.0          # içerik skoru sınırı Λ
D = 32             # gömme boyutu
RMAX = 2 ** 20     # her yönde en fazla token sayısı

def content_weights(m):
    """a_i = exp(s_i), |s_i| <= Λ; s_i = Λ cos(q, k_i) (LayerNorm benzeri normalize gömmeler)."""
    q = rng.standard_normal(D)
    q /= np.linalg.norm(q)
    k = rng.standard_normal((m, D)).astype(np.float32)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    return np.exp(LAM * (k @ q))


# İki yönlü token'lar: i = 1..RMAX sağda ve solda; ağırlıklar her iki taraf için ayrı.
i = np.arange(1, RMAX + 1, dtype=np.float64)
a_right, a_left = content_weights(RMAX), content_weights(RMAX)
a_both = a_right + a_left            # simetrik |i| katkısı (f0 radyal)

Rs = 2 ** np.arange(4, 21)           # R = 16 ... 2^20

# ---------------- Deney 1: Teorem C (λ > 0) ----------------
results = []
for lam in (0.2, 0.5, 0.8):
    for p in (1, 2):
        astar = (1 - lam) / p
        delta = min(0.15, astar / 2, (1 - astar) / 2)
        for regime, alpha in (("alt", astar - delta), ("kritik", astar), ("ust", astar + delta)):
            # f = η f0, η ~ U[0,1] rastgele (|f| <= f0 olduğundan f ∈ L^{p,λ})
            eta = rng.random(RMAX)
            terms = a_both * i ** (alpha - 1) * i ** (-astar) * eta   # a |i|^{α-n} |f(i)|
            cF = np.cumsum(terms)
            F = cF[Rs - 1]
            lo = Rs >= 2 ** 12
            slope = np.polyfit(np.log(Rs[lo]), np.log(F[lo]), 1)[0]
            # diyadik halka katkıları: A_j = Σ_{2^j <= i < 2^{j+1}} ; teori: A_j ~ 2^{j(α-α*)}
            jj = np.arange(4, 20)
            A = cF[2 ** (jj + 1) - 2] - cF[2 ** jj - 2]
            ring_exp = np.polyfit(jj * np.log(2), np.log(A), 1)[0]
            rec = dict(lam=lam, p=p, astar=round(astar, 4), alpha=round(alpha, 4),
                       regime=regime, loglog_slope=round(float(slope), 4),
                       ring_exponent=round(float(ring_exp), 4),
                       theory_ring_exponent=round(alpha - astar, 4))
            if regime == "kritik":
                c, b = np.polyfit(np.log(Rs[lo]), F[lo], 1)
                pred = c * np.log(Rs[lo]) + b
                r2 = 1 - np.sum((F[lo] - pred) ** 2) / np.sum((F[lo] - F[lo].mean()) ** 2)
                rec["lnR_fit_slope"] = round(float(c), 4)
                rec["lnR_fit_R2"] = round(float(r2), 6)
            if regime == "alt":
                rec["rel_change_2^16_to_2^20"] = round(float(F[-1] / F[Rs == 2 ** 16][0] - 1), 5)
            results.append(rec)

# ---------------- Deney 1b: Not C.1 (λ = 0, p = 2, α = α* = 1/2) ----------------
p, alpha = 2, 0.5
astar = 1 / p
j_idx = np.floor(np.log2(i)).astype(int)            # halka indeksi: 2^j <= i < 2^{j+1}
g = i ** (-1 / p)
crit_rows = []
for J in range(4, 20):
    m = 2 ** (J + 1) - 1
    gj = g[:m]; jj = j_idx[:m]
    # her halkada (iki taraf) L^p normu 1 olacak şekilde ölçekle, sonra (J+1)^{-1/p} ile
    ring_norm = np.bincount(jj, weights=2 * gj ** p) ** (1 / p)
    fR = gj / ring_norm[jj] * (J + 1) ** (-1 / p)
    norm_p = (np.sum(2 * fR ** p)) ** (1 / p)
    F = np.sum(a_both[:m] * i[:m] ** (alpha - 1) * fR)
    crit_rows.append(dict(J=J, R=2 ** (J + 1), Lp_norm=round(float(norm_p), 6),
                          F=float(F), ratio_to_sqrt_logR=float(F / np.sqrt(np.log(2 ** (J + 1))))))

# ---------------- Deney 2: Önerme C' (seyrelme) ----------------
rho = 16
dil = {}
kernels = {
    "Kesirli α = 0.25": lambda x: x ** (0.25 - 1),
    "Kesirli α = 0.5": lambda x: x ** (0.5 - 1),
    "Kesirli α = 0.75": lambda x: x ** (0.75 - 1),
    "Softmax (mesafe çekirdeği yok)": lambda x: np.ones_like(x),
    "ALiBi (eğim 1/16)": lambda x: np.exp(-x / 16),
}
for name, ker in kernels.items():
    w = a_both * ker(i)
    cw = np.cumsum(w)
    dil[name] = (cw[rho - 1] / cw[Rs - 1]).tolist()

# Önerme C' zarfı (sürekli modelden): e^{±2Λ} (ρ/R)^α ; ayrık toplamda sabitler hafif değişir.
env = {}
for a_ in (0.25, 0.5, 0.75):
    share = np.array(dil[f"Kesirli α = {a_}"])
    env[a_] = dict(min_ratio=float(np.min(share / (rho / Rs) ** a_)),
                   max_ratio=float(np.max(share / (rho / Rs) ** a_)),
                   bound_low=float(np.exp(-2 * LAM)), bound_high=float(np.exp(2 * LAM)),
                   loglog_slope=float(np.polyfit(np.log(Rs[Rs >= 2 ** 10]),
                                                 np.log(share[Rs >= 2 ** 10]), 1)[0]))

# ---------------- Teorem A sağlama: |N f(x)| / Mf(x) ----------------
L_seq, Rw = 8192, 512
f = rng.standard_cauchy(L_seq) * (rng.random(L_seq) < 0.05)   # seyrek, ağır kuyruklu girdi
f += 0.1 * rng.standard_normal(L_seq)
af = np.abs(f)
cs = np.concatenate([[0], np.cumsum(af)])
theoremA = {}
for alpha in (0.25, 0.5, 0.75):
    ratios = []
    for x in rng.integers(Rw + 1, L_seq - Rw - 1, size=400):
        idx = np.arange(x - Rw, x + Rw + 1); idx = idx[idx != x]
        dist = np.abs(idx - x).astype(float)
        kq = rng.standard_normal(D); kq /= np.linalg.norm(kq)
        kk = rng.standard_normal((idx.size, D)); kk /= np.linalg.norm(kk, axis=1, keepdims=True)
        w = np.exp(LAM * (kk @ kq)) * dist ** (alpha - 1)
        Nf = np.sum(w * f[idx]) / np.sum(w)
        r = np.arange(1, Rw + 1)
        Mf = np.max((cs[x + r + 1] - cs[x - r]) / (2 * r + 1))
        ratios.append(abs(Nf) / Mf)
    CA = 2 * alpha / (1 * (2 ** alpha - 1))
    theoremA[alpha] = dict(max_ratio=float(np.max(ratios)), mean_ratio=float(np.mean(ratios)),
                           bound=float(CA * np.exp(2 * LAM)))

json.dump(dict(deney1=results, deney1b=crit_rows, deney2_envelope=env, deney2_share=dil, Rs=Rs.tolist(), teoremA=theoremA),
          open(f"{OUT}/sonuclar.json", "w"), ensure_ascii=False, indent=1)

print(json.dumps(dict(deney1=results, deney2_envelope=env, teoremA=theoremA), ensure_ascii=False, indent=1))
print("deney1b:", [(r["R"], round(r["F"], 3), round(r["ratio_to_sqrt_logR"], 3), r["Lp_norm"]) for r in crit_rows])
