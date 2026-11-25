# =====================================================================
# Deney 3: Küçük Transformer'da kesirli power-law attention
# Görev: uzun menzilli anahtar–değer geri çağırma, uzunluk genelleştirmesi
# Modeller: softmax (mesafe yanlılığı yok), ALiBi, kesirli α ∈ {0.25, 0.5, 0.75},
#           öğrenilebilir α (baş başına; devam makalesi 3 için ön deney)
# =====================================================================
import math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- Ayarlar --------------------------------
CFG = dict(
    n_keys=64,          # anahtar sözlüğü
    n_vals=64,          # değer sözlüğü (çıktı sınıfı sayısı)
    n_filler=256,       # dolgu (gürültü) token sözlüğü
    n_pairs=8,          # dizideki anahtar–değer çifti sayısı
    train_len=64,       # eğitim dizi uzunluğu (CPU'da doğrulandı: 4000 adımda yakınsıyor)
    test_lens=[64, 256, 1024, 4096, 8192],   # 128 kata kadar uzunluk genelleştirmesi
    d_model=64, n_heads=4, n_layers=2, d_ff=128,
    max_rep=3,          # sorgu anahtarının en fazla kaç kez geçtiği
    steps=4000, batch=64, lr=3e-3, weight_decay=0.01,
    eval_samples=256,   # her uzunlukta değerlendirilen örnek sayısı
    eval_attn=2 ** 27,  # bellek sınırı: batch * L^2 <= eval_attn (T4 16 GB için güvenli)
    seeds=[0, 1, 2],
    noise_eps=0.05,     # Teorem D ampirik sınaması: gömme gürültüsü
)

# ----------------------------- Veri -----------------------------------
# Token kimliği yerine doğrudan (anahtar, değer) çiftleri üretiyoruz:
# çift token'ı = E_key[k] + E_val[v]; dolgu = E_fill[f]; sorgu = E_key[q] + e_query.
# Çiftler dizi boyunca rastgele konumlara yerleşir; sorgu son konumdadır.
# Doğru cevap, sorgu anahtarına eşleşen çiftin değeridir.

def make_batch(B, L, cfg, gen):
    """Görev 'en son değer': sorgu anahtarı dizide r ∈ {1, ..., max_rep} kez farklı
    değerlerle geçer; doğru cevap en son (sorguya en yakın) geçtiği yerdeki değerdir.
    Diğer çiftlerin anahtarları farklıdır. Bu görev hem yakınlık (en sonu seçmek)
    hem uzun menzil (tek geçiş uzaktaysa onu bulmak) gerektirir."""
    P, R = cfg["n_pairs"], cfg.get("max_rep", 3)
    keys = torch.stack([torch.randperm(cfg["n_keys"], generator=gen)[:P] for _ in range(B)])
    reps = torch.randint(1, R + 1, (B,), generator=gen)
    slot = torch.arange(P)[None, :]
    rep_mask = slot < reps[:, None]                       # ilk r yuva sorgu anahtarını taşır
    q_key = keys[:, 0].clone()
    keys = torch.where(rep_mask, q_key[:, None], keys)
    vals = torch.randint(0, cfg["n_vals"], (B, P), generator=gen)
    filler = torch.randint(0, cfg["n_filler"], (B, L), generator=gen)
    pos = torch.stack([torch.randperm(L - 1, generator=gen)[:P] for _ in range(B)])
    last = torch.where(rep_mask, pos, torch.full_like(pos, -1)).argmax(dim=1)
    target = vals[torch.arange(B), last]
    q_dist = (L - 1) - pos[torch.arange(B), last]          # sorgu–hedef mesafesi
    return dict(keys=keys, vals=vals, filler=filler, pos=pos, q_key=q_key,
                target=target, q_dist=q_dist, reps=reps)

# ----------------------------- Model ----------------------------------
class Attn(nn.Module):
    def __init__(self, d, H, kind, alpha=None):
        super().__init__()
        self.H, self.dh, self.kind = H, d // H, kind
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if kind == "alibi":
            slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)])
            self.register_buffer("slopes", slopes)
        if kind == "frac":
            self.register_buffer("alpha", torch.full((H,), float(alpha)))
        if kind == "frac_learn":
            # α = sigmoid(θ) ∈ (0,1); başlangıç α = 0.5
            self.theta = nn.Parameter(torch.zeros(H))

    def get_alpha(self):
        if self.kind == "frac":
            return self.alpha
        if self.kind == "frac_learn":
            return torch.sigmoid(self.theta)
        return None

    def bias(self, L, device):
        i = torch.arange(L, device=device)
        dist = (i[:, None] - i[None, :]).clamp(min=0).float()   # nedensel: j <= i
        if self.kind == "softmax":
            return None
        if self.kind == "alibi":
            return -self.slopes[:, None, None] * dist[None]
        # kesirli çekirdek |i-j|^{α-n}, n = 1; öz-dikkat için |i-j|+1
        a = self.get_alpha()
        return (a - 1.0)[:, None, None] * torch.log1p(dist)[None]

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        b = self.bias(L, x.device)
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        if b is None:
            attn_mask = torch.zeros(1, 1, L, L, device=x.device)
        else:
            attn_mask = b[None]
        attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(y.transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d, H, dff, kind, alpha):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, H, kind, alpha)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class Model(nn.Module):
    def __init__(self, cfg, kind, alpha=None):
        super().__init__()
        d = cfg["d_model"]
        self.E_key = nn.Embedding(cfg["n_keys"], d)
        self.E_val = nn.Embedding(cfg["n_vals"], d)
        self.E_fill = nn.Embedding(cfg["n_filler"], d)
        self.e_query = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["n_heads"], cfg["d_ff"], kind, alpha)
                                     for _ in range(cfg["n_layers"])])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["n_vals"])

    def embed(self, bt):
        x = self.E_fill(bt["filler"]).clone()                  # B, L, d
        B, L, d = x.shape
        pair = self.E_key(bt["keys"]) + self.E_val(bt["vals"])  # B, P, d
        x.scatter_(1, bt["pos"][..., None].expand(-1, -1, d), pair)
        x[:, -1] = self.E_key(bt["q_key"]) + self.e_query
        return x

    def forward(self, bt, noise=0.0):
        x = self.embed(bt)
        if noise > 0:
            x = x + noise * x.norm(dim=-1, keepdim=True) * torch.randn_like(x) / math.sqrt(x.shape[-1])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x[:, -1]))

# ----------------------------- Eğitim / değerlendirme -----------------
def to_dev(bt):
    return {k: v.to(DEV) for k, v in bt.items()}


def train(kind, alpha, seed, cfg, log_every=500):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    model = Model(cfg, kind, alpha).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"])
    model.train()
    for step in range(1, cfg["steps"] + 1):
        bt = to_dev(make_batch(cfg["batch"], cfg["train_len"], cfg, gen))
        loss = F.cross_entropy(model(bt), bt["target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log_every and step % log_every == 0:
            print(f"  [{kind}{'' if alpha is None else ' α=' + str(alpha)} seed={seed}] adım {step}: kayıp {loss.item():.3f}")
    return model


@torch.no_grad()
def evaluate(model, cfg, seed):
    model.eval()
    gen = torch.Generator().manual_seed(5000 + seed)
    out = {}
    for L in cfg["test_lens"]:
        B = max(1, min(cfg["batch"], cfg["eval_attn"] // (L * L)))
        n_batches = math.ceil(cfg["eval_samples"] / B)
        correct = total = 0
        far_c = far_t = 0
        mul_c = mul_t = 0
        rel_change = []
        for _ in range(n_batches):
            bt = to_dev(make_batch(B, L, cfg, gen))
            logits = model(bt)
            pred = logits.argmax(-1)
            ok = (pred == bt["target"])
            correct += ok.sum().item(); total += B
            far = bt["q_dist"] > L // 2               # hedef dizinin ilk yarısında
            far_c += ok[far].sum().item(); far_t += far.sum().item()
            mul = bt["reps"] > 1                      # birden çok geçiş: yakınlık gerekir
            mul_c += ok[mul].sum().item(); mul_t += mul.sum().item()
            noisy = model(bt, noise=cfg["noise_eps"])
            rel_change.append(((noisy - logits).norm(dim=-1) / logits.norm(dim=-1)).mean().item())
        out[L] = dict(acc=correct / total, acc_far=far_c / max(far_t, 1),
                      acc_multi=mul_c / max(mul_t, 1),
                      rel_change=float(np.mean(rel_change)))
    return out


def learned_alphas(model):
    res = []
    for blk in model.blocks:
        a = blk.attn.get_alpha()
        res.append(None if a is None else [round(float(t), 3) for t in a.detach()])
    return res


VARIANTS = [("softmax", None), ("alibi", None), ("frac", 0.25), ("frac", 0.5),
            ("frac", 0.75), ("frac_learn", None)]


def run_all(cfg, variants=VARIANTS):
    results = []
    for kind, alpha in variants:
        for seed in cfg["seeds"]:
            t0 = time.time()
            model = train(kind, alpha, seed, cfg)
            ev = evaluate(model, cfg, seed)
            rec = dict(kind=kind, alpha=alpha, seed=seed, eval=ev,
                       learned_alpha=learned_alphas(model) if kind == "frac_learn" else None,
                       seconds=round(time.time() - t0, 1))
            results.append(rec)
            accs = " ".join(f"L={L}:{v['acc']:.2f}" for L, v in ev.items())
            print(f"{kind:10s} α={alpha} seed={seed}  {accs}  ({rec['seconds']} s)")
    return results


def name(kind, alpha):
    return {"softmax": "Softmax", "alibi": "ALiBi", "frac_learn": "Kesirli, öğrenilebilir α"}.get(
        kind, f"Kesirli, α = {alpha}")


def summary_tables(results, cfg):
    lens = cfg["test_lens"]
    groups = {}
    for r in results:
        groups.setdefault((r["kind"], r["alpha"]), []).append(r)
    lines = ["Tablo A. Doğruluk (ortalama ± std, %d tohum); eğitim uzunluğu %d" % (len(cfg["seeds"]), cfg["train_len"]),
             "| Model | " + " | ".join(f"L = {L}" for L in lens) + " |",
             "| --- | " + " | ".join("---" for _ in lens) + " |"]
    for (kind, alpha), rs in groups.items():
        cells = []
        for L in lens:
            a = np.array([r["eval"][L]["acc"] for r in rs]) * 100
            cells.append(f"{a.mean():.1f} ± {a.std():.1f}")
        lines.append(f"| {name(kind, alpha)} | " + " | ".join(cells) + " |")
    lines += ["", "Tablo B. Uzak hedeflerde doğruluk (hedef dizinin ilk yarısında), %",
              "| Model | " + " | ".join(f"L = {L}" for L in lens) + " |",
              "| --- | " + " | ".join("---" for _ in lens) + " |"]
    for (kind, alpha), rs in groups.items():
        cells = [f"{np.mean([r['eval'][L]['acc_far'] for r in rs]) * 100:.1f}" for L in lens]
        lines.append(f"| {name(kind, alpha)} | " + " | ".join(cells) + " |")
    lines += ["", "Tablo B2. Sorgu anahtarı birden çok kez geçtiğinde doğruluk (yakınlık), %",
              "| Model | " + " | ".join(f"L = {L}" for L in lens) + " |",
              "| --- | " + " | ".join("---" for _ in lens) + " |"]
    for (kind, alpha), rs in groups.items():
        cells = [f"{np.mean([r['eval'][L]['acc_multi'] for r in rs]) * 100:.1f}" for L in lens]
        lines.append(f"| {name(kind, alpha)} | " + " | ".join(cells) + " |")
    lines += ["", f"Tablo C. Gömme gürültüsüne (ε = {cfg['noise_eps']}) karşı çıktı göreli değişimi",
              "| Model | " + " | ".join(f"L = {L}" for L in lens) + " |",
              "| --- | " + " | ".join("---" for _ in lens) + " |"]
    for (kind, alpha), rs in groups.items():
        cells = [f"{np.mean([r['eval'][L]['rel_change'] for r in rs]):.3f}" for L in lens]
        lines.append(f"| {name(kind, alpha)} | " + " | ".join(cells) + " |")
    la = [r["learned_alpha"] for r in results if r["learned_alpha"] is not None]
    if la:
        lines += ["", "Öğrenilen α değerleri (katman başına, baş başına):"]
        for r in results:
            if r["learned_alpha"] is not None:
                lines.append(f"  tohum {r['seed']}: {r['learned_alpha']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--hizli" in sys.argv:   # yerel duman testi
        CFG.update(steps=60, batch=16, test_lens=[128, 256], train_len=128, seeds=[0],
                   eval_samples=32, d_model=32, d_ff=64)
    print("Cihaz:", DEV)
    res = run_all(CFG)
    print(summary_tables(res, CFG))
    json.dump(res, open("deney3_sonuclar.json", "w"), ensure_ascii=False, indent=1, default=str)
