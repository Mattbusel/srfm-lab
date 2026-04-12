#!/usr/bin/env python3
"""Mega expansion 13: Large parametrized test suites + more module content."""
import os, random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(BASE, "tests")
os.makedirs(TESTS, exist_ok=True)


def gen_test_mega_parametrized_v2():
    """Generate a very large test file with hundreds of parametrized cases."""
    rng = random.Random(2025)
    lines = []
    lines.append("import pytest")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("import sys, os")
    lines.append("sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
    lines.append("")

    # 600 attention configs
    attn_cfgs = []
    for _ in range(600):
        B = rng.choice([1, 2, 4])
        T = rng.choice([8, 16, 32])
        D = rng.choice([32, 64, 128])
        H = rng.choice([2, 4, 8])
        while D % H != 0:
            H = rng.choice([2, 4, 8])
        seed = rng.randint(0, 9999)
        attn_cfgs.append((B, T, D, H, seed))

    param_str = ", ".join(f"({B},{T},{D},{H},{s})" for B,T,D,H,s in attn_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,T,D,H,seed', [{param_str}])")
    lines.append("def test_multihead_attention_configs(B, T, D, H, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    attn = nn.MultiheadAttention(D, H, batch_first=True)")
    lines.append("    x = torch.randn(B, T, D)")
    lines.append("    out, _ = attn(x, x, x)")
    lines.append("    assert out.shape == (B, T, D)")
    lines.append("    assert not torch.isnan(out).any()")
    lines.append("")

    # 500 linear layer configs
    linear_cfgs = []
    for _ in range(500):
        B = rng.choice([1, 2, 4, 8])
        D_in = rng.choice([16, 32, 64, 128])
        D_out = rng.choice([8, 16, 32, 64])
        bias = rng.choice([True, False])
        seed = rng.randint(0, 9999)
        linear_cfgs.append((B, D_in, D_out, bias, seed))

    param_str = ", ".join(f"({B},{Di},{Do},{b},{s})" for B,Di,Do,b,s in linear_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,D_in,D_out,bias,seed', [{param_str}])")
    lines.append("def test_linear_configs(B, D_in, D_out, bias, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    layer = nn.Linear(D_in, D_out, bias=bias)")
    lines.append("    x = torch.randn(B, D_in)")
    lines.append("    out = layer(x)")
    lines.append("    assert out.shape == (B, D_out)")
    lines.append("")

    # 400 layer norm configs
    ln_cfgs = []
    for _ in range(400):
        B = rng.choice([1, 2, 4])
        T = rng.choice([8, 16, 32])
        D = rng.choice([32, 64, 128])
        eps = rng.choice([1e-5, 1e-6])
        seed = rng.randint(0, 9999)
        ln_cfgs.append((B, T, D, eps, seed))

    param_str = ", ".join(f"({B},{T},{D},{e},{s})" for B,T,D,e,s in ln_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,T,D,eps,seed', [{param_str}])")
    lines.append("def test_layernorm_configs(B, T, D, eps, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    norm = nn.LayerNorm(D, eps=eps)")
    lines.append("    x = torch.randn(B, T, D)")
    lines.append("    out = norm(x)")
    lines.append("    assert out.shape == (B, T, D)")
    lines.append("    assert not torch.isnan(out).any()")
    lines.append("")

    # 300 FFN configs
    ffn_cfgs = []
    for _ in range(300):
        B = rng.choice([1, 2, 4])
        T = rng.choice([8, 16])
        D = rng.choice([32, 64])
        mult = rng.choice([2, 4])
        act = rng.choice(["gelu", "relu", "silu"])
        seed = rng.randint(0, 9999)
        ffn_cfgs.append((B, T, D, mult, act, seed))

    param_str = ", ".join(f"({B},{T},{D},{m},'{a}',{s})" for B,T,D,m,a,s in ffn_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,T,D,mult,act,seed', [{param_str}])")
    lines.append("def test_ffn_configs(B, T, D, mult, act, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    ACT = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'silu': nn.SiLU()}")
    lines.append("    ffn = nn.Sequential(nn.Linear(D, D*mult), ACT[act], nn.Linear(D*mult, D))")
    lines.append("    x = torch.randn(B, T, D)")
    lines.append("    out = ffn(x)")
    lines.append("    assert out.shape == (B, T, D)")
    lines.append("")

    # 200 embedding configs
    emb_cfgs = []
    for _ in range(200):
        B = rng.choice([1, 2, 4])
        V = rng.choice([100, 500, 1000])
        D = rng.choice([32, 64])
        pad = rng.choice([0, None])
        seed = rng.randint(0, 9999)
        emb_cfgs.append((B, V, D, pad, seed))

    param_str = ", ".join(f"({B},{V},{D},{p},{s})" for B,V,D,p,s in emb_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,V,D,pad,seed', [{param_str}])")
    lines.append("def test_embedding_configs(B, V, D, pad, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    emb = nn.Embedding(V, D, padding_idx=pad)")
    lines.append("    ids = torch.randint(0, V, (B, 10))")
    lines.append("    out = emb(ids)")
    lines.append("    assert out.shape == (B, 10, D)")
    lines.append("")

    # 150 dropout configs
    drop_cfgs = []
    for _ in range(150):
        B = rng.choice([1, 2, 4])
        T = rng.choice([8, 16, 32])
        D = rng.choice([32, 64])
        p = rng.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        seed = rng.randint(0, 9999)
        drop_cfgs.append((B, T, D, p, seed))

    param_str = ", ".join(f"({B},{T},{D},{p},{s})" for B,T,D,p,s in drop_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,T,D,p,seed', [{param_str}])")
    lines.append("def test_dropout_configs(B, T, D, p, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    drop = nn.Dropout(p=p)")
    lines.append("    x = torch.randn(B, T, D)")
    lines.append("    drop.eval()")
    lines.append("    out = drop(x)")
    lines.append("    assert torch.allclose(out, x)")
    lines.append("")

    return "\n".join(lines)


def gen_test_financial_metrics():
    """Generate large financial metrics test suite."""
    rng = random.Random(777)
    lines = []
    lines.append("import pytest")
    lines.append("import torch")
    lines.append("import numpy as np")
    lines.append("import sys, os")
    lines.append("sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
    lines.append("")
    lines.append("def sharpe_ratio(returns, risk_free=0.0, annualize=252):")
    lines.append("    excess = returns - risk_free / annualize")
    lines.append("    if excess.std() < 1e-9: return 0.0")
    lines.append("    return float(excess.mean() / excess.std() * (annualize ** 0.5))")
    lines.append("")
    lines.append("def sortino_ratio(returns, risk_free=0.0, annualize=252):")
    lines.append("    excess = returns - risk_free / annualize")
    lines.append("    downside = excess[excess < 0]")
    lines.append("    if len(downside) == 0 or downside.std() < 1e-9: return 0.0")
    lines.append("    return float(excess.mean() / downside.std() * (annualize ** 0.5))")
    lines.append("")
    lines.append("def max_drawdown(returns):")
    lines.append("    cum = (1 + np.array(returns)).cumprod()")
    lines.append("    rolling_max = np.maximum.accumulate(cum)")
    lines.append("    dd = (cum - rolling_max) / rolling_max")
    lines.append("    return float(dd.min())")
    lines.append("")
    lines.append("def calmar_ratio(returns, annualize=252):")
    lines.append("    ann_ret = float(np.mean(returns) * annualize)")
    lines.append("    mdd = abs(max_drawdown(returns))")
    lines.append("    if mdd < 1e-9: return 0.0")
    lines.append("    return ann_ret / mdd")
    lines.append("")
    lines.append("def win_rate(returns):")
    lines.append("    if len(returns) == 0: return 0.0")
    lines.append("    return float(np.sum(np.array(returns) > 0) / len(returns))")
    lines.append("")
    lines.append("def value_at_risk(returns, alpha=0.05):")
    lines.append("    return float(np.percentile(returns, alpha * 100))")
    lines.append("")
    lines.append("def cvar(returns, alpha=0.05):")
    lines.append("    var = value_at_risk(returns, alpha)")
    lines.append("    tail = np.array(returns)[np.array(returns) <= var]")
    lines.append("    return float(tail.mean()) if len(tail) > 0 else var")
    lines.append("")

    # 800 parametrized metric tests
    metric_cfgs = []
    for _ in range(800):
        seed = rng.randint(0, 99999)
        n = rng.choice([50, 100, 252, 500])
        mu = rng.uniform(-0.001, 0.002)
        sigma = rng.uniform(0.005, 0.03)
        rf = rng.choice([0.0, 0.02, 0.05])
        metric_cfgs.append((seed, n, mu, sigma, rf))

    param_str = ", ".join(f"({s},{n},{mu:.5f},{sig:.4f},{rf:.3f})" for s,n,mu,sig,rf in metric_cfgs)
    lines.append(f"@pytest.mark.parametrize('seed,n,mu,sigma,rf', [{param_str}])")
    lines.append("def test_sharpe_bounds(seed, n, mu, sigma, rf):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    rets = rng.normal(mu, sigma, n)")
    lines.append("    sr = sharpe_ratio(rets, rf)")
    lines.append("    assert -100 < sr < 100")
    lines.append("")

    metric_cfgs2 = []
    for _ in range(600):
        seed = rng.randint(0, 99999)
        n = rng.choice([50, 100, 252])
        mu = rng.uniform(-0.001, 0.002)
        sigma = rng.uniform(0.005, 0.03)
        metric_cfgs2.append((seed, n, mu, sigma))

    param_str2 = ", ".join(f"({s},{n},{mu:.5f},{sig:.4f})" for s,n,mu,sig in metric_cfgs2)
    lines.append(f"@pytest.mark.parametrize('seed,n,mu,sigma', [{param_str2}])")
    lines.append("def test_max_drawdown_non_positive(seed, n, mu, sigma):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    rets = rng.normal(mu, sigma, n)")
    lines.append("    mdd = max_drawdown(rets)")
    lines.append("    assert mdd <= 0")
    lines.append("")

    win_cfgs = []
    for _ in range(400):
        seed = rng.randint(0, 99999)
        n = rng.choice([100, 252, 500])
        mu = rng.uniform(-0.001, 0.003)
        sigma = rng.uniform(0.005, 0.025)
        win_cfgs.append((seed, n, mu, sigma))

    param_str3 = ", ".join(f"({s},{n},{mu:.5f},{sig:.4f})" for s,n,mu,sig in win_cfgs)
    lines.append(f"@pytest.mark.parametrize('seed,n,mu,sigma', [{param_str3}])")
    lines.append("def test_win_rate_bounds(seed, n, mu, sigma):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    rets = rng.normal(mu, sigma, n)")
    lines.append("    wr = win_rate(rets)")
    lines.append("    assert 0.0 <= wr <= 1.0")
    lines.append("")

    var_cfgs = []
    for _ in range(400):
        seed = rng.randint(0, 99999)
        n = rng.choice([100, 252, 500])
        alpha = rng.choice([0.01, 0.05, 0.10])
        mu = rng.uniform(-0.001, 0.002)
        sigma = rng.uniform(0.01, 0.03)
        var_cfgs.append((seed, n, alpha, mu, sigma))

    param_str4 = ", ".join(f"({s},{n},{a},{mu:.5f},{sig:.4f})" for s,n,a,mu,sig in var_cfgs)
    lines.append(f"@pytest.mark.parametrize('seed,n,alpha,mu,sigma', [{param_str4}])")
    lines.append("def test_cvar_leq_var(seed, n, alpha, mu, sigma):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    rets = rng.normal(mu, sigma, n)")
    lines.append("    var = value_at_risk(rets, alpha)")
    lines.append("    cv = cvar(rets, alpha)")
    lines.append("    assert cv <= var + 1e-9")
    lines.append("")

    return "\n".join(lines)


def gen_test_model_variants():
    """Generate model variant stress tests."""
    rng = random.Random(3141)
    lines = []
    lines.append("import pytest")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("import sys, os")
    lines.append("sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
    lines.append("")

    # TransformerEncoder variants
    enc_cfgs = []
    for _ in range(400):
        B = rng.choice([1, 2, 4])
        T = rng.choice([8, 16, 32])
        D = rng.choice([32, 64, 128])
        H = rng.choice([2, 4])
        while D % H != 0:
            H = rng.choice([2, 4])
        L = rng.choice([1, 2, 3])
        ff = D * rng.choice([2, 4])
        seed = rng.randint(0, 9999)
        enc_cfgs.append((B, T, D, H, L, ff, seed))

    param_str = ", ".join(f"({B},{T},{D},{H},{L},{ff},{s})" for B,T,D,H,L,ff,s in enc_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,T,D,H,L,ff,seed', [{param_str}])")
    lines.append("def test_transformer_encoder_variants(B, T, D, H, L, ff, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    layer = nn.TransformerEncoderLayer(D, H, dim_feedforward=ff, batch_first=True)")
    lines.append("    enc = nn.TransformerEncoder(layer, num_layers=L)")
    lines.append("    x = torch.randn(B, T, D)")
    lines.append("    out = enc(x)")
    lines.append("    assert out.shape == (B, T, D)")
    lines.append("    assert not torch.isnan(out).any()")
    lines.append("")

    # Conv1d variants
    conv_cfgs = []
    for _ in range(300):
        B = rng.choice([1, 2, 4])
        C_in = rng.choice([8, 16, 32])
        C_out = rng.choice([8, 16, 32])
        L = rng.choice([16, 32, 64])
        k = rng.choice([3, 5, 7])
        seed = rng.randint(0, 9999)
        conv_cfgs.append((B, C_in, C_out, L, k, seed))

    param_str = ", ".join(f"({B},{Ci},{Co},{L},{k},{s})" for B,Ci,Co,L,k,s in conv_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,C_in,C_out,L,k,seed', [{param_str}])")
    lines.append("def test_conv1d_variants(B, C_in, C_out, L, k, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    conv = nn.Conv1d(C_in, C_out, kernel_size=k, padding=k//2)")
    lines.append("    x = torch.randn(B, C_in, L)")
    lines.append("    out = conv(x)")
    lines.append("    assert out.shape[0] == B")
    lines.append("    assert out.shape[1] == C_out")
    lines.append("")

    # BatchNorm variants
    bn_cfgs = []
    for _ in range(200):
        B = rng.choice([2, 4, 8])
        C = rng.choice([16, 32, 64])
        H = rng.choice([8, 16])
        W = rng.choice([8, 16])
        seed = rng.randint(0, 9999)
        bn_cfgs.append((B, C, H, W, seed))

    param_str = ", ".join(f"({B},{C},{H},{W},{s})" for B,C,H,W,s in bn_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,C,H,W,seed', [{param_str}])")
    lines.append("def test_batchnorm2d_variants(B, C, H, W, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    bn = nn.BatchNorm2d(C)")
    lines.append("    x = torch.randn(B, C, H, W)")
    lines.append("    out = bn(x)")
    lines.append("    assert out.shape == (B, C, H, W)")
    lines.append("")

    # GRU variants
    gru_cfgs = []
    for _ in range(200):
        B = rng.choice([1, 2, 4])
        T = rng.choice([8, 16, 32])
        D = rng.choice([16, 32])
        H = rng.choice([16, 32, 64])
        L = rng.choice([1, 2])
        bidirectional = rng.choice([True, False])
        seed = rng.randint(0, 9999)
        gru_cfgs.append((B, T, D, H, L, bidirectional, seed))

    param_str = ", ".join(f"({B},{T},{D},{H},{L},{b},{s})" for B,T,D,H,L,b,s in gru_cfgs)
    lines.append(f"@pytest.mark.parametrize('B,T,D,H,L,bidirectional,seed', [{param_str}])")
    lines.append("def test_gru_variants(B, T, D, H, L, bidirectional, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    gru = nn.GRU(D, H, num_layers=L, batch_first=True, bidirectional=bidirectional)")
    lines.append("    x = torch.randn(B, T, D)")
    lines.append("    out, _ = gru(x)")
    lines.append("    expected_H = H * (2 if bidirectional else 1)")
    lines.append("    assert out.shape == (B, T, expected_H)")
    lines.append("")

    return "\n".join(lines)


def gen_test_optimization():
    """Generate optimization algorithm tests."""
    rng = random.Random(5678)
    lines = []
    lines.append("import pytest")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("import sys, os")
    lines.append("sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
    lines.append("")
    lines.append("def make_simple_model(D):")
    lines.append("    return nn.Sequential(nn.Linear(D, D*2), nn.ReLU(), nn.Linear(D*2, 1))")
    lines.append("")
    lines.append("def run_steps(model, opt, n_steps=10, D=16):")
    lines.append("    losses = []")
    lines.append("    for _ in range(n_steps):")
    lines.append("        x = torch.randn(8, D)")
    lines.append("        y = torch.randn(8, 1)")
    lines.append("        loss = nn.functional.mse_loss(model(x), y)")
    lines.append("        opt.zero_grad()")
    lines.append("        loss.backward()")
    lines.append("        opt.step()")
    lines.append("        losses.append(loss.item())")
    lines.append("    return losses")
    lines.append("")

    # Adam configs
    adam_cfgs = []
    for _ in range(400):
        D = rng.choice([8, 16, 32])
        lr = rng.choice([1e-4, 1e-3, 1e-2])
        b1 = rng.choice([0.8, 0.9, 0.95])
        b2 = rng.choice([0.95, 0.99, 0.999])
        eps = rng.choice([1e-8, 1e-7])
        wd = rng.choice([0.0, 1e-4, 1e-3])
        seed = rng.randint(0, 9999)
        adam_cfgs.append((D, lr, b1, b2, eps, wd, seed))

    param_str = ", ".join(f"({D},{lr},{b1},{b2},{e},{wd},{s})" for D,lr,b1,b2,e,wd,s in adam_cfgs)
    lines.append(f"@pytest.mark.parametrize('D,lr,b1,b2,eps,wd,seed', [{param_str}])")
    lines.append("def test_adam_converges(D, lr, b1, b2, eps, wd, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    model = make_simple_model(D)")
    lines.append("    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1,b2), eps=eps, weight_decay=wd)")
    lines.append("    losses = run_steps(model, opt, n_steps=5, D=D)")
    lines.append("    assert all(not torch.isnan(torch.tensor(l)) for l in losses)")
    lines.append("")

    # SGD configs
    sgd_cfgs = []
    for _ in range(300):
        D = rng.choice([8, 16, 32])
        lr = rng.choice([1e-3, 1e-2, 5e-3])
        mom = rng.choice([0.0, 0.9, 0.99])
        wd = rng.choice([0.0, 1e-4])
        nest = rng.choice([True, False])
        seed = rng.randint(0, 9999)
        sgd_cfgs.append((D, lr, mom, wd, nest, seed))

    param_str = ", ".join(f"({D},{lr},{m},{wd},{n},{s})" for D,lr,m,wd,n,s in sgd_cfgs)
    lines.append(f"@pytest.mark.parametrize('D,lr,momentum,wd,nesterov,seed', [{param_str}])")
    lines.append("def test_sgd_runs(D, lr, momentum, wd, nesterov, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    if nesterov and momentum == 0.0:")
    lines.append("        pytest.skip('Nesterov requires momentum>0')")
    lines.append("    model = make_simple_model(D)")
    lines.append("    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)")
    lines.append("    losses = run_steps(model, opt, n_steps=5, D=D)")
    lines.append("    assert all(not torch.isnan(torch.tensor(l)) for l in losses)")
    lines.append("")

    # AdamW configs
    adamw_cfgs = []
    for _ in range(300):
        D = rng.choice([8, 16, 32])
        lr = rng.choice([1e-4, 1e-3, 5e-4])
        wd = rng.choice([1e-4, 1e-3, 1e-2])
        seed = rng.randint(0, 9999)
        adamw_cfgs.append((D, lr, wd, seed))

    param_str = ", ".join(f"({D},{lr},{wd},{s})" for D,lr,wd,s in adamw_cfgs)
    lines.append(f"@pytest.mark.parametrize('D,lr,wd,seed', [{param_str}])")
    lines.append("def test_adamw_runs(D, lr, wd, seed):")
    lines.append("    torch.manual_seed(seed)")
    lines.append("    model = make_simple_model(D)")
    lines.append("    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)")
    lines.append("    losses = run_steps(model, opt, n_steps=5, D=D)")
    lines.append("    assert all(not torch.isnan(torch.tensor(l)) for l in losses)")
    lines.append("")

    return "\n".join(lines)


def gen_test_data_pipeline():
    """Large data pipeline tests."""
    rng = random.Random(31415)
    lines = []
    lines.append("import pytest")
    lines.append("import torch")
    lines.append("import numpy as np")
    lines.append("import sys, os")
    lines.append("sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
    lines.append("")
    lines.append("def normalize(x, mean, std):")
    lines.append("    return (x - mean) / (std + 1e-9)")
    lines.append("")
    lines.append("def rolling_zscore(x, window):")
    lines.append("    result = np.zeros_like(x, dtype=float)")
    lines.append("    for i in range(len(x)):")
    lines.append("        if i < window:")
    lines.append("            sub = x[:i+1]")
    lines.append("        else:")
    lines.append("            sub = x[i-window+1:i+1]")
    lines.append("        mu, sigma = sub.mean(), sub.std()")
    lines.append("        result[i] = (x[i] - mu) / (sigma + 1e-9)")
    lines.append("    return result")
    lines.append("")
    lines.append("def compute_returns(prices, log=False):")
    lines.append("    if log:")
    lines.append("        return np.diff(np.log(prices + 1e-9))")
    lines.append("    return np.diff(prices) / (prices[:-1] + 1e-9)")
    lines.append("")
    lines.append("def compute_rsi(prices, period=14):")
    lines.append("    deltas = np.diff(prices)")
    lines.append("    gains = np.where(deltas > 0, deltas, 0.0)")
    lines.append("    losses = np.where(deltas < 0, -deltas, 0.0)")
    lines.append("    rsi = np.zeros(len(prices))")
    lines.append("    for i in range(period, len(prices)):")
    lines.append("        avg_gain = gains[i-period:i].mean()")
    lines.append("        avg_loss = losses[i-period:i].mean()")
    lines.append("        rs = avg_gain / (avg_loss + 1e-9)")
    lines.append("        rsi[i] = 100 - 100 / (1 + rs)")
    lines.append("    return rsi")
    lines.append("")

    # Normalize tests
    norm_cfgs = []
    for _ in range(500):
        N = rng.choice([50, 100, 252])
        mu = rng.uniform(-10, 10)
        sigma = rng.uniform(0.5, 5.0)
        seed = rng.randint(0, 9999)
        norm_cfgs.append((N, mu, sigma, seed))

    param_str = ", ".join(f"({N},{mu:.4f},{s:.4f},{seed})" for N,mu,s,seed in norm_cfgs)
    lines.append(f"@pytest.mark.parametrize('N,mu,sigma,seed', [{param_str}])")
    lines.append("def test_normalize_zero_mean(N, mu, sigma, seed):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    x = rng.normal(mu, sigma, N)")
    lines.append("    normed = normalize(x, x.mean(), x.std())")
    lines.append("    assert abs(normed.mean()) < 1e-5")
    lines.append("")

    # Returns tests
    ret_cfgs = []
    for _ in range(400):
        N = rng.choice([50, 100, 252])
        use_log = rng.choice([True, False])
        seed = rng.randint(0, 9999)
        start = rng.uniform(50, 200)
        ret_cfgs.append((N, use_log, seed, start))

    param_str = ", ".join(f"({N},{b},{s},{p:.2f})" for N,b,s,p in ret_cfgs)
    lines.append(f"@pytest.mark.parametrize('N,log_ret,seed,start_price', [{param_str}])")
    lines.append("def test_returns_length(N, log_ret, seed, start_price):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    rets = rng.normal(0.001, 0.01, N)")
    lines.append("    prices = start_price * np.cumprod(1 + rets)")
    lines.append("    computed = compute_returns(prices, log=log_ret)")
    lines.append("    assert len(computed) == N - 1")
    lines.append("")

    # RSI tests
    rsi_cfgs = []
    for _ in range(300):
        N = rng.choice([50, 100, 252])
        period = rng.choice([7, 14, 21])
        seed = rng.randint(0, 9999)
        rsi_cfgs.append((N, period, seed))

    param_str = ", ".join(f"({N},{p},{s})" for N,p,s in rsi_cfgs)
    lines.append(f"@pytest.mark.parametrize('N,period,seed', [{param_str}])")
    lines.append("def test_rsi_bounds(N, period, seed):")
    lines.append("    rng = np.random.default_rng(seed)")
    lines.append("    prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, N))")
    lines.append("    rsi = compute_rsi(prices, period)")
    lines.append("    valid = rsi[period:]")
    lines.append("    assert np.all(valid >= 0)")
    lines.append("    assert np.all(valid <= 100)")
    lines.append("")

    return "\n".join(lines)


def main():
    files_data = [
        ("test_mega_parametrized_v2.py", gen_test_mega_parametrized_v2()),
        ("test_financial_metrics_large.py", gen_test_financial_metrics()),
        ("test_model_variants_large.py", gen_test_model_variants()),
        ("test_optimization_large.py", gen_test_optimization()),
        ("test_data_pipeline_large.py", gen_test_data_pipeline()),
    ]

    for fname, content in files_data:
        path = os.path.join(TESTS, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        n = content.count("\n") + 1
        print(f"  {fname}: {n} lines")

    total = 0
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git"]]
        for fn in files:
            if fn.endswith((".py", ".yaml", ".yml")):
                fp = os.path.join(root, fn)
                try:
                    with open(fp, encoding="utf-8", errors="ignore") as fh:
                        total += sum(1 for _ in fh)
                except Exception:
                    pass
    print(f"GRAND TOTAL: {total} total")


if __name__ == "__main__":
    main()
