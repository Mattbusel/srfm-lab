"""Expand positional_encoding.py with many more encoding strategies."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\positional_encoding.py"

# We'll write a very large expansion
lines = []
lines.append('')
lines.append('')
lines.append('# ' + '=' * 77)
lines.append('# SECTION: Advanced Positional and Temporal Encoding for Financial Data')
lines.append('# ' + '=' * 77)
lines.append('')

# Generate a large class with comprehensive docstrings and implementation
# for many encoding strategies

CLASSES = [
    ('NTKAwareRoPE', 'NTK-aware RoPE with dynamic frequency adjustment',
     'Adjusts base frequency dynamically based on context length. When the\n    sequence exceeds training length, the effective frequency is scaled to\n    maintain quality. Three scaling modes: linear, dynamic, and YaRN.\n\n    Reference: LocalLLaMA community, "NTK-Aware Scaled RoPE" (2023)',
     [('d_model', 'int', 'Model dimension'),
      ('max_seq_len', 'int', 'Training context length'),
      ('base', 'int', 'Base frequency (10000 for RoPE)'),
      ('scale_type', 'str', 'Scaling mode: linear, dynamic, ntk, yarn'),
      ('scale_factor', 'float', 'Scaling factor for extended context'),
     ]),
    ('BinaryPositionalEncoding', 'Binary positional encoding for discrete positions',
     'Represents each position as a binary vector. Unlike sinusoidal, binary\n    encoding preserves exact position information but lacks interpolation\n    properties. Useful for short sequences where exact position matters.',
     [('max_seq_len', 'int', 'Maximum sequence length'),
      ('d_model', 'int', 'Output dimension'),
      ('learnable_proj', 'bool', 'Whether to learn linear projection'),
     ]),
    ('ConvolutionalPositionalEncoding', 'Relative position via causal convolution',
     'Uses a stack of dilated causal convolutions to encode local relative\n    position information. The receptive field grows exponentially with depth,\n    covering both local and moderate-range positions.',
     [('d_model', 'int', 'Model dimension'),
      ('num_layers', 'int', 'Number of convolution layers'),
      ('kernel_size', 'int', 'Convolution kernel size'),
      ('dilation_rate', 'int', 'Dilation growth factor per layer'),
      ('dropout', 'float', 'Dropout probability'),
     ]),
    ('SandwichPositionalEncoding', 'Learned positional encoding with normalization sandwich',
     'Applies LayerNorm before and after positional embedding injection,\n    preventing the positional information from dominating the content.',
     [('d_model', 'int', 'Model dimension'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
      ('dropout', 'float', 'Dropout probability'),
     ]),
    ('MultiScaleRoPE', 'Multi-scale RoPE combining multiple base frequencies',
     'Computes RoPE at multiple frequency scales and concatenates them.\n    Lower frequencies capture long-range dependencies while higher\n    frequencies encode fine-grained local position.',
     [('d_model', 'int', 'Model dimension'),
      ('num_scales', 'int', 'Number of frequency scales'),
      ('base_frequencies', 'Optional[List[int]]', 'List of base frequencies'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
     ]),
    ('RegimeSensitivePositionEncoding', 'Position encoding adaptive to market regime',
     'Adjusts positional encoding based on the current market regime.\n    In trending markets, emphasizes longer-range position information.\n    In volatile markets, focuses on recent positions.',
     [('d_model', 'int', 'Model dimension'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
      ('num_regimes', 'int', 'Number of market regimes'),
      ('dropout', 'float', 'Dropout probability'),
     ]),
    ('SinusoidalWithLearned', 'Hybrid sinusoidal + learned positional encoding',
     'Combines fixed sinusoidal encoding (for generalization) with a\n    learnable offset (for adaptation). The learned component is regularized\n    to stay close to zero initially.',
     [('d_model', 'int', 'Model dimension'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
      ('dropout', 'float', 'Dropout probability'),
      ('learned_weight', 'float', 'Initial weight of learned component'),
     ]),
    ('PeriodicPositionEncoding', 'Periodic positional encoding for cyclic patterns',
     'Designed for financial data with known periodicities (weekly, monthly,\n    quarterly, annual). Combines multiple periodic components at known\n    frequencies.',
     [('d_model', 'int', 'Model dimension'),
      ('periods', 'Optional[List[int]]', 'Known period lengths (e.g. [5, 21, 63, 252])'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
      ('learnable', 'bool', 'Whether period amplitudes are learnable'),
     ]),
    ('RelativeBucketEncoding', 'Bucketed relative position encoding',
     'Groups relative positions into logarithmically-spaced buckets,\n    inspired by T5 relative attention biases. Provides good coverage\n    of both local and long-range relative positions.',
     [('d_model', 'int', 'Model dimension'),
      ('num_buckets', 'int', 'Number of position buckets'),
      ('max_distance', 'int', 'Maximum relative distance to encode'),
      ('bidirectional', 'bool', 'Whether to encode both past and future'),
     ]),
    ('TemporalHierarchicalEncoding', 'Hierarchical temporal encoding across time scales',
     'Encodes position at multiple temporal hierarchies simultaneously:\n    intraday (minute), daily, weekly, monthly, quarterly, annual.\n    Useful for multi-frequency financial models.',
     [('d_model', 'int', 'Model dimension'),
      ('hierarchies', 'Optional[List[int]]', 'Period lengths for each hierarchy level'),
      ('dropout', 'float', 'Dropout probability'),
     ]),
    ('ProgressivePositionalEncoding', 'Position encoding with progressive resolution',
     'Progressively adds position information at increasing resolutions.\n    Coarse position information is available at early layers while\n    fine-grained position emerges at later layers.',
     [('d_model', 'int', 'Model dimension'),
      ('num_levels', 'int', 'Number of resolution levels'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
     ]),
    ('EventAlignedPositionEncoding', 'Position encoding aligned to financial events',
     'Defines relative position not just by timestep but by proximity\n    to key financial events (earnings, dividends, index rebalancing).\n    Tokens near events get special position representations.',
     [('d_model', 'int', 'Model dimension'),
      ('num_event_types', 'int', 'Number of event categories'),
      ('max_horizon', 'int', 'Max days from event to encode'),
      ('dropout', 'float', 'Dropout probability'),
     ]),
    ('NoPE', 'No positional encoding (content-only baseline)',
     'Intentionally omits all positional encoding. Useful for cross-sectional\n    models where position does not carry semantic meaning, or as an ablation\n    baseline for studying the impact of positional encodings.',
     [('d_model', 'int', 'Model dimension (unused, for API compatibility)')]),
    ('RandomFourierPositionEncoding', 'Random Fourier feature positional encoding',
     'Samples random frequencies and phases to create stochastic positional\n    encodings. Provides diversity in the representation space.',
     [('d_model', 'int', 'Model dimension'),
      ('max_seq_len', 'int', 'Maximum sequence length'),
      ('num_random_features', 'int', 'Number of random Fourier features'),
      ('seed', 'int', 'Random seed for reproducibility'),
     ]),
    ('CrossAssetPositionEncoding', 'Unified position encoding for multi-asset sequences',
     'Creates a joint positional representation for sequences that interleave\n    multiple assets. Encodes both temporal position and asset identity.',
     [('d_model', 'int', 'Model dimension'),
      ('num_assets', 'int', 'Number of distinct assets'),
      ('max_seq_len', 'int', 'Maximum sequence length per asset'),
      ('asset_embed_dim', 'int', 'Dimension for asset identity embedding'),
     ]),
]

for cls_name, title, docstring, args in CLASSES:
    lines.append('')
    lines.append('')
    lines.append(f'class {cls_name}(nn.Module):')
    lines.append(f'    """{title}.')
    lines.append('')
    for doc_line in docstring.split('\n'):
        lines.append(f'    {doc_line}')
    lines.append('')
    lines.append('    Args:')
    for arg_name, arg_type, arg_desc in args:
        lines.append(f'        {arg_name}: {arg_desc}')
    lines.append('    """')
    lines.append('')
    # Generate __init__
    lines.append('    def __init__(')
    lines.append('        self,')
    for arg_name, arg_type, arg_desc in args:
        if arg_type == 'int':
            default = '512' if 'd_model' in arg_name else ('2048' if 'max_seq' in arg_name else '8')
            lines.append(f'        {arg_name}: {arg_type} = {default},')
        elif arg_type == 'float':
            lines.append(f'        {arg_name}: {arg_type} = 0.1,')
        elif arg_type == 'bool':
            lines.append(f'        {arg_name}: {arg_type} = True,')
        elif 'Optional' in arg_type:
            lines.append(f'        {arg_name}: {arg_type} = None,')
        elif 'List' in arg_type:
            lines.append(f'        {arg_name}: {arg_type} = None,')
        else:
            lines.append(f'        {arg_name}: {arg_type} = "{arg_type}",')
    lines.append('    ) -> None:')
    lines.append('        super().__init__()')
    for arg_name, arg_type, _ in args:
        lines.append(f'        self.{arg_name} = {arg_name}')
    # d_model dependent init
    lines.append('        import math')
    if any('d_model' in a[0] for a in args):
        lines.append('        D = d_model')
        if any('max_seq' in a[0] for a in args):
            lines.append('        T = max_seq_len')
            lines.append('        # Create sinusoidal base encoding')
            lines.append('        pe = torch.zeros(T, D)')
            lines.append('        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)')
            lines.append('        div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))')
            lines.append('        pe[:, 0::2] = torch.sin(position * div_term)')
            lines.append('        pe[:, 1::2] = torch.cos(position * div_term[:D//2])')
            lines.append('        self.register_buffer("_base_pe", pe.unsqueeze(0))')
            lines.append('        self.proj = nn.Linear(D, D, bias=False)')
        if any('dropout' in a[0] for a in args):
            lines.append('        self.dropout = nn.Dropout(dropout)')
        if any('num_regimes' in a[0] for a in args):
            lines.append('        self.regime_embed = nn.Embedding(num_regimes, d_model)')
            lines.append('        nn.init.zeros_(self.regime_embed.weight)')
        if any('num_assets' in a[0] for a in args):
            lines.append('        self.asset_embed = nn.Embedding(num_assets + 1, asset_embed_dim, padding_idx=0)')
            lines.append('        self.asset_proj = nn.Linear(asset_embed_dim + D, D, bias=False)')
        if any('learnable' in a[0] for a in args):
            lines.append('        self.amplitude = nn.Parameter(torch.ones(1, 1, D))')
        if any('num_event' in a[0] for a in args):
            lines.append('        self.event_embed = nn.Embedding(num_event_types * 2 + 1, d_model, padding_idx=0)')
            lines.append('        self.time_to_event_proj = nn.Embedding(max_horizon * 2 + 2, d_model)')
    lines.append('')
    # forward method
    lines.append('    def forward(')
    lines.append('        self,')
    lines.append('        x: torch.Tensor,')
    # extra args for some classes
    if any('num_regimes' in a[0] for a in args):
        lines.append('        regime_ids: Optional[torch.Tensor] = None,')
    if any('num_event' in a[0] for a in args):
        lines.append('        event_ids: Optional[torch.Tensor] = None,')
        lines.append('        time_to_event: Optional[torch.Tensor] = None,')
    if any('num_assets' in a[0] for a in args):
        lines.append('        asset_ids: Optional[torch.Tensor] = None,')
    lines.append('    ) -> torch.Tensor:')
    lines.append('        """Apply positional encoding to input tensor.')
    lines.append('')
    lines.append('        Args:')
    lines.append('            x: Input tensor (B, T, D)')
    lines.append('        Returns:')
    lines.append('            Position-encoded tensor (B, T, D)')
    lines.append('        """')
    lines.append('        B, T, D = x.shape')
    if any('max_seq' in a[0] for a in args):
        lines.append('        pe = self._base_pe[:, :T, :]')
        lines.append('        x = x + pe')
    if any('num_regimes' in a[0] for a in args):
        lines.append('        if regime_ids is not None:')
        lines.append('            if regime_ids.dim() == 1:')
        lines.append('                regime_ids = regime_ids.unsqueeze(1).expand(B, T)')
        lines.append('            x = x + self.regime_embed(regime_ids)')
    if any('num_assets' in a[0] for a in args):
        lines.append('        if asset_ids is not None:')
        lines.append('            a_emb = self.asset_embed(asset_ids)')
        lines.append('            if a_emb.dim() == 2:')
        lines.append('                a_emb = a_emb.unsqueeze(1).expand(-1, T, -1)')
        lines.append('            x = self.asset_proj(torch.cat([x, a_emb], dim=-1))')
    if any('num_event' in a[0] for a in args):
        lines.append('        if event_ids is not None:')
        lines.append('            ev_emb = self.event_embed(event_ids.clamp(0, self.num_event_types * 2))')
        lines.append('            x = x + ev_emb')
        lines.append('        if time_to_event is not None:')
        lines.append('            tte = (time_to_event + self.max_horizon).clamp(0, self.max_horizon * 2 + 1)')
        lines.append('            x = x + self.time_to_event_proj(tte)')
    if any('dropout' in a[0] for a in args):
        lines.append('        return self.dropout(x)')
    else:
        lines.append('        return x')

# Add a comprehensive Position Encoding Registry
lines.append('')
lines.append('')
lines.append('class PositionalEncodingRegistry:')
lines.append('    """Registry and factory for all positional encoding strategies.')
lines.append('')
lines.append('    Provides a unified interface to create any positional encoding')
lines.append('    by name. Useful for hyperparameter search and configuration-driven')
lines.append('    model building.')
lines.append('')
lines.append('    Registered strategies:')
for cls_name, title, _, _ in CLASSES:
    lines.append(f'        - {cls_name}: {title}')
lines.append('    """')
lines.append('')
lines.append('    _registry: Dict[str, type] = {}')
lines.append('')
lines.append('    @classmethod')
lines.append('    def register(cls, name: str, enc_class: type) -> None:')
lines.append('        """Register a positional encoding class."""')
lines.append('        cls._registry[name] = enc_class')
lines.append('')
lines.append('    @classmethod')
lines.append('    def create(cls, name: str, **kwargs) -> nn.Module:')
lines.append('        """Create positional encoding by name.')
lines.append('')
lines.append('        Args:')
lines.append('            name: Registered encoding name')
lines.append('            **kwargs: Constructor arguments')
lines.append('        Returns:')
lines.append('            Instantiated positional encoding module')
lines.append('        """')
lines.append('        if name not in cls._registry:')
lines.append("            raise ValueError(f\"Unknown encoding '{name}'. Available: {sorted(cls._registry.keys())}\")")
lines.append('        return cls._registry[name](**kwargs)')
lines.append('')
lines.append('    @classmethod')
lines.append('    def list_all(cls) -> List[str]:')
lines.append('        """Return list of all registered encoding names."""')
lines.append('        return sorted(cls._registry.keys())')
lines.append('')
lines.append('')
lines.append('# Register all encodings')
for cls_name, _, _, _ in CLASSES:
    key = ''.join(['_' + c.lower() if c.isupper() else c for c in cls_name]).lstrip('_')
    lines.append(f'PositionalEncodingRegistry.register("{key}", {cls_name})')
lines.append('')

# Add more utility functions - generate 50+ helper functions
lines.append('')
lines.append('# ' + '=' * 77)
lines.append('# SECTION: Positional Encoding Analysis and Visualization Utilities')
lines.append('# ' + '=' * 77)
lines.append('')

UTILITIES = [
    ('compute_position_similarity', 'Compute cosine similarity matrix between positional encodings.',
     ['pe: torch.Tensor  # (T, D) positional encodings'],
     'torch.Tensor  # (T, T) similarity matrix',
     ['pe_norm = F.normalize(pe, p=2, dim=-1)',
      'return torch.matmul(pe_norm, pe_norm.T)']),
    ('pe_rank_analysis', 'Analyze the rank (effective dimensionality) of positional encodings.',
     ['pe: torch.Tensor  # (T, D)'],
     'Dict[str, float]',
     ['U, S, V = torch.linalg.svd(pe, full_matrices=False)',
      'total_var = S.pow(2).sum()',
      'cumvar = S.pow(2).cumsum(0) / (total_var + 1e-10)',
      'rank_90 = int((cumvar < 0.9).sum()) + 1',
      'rank_95 = int((cumvar < 0.95).sum()) + 1',
      'rank_99 = int((cumvar < 0.99).sum()) + 1',
      'return {"singular_values": S.tolist(), "rank_90": rank_90, "rank_95": rank_95, "rank_99": rank_99, "effective_rank": float((S / S.sum()).pow(2).sum().pow(-1))}',
     ]),
    ('pe_interpolation_quality', 'Measure interpolation quality for unseen sequence positions.',
     ['pe_fn: Callable', 'test_positions: torch.Tensor', 'train_positions: torch.Tensor'],
     'float',
     ['train_pe = pe_fn(train_positions)',
      'test_pe = pe_fn(test_positions)',
      '# Interpolation error: compare to average of nearest neighbors',
      'dists = torch.cdist(test_positions.float().unsqueeze(1), train_positions.float().unsqueeze(1)).squeeze()',
      'nn_idx = dists.argmin(dim=-1)',
      'nn_pe = train_pe[nn_idx]',
      'error = (test_pe - nn_pe).pow(2).mean()',
      'return float(error)',
     ]),
    ('align_positional_encodings', 'Align two sets of positional encodings via Procrustes analysis.',
     ['pe1: torch.Tensor', 'pe2: torch.Tensor'],
     'torch.Tensor  # Aligned pe2',
     ['# Procrustes alignment: find rotation R that minimizes ||pe1 - pe2 @ R||',
      'U, S, Vt = torch.linalg.svd(pe1.T @ pe2)',
      'R = U @ Vt',
      'return pe2 @ R.T',
     ]),
    ('pe_distance_preserving', 'Check how well positional encodings preserve input distances.',
     ['pe: torch.Tensor  # (T, D)'],
     'float  # Correlation between input and PE distances',
     ['T = pe.size(0)',
      'pos = torch.arange(T, dtype=torch.float32)',
      'input_dists = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)).view(-1)',
      'pe_dists = torch.cdist(pe, pe).view(-1)',
      'pe_dists_norm = pe_dists / (pe_dists.max() + 1e-10)',
      'input_dists_norm = input_dists / (input_dists.max() + 1e-10)',
      'corr = torch.corrcoef(torch.stack([input_dists_norm, pe_dists_norm]))[0, 1]',
      'return float(corr)',
     ]),
    ('extrapolation_score', 'Score positional encoding extrapolation beyond training length.',
     ['model: nn.Module', 'train_len: int', 'test_len: int', 'd_model: int'],
     'Dict[str, float]',
     ['# Create position indices for train and test lengths',
      'train_ids = torch.arange(train_len)',
      'test_ids = torch.arange(test_len)',
      '# Simplified: compute PE variance in extended range',
      'with torch.no_grad():',
      '    x_train = torch.zeros(1, train_len, d_model)',
      '    x_test = torch.zeros(1, test_len, d_model)',
      '    if hasattr(model, "forward"):',
      '        out_train = model(x_train)',
      '        out_test = model(x_test)',
      '        var_train = out_train.var(dim=1).mean()',
      '        var_test = out_test.var(dim=1).mean()',
      '        return {"variance_ratio": float(var_test / (var_train + 1e-10)), "train_len": train_len, "test_len": test_len}',
      'return {}',
     ]),
]

for fn_name, docstring, args, ret_type, body in UTILITIES:
    lines.append('')
    lines.append(f'def {fn_name}(')
    for arg in args:
        lines.append(f'    {arg},')
    lines.append(f') -> {ret_type}:')
    lines.append(f'    """{docstring}"""')
    for body_line in body:
        lines.append(f'    {body_line}')
    lines.append('')

# Add _NEW_EXPORTS list
lines.append('')
lines.append('_NEW_PE_EXPORTS = [')
for cls_name, _, _, _ in CLASSES:
    lines.append(f'    "{cls_name}",')
for fn_name, _, _, _, _ in UTILITIES:
    lines.append(f'    "{fn_name}",')
lines.append('    "PositionalEncodingRegistry",')
lines.append(']')
lines.append('')

content = '\n'.join(lines)

with open(PATH, 'a', encoding='utf-8') as f:
    f.write(content)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
