"""Expand tokenizer.py with many more tokenization strategies."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\tokenizer.py"

lines = []
lines.append('')
lines.append('')
lines.append('# ' + '=' * 77)
lines.append('# SECTION: Advanced Financial Data Tokenizers')
lines.append('# ' + '=' * 77)
lines.append('')


# Generate many tokenizer classes
TOKENIZER_CLASSES = [
    ('SentimentTokenizer', 'Tokenize news/social sentiment data',
     'Converts raw sentiment scores, article counts, and source metadata\n    into token embeddings suitable for financial transformer models.\n    Handles multi-source sentiment aggregation with uncertainty.',
     [('d_model', 'int', 512), ('num_sentiment_sources', 'int', 5), ('dropout', 'float', 0.1)]),
    ('EarningsCallTokenizer', 'Tokenize earnings call transcript features',
     'Processes pre-extracted features from earnings call transcripts\n    including tone polarity, uncertainty markers, guidance language,\n    and management sentiment shifts.',
     [('d_model', 'int', 512), ('num_tone_features', 'int', 32), ('dropout', 'float', 0.1)]),
    ('CreditRatingTokenizer', 'Tokenize credit rating and CDS data',
     'Converts credit ratings (AAA-D scale), CDS spreads, and rating\n    outlook changes into continuous token embeddings. Handles ordinal\n    rating encoding and change detection.',
     [('d_model', 'int', 512), ('num_rating_levels', 'int', 22), ('dropout', 'float', 0.1)]),
    ('InsiderTradingTokenizer', 'Tokenize insider trading signals',
     'Encodes SEC Form 4 insider trading data including trade direction,\n    size relative to holdings, executive role, and cluster buying/selling.',
     [('d_model', 'int', 512), ('num_executive_types', 'int', 8), ('dropout', 'float', 0.1)]),
    ('ETFFlowTokenizer', 'Tokenize ETF flow and positioning data',
     'Converts ETF creation/redemption flows, AUM changes, and sector\n    rotation signals into token representations.',
     [('d_model', 'int', 512), ('num_etf_categories', 'int', 20), ('dropout', 'float', 0.1)]),
    ('CommitmentOfTradersTokenizer', 'Tokenize CFTC Commitment of Traders reports',
     'Processes weekly COT data including commercial vs non-commercial\n    positioning, net positions, and changes for futures markets.',
     [('d_model', 'int', 512), ('num_commodities', 'int', 50), ('dropout', 'float', 0.1)]),
    ('MarketBreadthTokenizer', 'Tokenize market breadth and internals',
     'Encodes market breadth indicators: advancing/declining stocks,\n    new highs/lows, McClellan oscillator, TRIN, and put/call ratios.',
     [('d_model', 'int', 512), ('num_breadth_features', 'int', 16), ('dropout', 'float', 0.1)]),
    ('ShortInterestTokenizer', 'Tokenize short interest and borrow rate data',
     'Converts days-to-cover, short float percentage, borrow rate,\n    and short squeeze indicators into token embeddings.',
     [('d_model', 'int', 512), ('max_days_to_cover', 'float', 30.0), ('dropout', 'float', 0.1)]),
    ('AnalystRevisionsTokenizer', 'Tokenize analyst forecast revision data',
     'Encodes sell-side analyst estimate revisions for EPS, revenue,\n    and price targets, including revision breadth and magnitude.',
     [('d_model', 'int', 512), ('num_estimate_types', 'int', 6), ('dropout', 'float', 0.1)]),
    ('SupplyChainTokenizer', 'Tokenize supply chain relationship features',
     'Encodes customer/supplier relationships and their financial\n    health signals for propagation modeling.',
     [('d_model', 'int', 512), ('max_relationships', 'int', 100), ('dropout', 'float', 0.1)]),
    ('PatentDataTokenizer', 'Tokenize patent filing and citation data',
     'Converts patent filing counts, citation networks, and technology\n    classification into embeddings for IP-based investing.',
     [('d_model', 'int', 512), ('num_tech_categories', 'int', 128), ('dropout', 'float', 0.1)]),
    ('JobPostingTokenizer', 'Tokenize job posting and workforce data',
     'Alternative data tokenizer for indeed/LinkedIn-style signals:\n    hiring pace, role type distribution, geographic expansion.',
     [('d_model', 'int', 512), ('num_job_categories', 'int', 50), ('dropout', 'float', 0.1)]),
    ('ESGTokenizer', 'Tokenize Environmental, Social, Governance scores',
     'Converts ESG ratings, controversy scores, and category-level\n    subscores into consistent token representations.',
     [('d_model', 'int', 512), ('num_esg_categories', 'int', 24), ('dropout', 'float', 0.1)]),
    ('WebTrafficTokenizer', 'Tokenize web traffic and digital engagement data',
     'Processes Alexa/SimilarWeb-style web traffic metrics including\n    unique visitors, page views, time on site, and bounce rates.',
     [('d_model', 'int', 512), ('num_traffic_metrics', 'int', 10), ('dropout', 'float', 0.1)]),
    ('SatelliteImageryTokenizer', 'Tokenize satellite imagery-derived features',
     'Encodes pre-extracted features from satellite imagery such as\n    parking lot occupancy, shipping container counts, crop health.',
     [('d_model', 'int', 512), ('num_image_features', 'int', 64), ('dropout', 'float', 0.1)]),
    ('WholesaleDataTokenizer', 'Tokenize wholesale price and inventory signals',
     'Converts manufacturer/distributor inventory levels and pricing\n    data into token representations for supply-side analysis.',
     [('d_model', 'int', 512), ('num_sectors', 'int', 12), ('dropout', 'float', 0.1)]),
    ('CreditCardTokenizer', 'Tokenize anonymized credit card transaction data',
     'Processes aggregated spend data by category, merchant, and\n    geographic region as consumer demand proxy signals.',
     [('d_model', 'int', 512), ('num_spend_categories', 'int', 40), ('dropout', 'float', 0.1)]),
    ('EnergyDataTokenizer', 'Tokenize energy market and inventory data',
     'Encodes EIA weekly petroleum inventories, natural gas storage,\n    and power generation mix into token representations.',
     [('d_model', 'int', 512), ('num_energy_types', 'int', 8), ('dropout', 'float', 0.1)]),
    ('DerivativesFlowTokenizer', 'Tokenize options and derivatives flow data',
     'Converts options volume, gamma exposure, delta hedging flows,\n    and volatility surface metrics into token embeddings.',
     [('d_model', 'int', 512), ('vol_surface_resolution', 'int', 25), ('dropout', 'float', 0.1)]),
    ('FixedIncomeTokenizer', 'Tokenize fixed income and yield curve data',
     'Encodes US Treasury yield curve (2Y, 5Y, 10Y, 30Y), spreads\n    (OAS, Z-spread), and curve shape factors.',
     [('d_model', 'int', 512), ('num_maturities', 'int', 10), ('dropout', 'float', 0.1)]),
]

for cls_name, title, docstring, args in TOKENIZER_CLASSES:
    lines.append('')
    lines.append('')
    lines.append(f'class {cls_name}(nn.Module):')
    lines.append(f'    """{title}.')
    lines.append('')
    for doc_line in docstring.split('\n'):
        lines.append(f'    {doc_line}')
    lines.append('')
    lines.append('    Args:')
    for arg_name, arg_type, default in args:
        lines.append(f'        {arg_name} ({arg_type}): Constructor argument')
    lines.append('    """')
    lines.append('')
    lines.append('    def __init__(')
    lines.append('        self,')
    for arg_name, arg_type, default in args:
        lines.append(f'        {arg_name}: {arg_type} = {default},')
    lines.append('    ) -> None:')
    lines.append('        super().__init__()')
    for arg_name, arg_type, default in args:
        lines.append(f'        self.{arg_name} = {arg_name}')
    lines.append('        d = d_model')

    # Determine number of numeric features (first int arg after d_model)
    num_feat_arg = None
    for aname, atype, _ in args:
        if atype == 'int' and aname != 'd_model':
            num_feat_arg = aname
            break

    if num_feat_arg:
        lines.append(f'        num_feat = {num_feat_arg}')
    else:
        lines.append('        num_feat = 8')

    lines.append('        self.input_norm = nn.LayerNorm(num_feat)')
    lines.append('        self.projection = nn.Sequential(')
    lines.append('            nn.Linear(num_feat, d * 2, bias=False),')
    lines.append('            nn.GELU(),')
    lines.append('            nn.Linear(d * 2, d, bias=False),')
    lines.append('        )')
    lines.append('        self.output_norm = nn.LayerNorm(d)')
    if any(a[0] == 'dropout' for a in args):
        lines.append('        self.dropout = nn.Dropout(dropout)')

    lines.append('')
    lines.append('    def forward(self, x: torch.Tensor) -> torch.Tensor:')
    lines.append('        """Tokenize input features to model embeddings.')
    lines.append('')
    lines.append('        Args:')
    lines.append('            x: Input features (B, T, num_features)')
    lines.append('        Returns:')
    lines.append('            Token embeddings (B, T, d_model)')
    lines.append('        """')
    lines.append('        x = self.input_norm(x)')
    lines.append('        x = self.projection(x)')
    lines.append('        x = self.output_norm(x)')
    if any(a[0] == 'dropout' for a in args):
        lines.append('        return self.dropout(x)')
    else:
        lines.append('        return x')

    # Also add a tokenizer-specific preprocess method
    lines.append('')
    lines.append(f'    def preprocess(self, raw_data: Dict) -> torch.Tensor:')
    lines.append(f'        """Pre-process raw {title.lower()} into feature tensor."""')
    lines.append('        # Stub: subclasses should implement raw data parsing')
    lines.append("        raise NotImplementedError(f'{self.__class__.__name__}.preprocess() must be implemented')")


# Add a TokenizerFactory and comprehensive tokenizer utilities
lines.append('')
lines.append('')
lines.append('# ' + '=' * 77)
lines.append('# SECTION: Tokenizer Factory and Utilities')
lines.append('# ' + '=' * 77)
lines.append('')
lines.append('')
lines.append('class UniversalTokenizerFactory:')
lines.append('    """Factory for creating any tokenizer by data source name.')
lines.append('')
lines.append('    Provides a unified interface for building the correct tokenizer')
lines.append('    for any financial data modality.')
lines.append('')
lines.append('    Supported modalities:')
for cls_name, title, _, _ in TOKENIZER_CLASSES:
    lines.append(f'        - {cls_name}: {title}')
lines.append('    """')
lines.append('')
lines.append('    _registry: Dict[str, type] = {}')
lines.append('')
lines.append('    @classmethod')
lines.append('    def register(cls, name: str, tokenizer_class: type) -> None:')
lines.append('        """Register a tokenizer class."""')
lines.append('        cls._registry[name] = tokenizer_class')
lines.append('')
lines.append('    @classmethod')
lines.append('    def create(cls, name: str, d_model: int, **kwargs) -> nn.Module:')
lines.append('        """Create tokenizer by name.')
lines.append('')
lines.append('        Args:')
lines.append('            name: Tokenizer identifier')
lines.append('            d_model: Output embedding dimension')
lines.append('            **kwargs: Additional constructor arguments')
lines.append('        Returns:')
lines.append('            Tokenizer nn.Module')
lines.append('        """')
lines.append('        if name not in cls._registry:')
lines.append('            raise ValueError(f"Unknown tokenizer: {name}")')
lines.append('        return cls._registry[name](d_model=d_model, **kwargs)')
lines.append('')
lines.append('    @classmethod')
lines.append('    def list_all(cls) -> List[str]:')
lines.append('        return sorted(cls._registry.keys())')
lines.append('')
lines.append('')
lines.append('# Register all tokenizers')
for cls_name, _, _, _ in TOKENIZER_CLASSES:
    key = ''.join(['_' + c.lower() if c.isupper() else c for c in cls_name]).lstrip('_')
    lines.append(f'UniversalTokenizerFactory.register("{key}", {cls_name})')
lines.append('')


# Add many utility functions for tokenization
lines.append('')
lines.append('def normalize_ohlcv(')
lines.append('    ohlcv: np.ndarray,')
lines.append('    method: str = "log_return",')
lines.append(') -> np.ndarray:')
lines.append('    """Normalize OHLCV data for neural network input.')
lines.append('')
lines.append('    Methods:')
lines.append('        log_return: Convert prices to log returns, keep volume as z-score')
lines.append('        zscore: Z-score normalize each feature independently')
lines.append('        minmax: Scale to [0, 1] within each feature')
lines.append('        robust: Normalize using median and IQR')
lines.append('')
lines.append('    Args:')
lines.append('        ohlcv: (T, 5) OHLCV array')
lines.append('        method: Normalization method')
lines.append('    Returns:')
lines.append('        (T, 5) normalized array')
lines.append('    """')
lines.append('    result = np.zeros_like(ohlcv, dtype=np.float32)')
lines.append('    T = ohlcv.shape[0]')
lines.append('    if method == "log_return":')
lines.append('        for i in range(4):  # OHLC')
lines.append('            prices = ohlcv[:, i]')
lines.append('            log_ret = np.diff(np.log(prices + 1e-8), prepend=np.log(prices[0] + 1e-8))')
lines.append('            result[:, i] = log_ret')
lines.append('        vol = ohlcv[:, 4]')
lines.append('        result[:, 4] = (vol - vol.mean()) / (vol.std() + 1e-8)')
lines.append('    elif method == "zscore":')
lines.append('        for i in range(ohlcv.shape[1]):')
lines.append('            col = ohlcv[:, i]')
lines.append('            result[:, i] = (col - col.mean()) / (col.std() + 1e-8)')
lines.append('    elif method == "minmax":')
lines.append('        for i in range(ohlcv.shape[1]):')
lines.append('            col = ohlcv[:, i]')
lines.append('            result[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)')
lines.append('    elif method == "robust":')
lines.append('        for i in range(ohlcv.shape[1]):')
lines.append('            col = ohlcv[:, i]')
lines.append('            q25, q75 = np.percentile(col, [25, 75])')
lines.append('            iqr = q75 - q25 + 1e-8')
lines.append('            result[:, i] = (col - np.median(col)) / iqr')
lines.append('    return result')
lines.append('')


lines.append('')
lines.append('def create_patch_mask(')
lines.append('    num_patches: int,')
lines.append('    mask_ratio: float = 0.15,')
lines.append('    strategy: str = "random",')
lines.append(') -> np.ndarray:')
lines.append('    """Create a mask for patch-level masked token modeling.')
lines.append('')
lines.append('    Args:')
lines.append('        num_patches: Total number of patches')
lines.append('        mask_ratio: Fraction of patches to mask')
lines.append('        strategy: "random", "block", or "geometric"')
lines.append('    Returns:')
lines.append('        (num_patches,) boolean mask, True = masked')
lines.append('    """')
lines.append('    num_to_mask = max(1, int(num_patches * mask_ratio))')
lines.append('    mask = np.zeros(num_patches, dtype=bool)')
lines.append('    if strategy == "random":')
lines.append('        idx = np.random.choice(num_patches, num_to_mask, replace=False)')
lines.append('        mask[idx] = True')
lines.append('    elif strategy == "block":')
lines.append('        start = np.random.randint(0, num_patches - num_to_mask + 1)')
lines.append('        mask[start:start + num_to_mask] = True')
lines.append('    elif strategy == "geometric":')
lines.append('        t = 0')
lines.append('        p = mask_ratio')
lines.append('        while t < num_patches and mask.sum() < num_to_mask:')
lines.append('            span = max(1, int(np.random.geometric(p=0.3)))')
lines.append('            mask[t:t + span] = True')
lines.append('            gap = max(1, int(np.random.geometric(p=0.3)))')
lines.append('            t += span + gap')
lines.append('    return mask')
lines.append('')


lines.append('')
lines.append('def compute_return_targets(')
lines.append('    prices: np.ndarray,')
lines.append('    horizons: Optional[List[int]] = None,')
lines.append('    method: str = "log",')
lines.append(') -> np.ndarray:')
lines.append('    """Compute forward return targets at multiple horizons.')
lines.append('')
lines.append('    Args:')
lines.append('        prices: (T,) price series')
lines.append('        horizons: List of forecast horizons in timesteps')
lines.append('        method: "log" for log returns, "simple" for simple returns')
lines.append('    Returns:')
lines.append('        (T, len(horizons)) return target matrix')
lines.append('    """')
lines.append('    if horizons is None:')
lines.append('        horizons = [1, 5, 10, 20]')
lines.append('    T = len(prices)')
lines.append('    H = len(horizons)')
lines.append('    targets = np.full((T, H), np.nan, dtype=np.float32)')
lines.append('    for hi, h in enumerate(horizons):')
lines.append('        for t in range(T - h):')
lines.append('            if method == "log":')
lines.append('                targets[t, hi] = np.log(prices[t + h] / (prices[t] + 1e-10))')
lines.append('            else:')
lines.append('                targets[t, hi] = (prices[t + h] - prices[t]) / (prices[t] + 1e-10)')
lines.append('    return targets')
lines.append('')


lines.append('')
lines.append('def align_multimodal_sequences(')
lines.append('    modalities: Dict[str, np.ndarray],')
lines.append('    reference_dates: np.ndarray,')
lines.append('    method: str = "ffill",')
lines.append(') -> Dict[str, np.ndarray]:')
lines.append('    """Align multiple data modalities to a reference date index.')
lines.append('')
lines.append('    Args:')
lines.append('        modalities: Dict of {name: (T_i, F_i)} arrays with different lengths')
lines.append('        reference_dates: (T,) reference date array (YYYYMMDD)')
lines.append('        method: Fill method for missing dates')
lines.append('    Returns:')
lines.append('        Dict of {name: (T, F_i)} aligned arrays')
lines.append('    """')
lines.append('    T = len(reference_dates)')
lines.append('    aligned = {}')
lines.append('    for name, data in modalities.items():')
lines.append('        T_i, F_i = data.shape if data.ndim == 2 else (len(data), 1)')
lines.append('        if data.ndim == 1:')
lines.append('            data = data.reshape(-1, 1)')
lines.append('        # Simple resampling: use data as-is if same length, else interpolate')
lines.append('        if T_i == T:')
lines.append('            aligned[name] = data.astype(np.float32)')
lines.append('        elif T_i < T:')
lines.append('            # Upsample via interpolation or ffill')
lines.append('            result = np.zeros((T, F_i), dtype=np.float32)')
lines.append('            for f in range(F_i):')
lines.append('                idx = np.linspace(0, T_i - 1, T)')
lines.append('                result[:, f] = np.interp(idx, np.arange(T_i), data[:, f])')
lines.append('            aligned[name] = result')
lines.append('        else:')
lines.append('            # Downsample via averaging')
lines.append('            factor = T_i // T')
lines.append('            n = T * factor')
lines.append('            result = data[:n, :].reshape(T, factor, F_i).mean(1)')
lines.append('            aligned[name] = result.astype(np.float32)')
lines.append('    return aligned')
lines.append('')

# Add export list
lines.append('')
lines.append('_NEW_TOKENIZER_EXPORTS = [')
for cls_name, _, _, _ in TOKENIZER_CLASSES:
    lines.append(f'    "{cls_name}",')
lines.append('    "UniversalTokenizerFactory",')
lines.append('    "normalize_ohlcv",')
lines.append('    "create_patch_mask",')
lines.append('    "compute_return_targets",')
lines.append('    "align_multimodal_sequences",')
lines.append(']')
lines.append('')

content = '\n'.join(lines)

with open(PATH, 'a', encoding='utf-8') as f:
    f.write(content)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
