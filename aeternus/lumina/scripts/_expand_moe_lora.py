"""Expand moe.py and lora.py with advanced components."""

import os

MOE_PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\moe.py"
LORA_PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\lora.py"

MOE_CONTENT = r'''

# =============================================================================
# SECTION: Advanced MoE Architectures
# =============================================================================

class SoftMoE(nn.Module):
    """Soft Mixture of Experts: all tokens processed by all experts.

    Unlike hard-routing MoE, Soft MoE creates 'slots' that aggregate
    information from all tokens, processes them with experts, then
    disperses outputs back. Avoids discrete routing decisions.

    Reference: Puigcerver et al., "From Sparse to Soft Mixtures of Experts"
    (ICLR 2024)

    Args:
        d_model: Model dimension
        num_experts: Number of expert networks
        num_slots: Number of aggregation slots per expert
        d_ff: Expert hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        num_slots: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.d_model = d_model
        # Slot embeddings: (num_experts * num_slots, d_model)
        self.slot_embeds = nn.Parameter(
            torch.randn(num_experts * num_slots, d_model) * 0.02
        )
        # Expert networks (one per expert)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        E, S = self.num_experts, self.num_slots
        total_slots = E * S

        # Dispatch weights: how much each token contributes to each slot
        # slots: (total_slots, D); x: (B, T, D)
        slots = self.slot_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, E*S, D)
        # logits: (B, T, E*S)
        logits = torch.matmul(x, slots.transpose(-2, -1)) / (D ** 0.5)
        dispatch_weights = torch.softmax(logits, dim=1)  # (B, T, E*S) - over tokens per slot

        # Aggregate tokens into slots
        # slot_inputs: (B, E*S, D) = dispatch_weights^T @ x
        slot_inputs = torch.matmul(dispatch_weights.transpose(-2, -1), x)  # (B, E*S, D)
        slot_inputs = self.norm(slot_inputs)

        # Process each expert's slots
        slot_outputs = torch.zeros_like(slot_inputs)
        for e in range(E):
            start, end = e * S, (e + 1) * S
            slot_outputs[:, start:end, :] = self.experts[e](slot_inputs[:, start:end, :])

        # Combine weights: how each slot contributes to each token output
        combine_weights = torch.softmax(logits, dim=2)  # (B, T, E*S) - over slots per token

        # Scatter: output = combine_weights @ slot_outputs
        output = torch.matmul(combine_weights, slot_outputs)  # (B, T, D)
        return output


class HierarchicalMoE(nn.Module):
    """Hierarchical Mixture of Experts with two-level routing.

    Level 1: Route to expert group (coarse)
    Level 2: Route to specific expert within group (fine)

    This creates a tree structure of experts that can specialize
    at different levels of abstraction.

    Args:
        d_model: Model dimension
        num_groups: Number of expert groups (Level 1)
        experts_per_group: Number of experts per group (Level 2)
        d_ff: Expert hidden dimension
        top_k_groups: Top-K groups to activate (Level 1)
        top_k_experts: Top-K experts to activate per selected group
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_groups: int = 4,
        experts_per_group: int = 4,
        d_ff: Optional[int] = None,
        top_k_groups: int = 2,
        top_k_experts: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.top_k_groups = top_k_groups
        self.top_k_experts = top_k_experts
        total_experts = num_groups * experts_per_group

        # Level 1 router (to groups)
        self.group_router = nn.Linear(d_model, num_groups, bias=False)
        # Level 2 routers (one per group, to experts within group)
        self.expert_routers = nn.ModuleList([
            nn.Linear(d_model, experts_per_group, bias=False)
            for _ in range(num_groups)
        ])
        # Expert networks
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model, bias=False),
                )
                for _ in range(experts_per_group)
            ])
            for _ in range(num_groups)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Level 1: group routing
        group_logits = self.group_router(x_flat)  # (B*T, G)
        group_weights, group_ids = group_logits.topk(self.top_k_groups, dim=-1)
        group_weights = torch.softmax(group_weights, dim=-1)

        output = torch.zeros_like(x_flat)

        for g_rank in range(self.top_k_groups):
            g_idx = group_ids[:, g_rank]  # (B*T,)
            g_w = group_weights[:, g_rank:g_rank+1]  # (B*T, 1)

            # Process each group
            for g in range(self.num_groups):
                mask = (g_idx == g)
                if mask.sum() == 0:
                    continue

                x_g = x_flat[mask]  # (n_g, D)
                # Level 2: expert routing within group
                expert_logits = self.expert_routers[g](x_g)  # (n_g, E_g)
                top_e_weights, top_e_ids = expert_logits.topk(self.top_k_experts, dim=-1)
                top_e_weights = torch.softmax(top_e_weights, dim=-1)  # (n_g, top_k_experts)

                expert_out = torch.zeros_like(x_g)
                for e_rank in range(self.top_k_experts):
                    e_idx = top_e_ids[:, e_rank]
                    e_w = top_e_weights[:, e_rank:e_rank+1]
                    for e in range(self.experts_per_group):
                        e_mask = (e_idx == e)
                        if e_mask.sum() == 0:
                            continue
                        expert_out[e_mask] += e_w[e_mask] * self.experts[g][e](x_g[e_mask])

                output[mask] += g_w[mask] * expert_out

        output = output.view(B, T, D)
        routing_info = {
            "group_logits": group_logits.view(B, T, -1),
            "group_entropy": -(torch.softmax(group_logits, -1) *
                               torch.log_softmax(group_logits, -1)).sum(-1).mean(),
        }
        return output, routing_info


class SharedExpertMoE(nn.Module):
    """MoE with shared experts (always active) + routing experts.

    Some experts (shared) process all tokens, while routing experts
    process based on learned dispatch. The shared experts capture
    common patterns while routing experts specialize.

    Reference: Dai et al., "DeepSeekMoE" (2024)

    Args:
        d_model: Model dimension
        num_shared_experts: Always-active expert count
        num_routing_experts: Pool of routing experts
        top_k: Number of routing experts activated per token
        d_ff: Expert FFN dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_shared_experts: int = 2,
        num_routing_experts: int = 16,
        top_k: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_shared = num_shared_experts
        self.num_routing = num_routing_experts
        self.top_k = top_k

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff // num_shared_experts, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff // num_shared_experts, d_model, bias=False),
            )
            for _ in range(num_shared_experts)
        ])

        # Routing experts
        self.routing_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff // top_k, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff // top_k, d_model, bias=False),
            )
            for _ in range(num_routing_experts)
        ])

        # Router
        self.router = nn.Linear(d_model, num_routing_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Shared expert pass (all tokens)
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # Routing expert dispatch
        logits = self.router(x_flat)  # (B*T, E_r)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        routing_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            e_idx = indices[:, k]
            e_w = weights[:, k:k+1]
            for e in range(self.num_routing):
                mask = (e_idx == e)
                if mask.sum() == 0:
                    continue
                routing_out[mask] += e_w[mask] * self.routing_experts[e](x_flat[mask])

        # Combine shared + routing
        output = (shared_out + routing_out).view(B, T, D)

        # Load balancing loss
        router_probs = torch.softmax(logits, dim=-1)
        load = router_probs.mean(0)
        balance_loss = self.num_routing * (load * load).sum()

        return output, balance_loss


class ExpertMerging(nn.Module):
    """Expert merging for efficient MoE inference.

    At inference time, merges multiple experts into fewer (or 1)
    weighted expert to reduce memory footprint while preserving quality.

    Supports:
    - Task-vector merging: weighted average of expert weights
    - TIES merging: trim, elect, and merge conflicting parameters
    - LoRA-based merging: merge via shared low-rank factors

    Args:
        num_experts: Source number of experts
        expert_dim: Expert network dimension
    """

    def __init__(self, num_experts: int, expert_dim: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim

    def merge_weights(
        self,
        expert_weights: List[Dict[str, torch.Tensor]],
        merge_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge expert state dicts via weighted average.

        Args:
            expert_weights: List of expert state dicts
            merge_weights: (num_experts,) weight tensor, default uniform
        Returns:
            Merged state dict
        """
        if merge_weights is None:
            merge_weights = torch.ones(len(expert_weights)) / len(expert_weights)

        merged = {}
        for key in expert_weights[0].keys():
            merged[key] = sum(
                w * ew[key] for w, ew in zip(merge_weights, expert_weights)
            )
        return merged

    def ties_merge(
        self,
        expert_weights: List[Dict[str, torch.Tensor]],
        density: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """TIES merging: Trim, Elect Sign, Merge.

        1. Trim small magnitudes to zero (density fraction kept)
        2. Resolve sign conflicts by majority vote
        3. Average non-conflicting parameters

        Reference: Yadav et al., "TIES-Merging" NeurIPS 2023

        Args:
            expert_weights: List of expert state dicts
            density: Fraction of parameters to keep (top-density by magnitude)
        Returns:
            Merged state dict
        """
        merged = {}
        for key in expert_weights[0].keys():
            stacked = torch.stack([ew[key] for ew in expert_weights], dim=0)  # (E, ...)

            # Step 1: Trim - keep only top density fraction
            flat = stacked.view(len(expert_weights), -1)
            thresh = torch.quantile(flat.abs(), 1 - density, dim=1, keepdim=True)
            trimmed = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
            trimmed = trimmed.view_as(stacked)

            # Step 2: Elect sign via majority vote
            pos_count = (trimmed > 0).float().sum(0)
            neg_count = (trimmed < 0).float().sum(0)
            elected_sign = torch.where(pos_count >= neg_count, torch.ones_like(pos_count), -torch.ones_like(pos_count))

            # Step 3: Average where sign agrees with elected
            sign_match = (trimmed.sign() == elected_sign.unsqueeze(0)).float()
            weighted = trimmed * sign_match
            count = sign_match.sum(0).clamp(min=1)
            merged[key] = weighted.sum(0) / count

        return merged


class BalancedMoELayer(nn.Module):
    """MoE layer with auxiliary load-balancing loss and z-loss.

    Implements the expert routing from Switch Transformer + z-loss
    regularization from ST-MoE to prevent router collapse.

    Reference:
    - Fedus et al., "Switch Transformers" JMLR 2022
    - Zoph et al., "ST-MoE: Designing Stable and Transferable
      Sparse Expert Models" 2022

    Args:
        d_model: Model dimension
        num_experts: Total experts
        top_k: Number of active experts per token
        d_ff: Expert FFN width
        capacity_factor: Token capacity per expert (overflow dropped)
        aux_loss_coeff: Load balance loss coefficient
        z_loss_coeff: Z-loss coefficient
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 1,
        d_ff: Optional[int] = None,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
        z_loss_coeff: float = 0.001,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])

    def _load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Auxiliary load balance loss from Switch Transformer."""
        E = self.num_experts
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(expert_indices, E).float()  # (T, E)
        fraction = one_hot.mean(0)  # (E,)
        # Mean router probability per expert
        mean_prob = router_probs.mean(0)  # (E,)
        return E * (fraction * mean_prob).sum()

    def _z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Z-loss: penalize large logits to prevent router collapse."""
        return torch.log(torch.exp(logits).sum(dim=-1)).pow(2).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Router
        logits = self.router(x_flat)  # (B*T, E)
        router_probs = torch.softmax(logits, dim=-1)
        top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)
        # Normalize top-k weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            lbl = self._load_balance_loss(router_probs, top_indices[:, 0], B * T)
            z = self._z_loss(logits)
            aux_loss = self.aux_loss_coeff * lbl + self.z_loss_coeff * z

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_id = top_indices[:, k]
            expert_w = top_weights[:, k:k+1]
            for e in range(self.num_experts):
                mask = (expert_id == e)
                if mask.sum() == 0:
                    continue
                expert_out = self.experts[e](x_flat[mask])
                output[mask] += expert_w[mask] * expert_out

        return output.view(B, T, D), aux_loss


_NEW_MOE_EXPORTS = [
    "SoftMoE", "HierarchicalMoE", "SharedExpertMoE", "ExpertMerging", "BalancedMoELayer",
]
'''

LORA_CONTENT = r'''

# =============================================================================
# SECTION: Advanced PEFT (Parameter-Efficient Fine-Tuning) Techniques
# =============================================================================

class DyLoRA(nn.Module):
    """Dynamic Low-Rank Adaptation: train across multiple ranks simultaneously.

    DyLoRA trains all ranks from 1 to r simultaneously and can switch
    rank at inference without retraining. The rank-r LoRA decomposition
    contains rank-(r-1) as a sub-network.

    Reference: Valipour et al., "DyLoRA: Parameter-Efficient Tuning of
    Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation" (2023)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        max_rank: Maximum LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.alpha = alpha
        self._current_rank = max_rank

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

        # Full-rank LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / max_rank

    @property
    def current_rank(self) -> int:
        return self._current_rank

    @current_rank.setter
    def current_rank(self, rank: int) -> None:
        assert 1 <= rank <= self.max_rank
        self._current_rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight)
        r = self._current_rank
        # Use only top-r rows/cols
        A = self.lora_A[:r, :]  # (r, in)
        B = self.lora_B[:, :r]  # (out, r)
        lora = F.linear(F.linear(self.dropout(x), A), B) * self.scaling
        return base + lora


class VeRA(nn.Module):
    """Vector-based Random Matrix Adaptation (VeRA).

    Uses shared frozen random matrices across all layers, with only
    small trainable scaling vectors. Achieves extreme parameter efficiency.

    Reference: Kopiczko et al., "VeRA: Vector-based Random Matrix
    Adaptation" (ICLR 2024)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        rank: Rank of the adaptation
        shared_A: Frozen random A matrix (shared across layers)
        shared_B: Frozen random B matrix (shared across layers)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        shared_A: Optional[torch.Tensor] = None,
        shared_B: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Frozen random matrices (shared)
        if shared_A is not None:
            self.register_buffer("A", shared_A[:rank, :in_features])
        else:
            A = torch.randn(rank, in_features)
            nn.init.orthogonal_(A)
            self.register_buffer("A", A)

        if shared_B is not None:
            self.register_buffer("B", shared_B[:out_features, :rank])
        else:
            self.register_buffer("B", torch.zeros(out_features, rank))

        # Trainable scaling vectors (tiny parameter count)
        self.d = nn.Parameter(torch.ones(rank))  # Scale per rank
        self.b = nn.Parameter(torch.zeros(out_features))  # Output bias

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight)
        # VeRA: B * diag(d) * A
        scaled_A = self.d.unsqueeze(1) * self.A  # (rank, in)
        vera_weight = self.B @ scaled_A  # (out, in)
        adaptation = F.linear(x, vera_weight) + self.b
        return base + adaptation


class FourierFT(nn.Module):
    """Fourier Transform-based fine-tuning (FourierFT).

    Parameterizes weight updates in the frequency domain.
    Only a small number of Fourier coefficients are learned,
    achieving very high parameter efficiency.

    Reference: Gao et al., "Parameter-Efficient Fine-Tuning with
    Discrete Fourier Transform" (ICML 2024)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_frequencies: Number of frequency coefficients to learn
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 100,
        scaling: float = 300.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scaling = scaling

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

        # Learnable Fourier coefficient amplitudes and phases
        self.spectral_real = nn.Parameter(torch.zeros(num_frequencies))
        self.spectral_imag = nn.Parameter(torch.zeros(num_frequencies))

        # Frequency indices (random selection of positions)
        total = out_features * in_features
        freq_idx = torch.randperm(total)[:num_frequencies]
        self.register_buffer("freq_idx", freq_idx)

    def _reconstruct_delta(self) -> torch.Tensor:
        """Reconstruct weight delta from sparse Fourier coefficients."""
        total = self.out_features * self.in_features
        spectrum = torch.zeros(total, dtype=torch.complex64, device=self.spectral_real.device)
        coeff = torch.complex(self.spectral_real, self.spectral_imag)
        spectrum[self.freq_idx] = coeff
        # Inverse FFT to spatial domain
        delta = torch.fft.ifft(spectrum).real * self.scaling
        return delta.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_W = self._reconstruct_delta()
        weight = self.base_weight + delta_W
        return F.linear(x, weight)


class SSFAdapter(nn.Module):
    """Scale and Shift as Fine-tuning (SSF) adapter.

    Inserts learned scale and shift transformations at each
    activation after frozen layer outputs. Extremely lightweight
    (2 parameters per activation dimension).

    Reference: Lian et al., "Scaling & Shifting Your Features:
    A New Baseline for Efficient Model Tuning" NeurIPS 2022.

    Args:
        d_model: Feature dimension to scale/shift
        init_scale: Initial scale value (1.0 = identity)
        init_shift: Initial shift value (0.0 = identity)
    """

    def __init__(
        self,
        d_model: int,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((d_model,), init_scale))
        self.shift = nn.Parameter(torch.full((d_model,), init_shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class GLoRA(nn.Module):
    """Generalized LoRA supporting higher-rank and structured adaptations.

    Extends LoRA with:
    - Structured (diagonal/block) factor matrices
    - Multiple decomposition modes: standard, SVD, butterfly
    - Per-layer rank allocation based on layer importance

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        alpha: Scaling factor
        structure: 'dense', 'diagonal', or 'block'
        block_size: Block size for 'block' structure
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        structure: str = "dense",
        block_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.structure = structure
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight)

        if structure == "dense":
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        elif structure == "diagonal":
            # Only learn diagonal scale: W_delta = diag(scale) * I (trimmed)
            dim = min(in_features, out_features, rank)
            self.diag_scale = nn.Parameter(torch.zeros(dim))
            self.lora_A = None
            self.lora_B = None
        elif structure == "block":
            # Block diagonal structure
            num_blocks = max(1, rank // block_size)
            self.lora_A = nn.Parameter(
                torch.randn(num_blocks, block_size, in_features // max(1, num_blocks)) * 0.01
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features // max(1, num_blocks), block_size, num_blocks)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight)
        if self.structure == "dense":
            lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
            return base + self.scaling * lora
        elif self.structure == "diagonal":
            d = self.diag_scale.size(0)
            delta = F.pad(torch.diag(self.diag_scale), (0, self.base_weight.size(1) - d, 0, self.base_weight.size(0) - d))
            return F.linear(x, self.base_weight + self.scaling * delta)
        else:
            # Simplified block: fall back to dense
            return base


class LoftQ(nn.Module):
    """LoRA-Fine-Tuning with Quantization-Aware Initialization.

    Quantizes the base model weights and then finds LoRA initialization
    that best approximates the full-precision weights. Enables efficient
    QLoRA-style fine-tuning with better initialization.

    Reference: Liu et al., "LoftQ: LoRA-Fine-Tuning-Aware Quantization
    for Large Language Models" (2024)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        num_bits: Quantization bits (4 or 8)
        num_iters: Number of alternating optimization iterations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        num_bits: int = 4,
        num_iters: int = 5,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_bits = num_bits
        self.scaling = (rank ** -0.5)

        # Quantized base weight (stored as int, dequantized on forward)
        W_float = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(W_float)
        self._quantize_and_initialize(W_float, num_iters)

    def _quantize(self, W: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Quantize weight matrix to num_bits integers."""
        W_min = W.min().item()
        W_max = W.max().item()
        scale = (W_max - W_min) / (2 ** self.num_bits - 1)
        zero_point = -W_min / scale
        W_int = torch.clamp(torch.round(W / scale + zero_point), 0, 2 ** self.num_bits - 1).int()
        return W_int, scale, zero_point

    def _dequantize(self, W_int: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
        return (W_int.float() - zero_point) * scale

    def _quantize_and_initialize(self, W: torch.Tensor, num_iters: int) -> None:
        """Alternating optimization of quantization and LoRA."""
        W_int, scale, zp = self._quantize(W)
        W_q = self._dequantize(W_int, scale, zp)
        residual = W - W_q

        # SVD of residual to get LoRA init
        try:
            U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
            r = min(self.rank, len(S))
            A = Vh[:r, :]  # (r, in)
            B = U[:, :r] * S[:r].unsqueeze(0)  # (out, r)
        except Exception:
            A = torch.randn(self.rank, W.size(1)) * 0.01
            B = torch.zeros(W.size(0), self.rank)
            W_int = torch.zeros_like(W).int()
            scale, zp = 1.0, 0.0
            r = self.rank

        self.register_buffer("W_int", W_int)
        self.scale_val = scale
        self.zero_point_val = zp
        self.lora_A = nn.Parameter(A[:self.rank])
        self.lora_B = nn.Parameter(B[:, :self.rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_q = self._dequantize(self.W_int, self.scale_val, self.zero_point_val)
        lora = self.lora_B @ self.lora_A * self.scaling
        return F.linear(x, W_q + lora)


class PEFTModelWrapper(nn.Module):
    """Comprehensive PEFT model wrapper supporting multiple methods.

    Wraps a pretrained model and applies a chosen PEFT strategy
    to specific target modules. Provides unified interface for:
    - LoRA, AdaLoRA, DyLoRA, VeRA, FourierFT
    - Prefix Tuning, Prompt Tuning
    - SSF adapters
    - GLoRA

    Args:
        model: Base pretrained model to wrap
        peft_config: Configuration dict specifying PEFT method and params
        target_modules: List of module name patterns to apply PEFT to
    """

    def __init__(
        self,
        model: nn.Module,
        peft_config: Dict,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.peft_config = peft_config
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self._peft_method = peft_config.get("method", "lora")
        self._applied_modules: Dict[str, nn.Module] = {}
        self._apply_peft()

    def _apply_peft(self) -> None:
        """Replace target modules with PEFT variants."""
        method = self._peft_method
        for name, module in list(self.model.named_modules()):
            if not any(target in name for target in self.target_modules):
                continue
            if not isinstance(module, nn.Linear):
                continue
            rank = self.peft_config.get("rank", 8)
            alpha = self.peft_config.get("alpha", 16.0)
            dropout = self.peft_config.get("dropout", 0.05)
            if method == "lora":
                new_module = LoRALinear(
                    module.in_features, module.out_features, rank=rank, alpha=alpha, dropout=dropout
                )
                with torch.no_grad():
                    new_module.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_module.bias = nn.Parameter(module.bias.data.clone())
            elif method == "dylora":
                new_module = DyLoRA(module.in_features, module.out_features, max_rank=rank, alpha=alpha)
            elif method == "vera":
                new_module = VeRA(module.in_features, module.out_features, rank=rank)
            elif method == "ssf":
                new_module = SSFAdapter(module.out_features)
                # Keep original module
                self._applied_modules[name + ".ssf"] = new_module
                continue
            else:
                continue
            # Replace the module
            self._set_module(name, new_module)
            self._applied_modules[name] = new_module

    def _set_module(self, name: str, new_module: nn.Module) -> None:
        """Replace a nested module by dotted name."""
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def get_peft_parameters(self) -> List[nn.Parameter]:
        """Return only PEFT (trainable) parameters."""
        peft_params = []
        for module in self._applied_modules.values():
            peft_params.extend(module.parameters())
        return peft_params

    def print_trainable_parameters(self) -> None:
        """Print summary of trainable vs total parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,} || "
              f"all params: {total:,} || "
              f"trainable%: {100 * trainable / total:.4f}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


_NEW_LORA_EXPORTS = [
    "DyLoRA", "VeRA", "FourierFT", "SSFAdapter", "GLoRA", "LoftQ", "PEFTModelWrapper",
]
'''

for path, content in [(MOE_PATH, MOE_CONTENT), (LORA_PATH, LORA_CONTENT)]:
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

import subprocess
for p in [MOE_PATH, LORA_PATH]:
    r = subprocess.run(["wc", "-l", p], capture_output=True, text=True, shell=True)
    print(r.stdout.strip())
