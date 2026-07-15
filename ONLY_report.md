# ONLY Method Analysis & Improvement Proposals

## 1. How ONLY Works

ONLY (One-Layer Intervention Sufficiently Mitigates Hallucinations) is a **training-free, single-forward-pass** decoding method for mitigating hallucinations in Vision-Language Models (VLMs).

### 1.1 Core Idea

Produce **two logit distributions** in one forward pass:
- **Normal logits** (`logits`): the standard model output
- **Contrastive logits** (`logits_cd`): a "head-suppressed" version where certain attention heads at layer 0 are dampened or zeroed out based on a per-head entropy ratio

These two are combined at each generation step via a **TVD-based switching rule**.

---

### 1.2 The Contrastive Hidden State

A tensor `hidden_states_cd` is threaded through the model alongside the normal hidden states.

#### Layer 0 — The Intervention Layer

At the chosen "enhance layer" (default: layer 0), the normal attention weights are computed as usual (`Q @ K^T / √d + mask → softmax`). Then:

1. **Clone** the attention weights → `attn_weights_cd`
2. **Separate** text positions and image positions in the sequence
3. **Clean** each head's distribution by zeroing out outlier values (> mean + std):

```python
attn_weights_text = attn_weights[:, :, text_positions]
attn_weights_img  = attn_weights[:, :, image_positions]
attn_weights_text = where(attn_weights_text > mean + std, 0, attn_weights_text)
attn_weights_img  = where(attn_weights_img > mean + std, 0, attn_weights_img)
```

4. **Normalize** per head to get proper distributions, then compute **entropy** for text and image:

```python
entropy_text = -Σ(p_text · log(p_text))   # per head
entropy_img  = -Σ(p_img  · log(p_img))    # per head
ratio = entropy_text / entropy_img
```

5. **Drop low-ratio heads** — heads where `ratio < ratio.mean()` are zeroed out:

```python
removed_heads = where(ratio < ratio.mean())
attn_weights_cd[removed_heads, :, :] = 0
```

6. Compute `attn_output_cd = attn_weights_cd @ V`, project through `o_proj`.

**Why drop low-ratio heads?** These heads have:
- Low text entropy → focused on specific text tokens
- High image entropy → diffuse over image patches
- They're "uncertain connectors" between text and image → removing them weakens spurious visual-text correlations

#### Layers 1–30 — Ghost Path

For layers 1 through 30, `hidden_states_cd` passes through **untouched**:

```python
# In the attention module:
attn_output_cd = hidden_states_cd  # literally a no-op pass-through

# In the decoder layer:
# hidden_states_cd is returned without going through:
#   - post-attention layernorm
#   - residual addition
#   - MLP
```

The tensor from layer 0's intervention ghosts silently through 30 layers.

#### Layer 31 — Final Processing

At the last layer, `hidden_states_cd` finally gets full decoder processing:

```python
hidden_states_cd = self.input_layernorm(hidden_states_cd)
hidden_states_cd = 0.2 * residual + hidden_states_cd  # residual is from NORMAL path
hidden_states_cd = self.post_attention_layernorm(hidden_states_cd)
hidden_states_cd = self.mlp(hidden_states_cd)
hidden_states_cd = residual_cd + hidden_states_cd
```

The `0.2 * residual` injection brings in normal-path information to stabilize the ghost signal. Then the MLP processes it into a proper representation.

#### After All Layers

```python
hidden_states_cd = self.norm(hidden_states_cd)           # final LN
hidden_states_cd = hidden_states_cd + 0.5 * normal_output  # another residual from normal
logits_cd = self.lm_head(hidden_states_cd)                # contrastive logits
```

---

### 1.3 Comparing & Combining the Two Logit Distributions

During sampling, at each autoregressive step:

```python
outputs, logits_cd = model(**model_inputs, use_only=True)
next_token_logits    = outputs.logits[:, -1, :]      # normal
next_token_logits_cd = logits_cd[:, -1, :]           # contrastive
```

#### Step 1: Compute TVD

```python
tvd = Σ | softmax(next_token_logits) - softmax(next_token_logits_cd) |
```

The TVD measures how much the head-suppressed path disagrees with the normal path.

#### Step 2: Switching Rule

- **If TVD < js_gamma** (default ~0.1): distributions are similar → **Complementive decoding** — add them:
  ```python
  diffs = next_token_logits + α_pos × next_token_logits_cd
  ```
  *Rationale:* The heads didn't matter much for this token. Both distributions agree → amplify the shared, robust signal.

- **If TVD ≥ js_gamma**: distributions diverge → **Contrastive decoding** — subtract:
  ```python
  diffs = (1 + α_neg) × next_token_logits - α_neg × next_token_logits_cd
  ```
  *Rationale:* The heads matter a lot. The contrastive path (with heads removed) is more language-prior-biased. Tokens it scores highly are suspect → penalize them.

#### Step 3: Adaptive Plausibility Constraint

```python
cutoff = log(ritual_beta) + max(next_token_logits)
diffs = diffs.masked_fill(next_token_logits < cutoff, -inf)
```

Discard tokens that are too low-probability under the normal distribution, preventing the correction from boosting unlikely tokens.

#### Step 4: Sample

```python
probs = softmax(diffs)
next_tokens = multinomial(probs)
```

---

### 1.4 Why This Works

The normal logits come from the full model — includes whatever hallucination biases exist.

The contrastive logits come from a representation where "entropy-imbalanced" heads at layer 0 were weakened. These heads are confident about specific image patches but uncertain about text → removing them makes the contrastive path **more reliant on language priors**, which actually makes it *more* prone to hallucinating.

The switching rule exploits this:
- When the two distributions agree → the suppression didn't change much → the prediction is robust → amplify it
- When they disagree → the normal path is being heavily influenced by suspect heads → suppress tokens that also score high under the (language-prior-biased) contrastive path

---

## 2. Improvement Proposals

The current ONLY method has a clear limitation: **only layer 0 contributes to the head mask**. Multi-layer attention dynamics are ignored. Each proposal below adds multi-layer information while controlling divergence to preserve the switching mechanism.

---

### Proposal 1: Damped / Exponential Mask Accumulation

**Idea:** Instead of computing the mask once at layer 0 and freezing it, compute a **fresh mask** at every layer and blend it into a running cumulative mask with an exponential moving average.

```python
# At layer 0:
mask_cumulative = compute_binary_head_mask(entropy_ratio_0, threshold)

# At layers 1..31:
mask_fresh = compute_binary_head_mask(entropy_ratio_k, threshold)
mask_cumulative = α × mask_fresh + (1 - α) × mask_cumulative
```

| α | Behavior |
|---|----------|
| 0.0 | Original ONLY (mask never changes after layer 0) |
| 1.0 | Full fresh mask at every layer (worst divergence) |
| 0.1–0.3 | Mask evolves slowly — each new layer nudges it |

**Mechanism via α:**
- α = 0.2 means the mask integrates information from roughly the last 5 layers (1/(1-0.2) ≈ 5-layer effective window)
- Heads that are **consistently** low-entropy across those layers get suppressed
- Heads that spike at only one layer do not significantly move the mask

**Cost:** ~0% overhead — one blend operation per layer (tensor add + scalar multiply).

**Advantage over original:** Multi-layer information, controlled by α. β can be tuned to find the sweet spot where the cd path diverges just enough for a meaningful TVD signal without saturating into the always-subtract regime.

---

### Proposal 2: Score-Based Accumulation

**Idea:** Instead of blending binary masks, track a **running continuous score** per head across layers. A head is suppressed only if its entropy ratio is consistently above threshold across many layers.

```python
# At each layer k:
entropy_ratio_i_k = text_entropy_i / image_entropy_i  # for head i
score_i += (entropy_ratio_i_k - threshold)

# Activation:
mask_i = σ(score_i / temperature)   # continuous in [0, 1]
# or: mask_i = 1 if score_i > 0 else 0  (hard)
```

**Properties:**
- A head must be **persistently** high-entropy-ratio to accumulate a positive score
- Single-layer outliers don't trigger suppression
- The continuous mask (sigmoid) gives softer suppression → less divergence per layer

**Why this is more principled than Proposal 1:**
- Proposal 1 blends mask values directly, which is position-dependent (masks from different layers have different semantics)
- This proposal tracks the **underlying cause** (the entropy ratio), not the mask itself
- More interpretable: the score directly answers "how consistently has this head been in the 'should suppress' regime?"

**Cost:** ~0% — one accumulate + one sigmoid per layer.

**Advanced variant:** Normalize by cross-layer variance:

```python
variance_i += (entropy_ratio_i_k - mean_i)^2
z_score_i = (mean_score_i - threshold) / sqrt(variance_i)
mask_i = σ(z_score_i / temperature)
```

A head with high variance needs stronger evidence before being suppressed. This handles the case where a head is sometimes useful and sometimes harmful — only consistently harmful heads get suppressed.

---

### Proposal 3: Residual Difference Tracking

**Idea:** At each layer, compute the **marginal effect** of that layer's mask on the attention output. Add a scaled fraction of this effect to the cd path. The cd path starts as the normal path and diverges only gradually.

```python
# At layer k:

# 1. Normal attention output (already computed)
attn_output_normal = attn_weights @ V   # (standard path)

# 2. Masked attention output using this layer's entropy ratio
mask_fresh = compute_binary_head_mask(entropy_ratio_k, threshold)  # [num_heads]
attn_weights_cd = attn_weights × mask_fresh.unsqueeze(-1)          # broadcast
attn_output_masked = attn_weights_cd @ V
attn_output_masked = attn_output_masked.o_proj()

# 3. Delta: how much this layer's mask changes the output
delta_k = attn_output_masked - attn_output_normal

# 4. Accumulate into cd path with per-layer scaling
hidden_states_cd = hidden_states_cd + λ_k × delta_k
```

**Initialization:**
```python
hidden_states_cd = hidden_states_normal_at_layer_0  # start equal to normal
# At layer 0 with λ_0 = 1: hidden_states_cd = normal + (masked - normal) = masked path
```

**Per-layer scaling schedule (λ_k):**
- **Constant:** λ_k = λ (e.g., 0.3) — simple, but divergence grows linearly
- **Decaying:** λ_k = λ₀ × γ^k (e.g., γ = 0.85) — later layers contribute less, bounding total divergence
- **Confidence-based:** λ_k ∝ confidence_of_mask_k (using variance or entropy ratio magnitude as confidence)

**Total divergence bound:**
```
total_delta = Σ λ_k × ||delta_k||
```
Since each `delta_k` is bounded by the attention output norm, the total divergence is directly controlled by Σ λ_k — a clean, interpretable dial.

**Why this is different from Proposals 1 & 2:**
- Proposals 1 & 2 modify **what mask** the cd path uses
- This proposal modifies **how much** each layer's mask changes the cd path's hidden state
- The mask itself is recomputed fresh each layer, but the **impact** on cd hidden state is scaled

**Cost:** ~0% — one masked weighted sum + one residual add per layer.

---

### Comparison Table

| Criterion | Original ONLY | Proposal 1 (Damped Mask) | Proposal 2 (Score Accumulation) | Proposal 3 (Residual Delta) |
|-----------|:------------:|:------------------------:|:-------------------------------:|:---------------------------:|
| Multi-layer information | ❌ | ✅ | ✅ | ✅ |
| Complementive regime | ✅ (low div.) | ✅ (tunable via α) | ✅ (tunable via threshold) | ✅ (tunable via λ schedule) |
| What is accumulated | — | Mask values | Entropy ratio scores | Attention output deltas |
| Interpretability | High | Moderate | High (score = "persistence") | High (delta = "marginal effect") |
| Divergence control | None needed | α ∈ [0, 1] | Temperature, threshold | λ_k schedule |
| Cost | — | ~0% | ~0% | ~0% |
| Hyperparameters | — | 1 (α) | 1 (+ temperature) | 1 (+ schedule) |
| Risk | One-layer signal is weak if layer 0 is uninformative | Mask values from different layers have different semantics — blending may be noisy | Most principled — tracks the cause, not the mask | Outer loop — normal path could "pull back" the cd path via shared residuals |

### Recommended Next Steps

1. **Start with Proposal 2 (Score Accumulation)** — it's the most principled (tracks the cause, not the mask), most interpretable (score = "how persistently bad is this head"), and gives the cleanest integration of multi-layer information.

2. **Add a β hyperparameter** to blend between original and new:
   ```python
   mask_final = β × mask_score_based + (1 - β) × mask_layer_0_only
   ```
   This lets you smoothly interpolate between original ONLY and the new method, making ablation clean.

3. **Evaluate the switching regime** — the key metric is: what fraction of TVD values fall below `js_gamma` (complementive regime) vs above (contrastive regime). Ideally, the distribution should straddle the threshold, not saturate on one side.

4. **If divergence is still too high** with any proposal, add a **temperature parameter** to the TVD switching rule:
   ```python
   probs_normal = softmax(next_token_logits / τ)
   probs_cd = softmax(next_token_logits_cd / τ)
   tvd = 0.5 × Σ |probs_normal - probs_cd|
   ```
   Increasing τ smooths both distributions, reducing the raw TVD and making the complementive regime more accessible.
