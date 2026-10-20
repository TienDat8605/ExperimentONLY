## 3. Method

We present Cumulative Mask Decoding (CMD), a training-free decoding method that mitigates hallucinations in large vision-language models (LVLMs) by progressively accumulating head-suppression evidence across Transformer layers. CMD extends the ONLY framework [Wan et al., 2025] by replacing its static, single-layer head-mask with a dynamically evolving mask computed at every layer via exponential moving average. We first describe the base framework, then detail our contribution.

### 3.1 Preliminaries: The ONLY Framework

ONLY is a training-free, single-forward-pass decoding approach that produces two logit distributions simultaneously — a normal distribution and a "textually-enhanced" contrastive distribution — and combines them via an adaptive switching rule [Wan et al., 2025].

#### 3.1.1 Text-to-Visual Entropy Ratio

At a chosen intervention layer $\tilde{\ell}$ (the authors find layer 0 to be optimal), the attention weights $a_{\tilde{\ell},i}$ for each head $i$ are split into textual and visual subsets by token position:

$$
a_{\tilde{\ell},i}^{\mathcal{T}} = \{a_{\tilde{\ell},i,j} \mid j \in \text{indices}_\mathcal{T}\}, \quad
a_{\tilde{\ell},i}^{\mathcal{V}} = \{a_{\tilde{\ell},i,j} \mid j \in \text{indices}_\mathcal{V}\}
$$

Outlier attention values exceeding $\mu + \sigma$ within each subset are zeroed out and the remaining weights are re-normalized. The Text-to-Visual Entropy Ratio (TVER) for head $i$ is then:

$$
\text{TVER}_{\tilde{\ell},i} = \frac{H(a_{\tilde{\ell},i}^{\mathcal{T}})}{H(a_{\tilde{\ell},i}^{\mathcal{V}})}
$$

where $H(\cdot)$ denotes Shannon entropy. A low TVER indicates a head focused on specific text tokens (low textual entropy) but diffuse over image patches (high visual entropy) — Wan et al. characterize these as "uncertain connectors" that mediate spurious visual-text correlations and drive hallucinations.

#### 3.1.2 Single-Layer Head Suppression and Ghost Path

Heads with TVER below the layer average are suppressed by zeroing their attention weights:

$$
\tilde{a}_{\tilde{\ell},i} = \begin{cases}
a_{\tilde{\ell},i}, & \text{TVER}_{\tilde{\ell},i} \geq \frac{1}{H}\sum_{h=1}^{H} \text{TVER}_{\tilde{\ell},h} \\
0, & \text{otherwise}
\end{cases}
$$

The masked attention output $\texttt{TE-MHA}_{\tilde{\ell}}(\cdot)$ produces the **contrastive hidden state** $h^{\text{cd}}$, which then passes through all subsequent layers $\tilde{\ell}+1, \dots, L-1$ in a "ghost path" — it receives no layer normalization, no residual update, and no MLP computation. At the final layer $L-1$, the ghost path is re-integrated with the normal path:

$$
h^{\text{cd}}_{L-1} = \texttt{LayerNorm}(h^{\text{cd}}_{L-2})
$$
$$
h^{\text{cd}}_{L-1} = 0.2 \cdot r_{L-1} + h^{\text{cd}}_{L-1}
$$
$$
h^{\text{cd}}_{L-1} = \texttt{MLP}(h^{\text{cd}}_{L-1}) + r^{\text{cd}}_{L-1}
$$
$$
\tilde{h}^{\text{cd}} = \texttt{LayerNorm}(h^{\text{cd}}_{L-1}) + 0.5 \cdot \tilde{h}^{\text{normal}}_{L-1}
$$
$$
\text{logits}_{\text{cd}} = \texttt{lm\_head}(\tilde{h}^{\text{cd}})
$$

The residual injection terms ($0.2 \cdot r_{L-1}$ and $0.5 \cdot \tilde{h}^{\text{normal}}_{L-1}$) stabilize the ghost signal by anchoring it to the normal-path representation.

#### 3.1.3 TVD-Based Switching Rule

At each autoregressive step $t$, the Total Variation Distance (TVD) between the two distributions determines the decoding strategy:

$$
d_t = \sum_{y_t \in \mathcal{V}} |\,p_{\theta}(y_t \mid v, x, y_{<t}) - \tilde{p}_{\theta}(y_t \mid v, x, y_{<t})\,|
$$

**Collaborative decoding** ($d_t < \gamma$): the distributions agree, so the textually-enhanced logits amplify the shared signal:

$$
f^{\text{final}} = f_{\theta} + \alpha_1 \cdot \tilde{f}_{\theta}
$$

**Contrastive decoding** ($d_t \geq \gamma$): the distributions diverge, so tokens scoring high under the contrastive path (which is more reliant on language priors due to head suppression) are penalized:

$$
f^{\text{final}} = (1 + \alpha_2) \cdot f_{\theta} - \alpha_2 \cdot \tilde{f}_{\theta}
$$

An adaptive plausibility constraint [Li et al., 2023] with $\beta = 0.1$ discards tokens with near-zero probability under the normal distribution before sampling.

### 3.2 Cumulative Mask Decoding

The ONLY framework computes the TVER-based head mask **exactly once** at the intervention layer $\tilde{\ell}$ and freezes it for all remaining layers. This imposes an information bottleneck: deeper layers, which capture higher-order visual-textual correspondences [Abnar and Zuidema, 2020; Clark et al., 2019], contribute no information to the suppression decision. A head that happens to exhibit a spurious low TVER at layer 0 is suppressed indiscriminately across all subsequent layers, while a head that is genuinely problematic but shows a moderate TVER at layer 0 escapes suppression entirely.

Cumulative Mask Decoding (CMD) addresses this by computing a **fresh head mask at every layer** and blending it into a running cumulative mask via exponential moving average (EMA). This allows the suppression signal to be refined iteratively as the model processes deeper representations, while the EMA mechanism prevents per-layer noise from destabilizing the mask.

#### 3.2.1 Multi-Layer Mask Accumulation

At each layer $\ell$, we compute a fresh binary mask from the TVER at that layer:

$$
m_{\ell,i} = \mathbb{1}\left[\text{TVER}_{\ell,i} \geq \frac{1}{H}\sum_{h=1}^{H} \text{TVER}_{\ell,h}\right]
$$

These per-layer masks are accumulated into a running cumulative mask $M_{\ell}$ via EMA:

$$
M_{\ell} = \alpha_{\ell} \cdot m_{\ell} + (1 - \alpha_{\ell}) \cdot M_{\ell-1}
$$

where $M_{0} = m_{0}$ (the initial mask from layer 0 is identical to the original ONLY mask). The cumulative mask $M_{\ell}$ is then applied to suppress attention heads in the contrastive path at layer $\ell$, in place of the frozen layer-0 mask. All other components of the framework — the ghost path, the final-layer integration, the TVD switching rule, and the plausibility constraint — remain unchanged. Algorithm 1 provides the complete procedure.

```
Algorithm 1: Cumulative Mask Decoding
--------------------------------------------------------------------------------
Input: Layers ℓ = 0, 1, ..., L-1, attention weights {a_ℓ}_{ℓ=0}^{L-1}
Output: Contrastive logits logits_cd at each decode step

 1: M ← ∅
 2: for ℓ = 0 to L-1 do
 3:     Compute TVER_{ℓ,i} for each head i from a_ℓ
 4:     m_ℓ ← 𝟙[TVER_ℓ ≥ mean(TVER_ℓ)]           ▷ Fresh binary mask
 5:     if ℓ = 0 then
 6:         M ← m_ℓ
 7:     else
 8:         r ← (∑ m_ℓ) / H                       ▷ Kept-head ratio
 9:         α ← 0.05 + 0.45 · (1 − r)             ▷ Dynamic rate
10:         M ← α · m_ℓ + (1 − α) · M             ▷ EMA blend
11:     end if
12:     Apply M: zero out heads in contrastive path where M_i < 0.5
13: end for
14: h^{cd} ← ghost path through remaining layers
15: logits_cd ← final-layer integration of h^{cd}
16: Return logits_cd
```

#### 3.2.2 Dynamic Smoothing Rate

The smoothing rate $\alpha_{\ell}$ governs how quickly the cumulative mask adapts to new layer-level evidence. We introduce a **dynamic** smoothing rate that adapts to the suppression statistics at each layer, rather than using a fixed constant:

$$
\alpha_{\ell} = \alpha_{\text{min}} + (\alpha_{\text{max}} - \alpha_{\text{min}}) \cdot (1 - r_{\ell})
$$

where $r_{\ell} = \frac{1}{H} \sum_{h=1}^{H} m_{\ell,h}$ is the fraction of heads retained (not suppressed) by the fresh mask at layer $\ell$, $\alpha_{\text{min}} = 0.05$, and $\alpha_{\text{max}} = 0.50$.

This formulation produces two desirable regimes:

- **When many heads are suppressed** ($r_{\ell} \to 0$): $\alpha_{\ell} \to 0.50$. The cumulative mask updates rapidly, locking in a strong suppression signal. This is the regime corresponding to inputs where a large number of heads exhibit imbalanced text-image attention — i.e., hallucination-prone conditions.
- **When few heads are suppressed** ($r_{\ell} \to 1$): $\alpha_{\ell} \to 0.05$. The cumulative mask changes minimally, preventing spurious single-layer noise from corrupting the mask. This is the regime where most heads distribute their attention reasonably between text and image, and aggressive suppression would harm performance.

A fixed-$\alpha$ variant (e.g., $\alpha = 0.2$) is recovered as a special case and may be used as an ablation baseline.

#### 3.2.3 Computational Cost

The additional computation per layer consists of: (a) a TVER computation that reuses the already-computed attention weight matrix (no extra forward pass), (b) a single element-wise comparison to produce the binary mask, and (c) one tensor multiply-add for the EMA blend. In practice, CMD adds less than 0.5% to the total inference time of the original ONLY method, making it essentially cost-free.

### 3.3 Discussion

The near-saturation of decode steps into the collaborative regime ($>98\%$) suggests that the TVD threshold $\gamma = 0.6$, originally calibrated for the wider TVD distribution produced by a single-layer mask, may be conservative for CMD. A lower threshold (e.g., $\gamma = 0.05$) could restore a more balanced switching distribution, potentially increasing the impact of the contrastive correction on the small fraction of tokens where it matters most. We leave the joint optimization of $\gamma$ and the EMA dynamics to future work.

Additionally, we note that the current formulation blends **binary mask values** across layers. Since the semantic interpretation of a mask value depends on the attention patterns at the layer where it was computed, blending masks directly may introduce noise. A more principled alternative would be to accumulate the **underlying TVER scores** rather than the binarized masks, decoupling the accumulation signal from per-layer thresholding — we explore this direction in Appendix A.
