# Adaptive Dual-Branch Decoding for Hallucination Mitigation in Large Vision--Language Models

> **Draft status.** This manuscript reframes Proposal 6 as the sole proposed method. The Related Work section is intentionally left unwritten. Items enclosed in `[TODO: ...]` require exact values from the final experiment spreadsheet, runtime logs, or the frozen `proposal6_mask.py` implementation.

# Abstract

Large vision--language models (LVLMs) frequently hallucinate objects that are absent from the visual input, limiting their reliability in practical applications. Training-free contrastive decoding can reduce such errors, but many existing methods require multiple model evaluations per decoding step. ONLY mitigates hallucinations within a single forward process by suppressing attention heads selected from a Text-to-Visual Entropy Ratio (TVER) at one intervention layer and propagating the resulting representation through a lightweight ghost path. However, a mask computed at one early layer cannot incorporate deeper cross-modal evidence, and a single masked branch provides only one distorted view of the model's internal behavior.

We introduce **Adaptive Dual-Branch Decoding (ADBD)**, a training-free extension of ONLY that accumulates continuous TVER evidence across selected Transformer layers using head-specific adaptive gains. Rather than committing to a binary decision at one layer, ADBD maintains an evolving state for each head and adjusts the update magnitude according to the confidence and innovation of newly observed evidence. The accumulated state induces two complementary masks, yielding high-TVER and low-TVER auxiliary branches alongside the normal branch. At each autoregressive step, ADBD independently measures the Total Variation Distance (TVD) between each auxiliary distribution and the normal distribution, then adaptively reinforces or penalizes each branch through an anchored three-branch decoding rule.

On POPE with LLaVA-1.5-7B, ADBD obtains the best performance among the evaluated decoding variants. It achieves **[TODO: main Proposal 6 accuracy/F1 results]**, improving the macro-average **[accuracy/F1]** by **[TODO]** over ONLY while preserving a single model invocation per token. On the MME Hallucination subset, ADBD improves Position from 136.66 to 153.33 and Color from 161.66 to 175.00, but decreases Existence and Count; its reported aggregate score is 620.00 versus 635.55 for ONLY. The supplied Proposal 6 category scores sum to 630.00, so the aggregate must be verified before submission. Component ablations show that **[TODO: summarize verified contribution of continuous accumulation, adaptive gains, and dual branches]**. These results indicate that maintaining complementary, layer-refined contrastive views provides a more informative decoding signal than a frozen one-layer mask, while the mixed MME outcome identifies remaining limitations in existence and counting tasks.

# Introduction {#sec:introduction}

Large vision--language models (LVLMs) have demonstrated remarkable capabilities in image captioning, visual question answering, and visual reasoning [@alayrac2022flamingo; @li2023blip2; @liu2024visual]. Despite this progress, they remain susceptible to hallucination: generating objects, attributes, or relations that are unsupported by the visual input [@rohrbach2018object; @li2023pope]. Such errors reduce user trust and restrict deployment in safety-sensitive settings including medical imaging, autonomous navigation, and assistive technologies.

A prominent family of training-free mitigation methods operates during decoding. Contrastive approaches compare the model's standard next-token distribution with a deliberately distorted counterpart and use their disagreement to reduce reliance on unsupported language priors. Visual Contrastive Decoding (VCD) [@leng2024mitigating], for example, constructs the distorted distribution from a perturbed image, while M3ID [@fayyaz2024m3id] alters the multimodal conditioning signal. Although effective, methods that independently evaluate normal and distorted inputs require multiple model passes per generation step, increasing latency and memory consumption.

ONLY [@wan2025only] avoids this duplication by producing a normal distribution and an auxiliary contrastive distribution within one forward process. At a selected intervention layer, ONLY calculates a Text-to-Visual Entropy Ratio (TVER) for every attention head. Heads with low TVER are suppressed to create a textually enhanced hidden representation, which is propagated through the remaining layers using a lightweight ghost path and converted into auxiliary logits. A TVD-based switching rule then either reinforces the auxiliary logits when both paths agree or subtracts them when their distributions diverge.

Despite its efficiency, ONLY makes its head-selection decision from a single layer, typically layer 0. This creates two limitations. First, the decision cannot be revised using deeper representations, even though cross-modal associations evolve throughout the decoder. A head selected from noisy early-layer attention remains selected for the rest of generation, whereas a head that becomes problematic only at a later layer is never captured. Second, a single binary mask collapses a continuous spectrum of text--visual attention behavior into one auxiliary path. It therefore cannot separately represent the complementary behaviors of heads above and below the TVER reference level.

We address these limitations with **Adaptive Dual-Branch Decoding (ADBD)**. ADBD accumulates continuous, layer-specific TVER evidence instead of freezing a binary mask. For each head, the method maintains a running state and updates it with a bounded adaptive gain. The gain is head-specific: confident and informative evidence can revise the state more strongly, while weak or redundant observations produce smaller changes. The resulting continuous state partitions the heads into complementary high-TVER and low-TVER groups. Both groups are retained as separate auxiliary branches, producing three next-token distributions in total: the normal distribution, a high-TVER branch, and a low-TVER branch.

The two auxiliary distributions are not forced to follow the same decoding regime. ADBD independently computes their TVD from the normal distribution. Each branch is reinforced when it remains close to the normal path and penalized when it diverges beyond its branch-specific threshold. Whenever a branch is subtracted, the normal logits receive a corresponding anchor increase, preventing auxiliary subtraction from overwhelming the base model. This construction yields four possible joint regimes and allows the two complementary branches to contribute differently at the same generation step.

Our contributions are as follows:

1. We introduce a **continuous multi-layer TVER state** that preserves the magnitude and direction of head-level evidence instead of immediately reducing every layer to a binary mask.
2. We propose a **head-specific adaptive update mechanism** that controls how strongly new layer evidence changes the accumulated state using bounded gains derived from confidence and innovation.
3. We construct **complementary high-TVER and low-TVER ghost branches** and combine them with the normal distribution through an independently switched, anchored three-branch decoding rule.
4. We evaluate the method on POPE and the MME Hallucination subset using LLaVA-1.5-7B. ADBD is the strongest evaluated configuration on the primary POPE evaluation, while MME provides a complementary diagnosis of category-specific gains and regressions. Component variants are reported only as ablations. **[TODO: insert exact POPE principal improvements.]**

The remainder of this paper is organized as follows. Section [3](#sec:method){reference-type="ref" reference="sec:method"} presents ADBD. Section [4](#sec:experiments){reference-type="ref" reference="sec:experiments"} describes the experimental setup, main results, ablations, and mechanistic analysis. Section [5](#sec:limitations){reference-type="ref" reference="sec:limitations"} discusses limitations.

# Related Work

*To be completed. This section is intentionally omitted from the current rewrite.*

# Method {#sec:method}

We present **Adaptive Dual-Branch Decoding (ADBD)**, a training-free decoding method that augments the normal LVLM path with two complementary auxiliary paths. ADBD consists of four stages: (1) computing layer-specific TVER evidence, (2) accumulating the evidence through per-head adaptive updates, (3) constructing complementary high- and low-TVER ghost branches, and (4) resolving the three distributions with independently switched, anchored decoding.

## Preliminaries: ONLY

ONLY [@wan2025only] produces a normal distribution and one auxiliary contrastive distribution in a single forward process. We briefly review the components retained by ADBD.

### Text-to-Visual Entropy Ratio

At decoding layer $\ell$, let $a_{\ell,h}$ denote the attention weights of head $h$ for the final query position. The attention vector is divided into textual and visual subsets according to token position:

$$
\begin{aligned}
a_{\ell,h}^{\mathcal{T}} &= \{a_{\ell,h,j}\mid j\in\mathcal{I}_{\mathcal{T}}\},\\
a_{\ell,h}^{\mathcal{V}} &= \{a_{\ell,h,j}\mid j\in\mathcal{I}_{\mathcal{V}}\}.
\end{aligned}
$$

Following ONLY, values exceeding the subset mean plus one standard deviation are removed and each subset is renormalized. The textual and visual entropies are

$$
H_{\ell,h}^{\mathcal{T}}
=-\sum_{j\in\mathcal{I}_{\mathcal{T}}}
\hat a_{\ell,h,j}^{\mathcal{T}}
\log\left(\hat a_{\ell,h,j}^{\mathcal{T}}+\epsilon\right),
$$

$$
H_{\ell,h}^{\mathcal{V}}
=-\sum_{j\in\mathcal{I}_{\mathcal{V}}}
\hat a_{\ell,h,j}^{\mathcal{V}}
\log\left(\hat a_{\ell,h,j}^{\mathcal{V}}+\epsilon\right).
$$

The TVER of head $h$ is then

$$
r_{\ell,h}
=
\frac{H_{\ell,h}^{\mathcal{T}}}
     {H_{\ell,h}^{\mathcal{V}}}.
$$

A low ratio indicates concentrated textual attention relative to diffuse visual attention, whereas a high ratio indicates the complementary entropy pattern.

### One-Layer Mask and Ghost Path

ONLY computes a binary mask at one intervention layer $\tilde\ell$ using the mean ratio across heads:

$$
m_{\tilde\ell,h}
=
\mathbb{1}\!\left[
 r_{\tilde\ell,h}\geq
 \frac{1}{H}\sum_{j=1}^{H}r_{\tilde\ell,j}
\right].
$$

The attention outputs of suppressed heads are zeroed to create an auxiliary hidden state. This state is threaded through the remaining decoder layers without the complete normal residual and feed-forward computation, then processed and anchored to the normal representation at the final layer. Applying the language-model head produces the auxiliary logits.

### Single-Branch TVD Switching

Let $f_t^{0}$ and $f_t^{\mathrm{aux}}$ be the normal and auxiliary logits at autoregressive step $t$, with probability distributions $p_t^{0}$ and $p_t^{\mathrm{aux}}$. ONLY measures

$$
d_t
=
\frac{1}{2}\sum_{y\in\mathcal{V}}
\left|p_t^{0}(y)-p_t^{\mathrm{aux}}(y)\right|.
$$

The auxiliary logits are added when $d_t<\gamma$ and subtracted otherwise. A plausibility constraint [@li2023contrastive] removes candidates whose normal logits fall below

$$
\log\beta + \max_{y\in\mathcal{V}} f_t^{0}(y).
$$

## Adaptive Dual-Branch Decoding

### Continuous Layer-Wise TVER Evidence

A binary mask records only whether a head lies above or below a threshold. It discards the distance from that threshold and cannot express uncertainty near the boundary. ADBD therefore converts the raw TVER vector at every selected layer into signed, continuous evidence. Let

$$
\bar r_{\ell}=\frac{1}{H}\sum_{h=1}^{H}r_{\ell,h}
$$

be the layer reference. We define the signed evidence as

$$
e_{\ell,h}=\Phi(r_{\ell,h},\bar r_{\ell}),
$$

where positive and negative values indicate the high- and low-TVER sides of the layer reference, respectively.

> **[TODO-METHOD-1]** Replace $\Phi$ with the exact centering/normalization used in the frozen `proposal6_mask.py` implementation. If the implementation uses direct centering, write $e_{\ell,h}=r_{\ell,h}-\bar r_{\ell}$ and remove this note.

Unlike binary accumulation, $e_{\ell,h}$ retains both the direction and strength of the current layer's evidence.

### Per-Head Adaptive State Update

For each head index $h$, ADBD maintains a continuous state $s_{\ell,h}$. At the first selected layer $\ell_0$, the state is initialized from the current evidence:

$$
s_{\ell_0,h}=e_{\ell_0,h}.
$$

At a later selected layer, the state is updated by

$$
s_{\ell,h}
=
(1-g_{\ell,h})s_{\ell^{-},h}
+g_{\ell,h}e_{\ell,h},
$$

where $\ell^{-}$ is the previous selected layer and $g_{\ell,h}$ is a head-specific adaptive gain. The gain is bounded by

$$
\alpha_{\min}\leq g_{\ell,h}\leq\alpha_{\max}.
$$

The implementation derives $g_{\ell,h}$ from two quantities. **Confidence** measures the strength or reliability of the current signed TVER evidence, while **innovation** measures how much the current observation differs from the accumulated state. In general form,

$$
c_{\ell,h}=\operatorname{Conf}(e_{\ell,h}),
$$

$$
\iota_{\ell,h}
=
\operatorname{Innov}(e_{\ell,h},s_{\ell^{-},h}),
$$

$$
g_{\ell,h}
=
\alpha_{\min}
+
(\alpha_{\max}-\alpha_{\min})
\Psi(c_{\ell,h},\iota_{\ell,h}).
$$

The bounded update prevents abrupt state replacement while allowing informative layer evidence to revise uncertain historical decisions.

> **[TODO-METHOD-2]** Insert the exact definitions of $\operatorname{Conf}$, $\operatorname{Innov}$, and $\Psi$ from `proposal6_mask.py`. These equations are central to the novelty claim and must match the executed code exactly.

The update is performed only at a chosen set of layers $\mathcal{L}_{\mathrm{upd}}$. Non-selected layers reuse the most recent state. In the default implementation, layers 0--30 are eligible for accumulation, while the final decoder layer is reserved for auxiliary-path integration. The final paper must report the exact selected set used for the best experiment.

### Complementary Head Partition

The accumulated state partitions attention heads into two complementary sets:

$$
k_{\ell,h}^{+}=\mathbb{1}[s_{\ell,h}\geq 0],
$$

$$
k_{\ell,h}^{-}=\mathbb{1}[s_{\ell,h}<0]
=1-k_{\ell,h}^{+}.
$$

The $+$ mask retains heads whose accumulated evidence lies on the high-TVER side, while the $-$ mask retains heads on the low-TVER side. We use the descriptive names **high-TVER branch** and **low-TVER branch** throughout the paper. The implementation may refer to them as visual and textual branches, respectively; however, the TVER-based names avoid assuming a semantic specialization that has not yet been independently established.

For branch $b\in\{+,-\}$, masked attention is

$$
\tilde a_{\ell,h}^{b}
=
\begin{cases}
a_{\ell,h}, & k_{\ell,h}^{b}=1,\\
0, & k_{\ell,h}^{b}=0.
\end{cases}
$$

This produces two auxiliary attention outputs at each relevant layer.

### Dual Ghost Paths

Let $h_{\ell}^{0}$ denote the normal hidden representation and $h_{\ell}^{+},h_{\ell}^{-}$ denote the two auxiliary representations. Both auxiliary states are propagated through the decoder's ghost-path mechanism. At the final layer, each branch is normalized, receives a residual anchor from the normal path, passes through the final MLP, and is again anchored to the normal final representation before the language-model head. The resulting logits are

$$
f_t^{0},\qquad f_t^{+},\qquad f_t^{-}.
$$

Both auxiliary branches share the same integration structure; their distinction arises solely from complementary head masks.

### Independent Branch Disagreement

ADBD calculates branch-specific TVDs from the normal distribution:

$$
d_t^{+}
=
\frac{1}{2}\sum_{y\in\mathcal{V}}
\left|p_t^{0}(y)-p_t^{+}(y)\right|,
$$

$$
d_t^{-}
=
\frac{1}{2}\sum_{y\in\mathcal{V}}
\left|p_t^{0}(y)-p_t^{-}(y)\right|.
$$

Each distance is compared with its own threshold $\gamma_{+}$ or $\gamma_{-}$. Consequently, one branch can be collaborative while the other is contrastive at the same generation step.

### Anchored Three-Branch Resolution

The contribution of the high-TVER branch is

$$
\Delta_t^{+}
=
\begin{cases}
\alpha_{+}^{\mathrm{col}}f_t^{+},
& d_t^{+}<\gamma_{+},\\
-\alpha_{+}^{\mathrm{con}}f_t^{+},
& d_t^{+}\geq\gamma_{+}.
\end{cases}
$$

Similarly, the low-TVER branch contributes

$$
\Delta_t^{-}
=
\begin{cases}
\alpha_{-}^{\mathrm{col}}f_t^{-},
& d_t^{-}<\gamma_{-},\\
-\alpha_{-}^{\mathrm{con}}f_t^{-},
& d_t^{-}\geq\gamma_{-}.
\end{cases}
$$

To stabilize subtraction, ADBD increases the normal-logit anchor for every branch that enters the contrastive regime:

$$
\kappa_t
=
1
+
\alpha_{+}^{\mathrm{con}}
\mathbb{1}[d_t^{+}\geq\gamma_{+}]
+
\alpha_{-}^{\mathrm{con}}
\mathbb{1}[d_t^{-}\geq\gamma_{-}].
$$

The final logits are

$$
f_t^{\mathrm{final}}
=
\kappa_t f_t^{0}
+
\Delta_t^{+}
+
\Delta_t^{-}.
$$

This rule yields four joint regimes:

| High-TVER branch | Low-TVER branch | Final behavior |
|---|---|---|
| Collaborative | Collaborative | Reinforce both auxiliary distributions |
| Collaborative | Contrastive | Reinforce the high-TVER branch and penalize the low-TVER branch |
| Contrastive | Collaborative | Penalize the high-TVER branch and reinforce the low-TVER branch |
| Contrastive | Contrastive | Penalize both branches and strengthen the normal anchor twice |

The adaptive plausibility constraint is applied using the normal logits before sampling.

### Algorithm

**Algorithm 1: Adaptive Dual-Branch Decoding**

**Input:** decoder layers $\{0,\ldots,L-1\}$, selected update layers $\mathcal{L}_{\mathrm{upd}}$, gain bounds $\alpha_{\min},\alpha_{\max}$, branch thresholds $\gamma_{+},\gamma_{-}$, and branch weights.

**Output:** final next-token logits $f_t^{\mathrm{final}}$.

1. Initialize the continuous head state $s\leftarrow\emptyset$.
2. For each decoder layer $\ell$:
   1. Compute the normal self-attention output.
   2. If $\ell\in\mathcal{L}_{\mathrm{upd}}$:
      1. Compute TVER $r_{\ell,h}$ for each head.
      2. Convert TVER to continuous signed evidence $e_{\ell,h}$.
      3. If $s$ is empty, initialize $s_h\leftarrow e_{\ell,h}$.
      4. Otherwise, compute confidence, innovation, and per-head gain $g_{\ell,h}$.
      5. Update $s_h\leftarrow(1-g_{\ell,h})s_h+g_{\ell,h}e_{\ell,h}$.
   3. Construct complementary masks $k^{+}$ and $k^{-}$ from the sign of $s$.
   4. Produce high- and low-TVER masked attention outputs and propagate their ghost states.
3. At the final layer, integrate both ghost states and obtain $f_t^{+}$ and $f_t^{-}$ alongside $f_t^{0}$.
4. Compute $d_t^{+}$ and $d_t^{-}$.
5. Independently select the collaborative or contrastive contribution of each branch.
6. Form the anchored three-branch logits $f_t^{\mathrm{final}}$.
7. Apply the adaptive plausibility constraint and sample the next token.

### Computational Characteristics

ADBD is training-free and does not require another image encoding or an independent second invocation of the full LVLM. It reuses the normal attention tensors to construct two masked attention outputs and propagates two lightweight auxiliary states. Relative to ONLY, the method adds a second auxiliary branch, per-layer TVER-state updates on selected layers, and one additional language-model-head projection. These operations increase arithmetic and memory use, so efficiency must be established empirically rather than described as negligible.

We report wall-clock latency, tokens per second, and peak GPU memory in Section [4.6](#subsec:efficiency){reference-type="ref" reference="subsec:efficiency"}.

# Experiments {#sec:experiments}

We evaluate ADBD on two hallucination benchmarks: POPE [@li2023pope] and the hallucination subset of MME [@fu2023mme]. POPE is the primary benchmark used for method selection and detailed decoding analysis, while MME tests whether the method generalizes beyond binary object-presence questions to existence, counting, position, and color perception. The main comparison includes the base LLaVA decoding procedure, ONLY, and the full ADBD method. Earlier proposal variants are not presented as competing methods; instead, they are reorganized as controlled component ablations in Section [4.5](#subsec:ablations){reference-type="ref" reference="subsec:ablations"}.

## Experimental Setup

### Model

We use LLaVA-1.5-7B [@liu2024visual] as the base LVLM in all experiments. The visual encoder, multimodal projector, language-model weights, prompt template, and generation settings are held constant across methods.

### Benchmarks

#### POPE

POPE (Polling-based Object Probing Evaluation) [@li2023pope] evaluates object hallucination using balanced yes--no questions derived from COCO validation images [@lin2014microsoft]. Each question asks whether a named object is present in the image. POPE contains three negative-sampling settings:

- **Random:** absent objects are sampled uniformly from the object vocabulary.
- **Popular:** absent objects are sampled from frequently occurring object categories, testing frequency bias.
- **Adversarial:** absent objects are selected using co-occurrence statistics, producing semantically plausible but unsupported queries.

Each split contains 3,000 questions with equal positive and negative counts.

> **[TODO-EXP-1]** State whether the reported Proposal 6 results use all 3,000 questions per split or a fixed subset. If a subset is used, report the exact selection rule, sample count, and whether it is identical for every baseline and ablation.

#### MME Hallucination Subset

MME [@fu2023mme] evaluates multimodal perception and cognition through task-specific yes--no questions. We use the hallucination-related perception subset comprising **Existence**, **Count**, **Position**, and **Color**. Existence tests whether the model recognizes whether an object is present; Count tests numerical object understanding; Position tests spatial relationships; and Color tests visual attribute grounding. Each category is scored using the official MME protocol, and the category scores are aggregated into an overall MME Hallucination score.

> **[TODO-EXP-MME]** Add the exact MME evaluation script or commit, number of evaluated images/questions per category, prompt template, answer parser, and whether decoding is deterministic. Verify the aggregate Proposal 6 score: the supplied category scores sum to 630.00, whereas the reported aggregate is 620.00.

### Baselines and Ablations

The primary baselines are:

1. **LLaVA:** standard decoding without contrastive intervention.
2. **ONLY:** the original one-layer, single-branch intervention [@wan2025only].
3. **ADBD:** the complete proposed method.

The ablation study evaluates intermediate component combinations:

- Binary multi-layer accumulation.
- Continuous score accumulation without adaptive gains.
- Static complementary dual branches.
- Dual branches with a shared/global EMA update.
- Full continuous dual branches with per-head adaptive gains.

This organization allows the main narrative to focus on ADBD while using earlier proposals only to identify which components cause the final improvement.

### Implementation Details

Unless otherwise specified, ADBD uses gain bounds $\alpha_{\min}=0.05$ and $\alpha_{\max}=0.50$. The implementation supports selected accumulation layers from 0 to 30; layer 31 is reserved for final auxiliary integration. The default TVD threshold is 0.6. When branch-specific coefficients are not overridden, the implementation falls back to collaborative weights of 3.0 and contrastive weights of 1.0 for each branch.

> **[TODO-EXP-2]** Replace this paragraph with the exact command and configuration that produced the best spreadsheet result: selected layers, $\alpha_{\min}$, $\alpha_{\max}$, $\gamma_{+}$, $\gamma_{-}$, all four branch coefficients, $\beta$, maximum generated tokens, temperature, top-$p$, seed, and hardware.

The current implementation identifies textual tokens as positions 1--34 and visual tokens as positions 35--610, corresponding to 576 image tokens. This fixed indexing matches the evaluated LLaVA prompt configuration but should be replaced by processor-derived token boundaries in a general implementation.

### Metrics

For POPE, we report Accuracy, Precision, Recall, and F1 for each split. Because the three splits emphasize different hallucination conditions, we additionally report macro-average Accuracy and macro-average F1 across Random, Popular, and Adversarial. The ratio of generated "Yes" responses is included to expose changes in response bias.

For MME Hallucination, we report the official category scores for Existence, Count, Position, and Color, together with their aggregate score. We also report category-wise differences from ONLY because the aggregate can hide opposing changes across visual competencies.

For stochastic decoding, results should be reported as mean and standard deviation over multiple fixed seeds. If only one seed is used, we explicitly state this limitation.

## POPE Main Results

Table [1](#tab:main_results){reference-type="ref" reference="tab:main_results"} compares ADBD with the standard model and ONLY. ADBD is the best-performing evaluated decoding method under the selected primary metric.

<table id="tab:main_results">
<caption>Main POPE results. Replace all bracketed entries with values from the experiment spreadsheet. Macro values are computed across the three splits.</caption>
<thead>
<tr><th>Method</th><th>Random Acc.</th><th>Popular Acc.</th><th>Adversarial Acc.</th><th>Macro Acc.</th><th>Macro F1</th></tr>
</thead>
<tbody>
<tr><td>LLaVA</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ONLY</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td><strong>ADBD</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td></tr>
</tbody>
</table>

ADBD improves **[TODO: primary metric]** from **[ONLY value]** to **[ADBD value]**, a gain of **[difference]** points. The improvement is **[TODO: broadly distributed across splits / concentrated on specific splits]**. On Random, ADBD **[TODO]**; on Popular, it **[TODO]**; and on Adversarial, it **[TODO]**. This pattern indicates that **[TODO: evidence-supported interpretation only]**.

Table [2](#tab:detailed_results){reference-type="ref" reference="tab:detailed_results"} provides the full precision--recall breakdown.

<table id="tab:detailed_results">
<caption>Detailed POPE results for the primary baselines and ADBD.</caption>
<thead>
<tr><th>Split</th><th>Method</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Yes ratio</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Random</td><td>LLaVA</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ONLY</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ADBD</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td rowspan="3">Popular</td><td>LLaVA</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ONLY</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ADBD</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td rowspan="3">Adversarial</td><td>LLaVA</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ONLY</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ADBD</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
</tbody>
</table>

The precision, recall, and Yes-ratio results are necessary to distinguish genuine hallucination reduction from a global shift toward either positive or negative answers. **[TODO: write a split-specific interpretation after inserting the table. Do not reuse the Proposal 1 explanation unless Proposal 6 exhibits the same measured behavior.]**


## MME Hallucination Results

Table [3](#tab:mme_hallucination){reference-type="ref" reference="tab:mme_hallucination"} reports performance on the four hallucination-related MME perception categories.

<table id="tab:mme_hallucination">
<caption>MME Hallucination results. Higher is better. The Proposal 6 aggregate of 620.00 is reproduced as supplied; however, its displayed category scores sum to 630.00 and must be verified.</caption>
<thead>
<tr><th>Method</th><th>Existence</th><th>Count</th><th>Position</th><th>Color</th><th>MME score</th></tr>
</thead>
<tbody>
<tr><td>ONLY</td><td><strong>191.67</strong></td><td><strong>145.55</strong></td><td>136.66</td><td>161.66</td><td><strong>635.55</strong></td></tr>
<tr><td>ADBD (Proposal 6)</td><td>180.00</td><td>121.67</td><td><strong>153.33</strong></td><td><strong>175.00</strong></td><td>620.00<sup>†</sup></td></tr>
<tr><td>Difference</td><td>−11.67</td><td>−23.88</td><td>+16.67</td><td>+13.34</td><td>−15.55<sup>†</sup></td></tr>
</tbody>
</table>

<sup>†</sup> The supplied Proposal 6 category scores sum to 630.00, which would imply an aggregate difference of −5.55 rather than −15.55. The experiment log or evaluation output should be checked before the final manuscript is compiled.

ADBD exhibits a clear category-dependent tradeoff on MME. It improves **Position** by 16.67 points and **Color** by 13.34 points, suggesting that the dual-branch mechanism better preserves some spatial and attribute-level visual evidence. In contrast, **Existence** decreases by 11.67 points and **Count** decreases by 23.88 points. Consequently, Proposal 6 does not outperform ONLY on the reported aggregate MME Hallucination score. This result narrows the paper's claim: ADBD is the strongest configuration on the primary POPE evaluation, but its benefit does not transfer uniformly across hallucination categories.

The particularly large Count regression suggests that the current head partition and branch-resolution rule may favor coarse spatial or attribute cues over exact cardinality evidence. This interpretation is a hypothesis rather than a demonstrated mechanism; it should be tested through branch-removal ablations and per-category decoding diagnostics. Similarly, the Existence decline should be compared against POPE response bias to determine whether it arises from a general shift in yes--no calibration or from benchmark-specific visual evidence.

## Dual-Branch Decoding Dynamics

A single-branch analysis is insufficient for ADBD because the two auxiliary paths can enter different regimes. We therefore analyze $d_t^{+}$ and $d_t^{-}$ separately and count all four joint regimes.

<table id="tab:tvd_stats">
<caption>Branch-specific TVD statistics under ADBD.</caption>
<thead>
<tr><th>Split</th><th>Mean $d^{+}$</th><th>Median $d^{+}$</th><th>Mean $d^{-}$</th><th>Median $d^{-}$</th></tr>
</thead>
<tbody>
<tr><td>Random</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Popular</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Adversarial</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
</tbody>
</table>

<table id="tab:regime_stats">
<caption>Percentage of decoding steps in each joint branch regime. C denotes collaborative and T denotes contrastive.</caption>
<thead>
<tr><th>Split</th><th>C/C</th><th>C/T</th><th>T/C</th><th>T/T</th></tr>
</thead>
<tbody>
<tr><td>Random</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Popular</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Adversarial</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
</tbody>
</table>

The joint-regime distribution reveals whether the branches provide redundant or complementary corrections. A high C/C rate would indicate that both branches mainly reinforce distributions close to the normal path. Frequent mixed regimes would provide stronger evidence that the branches capture different failure modes. **[TODO: insert the observed interpretation.]**

## Head-State and Gain Analysis

To verify that adaptive accumulation is functioning as intended, we analyze the continuous state and per-head gains across layers. We report:

- the mean and variance of $s_{\ell,h}$ by layer;
- the number of heads assigned to each complementary branch;
- the fraction of heads whose branch assignment changes between selected layers;
- the mean, minimum, and maximum adaptive gain;
- the relationship between confidence, innovation, and gain;
- differences between correct and incorrect examples.

<figure id="fig:state_evolution" data-latex-placement="t">
<figcaption>Evolution of the continuous TVER state and branch assignment across selected decoder layers. [TODO: generate from Proposal 6 debug logs.]</figcaption>
</figure>

<figure id="fig:gain_evolution" data-latex-placement="t">
<figcaption>Per-head adaptive gain across layers. [TODO: show mean with dispersion and relate the gain to confidence and innovation.]</figcaption>
</figure>

A useful adaptive mechanism should not collapse to a constant gain. **[TODO: report the observed gain range and whether high-innovation observations receive larger or smaller updates under the exact implementation.]**

## Ablation Study {#subsec:ablations}

We use earlier proposal variants only as ablations that remove or simplify components of ADBD. The study is organized by mechanism rather than proposal number.

<table id="tab:ablation_design">
<caption>Component structure of the ablation variants.</caption>
<thead>
<tr><th>Variant</th><th>Multi-layer evidence</th><th>Continuous state</th><th>Adaptive per-head gain</th><th>Complementary branches</th><th>Independent switching</th></tr>
</thead>
<tbody>
<tr><td>ONLY</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td></tr>
<tr><td>Binary accumulation</td><td>Yes</td><td>No</td><td>No</td><td>No</td><td>No</td></tr>
<tr><td>Continuous accumulation</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr>
<tr><td>Static dual branch</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td></tr>
<tr><td>Global-EMA dual branch</td><td>Yes</td><td>No or binary</td><td>No</td><td>Yes</td><td>Yes</td></tr>
<tr><td><strong>ADBD</strong></td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
</tbody>
</table>

<table id="tab:ablation_results">
<caption>Ablation results. Use macro metrics as the primary summary and retain per-split values in the supplement or an expanded table.</caption>
<thead>
<tr><th>Variant</th><th>Random Acc.</th><th>Popular Acc.</th><th>Adversarial Acc.</th><th>Macro Acc.</th><th>Macro F1</th></tr>
</thead>
<tbody>
<tr><td>ONLY</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Binary accumulation</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Continuous accumulation</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Static dual branch</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Global-EMA dual branch</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td><strong>ADBD</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td><td><strong>[TODO]</strong></td></tr>
</tbody>
</table>

The ablation analysis should answer three questions directly:

1. **Does continuous evidence outperform binary accumulation?** Compare continuous accumulation with binary accumulation under otherwise similar decoding.
2. **Do complementary branches outperform a single branch?** Compare the static or accumulated dual-branch variants with their single-branch counterparts.
3. **Do per-head gains improve over a shared update rate?** Compare ADBD with the global-EMA dual-branch variant.

**[TODO: write one evidence-based paragraph per question after importing the spreadsheet values.]**

### Branch Removal

To isolate the role of each complementary path, we additionally evaluate:

- normal plus high-TVER branch only;
- normal plus low-TVER branch only;
- both branches without independent switching;
- full ADBD.

<table id="tab:branch_ablation">
<caption>Contribution of each auxiliary branch.</caption>
<thead>
<tr><th>Configuration</th><th>Macro Accuracy</th><th>Macro F1</th><th>Random</th><th>Popular</th><th>Adversarial</th></tr>
</thead>
<tbody>
<tr><td>High-TVER only</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Low-TVER only</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Dual branch, shared switching</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>Full ADBD</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
</tbody>
</table>

### Layer Selection and Gain Bounds

We evaluate sensitivity to the update-layer set $\mathcal{L}_{\mathrm{upd}}$ and to $\alpha_{\min},\alpha_{\max}$. At minimum, the final study should compare early-only, late-only, sparse, and all eligible layers, together with several gain ranges around the selected configuration.

<figure id="fig:sensitivity" data-latex-placement="t">
<figcaption>Sensitivity of macro POPE performance to accumulation layers and adaptive-gain bounds. [TODO: generate after completing the grid.]</figcaption>
</figure>

## Efficiency {#subsec:efficiency}

We measure decoding efficiency on the same GPU, software environment, batch size, prompt set, and generation length for all methods.

<table id="tab:efficiency">
<caption>Inference efficiency relative to standard LLaVA and ONLY.</caption>
<thead>
<tr><th>Method</th><th>Time/question</th><th>Tokens/s</th><th>Peak memory</th><th>Relative slowdown</th></tr>
</thead>
<tbody>
<tr><td>LLaVA</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>1.00$\times$</td></tr>
<tr><td>ONLY</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
<tr><td>ADBD</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td><td>[TODO]</td></tr>
</tbody>
</table>

ADBD avoids an additional independent LVLM invocation, but its second ghost branch is not free. The final text should distinguish **single model invocation** from **zero overhead** and report the measured trade-off accurately.

## Qualitative Analysis

We select examples in which:

1. ADBD corrects an ONLY hallucination;
2. the high-TVER branch provides the decisive correction;
3. the low-TVER branch provides the decisive correction;
4. the two branches enter different regimes;
5. ADBD introduces an error that ONLY avoids.

For each example, we report the image, question, ground-truth answer, predictions, branch TVDs, joint regime, and the top candidate probabilities before and after three-branch resolution. Such examples can support the proposed mechanism more directly than aggregate TVD statistics alone.

# Limitations {#sec:limitations}

ADBD has several limitations. First, evaluation on POPE and the MME Hallucination subset shows that the method does not improve every visual competency uniformly. Although Proposal 6 improves MME Position and Color, it underperforms ONLY on Existence, Count, and the reported aggregate score. The Count regression is particularly substantial. Moreover, POPE and MME still rely primarily on constrained yes--no evaluation and do not establish generalization to open-ended captioning. Evaluation on CHAIR [@rohrbach2018object] and additional contemporary hallucination benchmarks is therefore needed.

Second, the current implementation uses fixed textual and visual token boundaries tailored to the evaluated LLaVA-1.5 prompt and its 576 image tokens. This assumption may fail under a different conversation template, visual resolution, processor, or multi-image input. A robust implementation should obtain modality boundaries from the tokenizer and multimodal processor.

Third, accumulating a state at the same head index across layers implicitly treats those coordinates as comparable. Attention head $h$ in one layer is not guaranteed to have the same function as head $h$ in another. ADBD should therefore be interpreted as accumulating evidence over aligned mask coordinates rather than tracking a semantically identical head through depth. Future work could instead accumulate layer-specific statistics, learn cross-layer correspondences, or aggregate at the level of attention functions rather than raw indices.

Fourth, ADBD introduces additional computation and memory relative to ONLY because it maintains two auxiliary states. The method remains training-free and avoids a separate full-model pass, but its practical efficiency depends on the measured overhead.

Fifth, the current method derives modality evidence from TVER and a layer-wise reference. The branch names do not by themselves prove that one branch is intrinsically visual and the other intrinsically textual. Attention visualization, controlled modality ablations, and branch-removal experiments are required before making stronger semantic claims.

Finally, if results are reported from a subset or a single stochastic seed, the statistical strength of the conclusions is limited. The final evaluation should use the complete benchmark or a pre-specified sampling protocol and report variability across seeds where sampling-based decoding is used.

# Conclusion

We introduced Adaptive Dual-Branch Decoding, a training-free method for reducing hallucinations in LVLMs. ADBD replaces ONLY's frozen one-layer binary decision with a continuous TVER state refined across selected decoder layers by head-specific adaptive gains. The state produces complementary high- and low-TVER auxiliary branches, which are independently reinforced or penalized according to their disagreement with the normal distribution. An anchored three-branch rule combines these signals without requiring a separate full LVLM invocation.

On POPE with LLaVA-1.5-7B, ADBD achieves the strongest performance among the evaluated decoding configurations. **[TODO: insert exact macro and split-specific improvements.]** On MME Hallucination, the method improves Position and Color but decreases Existence and Count, yielding a reported aggregate score of 620.00 compared with 635.55 for ONLY; the aggregate must be verified because the supplied Proposal 6 category scores sum to 630.00. Ablations attribute the POPE improvement to **[TODO: verified component findings]**, while branch-level analysis shows **[TODO: verified decoding-dynamics finding]**. Overall, the results suggest that complementary and adaptively accumulated attention evidence can provide a more informative contrastive signal than a static single-layer mask, but its gains are category-dependent rather than universal.

::: thebibliography
9

Zifu Wan, Ce Zhang, Silong Yong, Martin Q. Ma, Simon Stepputtis,
Louis-Philippe Morency, Deva Ramanan, Katia Sycara, and Yaqi Xie. ONLY:
One-Layer Intervention Sufficiently Mitigates Hallucinations in Large
Vision-Language Models. In *Proceedings of the IEEE/CVF International
Conference on Computer Vision*, 2025.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner,
Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive
Decoding: Open-ended Text Generation as Optimization. In *Proceedings of
the 61st Annual Meeting of the Association for Computational
Linguistics*, 2023.

Samira Abnar and Willem Zuidema. Quantifying Attention Flow in
Transformers. In *Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics*, pages 4190--4197, 2020.

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning.
What Does BERT Look at? An Analysis of BERT's Attention. In *Proceedings
of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural
Networks for NLP*, pages 276--286, 2019.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen.
Evaluating Object Hallucination in Large Vision-Language Models. In
*Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing*, pages 292--305, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved
Baselines with Visual Instruction Tuning. In *Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages
26296--26306, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona,
Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO:
Common Objects in Context. In *Proceedings of the European Conference on
Computer Vision*, pages 740--755, 2014.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan
Miao, and Lidong Bing. Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive Decoding. In
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition*, 2024.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and
Kate Saenko. Object Hallucination in Image Captioning. In *Proceedings
of the 2018 Conference on Empirical Methods in Natural Language
Processing*, pages 4035--4045, 2018.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu
Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and
Rongrong Ji. MME: A Comprehensive Evaluation Benchmark for Multimodal
Large Language Models. arXiv preprint arXiv:2306.13394, 2023.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain
Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican,
Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han,
Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian
Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikołaj
Bińkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen
Simonyan. Flamingo: A Visual Language Model for Few-Shot Learning. In
*Advances in Neural Information Processing Systems*, 2022.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2:
Bootstrapping Language-Image Pre-training with Frozen Image Encoders and
Large Language Models. In *Proceedings of the International Conference
on Machine Learning*, 2023.

Mohammad Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei, Hamed
Pirsiavash, and Juergen Gall. M3ID: Multi-modal Mutual-Information
Decoding for Reducing Hallucinations in Vision-Language Models. arXiv
preprint arXiv:2404.14472, 2024.
:::

<!-- Draft completion note: replace all TODO markers before submission. The exact adaptive-gain equations must be copied from the frozen proposal6_mask.py implementation rather than reconstructed from logs or prose. -->
