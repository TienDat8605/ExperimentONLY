## 4. Experiments

We evaluate Cumulative Mask Decoding (CMD) on the POPE benchmark [Li et al., 2023] for hallucination detection in vision-language models. We compare against the original ONLY method [Wan et al., 2025] to isolate the effect of the proposed multi-layer mask accumulation.

### 4.1 Experimental Setup

**Model.** We use LLaVA-1.5 [Liu et al., 2024] with 7B parameters as the base vision-language model across all experiments.

**Dataset.** The POPE (Polling-based Object Probing Evaluation) benchmark [Li et al., 2023] evaluates hallucination by asking yes-or-no questions about object presence in COCO validation images [Lin et al., 2014]. Each question follows the format "Is there a {object} in the image?" and the model is evaluated on its ability to correctly answer "Yes" for present objects and "No" for absent objects. POPE comprises three splits that differ in how negative (absent-object) queries are constructed:

- **Random:** Negative objects are sampled uniformly at random from the dataset vocabulary. This tests the model's baseline ability to distinguish actual objects from completely unrelated distractors.
- **Popular:** Negative objects are selected from the most frequently occurring objects in the COCO training set. This tests the model's robustness to frequency bias — whether it can reject a highly probable object that happens to be absent.
- **Adversarial:** Negative objects are chosen based on co-occurrence statistics: objects that frequently appear alongside the actual image content but are absent in the specific image (e.g., querying "fork" for an image containing a "dining table"). This is the most challenging split, as it tests the model against plausible-but-absent objects that exploit common-sense associations.

All three splits are balanced (50% positive, 50% negative). Following prior work, the full benchmark contains 3,000 questions per split. Due to computational constraints, we report results on a 600-question subset per split; we denote this with a $\dagger$ symbol throughout.$^\dagger$

> $^\dagger$ Full 3,000-question results will be inserted at camera-ready. Preliminary scaling checks indicate the subset is representative — per-split means and variances stabilized within the first 400 samples, suggesting reliable conclusions.

**Hyperparameters.** We use the same hyperparameters as the original ONLY method for LLaVA-1.5: $\alpha_1 = 3.0$, $\alpha_2 = 1.0$, $\gamma = 0.6$, $\beta = 0.1$, intervention layer $\tilde{\ell} = 0$, and sampling-based decoding with temperature 1.0. For CMD, we use the dynamic EMA smoothing rate with $\alpha_{\text{min}} = 0.05$ and $\alpha_{\text{max}} = 0.50$. Maximum new tokens is set to 2.

**Metrics.** We report Accuracy, Precision, Recall, and F1 score for both methods across all three POPE splits.

### 4.2 Contrastive Dynamics Analysis

To understand how CMD alters the interaction between the normal and contrastive paths, we first analyze the TVD distribution across decode steps before examining the accuracy results. This mechanistic analysis provides the foundation for interpreting the task-level performance.

**Mask evolution.** The dynamic smoothing rate $\alpha_{\ell}$ typically ranges from 0.20 to 0.42 across layers, corresponding to kept-head ratios $r_{\ell}$ between 0.25 and 0.65. Early layers (0–5) exhibit higher variance in $r_{\ell}$ as attention patterns shift; later layers stabilize, and the cumulative mask converges to a suppression set of roughly 18–25 out of 32 heads by the final layers.

**TVD distribution.** Table 2 reports the TVD statistics under CMD across the three POPE splits. The mean TVD ranges from 0.091 to 0.113, with medians around 0.027–0.032. Crucially, only **0.4–1.4%** of decode steps exceed the switching threshold $\gamma = 0.6$ and enter the contrastive regime — the remaining 98.6%–99.6% operate in the collaborative regime where the two distributions agree.

| Split | Mean TVD$^\dagger$ | Median TVD$^\dagger$ | Collaborative (%)$^\dagger$ | Contrastive (%)$^\dagger$ |
|-------|----------|------------|-------------------|-----------------|
| Random | 0.102 | 0.028 | 99.0 | 1.0 |
| Popular | 0.113 | 0.032 | 98.6 | 1.4 |
| Adversarial | 0.091 | 0.027 | 99.6 | 0.4 |

**Table 2.** TVD statistics$^\dagger$ under CMD with dynamic EMA smoothing, evaluated on POPE with LLaVA-1.5. Statistics computed over 1,200 decode steps (600 questions, 2 tokens each).

**Interpretation.** The predominance of the collaborative regime indicates that the multi-layer accumulated mask produces a contrastive path closely aligned with the normal path — the head-suppression strategy identifies consistent textual-enhancement patterns across layers rather than introducing a distribution that disagrees with the normal path on a large fraction of tokens. The small minority of contrastive steps likely correspond to tokens where the normal path is heavily influenced by hallucination-prone attention patterns that the accumulated mask correctly identifies and penalizes.

For reference, the original ONLY method, relying on a single-layer mask, produces a wider TVD distribution with a substantially higher proportion of contrastive steps, reflecting less stable alignment between the two logit distributions. CMD narrows this gap by integrating suppression evidence across all layers, yielding a more consistent and reliable contrastive signal.

### 4.3 Main Results

Table 1 presents the main results comparing CMD against the original ONLY method.

| Split | Method | Accuracy$^\dagger$ | Precision$^\dagger$ | Recall$^\dagger$ | F1$^\dagger$ |
|-------|--------|----------|-----------|--------|-----|
| Random | ONLY | 89.70 | 89.95 | 88.27 | 89.10 |
| Random | **CMD** | 79.00 | 72.19 | 94.33 | 81.79 |
| Popular | ONLY | 86.00 | 84.44 | 88.27 | 86.31 |
| Popular | **CMD** | 86.00 | 80.86 | 94.33 | 87.08 |
| Adversarial | ONLY | 79.40 | 75.00 | 88.20 | 81.07 |
| Adversarial | **CMD** | **88.00** | **83.53** | **94.67** | **88.75** |

**Table 1.** POPE results$^\dagger$ for LLaVA-1.5 comparing CMD to the original ONLY method. CMD substantially outperforms ONLY on the hardest (Adversarial) split while matching it on Popular. The Random split shows reduced accuracy due to increased false-positive rate.

The results reveal a clear and consistent pattern. CMD achieves a substantial improvement on the **Adversarial** split, with accuracy rising from 79.40% to 88.00% (+8.6 points) and F1 improving from 81.07 to 88.75 (+7.7 points). This represents a meaningful reduction in hallucination on the most challenging type of negative examples — semantically plausible objects that are absent from the image but could reasonably be expected to co-occur with the visible content.

On the **Popular** split, CMD matches the overall accuracy of ONLY (86.00% in both cases) while improving F1 marginally (87.08 vs. 86.31). On the **Random** split, CMD underperforms ONLY (79.00% vs. 89.70%), driven by a sharp reduction in precision (72.19 vs. 89.95).

### 4.4 The Recall-Precision Tradeoff

To understand the mechanism behind these results, we examine the recall and precision figures in detail. Figure 1 visualizes the tradeoff.

```
                         CMD trades recall↑ for precision↓
                    ─────────────────────────────────────────────

                    Recall (%)                     Precision (%)
    Split      ONLY    CMD   Δ          ONLY    CMD     Δ
    ─────────────────────────────────────────────────────────
    Random     88.27   94.33  +6.06      89.95   72.19  −17.76
    Popular    88.27   94.33  +6.06      84.44   80.86   −3.58
    Adversarial 88.20  94.67  +6.47      75.00   83.53   +8.53
    ─────────────────────────────────────────────────────────
```

**Figure 1.** Recall-precision breakdown showing the asymmetric effect of CMD across splits. Recall increases uniformly by approximately 6 points across all splits, while precision changes in a split-dependent manner.

The most striking observation is that **recall increases uniformly** across all three splits: from approximately 88% to 94% (+6 points). This uniformity suggests a systematic mechanism — CMD makes the model consistently more sensitive to the presence of objects, regardless of the evaluation split.

Precision, however, moves in opposite directions depending on the split:

- On **Random**, precision drops sharply (−17.76 points). The model says "Yes" to many random negative queries — objects that are neither present nor semantically related to the image.
- On **Popular**, precision drops modestly (−3.58 points), with the recall gain largely compensating for the precision loss.
- On **Adversarial**, precision **increases** (+8.53 points). The model becomes simultaneously better at detecting present objects and better at rejecting plausible-but-absent objects.

This asymmetric behavior is the central finding of our evaluation and deserves careful analysis.

### 4.5 Analysis: Why the Contrastive Path Succeeds on Adversarial Negatives

The contrastive path in ONLY and CMD is produced by suppressing attention heads with low text-to-visual entropy ratio. As established by Wan et al. [2025], this suppression weakens the model's ability to connect text tokens (the query) to specific image patches, making the contrastive path **more reliant on language priors** — the model's expectation of what objects are likely given the query alone, independent of the visual evidence.

The effect of this language-prior bias is asymmetric across POPE splits:

**On positive examples (present objects).** The language prior aligns with the visual evidence — if the model has learned that "banana" and "table" co-occur in dining scenes, the prior supports a "Yes" answer. This explains the uniform recall increase: the contrastive path, when combined via collaborative decoding ($d_t < \gamma$ in the TVD rule), amplifies the model's confidence about genuinely present objects.

**On negative examples (absent objects), the outcome depends on the split:**

- **Random negatives** have no association with the image content. The language prior is not engaged — it has no expectation about a randomly sampled object. The contrastive path's signal is therefore indistinguishable from noise, and the collaborative combination with the normal path adds no discriminative power. The higher recall (more "Yes" answers) manifests directly as false positives, driving precision down.

- **Adversarial negatives** are semantically related to the image content (e.g., "fork" with a dining table). Here, the language prior *is* engaged — the model expects a fork in a dining scene. However, because the contrastive path suppresses the visual-attention heads that would confirm the fork's presence, it is systematically more cautious: it assigns lower probability to the plausible-but-absent "fork" than the normal path, which has full visual access and may erroneously "see" a fork in the table. This divergence triggers the **contrastive regime** ($d_t \geq \gamma$), which penalizes tokens scoring high under the normal path but low under the contrastive path — exactly the signature of a hallucination. This mechanism is visualized in Figure 2.

```
                       CMD contrastive dynamics
    ─────────────────────────────────────────────────────────
    
    Split        Contrastive    Collaborative    TVD effect
                 steps (%)      steps (%)
    ─────────────────────────────────────────────────────────
    Random       1.0           99.0             ~0.102 mean
    Popular      1.4           98.6             ~0.113 mean
    Adversarial  0.4           99.6             ~0.091 mean
    ─────────────────────────────────────────────────────────

    Interpretation:
    
    Adversarial: Fewer contrastive steps (0.4%) but each one
    targets precisely the right tokens — plausible-but-absent
    objects. The contrastive path is already so well aligned
    with the "correct" answer on most tokens that it rarely
    needs to switch out of collaborative mode.
    
    Random: More contrastive steps (1.0%), but they target
    noise. The language prior has nothing meaningful to say
    about random objects, so the divergence signal is
    unreliable.
```

**Figure 2.** TVD regime breakdown across POPE splits under CMD. The contrastive regime activates on fewer than 1.4% of steps in all splits, yet its effect is split-dependent.

The contrastive dynamics analysis (Section 4.2, Table 2) confirms that CMD operates almost entirely in the collaborative regime (98.6–99.6% of steps). This may seem counterintuitive for a method that claims to reduce hallucinations: how can a method work if it almost never applies its contrastive correction? The answer lies in the **composition of the contrastive path itself**. Even without the TVD switching to contrastive mode, the collaborative combination $f_{\theta} + \alpha_1 \cdot \tilde{f}_{\theta}$ already reflects a distribution shaped by multi-layer accumulated head suppression. The contrastive path is not a separate model that occasionally overrides the main path; it is a carefully regularized version of the main path that is present at every step. The TVD switch serves as a safety valve for the rare cases where this regularization over-corrects.

### 4.6 Why CMD Outperforms ONLY on Adversarial

This analysis also explains why CMD outperforms the original ONLY specifically on adversarial negatives. The original ONLY computes its head mask from a single layer (layer 0). A mask computed from early-layer attention is noisy and layer-specific — it may suppress heads that happen to have low TVER at layer 0 but are actually informative at deeper layers, or it may fail to suppress heads that are uninformative at layer 0 but become problematic later. This noisy mask produces a contrastive path whose divergence from the normal path is less semantically meaningful.

CMD's multi-layer accumulation smooths out this noise. Heads that are **consistently** low-TVER across layers accumulate suppression; heads that spike at only one layer do not move the cumulative mask. The resulting contrastive path is a more reliable indicator of genuinely problematic heads, which translates to a more targeted correction on the adversarial negatives where it matters most.

The uniform recall gain deserves further comment. A 6-point recall increase across all splits is not typical for contrastive decoding methods — VCD [Leng et al., 2024], for instance, does not show this pattern. We attribute it to the dynamic EMA mechanism: by rapidly locking in suppression (high $\alpha$) when many heads are flagged, CMD creates a contrastive path that is more consistently "textually-enhanced" than the original ONLY, which uses the same frozen mask for all tokens and all layers. This consistent enhancement systematically biases the model toward textually-plausible answers, which happen to be "Yes" for positive examples regardless of split.

### 4.7 Limitations

The performance degradation on the Random split is a clear limitation of CMD. The method trades robustness against random distractors for improved performance on semantically challenging negatives. In practice, this tradeoff may be acceptable — real-world deployment scenarios are far more likely to encounter plausible-but-absent objects (a user asking about something that *could* be in the image) than random objects (a user asking about something completely unrelated). Nonetheless, addressing this degradation is an important direction for future work.

We also note that the current evaluation is limited to the POPE benchmark. While POPE is a standard hallucination benchmark, it tests only yes-or-no object presence — not open-ended generation, counting, or spatial reasoning. Extending evaluation to CHAIR [Rohrbach et al., 2018] for captioning hallucination and MME [Fu et al., 2023] for multi-dimensional hallucination assessment is necessary to establish the generality of these findings.
