# Tabular deep learning 2024-2026: a realistic path to beating a GBM+GAM ensemble on ~40k rows?

## What is the 2024-2026 state of the art in tabular deep learning, and how does it compare with GBMs on recent public benchmarks?

### Takeaway
The most recent living benchmarks (2024-2026) conclude that there is **no universal winner** among boosted trees (GBDT), neural networks and tabular foundation models: GBDTs remain very strong in practice, deep learning has caught up mainly when given a large compute budget and configuration ensembling, and foundation models (TabPFN) excel specifically on small datasets.

### Cited Findings
- TabArena (Erickson, Purucker, Tschalzev, Holzmüller, Desai, Salinas, Hutter — arXiv:2506.16791, 2025), a living benchmark with 51 curated datasets (from a pool of 1053 across 14 earlier benchmarks) and 16 model families: "gradient-boosted trees remain strong on practical tabular datasets. Deep learning has caught up with larger time budgets, notably through ensembling of hyperparameter configurations"; "the validation protocol and post-hoc ensembling change the rankings more than the choice of architecture"; foundation models "excel on small datasets" — [TabArena (arXiv 2506.16791)](https://arxiv.org/abs/2506.16791)
- "Tabular Data: Is Deep Learning all you need?" (Zabërgja, Kadra, Frey, Grabocka — arXiv:2402.03970, posted 2024, last revised Oct. 2025), 68 datasets, 17 methods tested: reverses the conclusion of the 2021 paper ("Deep Learning is Not All You Need") and claims that deep learning outperforms classical approaches **provided there is extensive hyperparameter optimization and a crucial "refitting" step**, with meta-learned foundation models achieving the best results — [arXiv:2402.03970](https://arxiv.org/pdf/2402.03970)
- OmniTabBench (2026), the largest tabular benchmark to date with 3030 datasets, with analysis by metadata (size, variable types, skewness/kurtosis): finds "the absence of a dominant winner" among tree ensembles, neural networks and foundation models — [alphaxiv 2604.06814](https://www.alphaxiv.org/abs/2604.06814v1)
- "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" (McElfresh et al., NeurIPS 2023, arXiv:2305.02997 — predates 2024 but still the reference cited in 2024-2026 work): across 176 datasets, the GBDT/NN performance difference is often negligible or dominated by light hyperparameter tuning of the GBDT; GBDTs handle irregular feature distributions (skewed, heavy-tailed) much better; GBDTs perform comparatively better than NNs **as dataset size increases** — [arXiv:2305.02997](https://arxiv.org/pdf/2305.02997)
- XTab (2023, 84 AutoML Benchmark datasets): CatBoost ranks first across all methods, XGBoost and LightGBM rank around 5th on average — a reminder that differences exist even among GBDTs, and that the tabular transformers of that time already did not dominate — [arXiv:2305.06090](https://arxiv.org/pdf/2305.06090)
- 2025 cardiovascular study (medRxiv, non-peer-reviewed preprint) comparing FT-Transformer, SAINT, TabNet, TabTransformer vs. classical ML on 3 CVD datasets, one of them imbalanced (Framingham, with SMOTETomek): "transformer models show strong potential on structured health data but remain sensitive to imbalance, where conventional ML retains an advantage" — [medRxiv 2025.08.03.25332878](https://www.medrxiv.org/content/10.1101/2025.08.03.25332878.full.pdf)

### Inferences
- The 2024-2026 literature is much less clear-cut than in 2021 ("GBM > DL everywhere"), but deep learning's gains almost systematically come from a tuning and ensembling effort far heavier than what a well-tuned GBM requires, not from a clear intrinsic architectural advantage.
- None of the sources shows a consistent, large gap in favor of DL at medium size (tens of thousands of rows); the consensus is rather "parity with a lot of effort" or "a DL advantage concentrated on very small datasets (foundation models) or on very large volumes/settings with complex interactions".

### Gaps
- No source provides a single aggregated figure "DL beats GBM in X% of cases" specifically for the 30-50k row range with an 85/15 imbalanced binary target — this is a gap that cuts across all the questions below.

## At what size/complexity does deep learning actually outperform a well-tuned GBM, for datasets of tens of thousands of rows with an imbalanced binary target?

### Takeaway
No source sets a precise numerical crossover threshold, but all sources converge on a trend: deep learning's advantage grows with **size and above all with the amount of compute/tuning invested**, while foundation models (TabPFN) have the opposite advantage — they excel below ~10,000 rows and lose their documented advantage beyond that. A 40,000-row dataset falls into an intermediate zone poorly covered by the literature, where no source claims a clear advantage for generic DL.

### Cited Findings
- TabPFN v2 (Hollmann et al., Nature 2025): Prior Labs' official documentation explicitly recommends TabPFNv2 for "datasets of up to 10,000 samples and 500 features" — [docs.priorlabs.ai, TabPFN-2](https://priorlabs.ai/tabpfn-2)
- The R package `tabpfn` applies by default a training limit (`training_set_limit`) of 10,000 rows, which can only be disabled explicitly — a sign that even software integrations treat 10k as a hard default limit — [CRAN tabpfn](https://cloud.r-project.org/web/packages/tabpfn/refman/tabpfn.html)
- TabPFN-2.5 "Scaling Mode" (Prior Labs, technical report, Nov. 2025): lifts the fixed row limit and was tested between 1M and 10M rows against CatBoost/XGBoost/LightGBM; the report's explicit conclusion: **"there is no evidence that the gap with gradient boosting narrows as scale increases"** — [Prior Labs, Large Data Model](https://priorlabs.ai/technical-reports/large-data-model)
- McElfresh et al. (NeurIPS 2023): GBDTs "perform comparatively better than NNs as dataset size increases"; TabPFN (original version, limited to ~3000 training rows) is the only notable exception, performing best on average even when subsampling to 3000 points — [arXiv:2305.02997](https://arxiv.org/pdf/2305.02997)
- Systematic review of GAMs vs. neural networks (Doohan, Kook, Burke — 143 articles, 430 datasets, Expert Systems with Applications, to appear 2026): "Large" category defined from 2800 observations; in this category NNs show only a "slight tendency" to perform better, and this effect becomes **non-significant in recent years** — [arXiv:2510.24601](https://arxiv.org/html/2510.24601v2)
- Practical guide oriented toward Kaggle competitions: "gradient boosting solutions still clearly dominate tabular competitions, with neural networks able to add signal in an ensemble"; GBM+NN blending works well because they capture different signals — [Notion, Neural networks for tabular competitions](https://decisive-willow-f4d.notion.site/Neural-networks-for-tabular-competitions-a72d7faa776f4e85ac1978478f120213)

### Inferences
- 40,000 rows exceeds TabPFN v2's documented comfort zone (10k) but remains in a range where no source documents a clear advantage for generic DL over a well-tuned GBM; it is a "gray zone" between the sweet spot of foundation models (small) and the relative sweet spot of GBDTs (large/irregular).
- The 85/15 imbalance is not directly treated by any of the sources found as a decisive factor in the choice of architecture; the only available hint (the cardiovascular study) points toward a somewhat greater fragility of transformer architectures under imbalance, not an advantage.

### Gaps
- No study found specifically tests the 30-50k row range with ~85/15 imbalance so as to isolate the effect of size from that of imbalance.
- No quantitative threshold ("beyond N rows, DL beats GBM by X points of AUC/logloss") published in the sources consulted.

## Do TabPFN or pre-trained tabular foundation models bring a particular advantage when the underlying function is additive and the noise Gaussian (an "easy" case for GAM/GBM)?

### Takeaway
No source directly tests this exact scenario (additive signal + Gaussian noise + probit link), but the adjacent literature comparing GAMs and neural networks strongly suggests that, when the true function is additive, a model with an additive inductive bias (GAM, or boosted depth-1 trees) is already close to optimal, and that a generic neural network (or a non-specialized foundation model) brings no structural advantage — on the contrary, it must "re-learn" the absence of interactions from limited data, which is a disadvantage on a small sample.

### Cited Findings
- Systematic review of GAMs vs. NNs (Doohan, Kook, Burke, 2025/2026, 143 articles/430 datasets): "neither GAMs nor NNs are systematically superior" on the usual metrics (RMSE, R², AUC); "GAMs remain competitive, particularly on small data, and offer interpretability"; the authors describe the two families as complementary, with an often modest performance trade-off for tabular data — [arXiv:2510.24601](https://arxiv.org/html/2510.24601v2)
- ProtoNAM (Xiong et al., ACM TKDD, Oct. 2025), a prototype-based neural additive model: claims to outperform existing neural GAMs, but explicitly notes that "neural GAMs still lag behind tree-based GAMs in many tabular applications" — [arXiv:2410.04723](https://arxiv.org/html/2410.04723v1)
- GAMformer (Mueller, Siems, Nori, Salinas, Zela, Caruana, Hutter — arXiv:2410.04560, NeurIPS 2024 workshop): the first tabular foundation model for GAMs, trained only on synthetic data, estimates a GAM's shape functions in a single forward pass (in-context learning) instead of iterative fitting; claims only to match the best existing GAMs on classification benchmarks — **no comparison against GBMs found in the sources consulted**, and the paper was only presented at a NeurIPS workshop (not yet independently validated at scale) — [arXiv:2410.04560](https://arxiv.org/html/2410.04560v1)
- TabPFN v2/2.5: none of the official pages consulted (docs.priorlabs.ai, "Large Data Model" report) mentions special handling or a documented advantage for targets with additive structure or Gaussian noise — these models are presented as general-purpose, without an explicit additive inductive bias — [docs.priorlabs.ai](https://priorlabs.ai/tabpfn-2), [Prior Labs technical report](https://priorlabs.ai/technical-reports/large-data-model)

### Inferences
- The current pipeline (depth-1 LightGBM/CatBoost = essentially boosted additive "stumps", + P-spline GAM) already imposes an additive inductive bias that exactly matches the described generative process (additive score + Gaussian noise, probit link). This is conceptually very close to what GAMformer does (imposing additivity through architecture) — but GAMformer itself claims only parity with classical GAMs, not superiority over hybrid GBM/GAM.
- A generic foundation model without an additive bias (classic TabPFN, FT-Transformer, SAINT) has *a priori* no structural advantage here: it must learn from the data that there are no significant interactions, whereas the GAM/depth-1 GBM starts from that assumption. On ~40k rows and a rare target (~15% positives, i.e. ~6000 positive examples), this "learning cost" of the additive constraint is probably not recouped.
- Overall, the very structure of the problem (additive, Gaussian noise, already confirmed experimentally by the user as optimal with a probit link) corresponds to the case where the GAM/NN literature shows the smallest performance gap and the one least favorable to generic DL.

### Gaps
- No study found explicitly simulates an additive+Gaussian synthetic dataset to compare TabPFN/GAMformer vs. GBM+GAM — a direct gap on the central question. A proper empirical (rather than theoretical) validation would require testing GAMformer or TabPFN-2.5 directly on this Zindi dataset.
- No source found on TabPFN's performance when the true signal is confirmed to be additive (vs. the general case where TabPFN must detect the structure itself).

## Are there hybrid architectures (boosting + neural network: NODE, GrowNet, boosted neural networks) with documented results on this type of structure?

### Takeaway
The two historical hybrid architectures (NODE, GrowNet) exist and are conceptually additive by construction, but neither has recent (2024-2026) documented results on datasets with confirmed additive structure + Gaussian noise, and neither appears among the model families highlighted in the most recent living benchmarks (TabArena), which suggests they have been superseded by newer architectures (RealMLP, TabM, TabICL, TabDPT, TabPFN) rather than validated on this specific use case.

### Cited Findings
- GrowNet (Badirli, Gu, Singh, Marculescu — arXiv:2002.07971, 2020): second-order gradient boosting where the weak learners are shallow neural networks instead of trees, with a global corrective step to address the greedy-approximation shortcoming of classical boosting; by construction, the final model remains an (additive) sum of sub-networks — [arXiv:2002.07971](https://arxiv.org/pdf/2002.07971)
- NODE (Popov et al. — arXiv:1909.06312, 2019): differentiable oblivious decision trees, trained end-to-end by gradient descent, combining tree-ensemble principles with differentiable optimization; originally claimed to outperform GBDTs on the original paper's benchmarks — [paperswithcode / GrowNet summary](https://paperswithcode.com/method/grownet)
- TabArena (2025), which lists the current reference model families for tabular deep learning (RealMLP, TabM, TabICL, TabDPT, TabPFN), **mentions neither NODE nor GrowNet** among the curated models of the living benchmark — [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)

### Inferences
- The absence of NODE/GrowNet from the 2024-2026 living benchmarks (TabArena, OmniTabBench) suggests that they are no longer considered competitive by the community, and that no recent validation specific to additive structures has been published — otherwise it would probably be cited in these survey works.
- GrowNet, by replacing trees with neural networks in the boosting loop, remains conceptually an additive model; it therefore brings no bias different from the one already exploited by depth-1 LightGBM/CatBoost — at best expected parity, with no documented justification for a gain.

### Gaps
- No recent (2024-2026) quantitative results found for NODE or GrowNet on an imbalanced binary-target dataset of comparable size (tens of thousands of rows). It is impossible to confirm or refute a precise empirical advantage/disadvantage for these architectures in this case.
- No source found on more recent (2024-2026) hybrid neural boosting variants beyond GrowNet/NODE — this research niche may have stagnated since their publication (2019-2020), with community interest shifting to foundation models (TabPFN) and regularized MLPs (RealMLP).

---

## Overall verdict for this specific case (synthesis note, not a formally requested research section but useful to the author of the final report)

In light of the sources above, 2024-2026 tabular deep learning does **not appear to be a realistic path** to surpassing the already bagged/blended GBM (depth 1, probit link) + P-spline GAM ensemble on this specific problem, for three convergent, sourced reasons:
1. The most recent living benchmarks (TabArena, OmniTabBench) conclude there is no universal winner, with DL only beating GBMs given much more expensive tuning and ensembling, with no guarantee of a net gain.
2. The dataset (~40k rows) falls outside the documented comfort zone of foundation models (TabPFN v2: ≤10k rows), and the successor that lifts this limit (TabPFN-2.5 Scaling Mode) itself states that it does not narrow the gap with gradient boosting as scale increases.
3. The signal structure (additive, Gaussian noise, confirmed probit link) exactly matches the inductive bias already exploited by the current pipeline; the GAM-vs-NN literature shows that it is precisely in this regime that the advantage of generic deep learning is smallest, or even non-significant in recent studies.

The only direction that remains defensible, and that is explicitly supported by the literature (the Kaggle practical guide, and the spirit of the blending already practiced in the current pipeline), is to add **a single additional neural model (a small regularized MLP, or even GAMformer) as a blend component with low error correlation**, not as a replacement — in the spirit of what the pipeline already does by bagging/blending LightGBM+CatBoost+GAM. Given that the gap to close to reach rank 1 is very small (logloss 0.227 → 0.223771, AUC 0.922 → 0.925), such a marginal blend is plausible as an incremental gain, but no source documents a similar case (additive + Gaussian signal + small imbalanced sample) where a generic neural network made it possible to close this kind of residual gap.
