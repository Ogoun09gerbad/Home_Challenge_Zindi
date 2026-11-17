# Home_Challenge_Zindi

My solutions to [Zindi](https://zindi.africa) data science competitions: code, experiments and reproducible pipelines for machine learning challenges across speech, tabular data and more.

Each folder is a self-contained project with its own code, notes and (when available) its own README explaining how to reproduce the results.

## Challenges

| Folder | Domain | Approach | Result |
|---|---|---|---|
| [WAXAL ASR Challenge](./WAXAL%20ASR%20Challenge) | Speech recognition (Lingala, Shona) | Ensemble of w2v-BERT 2.0 CTC models, text-based language routing, Minimum-Bayes-Risk selection, rule-based post-processing | Private score 0.7711, **rank 6** |
| [zindi_compet](./zindi_compet) | Tabular ML: financial stress prediction from mobile money transactions | Feature engineering, LightGBM + CatBoost + GAM blend | — |
| [AssemblyAI_copy](./AssemblyAI_copy) | Real-time streaming speech-to-text | Whisper + VAD + local-agreement streaming over FastAPI WebSockets | — |

## Notes

- Competition datasets, model weights and large artifacts are **not** included (size limits and Zindi data rules). Download the data from each competition page on Zindi.
- Secrets (e.g. Zindi credentials) are read from environment variables; see `.env.example` where provided.
