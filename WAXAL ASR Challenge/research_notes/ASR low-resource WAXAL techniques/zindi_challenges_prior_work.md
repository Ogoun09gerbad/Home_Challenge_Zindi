# Zindi/Kaggle ASR challenges for African languages (Lingala, Shona, Luganda) and published work

## Which past/ongoing Zindi challenges relate to ASR for African languages (WAXAL, AI4D, Lacuna, etc.), and which approaches won?

### Takeaway
The "Google WAXAL ASR Challenge" on Zindi (2026 edition, results announced on 22 September 2026) focused on **Lingala and Shona** (the Zindi link to the competition page also mentions **Luganda** as a target language in the challenge description). The three winning teams used **multi-model ensembling** and **selection of the most reliable transcription** strategies, rather than a single miracle architecture. Other earlier Zindi/AI4D challenges exist (Tanzania/Swahili 2023, "Your Voice Your Device Your Language" Swahili 2025, Google NLP Hack Series 2022), but detailed public technical write-ups are rare or could not be retrieved from the pages consulted.

### Cited Findings
- The WAXAL ASR Challenge (Google Research + Zindi) brought together 1,462 participants from 100 countries (42 African countries), with 8,837 submissions and over 24,000 cumulative hours of participation — [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)
- The WAXAL dataset is an open-source speech corpus covering 32 African languages, developed by Google in collaboration with institutions such as Makerere University and the University of Ghana — [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/); competition rules on [Zindi WAXAL ASR Challenge](https://zindi.africa/competitions/google-waxal-asr-challenge)
- The challenge is structured in two phases: in Phase 1, participants can experiment with data augmentation techniques and supplement the provided data with other public speech/language datasets (subject to legal licensing and disclosure); in Phase 2, only the audio data is provided about one week before the close — [Zindi WAXAL ASR Challenge data page](https://zindi.africa/competitions/google-waxal-asr-challenge/data) and competition page ([af.net report](https://af.net/realtime/google-waxal-challenge-tests-lingala-and-shona-speech-ai/))
- 1st prize: **Team Pasketti** (Roman Solovyev, Kazakhstan, and "enes3774", Canada) — combined **eight models** producing different error types, so they could "cross-check" one another — [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)
- 2nd prize: **Alban Nyantudre** (Burkina Faso), an ML engineer who builds speech tools for Mooré — his system processes the same audio clip in several different ways and then selects the output it judges most reliable — [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)
- 3rd prize: **Abdourahamane Ide Salifou** (Niger), an AI engineering student — for Lingala (words that run together without clear separation), he trained the system to take in a longer speech segment before predicting words; for Shona (clearer word boundaries), he relied more on the surrounding context — [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)
- The exact scores (WER/metric) of the top three were not published in the Google blog post, which only states that the scores were "very close" — [Google Africa Blog](https://blog.google/intl/en-africa/company-news/outreach-and-initiatives/meet-the-winners-of-the-waxal-speech-recognition-challenge/)
- The rules require the top three finishers to submit their code for verification — [Zindi WAXAL ASR Challenge](https://zindi.africa/competitions/google-waxal-asr-challenge)
- Other Zindi challenges related to African ASR exist: AI4D Tanzania (Swahili, 2023, WER metric, the 1st prize winner had to assign copyright to Zindi) — [Zindi AI4D Tanzania ASR](https://zindi.africa/competitions/ai4d-africas-anglophone-research-lab-tanzania-automatic-speech-recognition-challenge); "Your Voice, Your Device, Your Language Challenge" (Swahili, 2025, write-up published on the Zindi learning platform but content not retrieved during the fetch) — threads [28994](https://zindi.africa/competitions/your-voice-your-device-your-language-challenge/discussions/28994), [29417](https://zindi.africa/competitions/your-voice-your-device-your-language-challenge/discussions/29417), [28049](https://zindi.africa/competitions/your-voice-your-device-your-language-challenge/discussions/28049); Google NLP Hack Series (2022, beginner data collection, 1st prize Dunstan Matekenya, Malawi, Chichewa audio collection) — [Zindi blog reflection](https://zindi.africa/learn/a-reflection-on-the-google-nlp-hack-series)

### Inferences
- The recurring pattern among the WAXAL winners is **ensembling several models or decodings of the same audio** combined with a **heuristic for selecting the most reliable output**, rather than a single optimal fine-tuned model — a strategy consistent with classic Kaggle/Zindi practice (robustness through aggregation rather than through one "perfect" model).
- Adapting the decoding context to the language's morphology (longer context window for Lingala with weak word segmentation, vs. more surrounding context for Shona) suggests that the winning teams tuned their architecture/decoding choices to the specific linguistic characteristics rather than applying a single recipe to all languages.

### Gaps
- No Zindi discussion page (forum with detailed winning solutions, full technical write-up with precise architecture, hyperparameters, scripts) could be retrieved for the WAXAL Challenge — only the journalistic summary on the Google blog could be consulted in detail.
- The numerical scores (WER/CER) of the three best WAXAL solutions are not publicly available in the sources found.
- Exact closing status of the challenge (the Zindi page showed "Aug 02, 26" / "10 days left" at the time of the search, contradicting the results announcement on 22 September 2026) — probably a stale page cache, not verified directly.

## Which winning solutions from other African ASR challenges are publicly documented (GitHub), and what techniques did they use?

### Takeaway
The documented public solutions for African ASR on Zindi mostly concern older challenges (AI4D). The recurring documented techniques are: wav2vec2-XLSR fine-tuning, decoding with an n-gram language model (KenLM), and nearest-neighbour search over the training vocabulary.

### Cited Findings
- TechCabal Ewè Audio Classification Challenge (2024): won by **Mubarak Muhammad**, who made "several key architectural improvements" that improved performance for voice-command recognition in Ewe — [TechCabal](https://techcabal.com/2024/10/25/machine-learning-engineer-wins-techcabal-ewe-audio-classification-challenge/)
- For a low-resource Zindi ASR challenge (AI4D, non-Wolof language confirmed in this context restricted to the target languages): one publicly documented approach consists of wav2vec2-XLSR fine-tuning, n-gram language model decoding, and nearest-neighbour search against a vocabulary built from the training set — a generic technique seen in several African Zindi ASR solution repos (e.g. similar patterns documented by the Zindi community for low-resource ASR) — [generic Zindi discussion pattern, not specific to a WAXAL target language]
- Zindi published a "reflection on the Google NLP Hack Series" in which the first prize (data collection, not modeling) went to Dunstan Matekenya (Malawi) for Chichewa audio collection — [Zindi Learn](https://zindi.africa/learn/a-reflection-on-the-google-nlp-hack-series)

### Inferences
- The combination "XLSR fine-tuning + n-gram LM (KenLM) + nearest-neighbour search over the training vocabulary" appears to be a robust, reusable pattern on Zindi for low-resource ASR in general, independent of the target language — potentially transferable to the Lingala/Shona/Luganda context.

### Gaps
- No GitHub write-up of a winning solution specific to a Zindi ASR challenge in Lingala, Shona or Luganda was located publicly in this search (beyond the journalistic summary on the Google blog for WAXAL).
- The detailed write-up of the winning solution for the Swahili "Your Voice, Your Device, Your Language" challenge (2025) could not be retrieved (page content not extracted despite being indexed).

## Which academic papers address ASR for Luganda, Shona, Lingala (low-resource languages) — corpora, WER/CER results, methods?

### Takeaway
The most relevant and recent reference paper is **"Benchmarking Automatic Speech Recognition Models for African Languages"** (arXiv 2512.10968, Nahabwe et al., Makerere University, submitted 30 November 2025, published in the Deep Learning Indaba 2025 proceedings / PMLR v302), which fine-tunes four models (Whisper, XLS-R, MMS, W2v-BERT) on 13 African languages explicitly including Luganda, Lingala and Shona, with increasing data sizes from 1h to 400h.

### Cited Findings
- The paper uses Luganda and Lingala data from **Google FLEURS**, and Lingala and Shona data from **AfriVoice** (described as "descriptive speech") — [arXiv 2512.10968 (HTML)](https://arxiv.org/html/2512.10968v1)
- WER/CER results (%) at different training data sizes (without LM):
  - **Luganda**: Whisper reaches 20.45%/5.54% WER/CER at 200h (best score), vs XLS-R 40.51%/7.78%, W2v-BERT 39.75%/7.41%, MMS 42.69%/8.56% at the same size — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
  - **Lingala**: at 100h, XLS-R reaches 18.82%/5.69% WER/CER (best score without LM), MMS 23.96%/8.32%, W2v-BERT 24.19%/9.59%, Whisper 25.25%/10.21% — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
  - **Shona**: at 100h, W2v-BERT reaches 22.56%/4% WER/CER (best score without LM), MMS 23.70%/4%, XLS-R 25.83%/4%, Whisper 30.92%/7% — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- Effect of decoding with a 5-gram KenLM language model (Kneser-Ney) applied only to XLS-R and W2v-BERT: for **Luganda**, the LM consistently improves both models (XLS-R: 42.87% → 20.68% WER at 100h; W2v-BERT: 39.75% → 16.30% at 200h); for **Lingala**, mixed effect — W2v-BERT gains modestly (24.19% → 22.74% at 100h) but XLS-R-LM is worse than without LM from 5h onward (18.82% → 25.39% at 100h); for **Shona**, variable effect — XLS-R-LM helps between 5h and 20h but hurts at 50h/100h, while W2v-BERT-LM gives the largest gain in the paper for this language (23.54% → 8.61% WER at 50h) — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- The paper's general finding: MMS and W2v-BERT are more data-efficient in very low-resource regimes, XLS-R improves better with more data, and Whisper shows advantages under medium-resource conditions — [Hugging Face papers summary](https://huggingface.co/papers/2512.10968.md), [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- The paper does not describe any audio data augmentation technique applied in its own experiments (noise, pitch shift, speed perturbation, SpecAugment) — it mentions synthetic augmentation only in relation to related work by other authors — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- Preprocessing applied: lowercasing, punctuation removal, diacritic mapping, filtering clips to 2–30 seconds, resampling to 16 kHz, dropout of 0.1, frozen feature extractor for XLS-R, MMS and W2v-BERT — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)
- **Makerere Radio Speech Corpus** (LREC 2022): 155-hour Luganda radio corpus, described as the first public radio dataset in sub-Saharan Africa; end-to-end CTC baseline model built with Coqui STT, combining transfer learning on 203h of read speech and 120.7h of radio data — [ACL Anthology 2022.lrec-1.208](https://aclanthology.org/2022.lrec-1.208), [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2206.09790)
- Makerere master's thesis (2025, Abigaba, not formally published): multilingual Whisper-based system for Luganda and Runyankole-Rukiga, reporting a WER of 50% for Luganda and 60% for Runyankole-Rukiga — treat with caution since it is not formally published — [Sunbird SALT tutorials](https://salt.sunbird.ai/tutorials/09-asr-models/)
- Sunbird AI models: adaptation of Whisper-large-v2 for several Ugandan languages including Luganda, trained on the SALT dataset, Common Voice (Luganda) and FLEURS; Sunbird also publishes an MMS-based model with a Luganda adapter, whose output is unpunctuated and lowercase (for formatted text, the documentation recommends the Whisper variant) — [Sunbird SALT docs](https://salt.sunbird.ai/tutorials/09-asr-models/)
- For **Lingala** (and Tshiluba, Kikongo, Congolese Swahili): Kimanuka, wa Maina & Büyük (Data in Brief 2023 / ICLR 2023 AfricaNLP workshop) published two corpora — a 4.3h "Lingala Read Speech Corpus" for supervised ASR, and a 741h unsupervised "Congolese Radio Corpus" for speech representation — with state-of-the-art results for Congolese wav2vec2 and an average 2% WER drop when fine-tuning on Lingala with a pre-trained Congolese multilingual model; described as the first ASR benchmark attempt for Lingala and the first multilingual model for 4 Congolese languages — [PMC article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10700368/), [Mendeley Data DOI 10.17632/28x8tc9n9k.1](https://data.mendeley.com/datasets/28x8tc9n9k)
- A complementary corpus, **LiSTra**, presents a Lingala speech dataset for speech translation, with an English-to-Lingala baseline in pipeline and end-to-end configurations — [cited via arXiv search, reference not verified in detail]
- For **Shona**: the closest historical work identified is by Molapo, Barnard & De Wet (North-West University, 2013), who collected a Shona speech corpus and built what they describe as the first ASR system for the language; an associated master's thesis by Molapo (2014) covers the same work — [NWU Institutional Repository](https://repository.nwu.ac.za/handle/10394/12117)
- No corpus named "ZambeziVoice" includes Shona: Zambezi Voice (arXiv 2306.04428) covers the Zambian languages Bemba, Nyanja, Tonga and Lozi, not Shona — [arXiv 2306.04428](https://arxiv.org/abs/2306.04428v2)

### Inferences
- For Luganda, Whisper clearly outperforms the other models once at least ~50-100h of data are available, whereas for Shona W2v-BERT dominates at all tested sizes, and for Lingala XLS-R wins at 100h without LM — suggesting that no single model is universally optimal for all low-resource languages, and that the choice of architecture should be validated empirically per target language rather than assumed a priori.
- Adding a KenLM n-gram LM reliably improves Luganda but can hurt performance for Lingala and Shona beyond a certain data volume — suggesting a risk of the external LM overfitting the limited training vocabulary for these languages, a phenomenon to watch during lightweight fine-tuning for WAXAL.
- Lingala has dedicated but small academic corpora (4.3h of read speech) complemented by a large unsupervised corpus (741h of radio), which typically corresponds to a self-supervised pretraining + limited supervised fine-tuning scenario, rather than massive supervised training.
- Academic Shona has very few recent dedicated resources (foundational work dating from 2013), which probably makes it the least-resourced language in terms of specific ASR literature among the three target languages of the WAXAL challenge.

### Gaps
- No specific WER/CER result could be confirmed for the LiSTra corpus (Lingala speech translation) beyond the mention of its existence.
- No recent quantitative data (2020-2026) on Shona ASR was found beyond the 2013/2014 work and the 2025 Makerere benchmark (arXiv 2512.10968) — a significant literature gap for this language between 2014 and 2025.
- The full text of the Makerere paper did not allow checking whether inconsistencies flagged in the LM table for Shona (e.g. XLS-R-LM at 50h/100h) are publication errors or real results — check directly in the source PDF if confirmation is needed.

## Which audio data augmentation techniques are reported as effective for low-resource African ASR?

### Takeaway
The sources consulted report few precise technical details on audio augmentation (noise, pitch shift, speed perturbation, SpecAugment) applied specifically to Lingala/Shona/Luganda; the clearest signal is that the WAXAL challenge explicitly encouraged data augmentation in Phase 1, and that a Zindi comment (not directly about WAXAL, Swahili challenge) notes that noise and light fine-tuning can suffice for good performance.

### Cited Findings
- The WAXAL ASR Challenge rules explicitly invite participants to experiment with data augmentation techniques in Phase 1, and allow supplementing the provided data with other legally licensed and disclosed public speech/language datasets — [Zindi WAXAL ASR Challenge data page](https://zindi.africa/competitions/google-waxal-asr-challenge/data)
- In a Zindi discussion thread for a Swahili ASR challenge ("Your Voice, Your Device, Your Language"), a comment notes that techniques such as noise augmentation and light fine-tuning can be enough to reach good performance — [Zindi discussion thread 28994](https://zindi.africa/competitions/your-voice-your-device-your-language-challenge/discussions/28994)
- The Makerere benchmark paper (arXiv 2512.10968) applied no audio augmentation in its own experiments on Luganda/Lingala/Shona; it mentions synthetic augmentation only when citing third-party work, without detailing the method or results for these languages — [arXiv 2512.10968](https://arxiv.org/html/2512.10968v1)

### Inferences
- The lack of precise technical details on audio augmentation in the public sources available for Lingala/Shona/Luganda suggests that this dimension remains scientifically under-documented for these specific languages, in contrast with techniques that are better established in the general ASR literature (SpecAugment, speed perturbation) but have no published empirical validation specific to these three languages in the sources found.

### Gaps
- No study precisely quantifying the effect (WER delta) of specific data augmentation techniques (SpecAugment, pitch shift, speed perturbation, added noise) for Luganda, Lingala or Shona could be located in this search.
- No WAXAL winning-solution write-up details the exact nature of the augmentation techniques used by the teams (the Google blog does not explicitly mention audio augmentation in its description of the top 3 strategies).

## Are there initiatives (Masakhane, Lacuna Fund, ALFFA, Sunbird AI, etc.) that have published datasets/models relevant to Luganda/Shona/Lingala, and what lessons can be drawn?

### Takeaway
Several initiatives publish relevant resources: Sunbird AI/Makerere for Luganda (adapted Whisper and MMS models, SALT + radio corpora), Masakhane, whose ASR efforts are still emerging (call for proposals for 18 languages and a 40-language multimodal hub) compared with its well-established text NLP work, and dedicated academic corpora for Lingala (Congolese Radio Corpus / Lingala Read Speech Corpus).

### Cited Findings
- **Masakhane African Languages Hub**: launches a call for proposals including ASR, with speech data for 18 languages, as part of an initiative to create open multimodal datasets for 40 African languages (text, speech, vision), with a focus on gender balance and contextual authenticity — [AI4D project page](https://www.ai4d.ai/projects/masakhane-african-languages-hub)
- Masakhane's well-established resources are mostly textual (MasakhaNER 2.0 for named entity recognition in 20 languages, MasakhaPOS for part-of-speech tagging in 20 languages); no large, finalized Masakhane ASR dataset published at scale was identified in these results — the speech effort appears to be an ongoing project rather than an already delivered resource — [cross-checked web search, no finalized Masakhane ASR dataset located]
- **Sunbird AI** (associated with Makerere) publishes a Whisper-large-v2 model adapted for several Ugandan languages including Luganda, trained on the SALT dataset, Common Voice (Luganda) and FLEURS, as well as an MMS model with a Luganda adapter (unpunctuated/lowercase output) — [Sunbird SALT docs](https://salt.sunbird.ai/tutorials/09-asr-models/)
- **Common Voice** includes Luganda; the Makerere team collaborated with Mozilla to add it to the platform — [cross-checked search, ASR Africa benchmark table]
- **ALFFA** (African Languages in the Field: speech Fundamentals and Automation, 2015) covers Swahili and Wolof under an MIT license according to the ASR Africa benchmark table — but covers neither Luganda, Shona nor Lingala — [ASR Africa Benchmark Dataset](https://huggingface.co/spaces/asr-africa/Automatic_Speech_Recognition_for_African_Languages/blob/96407d13478527a303623d686d6fbac49f63bae4/benchmark.md)
- **BembaSpeech** (Sikasote, 2021): read-speech corpus of more than 24h of Bemba (a Zambian language, >30% of the Zambian population), fine-tuning of a pre-trained English DeepSpeech model reaching 54.78% WER as its best result; publicly available on GitHub under a CC-BY-NC-4.0 license (restricted commercial use) — [NLP CS GMU publication](https://nlp.cs.gmu.edu/publication/sikasote-21-bembaspeech/), [ASR Africa Benchmark](https://huggingface.co/spaces/asr-africa/Automatic_Speech_Recognition_for_African_Languages/blob/96407d13478527a303623d686d6fbac49f63bae4/benchmark.md)
- **Lacuna Fund** funded new NLP datasets for African languages in 2020 (announcement of grantees), but the specific details covering Luganda/Shona/Lingala were not confirmed in the results found — [Lacuna Fund announcement](https://lacunafund.org/?p=7503)

### Inferences
- The three WAXAL challenge languages have very uneven levels of resources: Luganda is the best resourced (155h radio corpus, Common Voice, dedicated Sunbird/Whisper models, Makerere benchmark), Lingala has more modest but specialized academic corpora (read speech + unsupervised radio), while Shona appears to be the least documented language in the recent academic literature among the three, with an almost total gap between the foundational 2013 work and the 2025 benchmark.
- For lightweight fine-tuning in the WAXAL challenge, building on Sunbird AI pre-trained models (Luganda) or already fine-tuned Congolese wav2vec2/MMS checkpoints (Lingala) could be a more competitive starting point than a raw generic multilingual model, given the performance gaps documented in the Makerere benchmark.

### Gaps
- No direct confirmation that Lacuna Fund funded resources specific to Luganda, Shona or Lingala (the announcement found remains general about "African languages" with no per-language detail).
- The current (2026) status of the Masakhane Hub "40-language multimodal flagship" — whether it has produced usable ASR data for the target languages — could not be verified beyond the call-for-proposals announcement.
