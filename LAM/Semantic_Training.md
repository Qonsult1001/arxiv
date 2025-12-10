# Semantic Training

## Overview

This document consolidates extracted text from the final batch of images and organizes it into clear columns for reference.

---

## I. How to Train Multilingual Models

**Column: Extracted Notes**

* Have suitable training data in target languages (e.g., question–answer pairs).
* Use translation bridging tasks:

  * English (question, answer) data
  * Parallel data: (english_sentence, german_sentence)
  * Alternate training between English and parallel data
* Approach used in mUSE.

---

## II. Multi-Dataset Training

**Column: Extracted Notes**

* Problem: Datasets vary massively in size (e.g., NQ: 100k vs Reddit: 600M+).
* Sampling Strategies:

  * Equal sampling (50% / 50%)
  * Temperature scaling (from NMT literature)
  * Cap maximum dataset size (e.g., 1M) → results in 10% NQ / 90% Reddit
* Consider cross-dataset batches.

---

## III. Multilingual Knowledge Distillation

**Column: Extracted Notes**

* Teacher model T (English-only, e.g., SBERT trained on STS).
* Student model S (multilingual vocab, e.g., XLM-R + mean pooling).
* Use parallel data pairs (s_i, t_i).
* Train S to approximate T:

  * S(s_i) ≈ T(s_i)
  * S(t_i) ≈ T(t_i)
* Uses MSE loss.

---

## IV. TPU Training Specialties

**Column: Extracted Notes**

* TPU builds and optimizes a graph for training steps.
* Slow initial optimization but cached afterwards.
* All tensors must have identical lengths.
* Workaround: allow padded sequence lengths (64, 128, 196, 256) → 4 cached graphs.
* Training with pairs/triplets in alternating setups may be difficult.

---

## V. Sentence Embeddings Model — Architecture

**Column: Extracted Notes**

* BERT produces contextualized word embeddings.
* Pooling yields fixed sentence embedding vectors.
* Architecture referenced from SBERT (EMNLP 2019).
* Normalization before final embedding simplifies cosine similarity.

---

## VI. Dot-Product vs Cosine Similarity

**Column: Extracted Notes**

* **Normalized Vectors**:

  * cos(a,a)=1
  * dot-product equals cosine similarity
  * Euclidean distance proportional to cos-sim
  * Supports k-means clustering
* **Unnormalized Dot-Product**:

  * Other vectors may exceed self-similarity
  * Sometimes slower for ANN
  * Not compatible with k-means

---

## VII. Optimizing Multiple-Negatives-Ranking-Loss

**Column: Extracted Notes**

* scaled_cos_sim(a,b) = C × cos_sim(a,b)

  * scale C often 14–20
  * ConverRT: increase 1 → 23 over first 10k steps
  * CLIP: exp(C) × cos_sim(a,b); C learnable
* Symmetric loss (A,P)+(P,A) / 2
* Additive margin:

  * sim(a_i,p_i) reduced by margin m
  * LaBSE uses margin 0.3
* Hard negatives matter

  * Mine via BM25 → pick from top 100
  * Ensure they are true negatives

---

## VIII. Improving Batch Quality

**Column: Extracted Notes**

* Cross-domain sampling can result in trivial positives.
* Better: keep domains clustered within batches.
* Example: StackExchange includes 140 subforums → mix accordingly.

---

## IX. Suitable Datasets + Zero-Shot Performance

**Column: Extracted Notes**

* MiniLM v6 pretraining: 2k steps, batch size 256.
* Best performing datasets include:

  * stackexchange_title_body_simal (59.83)
  * googq_pairs (59.76)
  * romance_query_passage_negative (59.06)
  * yahoo_answers_title_answer (58.85)

### **Full Dataset List (All Extracted Data)**

| Dataset Name                                   | Description                                    | Size (#Pairs) | Performance |
| ---------------------------------------------- | ---------------------------------------------- | ------------- | ----------- |
| stackexchange_title_body_simal                 | Title/Body pairs from StackExchange            | 304,951       | 59.83       |
| googq_pairs                                    | Google auto-suggest Q&A pairs                  | 3,012,490     | 59.76       |
| romance_query_passage_negative                 | MS MARCO: Query/Answer/Passage + hard negative | 9,144,553     | 59.06       |
| yahoo_answers_title_answer                     | Title → Answer from Yahoo Answers              | 1,136,263     | 58.85       |
| stackexchange_duplicate_questions              | Duplicate question titles                      | 304,505       | 58.47       |
| romance_query_passage                          | MS MARCO Query/Answer/Passage                  | 512,701       | 58.28       |
| qils_question_answer                           | ILS dataset Q&A pairs                          | 325,475       | 58.24       |
| yahoo_answers_title_question                   | Title / Question / Body                        | 659,896       | 58.05       |
| squad_pairs                                    | SQuAD QA + Passage                             | 97,909        | 58.02       |
| yahoo_answers_question_answer                  | Question / Body / Answer                       | 681,766       | 57.74       |
| NQ_train_pairs                                 | Natural Questions query + answer passage       | 130,231       | 57.48       |
| quora_duplicates                               | Duplicate questions                            | 133,563       | 57.38       |
| WikiAnswers_pairs                              | Massive duplicate question dataset             | 77,427,422    | 57.34       |
| stackexchange_duplicate_questions (Title+Body) | Dupes from StackExchange                       | 250,460       | 57.30       |
| S2ORC_citation_pairs                           | Scientific citation title pairs                | 52,603,982    | 57.28       |
| stackexchange_duplicate_questions (Body, Body) | Body-only dupe pairs                           | 250,519       | 57.26       |
| quora_duplicates_triplets                      | Triplets: anchor/dupe/hard negative            | 133,563       | 56.97       |
| APLI_1                                         | Combination of BaRT + MuMNI                    | 377,210       | 56.57       |
| MNLI                                           | Triplets (entailment / contradiction)          | Not visible   | Not visible |

## X. Summary

This document compiles all images' content into structured conceptual sections with clear column separation for extracted notes, enabling easy navigation and training reference.
