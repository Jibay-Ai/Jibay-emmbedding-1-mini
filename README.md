![Jibay Embedding 1 Mini](https://huggingface.co/JibayAi/jibay-emmbedding-1-mini/resolve/main/jibay-emmbedding-1-mini.jpg)

# Jibay Embedding 1 Mini

**Jibay Embedding 1 Mini** is a lightweight multilingual embedding model developed and trained by **JibayAI**.

Designed as a compact semantic representation model, it converts text into dense 128-dimensional vectors that can be used for semantic similarity, text retrieval, and other embedding-based applications.

## Model Overview

| Property | Value |
|---|---|
| Model Name | `jibay-emmbedding-1-mini` |
| Developer | JibayAI |
| Model Type | Text Embedding |
| Architecture | `JibayForEmmbeddingLM` |
| Training Duration | 6 hours |
| Total Parameters | 31,200 |
| Vocabulary Size | 512 |
| Embedding Dimension | 128 |
| Hidden Size | 32 |
| Transformer Layers | 1 |
| Attention Heads | 2 |
| Feed-Forward Network Size | 64 |
| Maximum Sequence Length | 64 tokens |
| Tokenization | Byte-Level BPE |
| Training Objective | Symmetric InfoNCE |
| Optimizer | AdamW |
| License | Apache-2.0 |

## Architecture

Jibay Embedding 1 Mini uses a compact Transformer-based architecture designed to generate dense semantic vectors with a very small parameter count.

Its architecture includes:

- **Token Embeddings:** Byte-Level BPE tokenization with a vocabulary of 512 tokens.
- **Transformer Encoder:** A single Transformer layer.
- **Multi-Head Attention:** 2 attention heads.
- **Feed-Forward Network:** 64-dimensional intermediate representation.
- **Hidden Dimension:** 32.
- **Embedding Projection:** Projects the learned representations into a 128-dimensional embedding space.

The model is designed to provide a lightweight foundation for multilingual semantic representation.

## Training

Jibay Embedding 1 Mini was developed and trained by **JibayAI**.

Training details:

- Training duration: approximately 6 hours.
- Training data: approximately 56,000 English–Persian text pairs.
- Dataset source: OPUS-100.
- Language pair: English–Persian (en-fa).
- Training objective: Symmetric InfoNCE.
- Optimization algorithm: AdamW.

The training process was designed to bring semantically corresponding English and Persian sentences closer together in the embedding space.

## Usage

The model generates 128-dimensional vectors for input text.

These embeddings can be used for:

- Semantic similarity
- Cross-lingual text matching
- Semantic search
- Information retrieval
- Text clustering
- Retrieval-Augmented Generation (RAG)
- Duplicate and related-text detection

## Model Versions

JibayAI develops multiple versions of its embedding models.

| Version | Description |
|---|---|
| Jibay Embedding 1 Mini | Lightweight model released for public use |
| Jibay Embedding Large | Larger model available through the official JibayAI API |

The **Large** Coming soon.

## Evaluation Results

We ran three small, lightweight evaluations on this model. They are not standard leaderboard benchmarks, so the numbers are only a rough guide and should not be compared directly with published MTEB results.

### 1. Persian manual benchmark

A small hand-made test set of Persian sentences with two parts:

- **Triplet accuracy:** 15 (anchor, positive, negative) triplets. A triplet passes when the anchor is closer by cosine similarity to the positive than to the negative.
- **Retrieval:** 10 queries against a pool of 10 documents, each query having one gold document. We report top-1 accuracy and Mean Reciprocal Rank (MRR).

| Metric | Value |
| --- | --- |
| Triplet accuracy | 0.800 (12/15) |
| Retrieval top-1 | 0.100 (1/10) |
| Retrieval MRR | 0.293 |
| **Score** | **54.6 / 100** |

Score = 100 × (0.5 × triplet accuracy + 0.5 × MRR)

### 2. English manual benchmark

The same procedure as the Persian benchmark, using English sentences.

| Metric | Value |
| --- | --- |
| Triplet accuracy | 0.400 (6/15) |
| Retrieval top-1 | 0.100 (1/10) |
| Retrieval MRR | 0.296 |
| **Score** | **34.8 / 100** |

### 3. Persian STS: Farsick (`mteb/Farsick`)

Semantic textual similarity on the test split of [Farsick](https://huggingface.co/datasets/mteb/Farsick), a Persian STS dataset from the FaMTEB collection. We embedded 200 sentence pairs, of which 199 were scored (1 was skipped because the API returned an error). For each pair we computed the cosine similarity between embeddings and measured Spearman correlation with the human similarity scores.

| Metric | Value |
| --- | --- |
| Pairs evaluated | 199 (1 skipped) |
| Spearman correlation | 0.2448 |
| **Score** | **24.5 / 100** |

Score = Spearman correlation × 100

### Summary

| Benchmark | Language | Metric | Score |
| --- | --- | --- | --- |
| Manual triplets + retrieval | Persian | Composite | 54.6 |
| Manual triplets + retrieval | English | Composite | 34.8 |
| Farsick STS | Persian | Spearman × 100 | 24.5 |

The model is designed to be lightweight (31,200 parameters, 128-dimensional embeddings, 64-token maximum sequence length), so these scores should be read in that context. The model was trained on English–Persian pairs, which may explain why Persian-to-Persian similarity is relatively stronger than English results on the manual test set. Larger models are available through the JibayAI API.

## Open Source & License

Jibay Embedding 1 Mini is released under the **Apache License 2.0**.

Everyone is free to:

- Use the model for personal or commercial projects.
- Modify and customize its architecture.
- Improve and optimize its weights.
- Fine-tune or retrain the model.
- Publish modified or improved versions.
- Redistribute the original or modified model.
- Integrate it into applications and other AI systems.

You are welcome to experiment with the model, improve it, build upon it, and publish your own versions.

Please follow the requirements of the Apache-2.0 license when redistributing the model.

See the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) for full terms.

## Developed By

**JibayAI**

Independent AI development by JibayAI.

Website: https://jibay.ir

## License

Copyright © JibayAI

Licensed under the Apache License, Version 2.0.
