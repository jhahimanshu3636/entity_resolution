# ENTITY RESOLUTION FLOW

## Overview

The resolver performs entity resolution for company names using a multi-stage pipeline:

1.  **Precompute Features** for unique company names (name + context).
2.  **Build Inverted Indices** (tokens, trigrams, initials) → O(1) candidate retrieval.
3.  **Find Similar Pairs** using:
    *   Candidate generation
    *   Fast prefilters
    *   Tiered similarity (name + context)
    *   Thresholding
4.  **Build Graph & Cluster** via connected components (DFS).
5.  **Select Parent Names** per cluster and compute canonical names.
6.  **Compute Confidence** per record against its cluster’s parent.
7.  **Map Results** back to the original DataFrame and print summaries.
8.  **Driver**: Load data (with encoding fallbacks), run resolver, save multi-sheet Excel outputs.

***

## **Stage 1 — Precompute Company Features**

### **Name Normalization**

*   Uppercase all names and strip whitespace.
*   Expand symbols:
    *   `& → AND`, `@ → AT`, `+ → PLUS`.
*   Remove non-alphanumeric characters; collapse multiple spaces.

### **Legal Suffix Removal**

Remove trailing legal entity suffixes (e.g., `LTD`, `LLC`, `INC`, `GMBH`, `PTE LTD`, `PLC`, `SARL`, `SAS`, `DMCC`, `TRADING CO`, `ENTERPRISES`, `INDUSTRIES`, `GROUP`, `HOLDINGS`, etc.) to get a **core name**.

### **Name Features (per unique name)**

*   `original`: Original name string.
*   `cleaned`: Normalized name.
*   `core`: Suffix-stripped name.
*   `tokens`: Words in core.
*   `token_set`: Unique tokens excluding noise words (`THE`, `OF`, `FOR`, `IN`, `ON`, `AT`, `TO`, `A`, `AN`).
*   `initials`: First letters of meaningful tokens.
*   `first_word`: First token.
*   `length`: Character length of core.
*   `char_signature`: Sorted unique characters of core (spaces removed).

### **Context Features (per unique name)**

*   `origin_country` and `dest_country`: Uppercased strings from the first occurrence.
*   `hs_chapters`: Extract 2-digit HS chapters from `hs_code` (handles comma-separated values).
*   `log_volume`: `log10(max(count, 1))` using the row’s count; fallback to 1 when missing.

### **Frequency Map**

*   Sum of `count` if numeric; else occurrence counts.

> **Note:** Feature computation is done once per unique name to accelerate comparisons.

***

## **Stage 2 — Inverted Index Blocking**

Construct inverted indices for fast candidate retrieval:

*   **Token Index**: `token → set(company_indices)`  
    For exact token overlaps.
*   **Trigram Index**: `trigram → set(company_indices)`  
    Trigrams from core (`TRADING` → `{TRA, RAD, ADI, DIN, ING}`) for fuzzy matches.
*   **Initials Index**: `initials → set(company_indices)`  
    Useful for acronym-like names.

**Candidate Retrieval**

*   Union of candidates from all indices.
*   Remove self from candidate set.
*   Truncate if too many candidates (e.g., max 1000).

***

## **Stage 3 — Candidate Filtering & Similarity**

### **Pre-filters (Fast Rejections)**

*   **Length Filter**: `min_len / max_len ≥ 0.5`.
*   **Token Overlap Filter**: ≥ 1 shared meaningful token.
*   **Character Signature Filter**: Jaccard ≥ 0.3.

### **Name Similarity (Tiered; Cached)**

*   **Token Jaccard**: `intersection / union`.
    *   Early exit: `≥ 0.8 → 0.90`, `< 0.2 → jaccard * 0.25`.
*   **Character-Level Similarity + Coverage**:
    *   Sequence ratio (with prefix bonus).
    *   Token coverage: `intersection / min(len(tokens1), len(tokens2))`.
*   **Composite Name Score**:
        name_sim = 0.50 * token_jaccard
                 + 0.30 * sequence_ratio_with_prefix_bonus
                 + 0.20 * token_coverage

### **Context Similarity**

Combines:

*   **Trade Route (30%)**
    *   Same origin & destination → 1.0
    *   Partial match → 0.5
*   **HS Chapters (40%)**
    *   Missing → 0.5
    *   Else Jaccard similarity.
*   **Trade Volume (30%)**
    *   Log-scale ratio; unknown → 0.5.

<!---->

    context_sim = 0.30 * route_score
                + 0.40 * product_score
                + 0.30 * volume_score

### **Composite Similarity**

    composite = name_weight * name_sim + context_weight * context_sim

*   `context_weight` configurable (e.g., 0.25).
*   Match if `composite ≥ similarity_threshold` (e.g., 0.85).

***

## **Stage 4 — Graph Construction & Clustering**

*   Build undirected graph: nodes = companies, edges = matched pairs.
*   Extract connected components via DFS → clusters.

**Report:**

*   Number of clusters
*   Multi-member clusters

***

## **Stage 5 — Parent Selection & Canonical Naming**

*   **Parent Name**: Highest `length * frequency` score.
*   **Canonical Name**: Normalize + remove suffixes again.
*   Store:
    *   `parent_name`
    *   `canonical_name`
    *   `cluster_size`
    *   `member_names`

***

## **Stage 6 — Match Confidence Scoring**

*   Singleton cluster → confidence = 1.0.
*   Else compute composite similarity with parent name.

***

## **Stage 7 — Result Mapping & Summary**

Map back to original DataFrame:

*   `cluster_id`
*   `parent_name`
*   `canonical_name`
*   `cluster_size`
*   `match_confidence`

**Print Summary:**

*   Input records
*   Unique companies
*   Clusters created
*   Multi-member clusters
*   Deduplication rate (%)
*   Processing time & speed

***
