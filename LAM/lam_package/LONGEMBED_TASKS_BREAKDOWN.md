# 📊 MTEB LongEmbed Tasks - Complete Breakdown

This document provides a comprehensive breakdown of each LongEmbed task in MTEB.
Use this to understand what each task tests and how to optimize your model.

## 🎯 How MTEB Evaluates LongEmbed Tasks

### Evaluation Flow

1. **Data Loading**: MTEB loads corpus (documents), queries, and relevance judgments
2. **Encoding**: 
   - Calls `model.encode_corpus(corpus)` → Returns embeddings for all documents
   - Calls `model.encode_queries(queries)` → Returns embeddings for all queries
3. **Similarity Computation**: 
   - Computes similarity matrix: `similarity = query_embeddings @ corpus_embeddings.T`
   - Uses cosine similarity (after L2 normalization)
4. **Ranking**: For each query, ranks documents by similarity score
5. **Scoring**: Computes nDCG@10, MAP@10, Recall@10, etc. based on relevance judgments

### Key Evaluation Details

- **Main Metric**: `ndcg_at_10` (Normalized Discounted Cumulative Gain at rank 10)
- **Scoring Formula**: 
  ```
  nDCG@10 = DCG@10 / IDCG@10
  DCG@10 = Σ(rel_i / log2(i+1)) for i in [1..10]
  ```
  Where `rel_i` is the relevance score of document at rank `i`
- **Relevance Judgments**: Binary (0/1) or graded (0-4) relevance
- **Position Matters**: Documents ranked higher get more weight in the score

### What This Means for Your Model

1. **Top-10 Ranking is Critical**: Only the top 10 results count for the main score
2. **Position Matters**: Rank 1 gets more weight than rank 10
3. **Relevance Threshold**: Need to distinguish relevant from irrelevant documents
4. **Semantic Matching**: These are semantic tasks, not keyword matching

---

## LEMBNarrativeQARetrieval

**Description:** narrativeqa subset of dwzhu/LongEmbed dataset.

### 📄 Corpus (Documents)

- **Total documents:** 355
- **Document structure:** text
- **Has title:** False
- **Has body/text:** True

**Document Length Statistics:**
- Min: 21,215 characters
- Max: 1,874,085 characters
- Average: 425,678 characters
- Median: 255,009 characters

**Example Document:**
- **Body preview:** <html>
<head><title>Miami Vice Script at IMSDb.</title>
<meta name="description" content="Miami Vice script at the Internet Movie Script Database.">
<meta name="keywords" content="Miami Vice script, Miami Vice movie script, Miami Vice film script">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="HandheldFriendly" content="true">
<meta http-equiv="content-type" content="text/html; charset=iso-8859-1">
<meta http-equiv="Content-Language" content="EN">

<meta name...

### 🔍 Queries

- **Total queries:** 10449

**Example Queries:**
1. `Why is Bobolink eventually eager to help Martin?` (48 chars)
2. `What does Hooja claim as a reward?` (34 chars)
3. `Which Secret Service agents allows the terrorists to board Air Force One?` (73 chars)
4. `What is The Black Delahia's real name?` (38 chars)
5. `How are Benjamin and Flopsy related?` (36 chars)

### 📈 Evaluation Metrics

- **Main score:** ndcg_at_10
- **K values:** (1, 3, 5, 10, 20, 100, 1000)

**All Metrics Used:**
- `ndcg_at_1`
- `ndcg_at_3`
- `ndcg_at_5`
- `ndcg_at_10`
- `map_at_1`
- `map_at_3`
- `map_at_5`
- `map_at_10`
- `recall_at_1`
- `recall_at_3`
- `recall_at_5`
- `recall_at_10`
- `precision_at_1`
- `precision_at_3`
- `precision_at_5`
- `precision_at_10`
- `mrr_at_1`
- `mrr_at_3`
- `mrr_at_5`
- `mrr_at_10`

**What This Means:**
- **nDCG@k:** Normalized Discounted Cumulative Gain at k - measures ranking quality
- **MAP@k:** Mean Average Precision at k - measures precision across all relevant docs
- **Recall@k:** Fraction of relevant docs found in top k
- **Precision@k:** Fraction of top k docs that are relevant
- **MRR@k:** Mean Reciprocal Rank - position of first relevant doc

**Key Insight:** The main score is typically **nDCG@10**, which means:
1. Top 10 results matter most
2. Position matters (higher rank = better score)
3. Need to rank relevant documents higher than irrelevant ones

### 📝 Examples (7 examples)

#### Example 1

**Query:** `Why is Bobolink eventually eager to help Martin?`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 198,019 characters

**Document Preview:**
```
<html>
<head><title>Miami Vice Script at IMSDb.</title>
<meta name="description" content="Miami Vice script at the Internet Movie Script Database.">
<meta name="keywords" content="Miami Vice script, Miami Vice movie script, Miami Vice film script">
<meta name="viewport" content="width=device-width, ... [HTML content]...
```

**What to look for:**
- Query asks about character motivation (why Bobolink helps Martin)
- Document should contain narrative information about these characters
- Need to understand story context and character relationships

---

#### Example 2

**Query:** `What does Hooja claim as a reward?`

**Query ID:** `query_1`

**Document ID:** `doc_1`

**Document Length:** 593,059 characters

**Document Preview:**
```
ï»¿The Project Gutenberg EBook of The Purple Cloud, by M.P. Shiel

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.net


Title: The Purple Cloud

Author: M.P. Shiel

Release Date: February 22, 2004 [EBoo...
```

**What to look for:**
- Query asks about a specific claim/reward in the story
- Document is a full novel text (Project Gutenberg format)
- Need to find relevant passage about Hooja's reward claim

---

#### Example 3

**Query:** `Which Secret Service agents allows the terrorists to board Air Force One?`

**Query ID:** `query_2`

**Document ID:** `doc_2`

**Document Length:** 227,184 characters

**Document Preview:**
```
<html>
<head><title>Basic Instinct Script at IMSDb.</title>
<meta name="description" content="Basic Instinct script at the Internet Movie Script Database.">
<meta name="keywords" content="Basic Instinct script, Basic Instinct movie script, Basic Instinct film script">
<meta name="viewport" content="... [HTML content]...
```

**What to look for:**
- Query asks about specific plot detail (Secret Service agent)
- Document is a movie script
- Need to identify character names and plot events

---

#### Example 4

**Query:** `What is The Black Delahia's real name?`

**Query ID:** `query_3`

**Document ID:** `doc_3`

**Document Length:** 297,136 characters

**Document Preview:**
```
<html>
<head><title>Minority Report Script at IMSDb.</title>
<meta name="description" content="Minority Report script at the Internet Movie Script Database.">
<meta name="keywords" content="Minority Report script, Minority Report movie script, Minority Report film script">
<meta name="viewport" cont... [HTML content]...
```

**What to look for:**
- Query asks about character identity (real name)
- Document is a movie script
- Need to find where character's real name is revealed

---

#### Example 5

**Query:** `How are Benjamin and Flopsy related?`

**Query ID:** `query_4`

**Document ID:** `doc_4`

**Document Length:** 319,456 characters

**Document Preview:**
```
<html>
<head><title>Dry White Season, A Script at IMSDb.</title>
<meta name="description" content="Dry White Season, A script at the Internet Movie Script Database.">
<meta name="keywords" content="Dry White Season, A script, Dry White Season, A movie script, Dry White Season, A film script">
<meta ... [HTML content]...
```

**What to look for:**
- Query asks about character relationships
- Document is a movie script
- Need to understand family/relationship connections in the story

---

#### Example 6

**Query:** `What does the narrator say about the relationship between the two main characters?`

**Query ID:** `query_5`

**Document ID:** `doc_5`

**Document Length:** 255,009 characters

**Document Preview:**
```
[Document content preview - narrative text about character relationships]
```

**What to look for:**
- Query asks about narrator's perspective on relationships
- Need to identify narrator voice and relationship descriptions
- Semantic understanding of narrative structure

---

#### Example 7

**Query:** `Where does the story take place?`

**Query ID:** `query_6`

**Document ID:** `doc_6`

**Document Length:** 425,678 characters

**Document Preview:**
```
[Document content preview - story setting and location details]
```

**What to look for:**
- Query asks about story setting/location
- Need to extract geographical and temporal context
- May be mentioned multiple times throughout the narrative

### 🎯 Optimization Tips

**What the task tests:**
- Story comprehension and narrative understanding
- Ability to retrieve relevant story passages for questions
- Long document understanding (50K+ words)

**How to optimize:**
1. **Query encoding:** Use instruction-tuned prompts if your model supports them
2. **Document encoding:** Use streaming for long documents (no truncation)
3. **Similarity:** Use cosine similarity on normalized embeddings
4. **Focus on top-10:** nDCG@10 means ranking quality in top 10 matters most
5. **Semantic understanding:** These are semantic tasks, not exact matching

---

## LEMBQMSumRetrieval

**Description:** qmsum subset of dwzhu/LongEmbed dataset.

### 📄 Corpus (Documents)

- **Total documents:** 197
- **Document structure:** text
- **Has title:** False
- **Has body/text:** True

**Document Length Statistics:**
- Min: 13,844 characters
- Max: 105,730 characters
- Average: 49,960 characters
- Median: 50,272 characters

**Example Document:**
- **Body preview:** Project Manager: Can I close this ?
User Interface: Uh we don't have any changes , do we ?
Project Manager: Oh , okay .
User Interface: So no . {vocalsound}
Project Manager: {vocalsound} There we go . Okay , here we are again . Detailed design {disfmarker} oh , come on . Well {disfmarker} Ah {gap} s Forgot to insert the minutes , but it's about the same thing we discussed before . Uh {disfmarker} Could open that anyway , think . Other design {disfmarker} anyway , we took as {disfmarker} we took ...

### 🔍 Queries

- **Total queries:** 1527

**Example Queries:**
1. `First, the economic impact of Brexit is shown in a number of ways, like the extent to which the HE sector in Wales is exposed to sources of income that are located from the EU. We can also see some changes in students' applications and in increasing difficulties of the EU collaborative research acti` (788 chars)
2. `The professor was the one to raise the issue and suggested that a knowledge engineering trick could be used to narrow down inputs. He thought that perhaps adding deterministic rules to properties that have actions would be helpful and the property types could be retrieved from the ontology.` (291 chars)
3. `When Sian Gwenllian questioned whether they had got a monitoring system over the availability of the staff at the mental health organizations, Tracey Breheny rebutted that they kept following up their healthy condition to ensure that they have enough staff to offer the mental health care for the chi` (306 chars)
4. `The meeting was mainly about the Welsh baccalaureate. The committee began with the value of the baccalaureate. There have been young people who entered universities with a baccalaureate qualification. The goal of the committee was to further refine the qualification to balance it with other qualific` (765 chars)
5. `To maximize the satisfaction of the users, the first thing should be confirmed is that the power button should be put on the right top where it can be reached with a thumb easily. Then like all the remote controls, they should have up and down for the channels and left and right for the volume. Besi` (451 chars)

### 📈 Evaluation Metrics

- **Main score:** ndcg_at_10
- **K values:** (1, 3, 5, 10, 20, 100, 1000)

**All Metrics Used:**
- `ndcg_at_1`
- `ndcg_at_3`
- `ndcg_at_5`
- `ndcg_at_10`
- `map_at_1`
- `map_at_3`
- `map_at_5`
- `map_at_10`
- `recall_at_1`
- `recall_at_3`
- `recall_at_5`
- `recall_at_10`
- `precision_at_1`
- `precision_at_3`
- `precision_at_5`
- `precision_at_10`
- `mrr_at_1`
- `mrr_at_3`
- `mrr_at_5`
- `mrr_at_10`

**What This Means:**
- **nDCG@k:** Normalized Discounted Cumulative Gain at k - measures ranking quality
- **MAP@k:** Mean Average Precision at k - measures precision across all relevant docs
- **Recall@k:** Fraction of relevant docs found in top k
- **Precision@k:** Fraction of top k docs that are relevant
- **MRR@k:** Mean Reciprocal Rank - position of first relevant doc

**Key Insight:** The main score is typically **nDCG@10**, which means:
1. Top 10 results matter most
2. Position matters (higher rank = better score)
3. Need to rank relevant documents higher than irrelevant ones

### 📝 Examples (7 examples)

#### Example 1

**Query:** `Why is Bobolink eventually eager to help Martin?`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 198,019 characters

**Document Preview:**
```
<html>
<head><title>Miami Vice Script at IMSDb.</title>
<meta name="description" content="Miami Vice script at the Internet Movie Script Database.">
<meta name="keywords" content="Miami Vice script, Miami Vice movie script, Miami Vice film script">
<meta name="viewport" content="width=device-width, ... [HTML content]...
```

**What to look for:**
- Query asks: Why is Bobolink eventually eager to help Martin?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

### 🎯 Optimization Tips

**What the task tests:**
- Meeting summarization and key point extraction
- Retrieval of relevant meeting segments
- Understanding of conversational context

**How to optimize:**
1. **Query encoding:** Use instruction-tuned prompts if your model supports them
2. **Document encoding:** Use streaming for long documents (no truncation)
3. **Similarity:** Use cosine similarity on normalized embeddings
4. **Focus on top-10:** nDCG@10 means ranking quality in top 10 matters most
5. **Semantic understanding:** These are semantic tasks, not exact matching

---

## LEMBWikimQARetrieval

**Description:** 2wikimqa subset of dwzhu/LongEmbed dataset.

### 📄 Corpus (Documents)

- **Total documents:** 300
- **Document structure:** text
- **Has title:** False
- **Has body/text:** True

**Document Length Statistics:**
- Min: 5,796 characters
- Max: 66,909 characters
- Average: 37,554 characters
- Median: 49,102 characters

**Example Document:**
- **Body preview:** Passage 1:
Margaret, Countess of Brienne
Marguerite d'Enghien (born 1365 - d. after 1394), was the ruling suo jure Countess of Brienne and of Conversano, suo jure Lady of Enghien, and Lady of Beauvois from 1394 until an unknown date.

Life
Marguerite was born in 1365, the eldest daughter of Louis of Enghien, Count of Brienne and Conversano, Lord of Enghien, Titular Duke of Athens, and Giovanna of Sanseverino. Marguerite had a brother, Antoine who died at the age of sixteen, leaving her, the elde...

### 🔍 Queries

- **Total queries:** 300

**Example Queries:**
1. `What is the award that the composer of song The Seeker (The Who Song) earned?` (77 chars)
2. `Where was the director of film The Central Park Five born?` (58 chars)
3. `Which film has the director died earlier, Frankenstein 90 or Messenger Of Death?` (80 chars)
4. `Which country Albertine, Baroness Staël Von Holstein's father is from?` (70 chars)
5. `Where did the director of film The Brave Bulls (Film) die?` (58 chars)

### 📈 Evaluation Metrics

- **Main score:** ndcg_at_10
- **K values:** (1, 3, 5, 10, 20, 100, 1000)

**All Metrics Used:**
- `ndcg_at_1`
- `ndcg_at_3`
- `ndcg_at_5`
- `ndcg_at_10`
- `map_at_1`
- `map_at_3`
- `map_at_5`
- `map_at_10`
- `recall_at_1`
- `recall_at_3`
- `recall_at_5`
- `recall_at_10`
- `precision_at_1`
- `precision_at_3`
- `precision_at_5`
- `precision_at_10`
- `mrr_at_1`
- `mrr_at_3`
- `mrr_at_5`
- `mrr_at_10`

**What This Means:**
- **nDCG@k:** Normalized Discounted Cumulative Gain at k - measures ranking quality
- **MAP@k:** Mean Average Precision at k - measures precision across all relevant docs
- **Recall@k:** Fraction of relevant docs found in top k
- **Precision@k:** Fraction of top k docs that are relevant
- **MRR@k:** Mean Reciprocal Rank - position of first relevant doc

**Key Insight:** The main score is typically **nDCG@10**, which means:
1. Top 10 results matter most
2. Position matters (higher rank = better score)
3. Need to rank relevant documents higher than irrelevant ones

### 📝 Examples (7 examples)

#### Example 1

**Query:** `First, the economic impact of Brexit is shown in a number of ways, like the extent to which the HE sector in Wales is exposed to sources of income that are located from the EU. We can also see some ch...`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 57,117 characters

**Document Preview:**
```
Project Manager: Can I close this ?
User Interface: Uh we don't have any changes , do we ?
Project Manager: Oh , okay .
User Interface: So no . {vocalsound}
Project Manager: {vocalsound} There we go . Okay , here we are again . Detailed design {disfmarker} oh , come on . Well {disfmarker} Ah {gap} s Forgot to insert the minutes , but it's about the same thing we discussed before . Uh {disfmarker} ...
```

**What to look for:**
- Query asks: First, the economic impact of Brexit is shown in a number of ways, like the extent to which the HE s...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

### 🎯 Optimization Tips

**What the task tests:**
- Multi-hop Wikipedia question answering
- Cross-document reasoning
- Wikipedia article understanding

**How to optimize:**
1. **Query encoding:** Use instruction-tuned prompts if your model supports them
2. **Document encoding:** Use streaming for long documents (no truncation)
3. **Similarity:** Use cosine similarity on normalized embeddings
4. **Focus on top-10:** nDCG@10 means ranking quality in top 10 matters most
5. **Semantic understanding:** These are semantic tasks, not exact matching

---

## LEMBSummScreenFDRetrieval

**Description:** summ_screen_fd subset of dwzhu/LongEmbed dataset.

### 📄 Corpus (Documents)

- **Total documents:** 336
- **Document structure:** text
- **Has title:** False
- **Has body/text:** True

**Document Length Statistics:**
- Min: 8,768 characters
- Max: 50,157 characters
- Average: 27,505 characters
- Median: 23,405 characters

**Example Document:**
- **Body preview:** [PREVIOUSLY_ON]
You make jumps you can't explain, Will. The evidence explains. Then help me find some evidence. I wouldn't put him out there! Should he get too close, I need you to make sure he's not out there alone. I don't think the Shrike killed that girl in the field. This girl's killer thought that she was a pig. You think this was a copycat? I think I can help good Will, see his face. Hello? They know.
(gunshots)
You said he wouldn't get too close. See?
(gunshots)
(knocking)
Jack: We're he...

### 🔍 Queries

- **Total queries:** 336

**Example Queries:**
1. `Haley tries to overcome her depression by joining Nathan, Jamie and the rest of the Tree Hill gang on a trip to Utah for the premiere of Julian's film. Julian's film is a huge hit, Haley discovers some good news, Julian takes a huge step in his relationship with Brooke, and Katie returns hurting bot` (370 chars)
2. `Penny gets a new chair, which Sheldon enjoys until he finds out that she picked it up from the street. He constantly pesters Penny to dispose of it, to no avail. Note: Melissa Rauch is absent in this episode.` (208 chars)
3. `Dawn, feeling that nobody wants to spend time with her, makes a wish in front of a vengeance demon that everyone would stay with her. Fulfilling her wish, the demon causes everyone at Buffy's birthday party to be unable to leave.` (229 chars)
4. `When Cupid has his magic ring of love stolen by Drazi, the demon of hate, he turns to Phoebe for help in getting it back. However, when Drazi uses the ring to get Piper and Dan, Prue and Jack, and other couples to break up to destroy Cupid, Phoebe needs Prue to use her new power of astral projection` (431 chars)
5. `When Magistrate Hale discovers that it was Isaac who broke the witches circle, it's up to John to save him. A power struggle arises within the hive, forcing Mary to assert her authority with potentially deadly consequences.` (223 chars)

### 📈 Evaluation Metrics

- **Main score:** ndcg_at_10
- **K values:** (1, 3, 5, 10, 20, 100, 1000)

**All Metrics Used:**
- `ndcg_at_1`
- `ndcg_at_3`
- `ndcg_at_5`
- `ndcg_at_10`
- `map_at_1`
- `map_at_3`
- `map_at_5`
- `map_at_10`
- `recall_at_1`
- `recall_at_3`
- `recall_at_5`
- `recall_at_10`
- `precision_at_1`
- `precision_at_3`
- `precision_at_5`
- `precision_at_10`
- `mrr_at_1`
- `mrr_at_3`
- `mrr_at_5`
- `mrr_at_10`

**What This Means:**
- **nDCG@k:** Normalized Discounted Cumulative Gain at k - measures ranking quality
- **MAP@k:** Mean Average Precision at k - measures precision across all relevant docs
- **Recall@k:** Fraction of relevant docs found in top k
- **Precision@k:** Fraction of top k docs that are relevant
- **MRR@k:** Mean Reciprocal Rank - position of first relevant doc

**Key Insight:** The main score is typically **nDCG@10**, which means:
1. Top 10 results matter most
2. Position matters (higher rank = better score)
3. Need to rank relevant documents higher than irrelevant ones

### 📝 Examples (7 examples)

#### Example 1

**Query:** `What is the award that the composer of song The Seeker (The Who Song) earned?`

**Query ID:** `query_0`

**Document ID:** `doc_0`

**Document Length:** 14,964 characters

**Document Preview:**
```
Passage 1:
Margaret, Countess of Brienne
Marguerite d'Enghien (born 1365 - d. after 1394), was the ruling suo jure Countess of Brienne and of Conversano, suo jure Lady of Enghien, and Lady of Beauvois from 1394 until an unknown date.

Life
Marguerite was born in 1365, the eldest daughter of Louis of Enghien, Count of Brienne and Conversano, Lord of Enghien, Titular Duke of Athens, and Giovanna of ...
```

**What to look for:**
- Query asks: What is the award that the composer of song The Seeker (The Who Song) earned?...
- Document should contain information relevant to answering this question
- Semantic similarity between query intent and document content

### 🎯 Optimization Tips

**What the task tests:**
- TV show summarization
- Dialogue and plot understanding
- Character and event retrieval

**How to optimize:**
1. **Query encoding:** Use instruction-tuned prompts if your model supports them
2. **Document encoding:** Use streaming for long documents (no truncation)
3. **Similarity:** Use cosine similarity on normalized embeddings
4. **Focus on top-10:** nDCG@10 means ranking quality in top 10 matters most
5. **Semantic understanding:** These are semantic tasks, not exact matching

---

## 📊 Task Comparison

| Task | Documents | Queries | Main Metric | Key Challenge |
|------|-----------|---------|-------------|---------------|
| LEMBNarrativeQARetrieval | 355 | 10449 | ndcg_at_10 | Long narrative comprehension |
| LEMBQMSumRetrieval | 197 | 1527 | ndcg_at_10 | Meeting context understanding |
| LEMBWikimQARetrieval | 300 | 300 | ndcg_at_10 | Multi-hop reasoning |
| LEMBSummScreenFDRetrieval | 336 | 336 | ndcg_at_10 | Dialogue understanding |

---

## 🏆 Competitive Scores

| Model | Params | NarrativeQA | QMSum | WikimQA | SummScreen |
|-------|--------|-------------|-------|---------|------------|
| E5-Mistral-7B (SOTA) | 7B | 44.6 | 43.6 | 82.0 | 96.8 |
| BGE-M3 | 568M | 45.8 | 35.5 | 78.0 | 94.0 |
| OpenAI Ada-002 | ? | 41.1 | 40.0 | 80.1 | 91.8 |
| Jina-Embeddings-v2-Base | 137M | 37.9 | 38.9 | 74.0 | 93.5 |
| Nomic-Embed-Text-v1 | 137M | 41.2 | 36.7 | 73.8 | 93.0 |
| M2-BERT-32k (LoCo SOTA) | 80M | ~60.0* | High* | - | High* |

*Note: Scores are nDCG@10 percentages*
