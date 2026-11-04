# Speaking script: 14 slides

Deck: `Thesis_Defence_Menon_v4.pptx`. Audience: Prof. Mauthe and Christopher Latz.

**How to use this.** Speak the indented text in quotes. Lines starting with *Heads off* are not spoken: they name the doubt the passage answers before anyone asks, with the matching entry in `QA_prep.md`. Timings are measured from the spoken words at about 144 words per minute. Bold words are the ones to stress.

| # | Slide | Part | Words | Time | Running total |
|---|---|---|---|---|---|
| 1 | Title | Intro | 34 | 0:15 | 0:15 |
| 2 | Motivation: three dimensions, hypothesis | Intro | 175 | 1:15 | 1:30 |
| 3 | Related work and gap | Intro | 241 | 1:40 | 3:10 |
| 4 | Research questions | Intro | 99 | 0:40 | 3:50 |
| 5 | Data and representation | Method | 351 | 2:25 | 6:15 |
| 6 | Architecture | Method | 185 | 1:15 | 7:30 |
| 7 | Design decisions | Method | 193 | 1:20 | 8:50 |
| 8 | Detection and gate (RQ2, RQ4) | Results | 185 | 1:15 | 10:05 |
| 9 | Ablation and substitution (RQ1, RQ2) | Results | 250 | 1:45 | 11:50 |
| 10 | Baselines under attack (RQ3) | Results | 206 | 1:25 | 13:15 |
| 11 | Unseen source (RQ5) | Results | 131 | 0:55 | 14:10 |
| 12 | Interpretability (RQ4) | Results | 130 | 0:55 | 15:05 |
| 13 | Conclusion | Conclusion | 169 | 1:10 | 16:15 |
| 14 | Thank you | Conclusion | 8 | 0:05 | 16:20 |

**Split:** introduction 3:50, methodology 5:00, results 6:15, conclusion 1:15. Methodology and results together take 11:15 of 16:20. Total 2357 spoken words at about 144 words a minute.

**Checkpoints.** When slide 8 appears you should be near 8:50, and when slide 11 appears near 13:15. If you are more than 30 seconds behind, use the *Shorter* line given for slides 3, 5 and 9.

---

## 1 · Title (0:15)

> "Good morning. My thesis is about detecting malware from what it **does** when it runs, in a way that still works when an attacker disguises that behaviour. Four parts: motivation, method, results, and conclusion."

---

## 2 · Motivation: three dimensions and the hypothesis (1:15)

> "Behavioural analysis records what a sample does in a sandbox. Behaviour has **three dimensions**. **Structure**: what a program does and what it acts on. **Meaning**: what each action is for. And **timing**: the order and pace in which it all unfolds.
>
> Most detectors read one, and each has a blind spot. A model that reads only structure is left guessing between a benign backup and a ransomware encryption phase: both write many files and call cryptographic APIs, and what separates them is largely **pace**. A model that reads only call names is defeated by API substitution: same behaviour, different symbols. And a flat sequence loses the topology of a multi-process loader.
>
> That gives the **hypothesis** of this thesis. If the weakness is always the dimension a detector ignores, then a detector that weighs structure, meaning and timing **together** leaves no single dimension whose manipulation is enough to slip past it. So the question for the method is how to read all three in **one model**, and decide for each sample how much each should count."

*Heads off:* "why does this matter?" Section 1.1 adds the asymmetry: a missed threat can encrypt or exfiltrate data, while a false alarm costs analyst time.

---

## 3 · Related work and gap (1:40)

> "Table 2.1 places existing work against the three dimensions. **Signatures** read none, **static machine learning** a little structure, and **sequential models** order, with calls as bare tokens.
>
> **HetGNN**, Zhang et al., models typed structure; the **Temporal Graph Network**, Rossi et al., adds time; **BEACON**, Perera and Jiang, reads meaning through language-model embeddings; each leaves out the rest.
>
> **GraphShield**, Amer et al., is the closest to this work, and it also uses a language model, BERT. But it uses BERT only once, **before** training, to sort similar call names into **groups**. After that, each node in its graph stands for a group, and the classifier learns from which groups appear and how they connect. The meaning BERT computed is used for sorting, then **dropped**. That matters: a renamed call, or one the grouping never saw, reaches its classifier as a symbol it has not learned, so the substitution meaning was meant to defeat **gets through anyway**, and their own paper concedes that API obfuscation can bypass it. Here, the meaning vector **is** the node's feature, so a renamed call lands next to its equivalent. It also has order but not pace. I compare what reaches each classifier; I did not re-run it.
>
> So the **gap**: each dimension is modelled well somewhere, but none carries all three into the decision. Two secondary gaps are on the slide: attribution, and separating how much a dimension is **used** from whether the decision **survives** without it."

*If pressed for evidence* (not timed; QA A2b): RQ1, where the decision follows meaning (0.7 vs 5.3 percent), and the out-of-context rename, where the framework missed none of 125. Grouping is a sensible way to shrink a vocabulary; the point is only where the meaning stops.

*Shorter (if behind):* drop the sentence on signatures, static ML and sequential models.

*Heads off:* A2 (GraphShield is "limited", not absent; the one-line version is "they use meaning to tidy the input, I use it to make the decision"), A3 (no direct re-run: the comparison is by paradigm, through the baselines).

---

## 4 · Research questions (0:40)

> "The hypothesis breaks into five questions, each with its experiment on the right. RQ1: does meaning help detection generalise to obfuscated or unseen behaviour? RQ2: do the three dimensions complement each other? RQ3, the direct test of the hypothesis: how does it compare with single-paradigm baselines, clean and **under attack**? RQ4: can a decision be attributed? RQ5: does detection hold on an independent source? Answering them needs a model that keeps the **meaning** of every call, the **typed relations** between entities, and the **timing** between calls, and that sets their balance per sample. That is what the method builds."

---

## 5 · Data and representation (2:25)

> "The corpus is **9,333** behavioural graphs from sandbox execution traces, in three classes: ransomware, other malware and benign, split 70, 15, 15, stratified. The primary source is the Carpenter and Luo multi-stage malware reports from the CAPE sandbox. I supplemented it from Hybrid Analysis: **1,987** benign samples across seven software categories, and **110** threats from a different sandbox and acquisition route, which never touch training; they are the RQ5 test.
>
> Each trace becomes one **heterogeneous** graph, with five node types and five relation types. A **node type** is the kind of thing in the graph: an **API call**, a **process**, a **file**, a **registry key**, or a **network endpoint**. A **relation type** is the kind of link between two of them: **temporal**, one call followed by the next; **executes**, a process issued a call; **operates on**, a call touched a file; **modifies**, a call changed a registry key; and **connects to**, a call contacted a network endpoint. Every API call is a node, and the temporal edges between consecutive calls form the spine of the graph; processes, files, registry keys and network endpoints attach to the calls that touch them.
>
> Why does typing matter? In an ordinary, homogeneous graph every link just means "connected", so a file write and a process spawn look like the same kind of edge, and the distinctions that carry much of a behaviour's meaning collapse. Here each relation type is learned with **its own weights**: its own hybrid GAT plus SAGE operator, two layers, holding 87 percent of the 16.1 million parameters. Inside each operator a gate *g*, between 0 and 1 for every node and every feature, mixes the two: **h = g ⊙ h_GAT + (1 − g) ⊙ h_SAGE**, and it sets *g* by reading both outputs and how much they disagree (Eq. 4.1–4.2). And it pays off in the results: the ablation finds that one relation, a process **executing** a call, is the only one whose removal costs threats; the file, registry and network relations can each be removed without changing a single decision.
>
> Every API node keeps the **768-dimensional MPNet embedding** of its name, plus seven metadata features, so meaning stays **on the node**, unlike GraphShield. And every temporal edge carries **14 timing features**: pacing, bursts, regularity, timeline position and process context. Graphs reach 1.5 million API nodes, so very large ones are compressed and capped, keeping the cross-type edges and timing spikes."

*Side note: if they ask "what is the difference between a homogeneous and a heterogeneous graph?"* (not timed; QA C12)

- **Homogeneous:** every node is the same kind of thing and every edge means just "connected", so one set of weights processes every link. A file write and a process spawn become the same kind of edge, which "collapses semantically meaningful distinctions" (§2.1.4, L532–536).
- **Heterogeneous:** every node and every edge carries a **type label** (Eq. 3.4, §3.3): here five node types and five relation types. The model keeps a **separate message function for each relation**, so it learns a process spawning a call differently from a call writing a file (§2.1.5).
- **One-line version:** "A homogeneous graph knows *that* two things are connected; a heterogeneous graph also knows *how*."
- **Why it matters here:** each relation's own weights are most of the 16.1 M parameters, and typing is what let the ablation isolate the process → API relation as the only one whose removal costs threats.

*Shorter:* drop the last sentence.

*Heads off:* the "5 node types · 5 typed relations" card (defined and justified above; schema in Table 4.1, argument in §2.2), F8 (why MPNet works on API names), F5 (the 1.5 M vs 1.4 × 10⁵ vs 30,000 sizes), B3 (provenance shortcut; returns on slide 13).

---

## 6 · Architecture (1:15)

> "The model is a single end-to-end network of about **16.1 million parameters**, the numbers it learns in training. The language model is not among them: MPNet runs once beforehand and its vectors are stored on the nodes. Most of the parameters, about 87 percent, sit in the structural message passing, because every relation type gets its own weights. From there, three branches read the same graph at different depths.
>
> The **semantic** branch is the mean of the projected API states **before** any message passing, so its gate weight reads meaning on its own. The **structural** branch runs two layers of hybrid GAT and SAGE message passing over the typed relations. The **temporal** branch runs a bidirectional LSTM over the message-passed call states in time order, with attention pooling. So the branches are not separate models: the temporal branch reads states that structure has already shaped.
>
> An **adaptive gate** then weighs the three branch embeddings for every sample, and a shared trunk with behavioural statistics feeds the class heads. It is trained on three classes and evaluated as threat versus benign; the next slide explains why.
>
> Follow one sample along the numbers in the boxes: each call arrives as 775 numbers and each step as 14 timing numbers; every node is projected into its own space; the three branches each condense the sample into 128 numbers; the gate weighs them, for example 0.66, 0.15 and 0.19, and adds them into one vector; the trunk adds 15 volume statistics, the heads give three class probabilities, and the highest decides."

*Heads off:* E4 (where the 16.1 M parameters sit), F1 (not an ensemble: one loss, shared projections, coupled branches), C1 (why the semantic branch is a mean), C3 (bidirectional is valid because the whole report is classified).

---

## 7 · Design decisions (1:20)

> "Three decisions shape the model. The first two are the contribution.
>
> First, the **hybrid aggregator**. Attention finds the single decisive edge, a process spawn among thousands of routine reads, but its weights are least reliable on unfamiliar neighbourhoods. GraphSAGE generalises to unseen entities but weighs all neighbours equally. I extended SAGE to use edge features, and a gate reads where the two disagree and decides, **per node and per feature**, which to trust.
>
> Second, the **adaptive fusion**. The dimension that decides a case varies: one sample gives itself away by what it calls, another by how fast it acts. So the gate sets the balance **per sample**, never one fixed blend, and the same weights are read out as attribution. Branch dropout and a soft entropy penalty stop it collapsing onto one branch.
>
> Third, **three-class training with a binary verdict**. Ransomware has its own timing signature, encryption bursts and mass file changes, which one merged threat class would dilute. Three classes force the model to learn what tells the threats apart, which matters most for threats it has never seen. The deployed question is binary, so either malicious class counts as a threat."

*Heads off:* C4 and G5 (three classes but binary reporting), F2 (does the entropy penalty force the balance? It only blocks collapse below 0.75), C7 (is the hybrid gate doing anything?).

---

## 8 · Detection and gate usage, RQ2 and RQ4 (1:15)

> "On the held-out test split of 1,384 samples, all **1,085 threats** are assigned to the threat side: **zero missed**. The cost is 16 of 299 benign samples flagged, a false-positive rate of 5.4 percent, and threat F1 is 0.993.
>
> In the three-class view, ransomware is the hardest class: recall **0.462** on only 39 samples. But look at where the errors go: all 21 misses are called **other malware, none benign**. So this is an error in naming the threat, never in detecting it, which is exactly what three-class training intended. With 39 samples that recall has a wide interval, which is why threat recall was fixed in advance as the headline.
>
> On the right, the first half of RQ2: does the model really use all three branches? Reading the gate on 440 samples, it is close to balanced overall, so **no branch is decorative**. Per class it adapts: for ransomware the gate puts **0.661** on semantics, which leads almost 99 percent of ransomware decisions; for benign software it shifts to **structure**, 0.450. This reading of the gate is also the first level of interpretability, RQ4."

*Heads off:* B1 (ransomware recall), F7 (is 5.4 % too high? Fixed argmax, no tuned threshold), F3 (the 440 are 330 held-out plus the 110 unseen), G4 (why RQ4 is marked here).

---

## 9 · Ablation and semantic substitution, RQ1 and RQ2 (1:45)

> "Second half of RQ2: what survives if a branch is removed outright? On 300 test graphs with 237 threats, the surprising result is semantics. Section 5.3 ranks it first by contribution share, 59 percent, yet removing it misses **not a single threat**; all 237 are still flagged. Its loss shows up somewhere else: the number of samples given their exact class, ransomware, other malware or benign, falls from 285 to 228. Without semantics the model still knows something is malicious, but it gets much worse at saying **which kind** of threat it is, and it flags more benign software. That costs analysts' workload, not safety. Removing structure is the only removal that misses threats, ten of them, a recall of 95.8 percent, and the finer ablation traces all ten to **one relation**, process to API. The ranking is reversed: semantics **names** the threat, structure **catches** it.
>
> So **contribution is not indispensability**. The branches overlap on whether a sample is a threat, which is why detection degrades gracefully, and they specialise on which threat: only the full model reaches 285 of 300; no single branch exceeds 279. This holds for a missing branch on clean traces; the attacked case is RQ3.
>
> On the right, RQ1: does the semantic branch understand what calls do, or has it memorised their vectors? It takes two steps. Step one: in the intact model, swapping almost every embedding barely moves anything, so there is **no over-dependence**. Step two: I silenced structure to force reliance on meaning, then swapped every embedding two ways. With a functional neighbour, meaning kept, missed threats go from 8 to 10 of the 226 threats, 4.4 percent, and 0.7 percent of all predictions change. With a random operation, meaning destroyed, missed threats rise to **21 of 226, 9.3 percent**, and 5.3 percent of predictions change. A memorising branch would be hurt equally by both. The gap shows it reads **what a call does**."

**Exact class: what 285 → 228 means** (Table 5.8, 300 test graphs: 237 threats, 63 benign)

| | Intact model | Semantic branch removed |
|---|---|---|
| Exact class correct (of 300) | **285** | **228** |
| Wrong overall | 15 | 72 |
| Threats missed (called benign) | 0 | **0** |
| False alarms (benign called threat) | 4 | 15 |
| Threats caught but given the wrong threat type* | 11 | **57** |

*Note:* **threat recall** only asks whether a threat was called a threat; ransomware labelled "other malware" still counts as caught. **Exact class** is stricter: the label must be precisely right, so that same case counts as wrong. Without semantics, wrong-type labels rise from 11 to 57 and false alarms from 4 to 15, while missed threats stay at 0. Semantics is the branch that reads what calls mean (for example, CryptEncrypt means encryption), which is what separates ransomware from other malware. \*The last row is derived by subtraction (wrong − missed − false alarms); it is not printed in Table 5.8, so say "derived" if asked.

*Shorter:* drop the sentence "This holds for a missing branch on clean traces; the attacked case is RQ3." (it is repeated on slide 13).

*Heads off:* D2 (structure alone matches threat recall on clean traces; the gain is exact class plus the attacked case), D3 (inference-time ablation, not trained single-view models), G3 (does step one show it ignores semantics?), F4 (why 237 here and 226 there), F11 ("survives" means 95.8 % at worst, not zero cost).

---

## 10 · Baselines under attack, RQ3 (1:25)

> "This is the central result. Four established detectors, each committed to one view: TF-IDF with XGBoost for call identity, a static GAT for structure, an MPNet embedding classifier for meaning, and a BiLSTM for order.
>
> On clean traces I **do not claim a lead**. XGBoost has the best macro-F1, 0.874 against the framework's 0.827, and the BiLSTM scores higher on its set while missing four threats where the framework misses none. On clean data the framework **matches** them.
>
> Then each baseline faces a **behaviour-preserving** attack aimed at its own dimension: the malware still does exactly the same thing. Renaming calls erases XGBoost: **all 275 threats missed**. Perturbing the graph drives the GAT to miss 144 of 275. An out-of-context rename makes the semantic classifier miss all 125. The BiLSTM is the one that survives its own attack, reordering, but under renaming it flags **all 200 benign traces**: it fails by crying wolf. Its zero missed threats there is not a success: a detector that calls everything a threat catches every threat trivially while distinguishing nothing.
>
> Under the same attacks, the framework misses **at most two threats** and never loses more than two points of macro-F1. An attack on one dimension leaves it two intact views and a single-view detector none. So, in the words of the abstract: on clean traces it matched them; **under attack it alone kept detecting**."

*Heads off:* D1 and G2 (no clean lead; the claim is under attack), G1 (the BiLSTM survives its own attack), F6 ("you crippled the classical baseline"), D6 (are these attacks realistic?), D7 (only one dimension attacked at a time).

---

## 11 · Unseen source, RQ5 (0:55)

> "RQ5. Before measuring detection, I measured how unfamiliar these 110 threats are, so the result cannot be memorisation. They come from a separate source with no overlap. **27.8 percent** of their distinct calls never appear in training, the Jensen-Shannon divergence between the call distributions is 0.534, and only six of the twenty most frequent calls are shared. The chart shows where it shifts: LdrGetProcedureAddress is a fifth of the unseen calls and almost absent from training.
>
> The result: **all 110 detected**, the least confident at 0.865, so none near the boundary, with 6 of 200 benign traces flagged. On these threats the gate leans more on **structure and timing**, while semantics keeps a quarter of the weight. This is one genuine shift of source and distribution, not breadth across many families."

*Heads off:* B2 (are these new families? Do not claim it), D10 (0.534 is a divergence, not scipy's distance), D9 (confidence is not the claim; zero misses is).

---

## 12 · Interpretability, RQ4 (0:55)

> "RQ4. An explanation answers two analyst questions: which **kind** of evidence carried the verdict, and **what** that evidence was. Here is one held-out malware sample, called a threat with its malware family at 0.95.
>
> Level one is the gate: it leaned on **timing**, 0.39, with structure and semantics close behind. Level two is gradient-times-input saliency on the calls: handle duplication, atom tables, socket configuration, event synchronisation, hashing and file attributes. They are spread across the trace, but each sits in a **burst**, which is why the gate weighs timing, and all are anchored to the sample's **own process**, the same relation the ablation found load-bearing.
>
> Both levels are quantities the model computed while deciding, not an explainer fitted afterwards. They are faithful to the computation, not a proof of cause."

*Heads off:* C8 (why not attention weights?), D11 (one sample: an illustration of a method that applies to every prediction), G6 (what "family probability" means), rule 4 (a gate weight is a coefficient, not a share).

---

## 13 · Conclusion (1:10)

> "**RQ1**: yes. The semantic branch reads the function of a call rather than its name, for substitutions that preserve behaviour. **RQ2**: yes. All three branches are used, any one can be lost, and only the full model names the class best. **RQ3**: on clean traces it matches the baselines; under attack it alone keeps detecting, because no single dimension carries the decision. **RQ4**: yes, at two levels, from the model's own quantities: the **gate weights** say which kind of evidence carried the verdict, and **gradient-times-input saliency** says which calls it was. Attention weights are not used as the explanation; attention shows where the model looked, not what decided. **RQ5**: yes, all 110 threats from an unseen source.
>
> The bounds, briefly. It classifies a complete trace, so the verdict comes after the program has run. The corpus is a two-source Windows snapshot with few ransomware samples, and the attacks are representative rather than adaptive, from a single training run. These qualify the **breadth** of the claims, not their validity.
>
> Every next step extends the same architecture. First, **detection on the fly**, with a continuous-time dynamic graph.
>
> The contribution, in one sentence: a **single end-to-end model** whose adaptive gate reweighs structure, meaning and timing for every decision."

*Heads off:* B8 (single run), G7 (memory horizon), G8 (why not built on the fly), E2 and E3 (Windows only, evasive samples). The full limitation and future-work lists are on the slide; let them read the rest.

---

## 14 · Thank you (0:05)

> "Thank you. I am happy to take questions."

Then move to the live demo (see the end of `QA_prep.md`).

---

## Words to avoid while speaking

- "ensemble", "combination", "unified" or "assembly" for the model. Say **one end-to-end model**.
- "real-time". Say **on the fly**.
- "better accuracy than the baselines". Say **matches them on clean traces; alone holds under attack**.
- "the temporal branch made 39 % of the decision". Say **the gate weighted timing at 0.39**.
- "obfuscation leaves runtime behaviour unchanged". Packing adds a visible unpacking stage. Say obfuscation keeps the **functionality**; the attacks in RQ3 alter the **recorded trace**.
- "new families" for the 110. Say **a new source and distribution**.
- "no single view suffices" with no condition. Say **under attack, every single-view detector fails**.
