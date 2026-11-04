# Defence Q&A preparation

Questions two careful readers of the final thesis are likely to ask, with the answer, the number to quote, and where it lives in the thesis. Every number here is taken from the submitted PDF (113 sheets, hash `05cb759f…`).

---

## Ground rules for answering

1. **Lead with the number, then the section.** "Zero missed, Table 5.2" beats a paragraph of reasoning.
2. **Never claim a clean-accuracy advantage.** Use the abstract's words for RQ3: "On clean traces it matched them. Under attack it alone kept detecting: each rival failed once the single dimension it depends on was disturbed." §5.6: the framework is "the only one of the five that continues to work".
3. **Never say "no single view suffices" without a condition.** On clean traces structure alone reaches threat recall 1.000 (Table 5.8). Say "on clean traces structure alone matches threat recall; under attack every single-view detector breaks".
4. **A gate weight is a coefficient, not a share of the decision** (§6.2). Do not say "the temporal branch made 39 % of the decision".
5. **Do not call the model a combination, assembly or unification of parts.** It is one end-to-end model; the adaptive per-sample gate is the novelty.
6. **Concede bounded limitations quickly, then state what still holds.** §6.2's last paragraph is the template: the limits qualify breadth, not validity.

---

## A. Novelty and positioning

**A1. Heterogeneous GNNs, temporal GNNs and sentence embeddings all exist. What is actually new?**
Three things, in one end-to-end model. (1) The call's meaning stays on every API node as a 768-d MPNet vector the classifier reasons over; it is never clustered away (§4.3.2). (2) A hybrid relational operator: GAT plus an edge-aware GraphSAGE I modified to inject edge features (m_uv = h_u + W_e e_uv), reconciled by a per-node, per-feature gate that reads their disagreement (Eq. 4.1–4.2). (3) An adaptive gate that sets the balance between meaning, structure and timing for each sample and doubles as attribution (Eq. 4.3). Table 2.1 shows no listed approach has all three dimensions as first-class.

**A2. GraphShield also uses BERT and a GNN. Why is it "Limited" on semantics?**
Their pipeline is BERT embeddings → clustering → graph construction; nodes are clusters of APIs, not calls. The classifier never sees a per-call semantic vector. Their own §8 lists "assigning semantic labels to nodes" and "combining GAT with LSTM" as future work, and concedes that API obfuscation can bypass detection. *Plain version:* they use meaning to tidy the input; this work uses meaning to make the decision. *If asked for an example:* CryptEncrypt and CryptEncryptEx have almost the same MPNet vector, so in this model a renamed call lands next to the original and the classifier sees that closeness directly. In a grouping approach the classifier sees which group a node belongs to, not a vector describing what the call does. Say "what reaches the classifier", never "their model would fail", because GraphShield was not run here. Two further contrasts: they encode execution **order** as graph topology, but not **pace** (order is not timing), and they explain through a **post-hoc explainer**, whereas the gate weights here are part of the decision itself. *Caveat to state if pressed:* I did not re-run GraphShield; the comparison is by paradigm, through the four baseline families.

**A2b. "So GraphShield drops the meaning after grouping. Why is that bad?"**
Because the grouping is fixed before training, from names alone. A renamed call, or one the grouping never saw, reaches the classifier as a group symbol it has not learned rather than a description of what the call does, so "the substitution that semantic modelling was meant to defeat succeeds anyway" (§2.2, L759–765). GraphShield's own §8 concedes that API obfuscation can bypass its detection. Here, the meaning vector itself reaches the classifier, so a renamed call lands next to its functional equivalent. Evidence: RQ1, where meaning-preserving swaps change 0.7 % of predictions and random swaps 5.3 % (Table 5.11), and the out-of-context rename, where the semantic baseline misses 125/125 and the framework 0 (Table 5.13). *Concede fairly:* grouping is a sensible way to shrink a vocabulary, and GraphShield was not run here; the claim is only about where meaning stops, before the decision there and inside it here.

**A3. Why not compare directly against GraphShield, TGN or HetGNN?**
The comparison is designed per paradigm (§4.6.5): each baseline commits to one view, so the gap is attributable to the view it omits. Reimplementing every published system on this corpus was out of scope; the static heterogeneous GCN/GAT is the closest structural counterpart.

**A4. Is "gate-based attribution" a fair entry in Table 2.1 when others are "partial"?**
The distinction is that the gate weights are the coefficients the model actually uses to combine branches, not an explainer fitted afterwards (§4.4.4, §5.8). It is scoped honestly in §6.2: a coefficient, not a share of the decision.

---

## B. Data and evaluation design

**B1. Ransomware recall is 0.462. Doesn't that undermine a ransomware-motivated thesis?**
Three points. (1) All 21 misses go to other malware, none to benign (Table 5.4), so every ransomware sample is still caught as a threat; the error is naming, not detection. (2) The class is recognised: the gate puts 0.661 on semantics for ransomware and semantics leads 98.7 % of ransomware decisions (Table 5.7). (3) n = 39, so the confidence interval is about 30 points wide; that is why threat recall was fixed as the headline metric in advance (§5.1.4). Training recall was 0.949 on 172 graphs (Table 5.5): learned, but too few examples to span the class. *Do not concede that the temporal branch is idle for ransomware:* its gate weight there is 0.209, a real minority stake, and removing the temporal branch costs a missed threat and an exact-class error (284 vs 285, Table 5.8).

**B2. Are the 110 "unseen" threats new families?**
Do not claim families. The thesis claims a shift of **source and distribution**: separate sandbox and acquisition route, no sample overlap, 71 of 255 distinct calls (27.8 %) never in training, JS divergence 0.534, 6 of top-20 calls shared (Table 5.14). §6.1 states it explicitly: "transfer across one genuine shift of source and distribution, not breadth across many threat families, which this corpus was not labelled to test."

**B3. Most benign samples come from the supplementary source. Couldn't the model learn "this sandbox = benign"?**
§6.2 addresses it: the 110 unseen threats came from that same sandbox and none was missed, so the decision is not a provenance shortcut. What remains open, and is stated: the benign side of the generalisation result was measured on familiar software, not on clean programs from the new source.

**B4. Random stratified splits leak near-duplicate family samples across train and test.**
Agreed, and §4.6.3 says so ("random splits over-estimate real-world performance"). That is exactly why RQ5 uses an independent source, where memorisation cannot explain detection.

**B5. Why are there two evaluation sets (475 and 325) in §5.6?**
Each baseline group was run against its own evaluation set, and the framework was run on both, so every comparison is on identical traces. Their class proportions differ from the test split, so macro-F1 there is not comparable with §5.2 (stated under Table 5.12).

**B6. Why does the framework's macro-F1 drop from 0.841 (test) to 0.827 and 0.763?**
Composition, not the model. Macro-F1 averages classes unweighted. Ransomware F1 actually rises across the sets; what falls is other-malware **precision**, because the same ransomware→other-malware confusions land on an other-malware class of 1,046, then 200, then only 50 samples.

**B7. Labels come from VirusTotal. How noisy are they?**
Vendor detections are revised over time (§6.2). The binary threat decision is robust to ransomware/other-malware label noise, which is one reason it is the headline.

**B8. Single training run, no variance. How do you know the results aren't luck?**
Concede it (§6.2), then: the headline outcomes are counts with large margins (0 of 1,085; 0 of 110; ≤ 2 of 275 under attack vs 275/275 for XGBoost). Seed repetition is in §6.3.

---

## C. Method

**C1. You argue that collapsing a trace loses structure, yet the semantic branch is a mean. Why?**
Deliberately. It is not a bag of counts: in MPNet space CryptEncrypt and CryptEncryptEx sit almost together, so renaming barely moves the centroid, whereas in a count vector they are unrelated dimensions. A mean rather than a sum stops trace length from dominating. And it is taken before message passing, so its gate weight is a reading of meaning alone and removing it removes meaning alone (§4.4.1). Structure and timing are carried by the other two branches, and the per-type behavioural statistics restore the scale that pooling removes (§4.4.5).

**C2. Why a BiLSTM rather than a transformer or Mamba?**
Maturity and stable optimisation on very long traces; attention is quadratic in length. The slot is modular (§4.4.3); Mamba and transformer comparisons are future work (§6.3). Mamba is not the reported model.

**C3. Bidirectional means future context. Isn't that unrealistic?**
For the task as defined, classification of a complete sandbox report, future context is legitimately available (§4.4.3). On-the-fly detection is the first limitation and the first future-work item.

**C4. Why train three classes but report binary?**
§4.6.1 (slide 7, third box). Learning: ransomware's temporal signature would be diluted in a merged threat class, and in a binary objective 172 ransomware graphs would be about 3.3 % of that class and could be ignored (L3395). Operational: deployment asks "threat or not", so headline results use the binary projection (Eq. 4.8). Evidence: all 21 ransomware errors go to other malware, none to benign. Full answer in G5.

**C5. Focal α is almost uniform. Why not weight ransomware higher?**
The oversampler already draws ransomware 9.0 / 1.2 ≈ 7.5× as often as other malware; favouring it twice would over-correct. The slightly higher α on other malware (0.42) compensates for it being the least oversampled (§4.5.1).

**C6. The recall floors 0.40 and 0.52 look arbitrary.**
They are selection constraints, set on validation, never on test, to stop a checkpoint that scores well in aggregate while neglecting the minority class (§4.5.2, Fig. 5.2). Be honest that they are hand-set operational thresholds, not tuned hyperparameters.

**C7. The hybrid gate's mean only moves from 0.30 to 0.28. Is the hybrid doing anything?**
The mean barely moves but the standard deviation doubles (§4.4.2): the mixture differentiates per node rather than migrating to one operator. That is the intended behaviour.

**C8. Why not use attention weights as the explanation?**
Attention shows where a model looked, not which dimension decided (§2.1.8, citing Bilot et al.). RQ4 uses gate weights and gradient × input saliency only.

**C9. Why three regularisers (dropout, temporal-feature noise, graph augmentation)? Isn't that excessive?**
Each targets a risk this architecture creates (§4.5.2). Dropout is generic. Noise on temporal edge features (p = 0.2, σ = 0.05, training only) exists because both aggregation operators consume edge features, so the model could otherwise key on exact sandbox timestamps. Graph augmentation exists because the oversampler draws ransomware graphs repeatedly; without variation they would be identical copies, which is why augmentation is applied more often to rarer classes.

**C10. Why size-aware batching instead of one small batch size?**
Graph sizes span four orders of magnitude. A fixed batch must be sized for the largest graph, which leaves every small batch far below capacity. Buckets keep per-batch memory roughly constant (64 graphs below 10k API nodes, down to one graph above 100k) without dropping any graph for being too large, which matters because the largest graphs are the loudest samples.

**C11. Isn't selecting the checkpoint on validation macro-F1 a leak?**
No. The recall floors and the selection both use validation; the test split is touched only once, after the model and all baselines are fixed (§5.1). Macro-F1 is the signal because it weights all three classes equally.

**C12. What is the difference between a homogeneous and a heterogeneous graph?**
Homogeneous: all nodes are one kind and all edges mean "connected", so one set of weights processes every link, and a file-write edge and a process-spawn edge are treated identically, which "collapses semantically meaningful distinctions" (§2.1.4). Heterogeneous: each node and edge carries a type label (Eq. 3.4, §3.3), and the model keeps a separate message function per relation (§2.1.5). Here: five node types (API call, process, file, registry key, network endpoint) and five relations (temporal, executes, operates on, modifies, connects to; Table 4.1). One line: "a homogeneous graph knows *that* two things are connected; a heterogeneous graph also knows *how*."
*Follow-up "couldn't you just add an edge-type feature to a homogeneous graph?":* simple picture: one reader with a sticky note on each message ("this is a file write") versus five specialist readers, one per relation. The note *tells* the shared reader the kind of link, but the same weights still process every relation, so the five compete for one function; the heterogeneous model *gives* each relation its own weights. Say: "An edge-type feature only labels the message; the same weights still process every relation. A heterogeneous model gives each relation its own weights, which is why the thesis calls treating all relations identically a strong and frequently inappropriate assumption" (§3.3). *Be fair if pushed:* a large enough shared function could in principle learn to treat types differently, so never say "impossible". Typed weights are a stronger built-in assumption (homogeneous graphs weaken "the inductive bias of the network", §2.1.4): the distinction is the default rather than something the model must discover, and it is what let the ablation isolate process → API.
*Follow-up "did you measure the gain from typing?":* no homogeneous variant was trained; even the GCN/GAT baselines are heterogeneous (§5.1). The evidence for typing is indirect: the relation-level ablation could isolate process → API as the only relation whose removal costs threats (Table 5.9), which a single untyped edge set could not show. Do not claim a measured improvement over a homogeneous model.

---

## D. Results

**D1. The framework isn't better on clean data. Why the extra complexity?**
Correct and stated (§5.6.1). The framework was never meant to raise clean accuracy; it removes the load-bearing dimension an attacker can target. Under behaviour-preserving attacks every single-view baseline breaks (XGBoost 275/275 missed, GAT 144/275, semantic MLP 125/125, BiLSTM 200/200 false alarms) while the framework misses at most 2 (Table 5.13). If challenged on "every", see G1: the BiLSTM survives its own attack but fails under renaming.

**D2. Structure alone gives threat recall 1.000. Then what does joint modelling add (RQ2)?**
On clean traces, nothing on the binary metric, and the thesis says so ("This holds for a branch that is missing, not one under attack", §6.1 RQ2). The gain on clean data is exact class: 285 vs at most 279 of 300 for any single branch (6 of 300). Structure alone is also alarm-prone (10 false alarms of 63 against the intact model's 4) and names worse (259 exact). Point out too that the binary task is saturated: five of the seven Table 5.8 rows miss 0 or 1 threat, so that metric cannot show complementarity. The decisive evidence is the attacked case, which the thesis assigns to RQ3: there the framework alone keeps detecting. *Precision:* the ablation was clean-trace only; the attack evidence comes from the separately trained baselines, so say "the structural baseline collapses", not "the structure-only branch collapses".

**D3. The keep-one ablation isn't a real single-view model.**
Agreed, and §4.6.6 and §5.4.2 say so: these are inference-time sensitivity analyses of one trained model. That is precisely why §5.6 trains separate single-paradigm baselines.

**D4. Contribution shares (Table 5.6) and gate weights (Table 5.7) disagree for benign. Which is right?**
Both; they measure different things. The gate measures how much the model draws on a branch while deciding; the share measures how much is lost when it is removed. For benign, the gate leans on structure (0.450) but silencing structure costs only 10.4 %: structure guides the decision but its evidence is recoverable from the other branches (§5.3.2).

**D5. Reversing or collapsing timestamps changes nothing. Is the temporal branch useless?**
No. Timing is read twice: engineered temporal features are folded into node states during message passing, and the BiLSTM reads order on top (§4.4.3). Disturbing one route leaves the other. Silencing the temporal branch outright still costs a threat (Table 5.8), and with structure silenced and semantics scrambled, timing alone holds detection above 90 % (§5.5.2).

**D6. Are the attacks realistic? Malware cannot rename what the sandbox records.**
They model real variation: wrapper and alias calls (CreateFileW vs NtCreateFile), dynamic resolution (LdrGetProcedureAddress dominates the unseen set), API-version and sandbox naming differences, reordering, junk-call padding (§5.6.2). They are behaviour-preserving by construction and deliberately cheap. *Precision:* obfuscation such as packing keeps the **functionality**, not the runtime trace, since unpacking adds a visible stage. The attacks here manipulate the **recorded trace** while the malware does the same thing; never say "packing leaves runtime behaviour unchanged". They are not an adaptive white-box adversary; that is a stated limitation and future work.

**D7. Each attack only touches one dimension, so the framework always has two clean views. Isn't that rigged?**
That is the claim being tested: an attack on one dimension leaves the framework two intact views and a single-view detector none (RQ3, §6.1). A two-dimension or adaptive attack is the natural next test. *If asked about the "combined" track in §4.6.4:* Table 5.13 reports the single-dimension attacks. Check your result files before quoting any combined-track numbers from development runs.

**D8. Why doesn't the framework break under graph perturbation when its structural branch uses GAT?**
Semantic and temporal views supply the evidence the corrupted graph no longer carries (§5.6.2), and within the structural branch the inductive SAGE path is the steadier operator under unfamiliar neighbourhoods.

**D9. You say unseen threats are detected "with margin" (lowest 0.865), but §5.5.2 says confidence stays ~0.90 even on errors. Contradiction?**
Fair. The detection claim rests on zero misses, not on confidence. The confidence figure describes the distribution; §5.5.2 is exactly why robustness is measured through paired outcome changes rather than confidence thresholds. Calibration under perturbation is listed in §6.3.

**D10. JS divergence 0.534: re-running it with scipy gives 0.731.**
`scipy.spatial.distance.jensenshannon` returns the **distance** (square root). 0.534 is the base-2 **divergence**, bounded 0–1, over the union of the two vocabularies. Both caption and §5.7.1 say "divergence".

**D11. The interpretability example is a single sample. Cherry-picked?**
It is an illustration of a method that applies to every prediction. Its faithfulness comes from using quantities the model computed (gate coefficients, gradient × input). Concede: no user study and no formal faithfulness test (such as deletion curves); §5.8 calls it "an interpretability aid … rather than a formal causal proof".

**D12. Why do 21 misses under random substitution appear with confident predictions?**
Softmax confidence on inputs unlike training is not a warning signal (§5.5.2): well-formed nonsense is consumed as evidence, whereas absent input is down-weighted. That is also why random substitution (21 missed) hurts more than zeroing (8).

---

## E. Deployment and scope

**E1. Can this run in production?** It classifies finished sandbox reports: suited to retrospective triage. Training needs a high-memory GPU (96 GB used; 80 GB minimum) because graphs reach over 1.5 M API nodes (§4.2) and each API node carries a 768-d embedding. ~16.1 M parameters.

**E2. Windows only?** Yes, every trace is Windows (§6.2). The schema is platform-agnostic, but this was not tested.

**E3. A sample that detects the sandbox?** Presents little behaviour to reason over; a limitation shared by every dynamic-analysis detector (§6.2).

**E4. What are the 16.1 million parameters, and where do they sit? Can you prove the number?**
*What they are:* the weights, the numbers each layer multiplies and adds. They start random, training adjusts them, and the final values are saved in the checkpoint. *Proof:* run `python presentation/count_parameters.py` from the repository root. It loads `results/checkpoints/best_model_1.pt`, adds up every learned weight (16,065,698), and checks one layer by hand: the 128 → 128 semantic projection has 128 × 128 weights + 128 biases = 16,512, which the script's count matches. The training code never prints the count, so the checkpoint is the evidence. Counted from the reported checkpoint (`best_model_1.pt`): 16,065,698, the "approximately 16.1 million" of §5.1. Structural message passing holds 13.99 M (87 %): two layers, five relation types, each with its own GAT (3.76 M in total), edge-aware SAGE (3.73 M) and the per-node, per-feature gate that arbitrates between them (6.50 M, the largest single part). The temporal BiLSTM and attention pooling hold 1.48 M (9 %). The projections, trunk, fusion gate (about 50 k) and three class heads hold the remaining ~0.6 M. **MPNet is not part of the count:** each API name is encoded once and the vector cached on the node (§4.3.2), so no language model is trained or run inside the detector. *If asked "is that large?":* it is small next to the MPNet encoder itself, which has roughly 110 M parameters; most of the size comes from giving every relation type its own weights, which is the point of a heterogeneous model.

---

## F. Further counters from a close reading (added 2026-10-11)

Each answer below was checked against `report_classic.tex`; the line references are to that file.

**F1. Isn't this just an ensemble or mixture of experts: three models and a weighting?**
No. Do not use the word "ensemble". It is one model with one loss, trained end to end. All three branches start from the same type-specific projections, and the branches are coupled: the temporal BiLSTM reads the API states *after* structural message passing, so it is not an independent model. The gate takes all three embeddings jointly and is trained with the classifier. An ensemble trains separate predictors and combines their outputs; here, the branches only ever produce an internal representation (§4.4).

**F2. Doesn't the entropy penalty force the near-balanced gate (0.35 / 0.35 / 0.29)? Then the "balance" is imposed, not learned.**
The penalty is a hinge: it is active only when a sample's normalised gate entropy falls *below* 0.75, with λ = 0.01, and it "deliberately does not force equal branch usage" (§4.4.4, L2448–2452). Concede honestly: it does rule out near one-hot weights (for example, 0.80 / 0.10 / 0.10 has normalised entropy ≈ 0.58 and would be penalised). Within that bound the gate moves a lot: semantic weight is 0.661 for ransomware but 0.223 for benign. Those per-class differences are learned, not imposed. Branch dropout (0.05) is the other safeguard.

**F3. Why is the gate analysed on 440 samples, when the test split has 1,384?**
The gate weights were logged during the generalisation evaluation: 330 held-out samples from the primary corpus plus the 110 unseen-source threats (§5.3.2, L3566). Do not describe the 440 as the test split.

**F4. The ablation uses 237 threats and the substitution 226. Why different samples, and why not the full test split?**
The ablation reuses the §5.3 sample of 300 test graphs (237 threats, 63 benign; L3689) so that attribution and ablation are measured on identical data. The substitution study uses an *independently drawn* 300 (226 threats, 74 benign; L4142). This is why "no structure" misses 10 in one and 8 in the other; the thesis calls the eight "in line with the ten recorded in Table 5.8 on an independent test sample" (L4172). If asked why a 300-graph sample was used rather than all 1,384, the thesis gives no reason. Answer honestly that it was a practical choice, and that the headline detection figures use the full split.

**F5. §4.2 says graphs reach 1.5 million API nodes and are capped to 30,000, yet Figure 5.1 says the largest is about 1.4 × 10⁵. Which is it?**
The figures measure different things. 1.5 M is the corpus-wide maximum (L2028). Figure 5.1 shows only the 1,356 graphs of the validation split (caption, L3060). Capping (graphs above 100,000 API nodes reduced to at most 30,000, protecting cross-type edge endpoints and time-delta spikes above the 95th percentile) is applied when a graph is loaded (`graph_construction/pytorch_dataset.py`), so stored graphs, and therefore the figure, keep their full size. The model sees the capped version. Capping is "a concession to available memory, not a modelling choice" (L3071).

**F6. You crippled the classical baseline by giving it only TF-IDF tokens.**
It is the standard representation for that family: TF-IDF unigrams and bigrams, 3,000 features. Two learners, bagged and boosted, were used to represent the paradigm "near its practical ceiling" (L3171). It is not weak: XGBoost has the *best* clean macro-F1 in the study (0.874). Its failure under renaming is a property of token-identity features, not of the tuning. The Random Forest fails the same way, with the same figures (L4477).

**F7. A 5.4 % false-positive rate is too high to deploy.**
That is 16 of 299 benign samples. Because threats outnumber benign samples, the FPR moves sharply with very few errors (L1810–1813). The binary decision uses the fixed argmax projection, with no threshold tuned on evaluation data (L3049), so no attempt was made to trade false alarms for recall. A missed threat costs far more than an analyst reviewing a false alarm (L323–326). On the unseen evaluation it is 6 of 200. Do not claim what a tuned threshold would achieve; it was not measured.

**F8. MPNet is a sentence encoder. API names are not sentences, so why should it work?**
Most identifiers are made of descriptive English fragments (CryptEncrypt, NtCreateFile), so cryptographic, file and network calls form separate neighbourhoods (§4.2, L1915–1925). Where a name carries no usable meaning, the decision rests on structure and timing. The empirical answer is RQ1: with structure silenced, a meaning-preserving swap changes 0.7 % of predictions while a random swap changes 5.3 %. That gap could not appear if the embeddings carried no functional meaning.

**F9. Why didn't you compare against R-GCN or HGT?**
The thesis does not evaluate them; say so plainly. The hybrid aggregator was chosen for a stated reason: attention finds the decisive edge, edge-aware SAGE generalises inductively, and a learned gate arbitrates between them. A direct comparison with other heterogeneous operators is a fair extension. Do not claim it would come out in your favour.

**F10. The framework's own temporal branch is a BiLSTM. Why doesn't it crash under renaming like the BiLSTM baseline?**
The baseline embeds call *tokens* from a vocabulary built on the training split alone, so a renamed call is out of vocabulary (L3178). The framework's BiLSTM reads API states that already carry the MPNet embedding of the name and the message-passed structural context, so a renamed call still lands near its functional equivalents. Under renaming, the framework misses no threats and flags 7 benign samples. The baseline flags all 200.

**F11. Slide 9 says detection survives losing any branch, but removing structure costs 10 threats.**
"Survives" means 95.8 % threat recall at worst, not zero cost. The 10 misses all trace to one relation, process → API (Fig. 5.5). The point is graceful degradation: no single removal collapses detection. Do not say that no branch matters.

**F12. Is the code available? Could someone reproduce this?**
Everything that determines the result is in the thesis: all hyperparameters (§4.5), seed 42 with deterministic execution, split protocol, and the checkpoint rule. Each experiment also saves its evaluation manifest and per-sample predictions, so every reported figure "can be regenerated exactly from the trained checkpoint" (L3211). The thesis does not state a public code release, so do not promise one on the spot. Offer to share the code with the examiners if asked.

**F13. What would you do differently with more time?**
From §6.3: repeated seeds to put an interval on every figure; an adaptive adversary optimising against the gate; breadth across sandboxes, platforms and sources; confidence calibration under perturbation; a zero-shot LLM baseline; and detection on the fly. As your own reflection (not in the thesis): a family-disjoint split and more ransomware samples, which follow from B4 and B1.

---

## G. Questions the v4 slides invite (added 2026-10-11)

These come from what the 14-slide deck now shows on screen. Each was checked against `report_classic.tex`.

**G1. Slide 10 says "every baseline breaks", but your BiLSTM survived the reordering attack aimed at it.**
Correct, and the thesis says so: the sequence model "is the one baseline that withstands the attack aimed at its own dimension", falling from 0.840 to 0.728 macro-F1 with 5 threats missed (§5.6.2, L4505). Its fragility is elsewhere: under renaming it "collapses not by missing threats but by crying wolf", flagging all 200 benign traces (macro-F1 0.136). Hence §6.1: "three of the four give way on the axis they were built around, and the fourth on a different one." Under the same rename, the framework misses no threats and flags 7 benign.

**G2. Your conclusion says the framework "alone holds under attack", yet XGBoost beats it on clean data. So is it better or not?**
Say it in the abstract's words: "On clean traces it matched them. Under attack it alone kept detecting: each rival failed once the single dimension it depends on was disturbed." Clean macro-F1 is 0.827 against XGBoost's 0.874 (Table 5.12), and you do not claim a clean lead. The claim is about manipulated traces: at most 2 missed threats and at most two points of macro-F1 lost, while every rival fails on some attack (Table 5.13). §5.6: "the only one of the five that continues to work."

**G3. Slide 9, step 1: random substitution barely moves the intact model (Table 5.10). Doesn't that show the model simply ignores semantics?**
That is exactly the ambiguity the thesis names, and why step 2 exists. On its own, the intact result "settles nothing: a detector leaning on structure instead would look exactly the same" (§6.1 RQ1). Silencing structure removes that alternative. The model then separates the two regimes: 0.7 % of predictions change under meaning-preserving swaps and 5.3 % under random ones (Table 5.11). A branch that ignored semantics would show no gap. Step 1 shows no over-dependence; step 2 shows real use.

**G4. Slide 8 is labelled RQ4 as well as RQ2. Why is a per-class gate average interpretability evidence?**
§6.1 RQ4 names two levels; the first is "established in Section 5.3 [and] reads the fusion gate to name the behavioural dimension that carried a verdict". The per-class means on slide 8 are that level-1 reading, aggregated. Slide 12 shows both levels for a single decision.

**G5. Slide 7: why train three classes if you only report binary? Isn't that inconsistent?**
§4.6.1 calls it "a deliberate design decision rather than an inconsistency". Learning: ransomware has its own temporal signature (encryption bursts, rapid mass file modification) that one merged threat class would dilute, and three classes force the model to learn what separates the two threat classes. A binary objective can be minimised with the patterns most malware shares; 172 ransomware graphs would be about 3.3 % of a merged class (L3395). Operational: the deployed question is binary, so either malicious class counts as threat (Eq. 4.8). Evidence: all 21 misclassified ransomware were called other malware, none benign (Table 5.4), and all 110 unseen threats were detected.

**G6. Slide 12 says "family probability 0.95". Is that the threat probability?**
It is the probability the model assigns to the predicted malware class: "classifies it as a threat and assigns its malware family a probability of 0.95" (§5.8). Under the binary projection (Eq. 4.8), that class counts as a threat.

**G7. Limitation "memory horizon": can a BiLSTM really use a trace of 30,000 calls?**
Concede it (§6.2): its effective memory is finite, so dependencies between distant events on the longest traces are unlikely to be carried in full. Mitigation stated in the same paragraph: the structural branch puts every call two steps from every other through the process that issued it, "which bounds the effect rather than removing it". Mamba (linear in length) and transformer backbones are the §6.3 response.

**G8. Why didn't you build the on-the-fly version, if it is the most consequential extension?**
§6.2: it would mean rebuilding and re-evaluating the graph repeatedly as a trace unfolds, and "the computational resources and time available to this work did not extend to that". It is left as a direction for development, "rather than a bound the design cannot cross"; the graph-native design makes a continuous-time dynamic graph the natural host (§6.3).

**G9. In one sentence, what is your contribution?**
§6.1: "a single, end-to-end model in which structural, semantic and temporal reasoning are learned together over one schema-aware heterogeneous graph and fused by an adaptive gate that reweighs the three for every individual decision." Empirically, it withstands the cheap behaviour-preserving evasions that defeat single-paradigm detectors, without the opacity such robustness usually costs.


---

## Numbers to have at your fingertips

| Fact | Value | Where |
|---|---|---|
| Corpus | 9,333 graphs; 70/15/15 stratified | §4.2.2 |
| Test split | 1,384 = 39 ransomware / 299 benign / 1,046 other | §5.2 |
| Threats missed (test) | 0 of 1,085; FPR 0.054 (16/299) | Table 5.2 |
| Threat F1 / precision | 0.993 / 0.985 | §5.2.1 |
| Three-class macro-F1 / accuracy | 0.841 / 0.969 | Table 5.3 |
| Ransomware recall test / train | 0.462 (n=39) / 0.949 (172 graphs) | Tables 5.3, 5.5 |
| Gate weights overall | 0.354 / 0.353 / 0.293 | Table 5.7 |
| Ablation worst case | 95.8 % threat recall (no structural, 10 missed) | Table 5.8 |
| Exact class intact vs best single branch | 285 vs 279 of 300 | Table 5.8 |
| Semantic swap (no structure) | 8 → 10 missed (0.7 % changed) vs 8 → 21 (5.3 %) | Table 5.11 |
| Attacks | XGBoost 275/275, GAT 144/275, MLP 125/125, BiLSTM 200/200 FP; framework ≤ 2 | Table 5.13 |
| Unseen source | 110/110; 71/255 calls new (27.8 %); JS 0.534; 6/20 shared; min p 0.865 | §5.7 |
| Model | ~16.1 M params; BiLSTM 126/dir; 6 heads; seed 42 | §4.4, §5.1.2 |
| Clean baselines vs framework (macro-F1) | XGBoost 0.874, BiLSTM 0.840 vs framework 0.827 (set A) / 0.763 (set B); BiLSTM misses 4, framework 0 | Table 5.12 |
| BiLSTM own attack (reorder) | 0.840 → 0.728, 5 missed; under rename all 200 benign flagged (0.136) | §5.6.2 |
| Intact-model substitution | test: 1 → 2 missed of 306 under either swap | Table 5.10 |
| Three-class rule | threat ⇔ ŷ ∈ {ransomware, other malware} | Eq. 4.8, §4.6.1 |

---

## Transition to the demo

After "Thank you", move straight to the live demo; it is not on the slides. One sentence is enough: "Let me show what this looks like on a real report, including what happens when I rename its calls." The rename attack in the web app mirrors the RQ3 slide, so it reinforces the central result.
