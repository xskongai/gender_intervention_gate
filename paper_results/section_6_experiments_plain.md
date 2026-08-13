# 6 Experiments

We evaluate overcorrection in gender-inclusive generation using Chinese gender-inclusive rewriting as the primary experimental setting. The source text provides an explicit reference for determining both whether intervention is warranted and whether the resulting modification is appropriate. Our experiments examine three questions:

1. How frequently do existing language models over-intervene?
2. Can the proposed Gate reduce unnecessary intervention without suppressing valid rewriting needs?
3. Can the proposed Rewriter improve the quality of necessary rewrites?

## 6.1 Experimental Design

We conduct three groups of experiments. The Golden-Negative set contains inputs whose gender information or superficially biased cues should be preserved in context, whereas the Golden-Positive set contains inputs that require gender-inclusive intervention.

The overcorrection evaluation is conducted on the complete Golden-Negative set. The Gate is evaluated on both datasets so that improved preservation cannot be achieved by indiscriminately predicting KEEP. The Rewriter is evaluated on the complete Golden-Positive set, where all instances require intervention.

| Experiment | Dataset | Compared Settings | Purpose |
|---|---|---|---|
| Overcorrection Evaluation | Golden-Negative | Zero-shot rewriting across 11 models | Measure unnecessary intervention |
| Gate Evaluation | Golden-Negative + Golden-Positive | Zero-shot baseline vs. Proposed Gate | Reduce overcorrection while retaining valid intervention recognition |
| Gate Ablation | Golden-Negative + Golden-Positive | LLM-only vs. Rule+LLM | Measure the contribution of programmatic protection rules |
| Rewriter Evaluation | Golden-Positive | Zero-shot vs. Full Rewriter | Evaluate improvement in necessary rewrite quality |
| Rewriter Ablation | Golden-Positive | Zero-shot vs. Semantic-Preservation Prompt vs. Full Rewriter | Isolate the contributions of semantic constraints and adaptive refinement |

## 6.2 Models and Baselines

### Models

We evaluate 11 language models covering different model families, scales, and deployment settings. The hosted API models are DeepSeek-V4-Pro, Qwen3.7-Plus, GLM-5.2, GPT-4o, and Gemini-3.6-Flash. The locally deployed models are Qwen3.5-9B, GLM-4-9B, DeepSeek-R1-8B, Llama-3.1-8B, Mistral-7B, and Gemma2-9B. All models are evaluated on the complete Golden-Negative set in the overcorrection experiment.

### Zero-shot Baseline

The zero-shot baseline uses a direct gender-inclusive rewriting instruction. The model is asked to rewrite the input only when a gender-inclusive problem is present and otherwise return the original text. This setting serves as the common baseline for overcorrection, Gate, and Rewriter comparisons. It is run separately on Golden-Negative and Golden-Positive when evaluating the Gate.

### Gate Variants

The main Gate comparison evaluates the zero-shot baseline against the complete Protection-Aware Intervention Gate. For the Gate ablation, we compare an LLM-only variant with the full Rule+LLM configuration. The former directly assigns all instances to contextual LLM judgment, whereas the latter first handles high-confidence protection cases using programmatic rules.

### Rewriter Variants

The main Rewriter comparison evaluates zero-shot direct rewriting against the complete Feedback-Guided Adaptive Rewriter. We further consider three progressively stronger configurations:

Zero-shot → Semantic-Preservation Prompt → Full Rewriter

The Semantic-Preservation Prompt adds explicit requirements for semantic preservation, factual faithfulness, and minimal necessary modification, but performs only one generation round. The Full Rewriter additionally uses verification, targeted feedback, adaptive refinement, early stopping, and fallback selection.

## 6.3 Evaluation Metrics

### Overcorrection

On Golden-Negative, the correct behavior is to preserve the input. We define the Overcorrection Rate as:

OCR = Number of unnecessary modifications / Number of Golden-Negative instances

We also report the corresponding Preservation Rate:

PR = 1 − OCR

Lower OCR and higher PR indicate better resistance to unnecessary intervention.

### Gate Performance

For Gate evaluation, Golden-Negative instances have the gold decision KEEP, while Golden-Positive instances have the gold decision REWRITE.

Keep Recall measures the proportion of Golden-Negative instances correctly predicted as KEEP:

Recall_KEEP = Number of correct KEEP decisions / Number of Golden-Negative instances

Rewrite Recall measures the proportion of Golden-Positive instances correctly predicted as REWRITE:

Recall_REWRITE = Number of correct REWRITE decisions / Number of Golden-Positive instances

Because either class can be favored by a biased decision policy, our primary aggregate metric is Balanced Accuracy:

BA = (Recall_KEEP + Recall_REWRITE) / 2

We additionally report Macro-F1 and the absolute reduction in overcorrection relative to the zero-shot baseline:

ΔOCR = OCR_baseline − OCR_gate

### Rewrite Quality

The Rewriter is evaluated along three dimensions:

1. Debiasing Effectiveness
2. Linguistic Naturalness
3. Content Quality

Content quality is evaluated according to the rewrite type. For local edits, we emphasize semantic preservation, factual faithfulness, and modification scope. For global reconstruction, we evaluate preservation of the source’s core content, relevance, and communicative purpose.

We also report the Overall Success Rate, defined as the proportion of outputs that satisfy all three quality dimensions:

OSR = Number of outputs satisfying all three dimensions / Number of Golden-Positive instances

### Efficiency

We report the proportion of instances handled directly by programmatic rules, the average number of refinement rounds, the first-round pass rate, the average number of model calls, and the average processing time.

## 6.4 Implementation Details

All experiments are conducted on the complete evaluation sets without sampling. Comparable systems use the same task prompt, temperature, and decoding settings. Models are instructed to return only the required output without explanations or intermediate reasoning.

The Gate follows a rule-first cascade. Inputs matching a high-confidence protection rule are directly assigned KEEP; all remaining inputs are passed to the contextual LLM decision module. The LLM-only ablation removes this rule-based routing while keeping the LLM prompt and decoding configuration unchanged.

The Full Rewriter performs at most three generation rounds. It terminates early when all verification requirements are satisfied. If no candidate passes all criteria within the maximum number of rounds, the candidate with the highest verification score is returned.

Hosted models are accessed through their respective APIs, while the 7B–9B models are deployed locally through Ollama. Complete prompts, exact model identifiers, decoding parameters, output-parsing rules, and additional implementation details are provided in the appendix.
