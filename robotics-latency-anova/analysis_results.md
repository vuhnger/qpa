# Questionnaire Latency Analysis (ANOVA + Post Hoc)

## Why these tests
- **Design**: Each participant rated all five latency levels (0, 50, 100, 150, 200 ms) for the same questions → within-subjects (repeated measures) factor.
- **Omnibus**: A repeated-measures ANOVA tests whether mean ratings differ across latency while partitioning participant variance from residual error.
- **Post hoc**: If the omnibus test is significant, paired contrasts identify which specific latencies differ. Tukey HSD is for independent groups, so we use paired t-tests with Holm correction to control family-wise error for repeated measures.

## How the scripts work
- `run_anova_questions.py`
  - Reads `manually_pruned_data.csv` (semicolon-separated).
  - Builds a participant × latency table per question.
  - Computes RM-ANOVA by hand (SS_latency, SS_participant, SSE, df, MS, F, p, partial eta²).
  - Prints results and writes `anova_results.csv`, including critical F-values and a “significant_at_alpha” flag.
- `posthoc_latency_tests.py`
  - Uses the same participant × latency tables.
  - Runs paired t-tests for every latency pair per question.
  - Applies Holm correction; reports raw p, adjusted p, t, df, Cohen’s d_z, and significance flags.
  - Writes `posthoc_latency_pairs.csv`.

Run commands (from repo root, after installing `requirements.txt`):
```bash
python run_anova_questions.py
python posthoc_latency_tests.py  # interpret only for questions with significant omnibus ANOVA
```

## Omnibus ANOVA results (last run)
- **Question: “I felt like I was controlling the movement of the robot”**
  - F(4,120)=2.59, p=0.040; partial eta²≈0.0795 (small effect).
  - Means: 0ms 4.226; 50ms 4.226; 100ms 4.129; 150ms 4.129; 200ms 3.774.
  - Interpretation: Ratings drop at 200 ms compared to 0–150 ms; overall latency effect is significant at α=0.05.
- **Question: “It felt like the robot was part of my body”**
  - F(4,120)=0.58, p=0.677; partial eta²≈0.019.
  - Means: 0ms 2.839; 50ms 2.968; 100ms 2.871; 150ms 3.000; 200ms 2.742.
  - Interpretation: No evidence of a latency effect; do not interpret post hoc contrasts.

## Post hoc summary (only meaningful where ANOVA is significant)
- Conducted for the significant question (“controlling the movement”):
  - Paired comparisons with Holm correction: no pair survived correction.
  - Largest raw differences involve 200 ms:
    - 0ms vs 200ms: raw p≈0.0082, Holm p≈0.082 → not significant after correction.
    - 50ms vs 200ms: raw p≈0.0108, Holm p≈0.097 → not significant after correction.
    - 100ms vs 200ms: raw p≈0.0463, Holm p≈0.370 → not significant after correction.
  - Interpretation: Trend suggests reduced control feeling at 200 ms, but pairwise effects are not reliable after multiple-comparison correction.
- For the embodiment question (non-significant ANOVA), post hoc testing is not warranted.***
