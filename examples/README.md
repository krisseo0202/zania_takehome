# Chunk-size comparison

Three runs of the 19 sample questions against `Nave-SOC2-Type-2-Report.pdf`,
varying `chunk_size` only. `chunk_overlap=150` and `top_k=5` are held constant,
so chunk size is the single variable.

```bash
CHUNK_SIZE=500 TOP_K=5 .venv/bin/python scripts/run_sample.py \
  Nave-SOC2-Type-2-Report.pdf samples/questions.json > examples/nave-chunk500-k5.json
```

| chunk_size | chunks | answered | abstained | chat input | embedding | est. cost |
|---|---|---|---|---|---|---|
| 500  | 751 | 9 | 10 | 21,135 | 94,227 | $0.0054 |
| 1000 | 333 | 8 | 11 | 31,489 | 78,255 | $0.0067 |
| 1500 | 229 | 8 | 11 | 40,698 | 74,422 | $0.0080 |

**Cost runs backwards from the intuition.** Smaller chunks mean more of them and
more embedding tokens, but `k=5` fixes the *number* of chunks sent to the chat
model, not their size, so chat input shrinks with chunk size and dominates the
bill: 500 is the cheapest config here, not the most expensive.

**Do not read the answered column as a ranking.** Only 4 of 19 questions differ
across the three, and they flip in both directions - Q8 and Q11 answer at 500
and 1500 but abstain at 1000; Q18 does the reverse. A separate run at
`chunk_size=1000` earlier the same day gave 7 answered, not 8, on identical
inputs. Run-to-run variance at temperature 0 is therefore about the same size
as the difference between configs, so this comparison is too small to call a
winner. It shows the cost curve, which is stable, and it does not show a
quality difference, which would need repeated runs and a labelled set.

## Overlap comparison

Three more runs at `chunk_size=500`, `top_k=5`, varying `chunk_overlap` only.

| overlap | % of chunk | chunks | answered | abstained | chat input | embedding | est. cost |
|---|---|---|---|---|---|---|---|
| 50  | 10% | 612 | 7 | 12 | 21,284 | 74,924 | $0.0050 |
| 100 | 20% | 676 | 6 | 13 | 21,066 | 83,627 | $0.0051 |
| 150 | 30% | 751 | 8 | 11 | 21,135 | 94,227 | $0.0054 |

**Overlap is a much smaller cost lever than chunk size.** Chat input is flat
across all three (~21.1k), because `top_k=5` sends five 500-char chunks
whatever the overlap. Only embedding grows, +26% from 10% to 30%, which is +8%
on the total: $0.0004 per run.

**Again, no quality signal.** The answered column goes 7, 6, 8 - not monotonic -
and the three configs disagree on only 3 of 19 questions. The 30% run here
returns 8 answered where the identical config in the chunk-size table above
returned 9. That is the same config twice, differing by one, which is as large
as any difference between the overlaps.

Kept at 150 (30%). At 500 characters that is roughly one or two sentences of
protection against a fact being split across a boundary, and the alternative
saves $0.0004 a run. It is a judgement call, not a measurement.

## Top-k comparison

Four more runs at `chunk_size=500`, `chunk_overlap=150`, varying `top_k` only.
`k=5` was re-run in the same session as a fresh baseline (`nave-chunk500-k5-b.json`);
the first row is the earlier run from the chunk-size table.

```bash
TOP_K=8 .venv/bin/python scripts/run_sample.py \
  Nave-SOC2-Type-2-Report.pdf samples/questions.json > examples/nave-chunk500-k8.json
```

| top_k | answered | abstained | chat input | chat output | embedding | est. cost |
|---|---|---|---|---|---|---|
| 5 (earlier) | 9 | 10 | 21,135 | 659 | 94,227 | $0.0054 |
| 5 | 8 | 11 | 21,135 | 580 | 94,227 | $0.0054 |
| 6 | 8 | 11 | 23,734 | 645 | 94,227 | $0.0058 |
| 7 | 6 | 13 | 26,261 | 536 | 94,227 | $0.0061 |
| 8 | 8 | 11 | 28,887 | 637 | 94,227 | $0.0066 |

**Cost is linear in k.** Each extra chunk adds ~2,600 chat input tokens per
19 questions (five 500-char chunks, nineteen times), about +$0.0004 per step;
embedding is untouched because the index is the same. `k=8` costs 22% more
than `k=5`.

**Quality does not move with k.** Six questions (Q4-Q8) answer `Partially`
at every k. The four that flip do so in both directions:

| | k=5 | k=5 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|
| Q2 unauthorized software | Partially | Partially | Yes | Yes | Partially |
| Q11 incident reporting | Partially | - | Partially | - | - |
| Q15 wireless monitoring | Partially | Partially | Partially | - | Partially |
| Q18 IoT asset inventory | - | - | - | - | Partially |
| Q19 IAM system | Partially | Partially | - | - | - |

Q18 answers only at `k=8`, where a passage ranked 8th finally reaches the
model. Q19 stops answering from `k=6` on, even though the `k=5` evidence is
still in front of the model: the extra passages dilute rather than help. And
`k=7`, with more context than 6, answers fewer questions than either
neighbour. The two `k=5` rows differ by one question on identical inputs, so
again the spread between configs (6-9) is about the run-to-run noise.

Kept at 5. It is the cheapest, and nothing above it shows a quality gain that
survives the noise; a bigger k mostly buys longer prompts.
