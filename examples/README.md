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
