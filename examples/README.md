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
