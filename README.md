# df-augment

A standalone Python package that turns one parquet dataset into a larger,
augmented parquet dataset. Every synthetic row records which original
sample(s) it came from, so augmented data can never leak into test sets
when used with [df-analyze](https://github.com/stfxecutables/df-analyze)
or [DeepTune](https://github.com/moayadeldin/deeptune).

## Install

```bash
pip install df-augment
```

## Quick start

```bash
df-augment --data input.parquet --out output.parquet --techniques flip jitter --multiplier 3
```

## Status

Under active development — Phase 0 setup.
