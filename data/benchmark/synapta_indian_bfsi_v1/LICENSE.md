# License: Creative Commons Attribution-ShareAlike 4.0 International

The Synapta Indian BFSI Benchmark v1 (the contents of this directory,
including `questions.jsonl`, `README.md`, `scoring.py` and
`build_benchmark.py`) is released under the
**Creative Commons Attribution-ShareAlike 4.0 International License
(CC-BY-SA-4.0)**.

The full license text is available at:
<https://creativecommons.org/licenses/by-sa/4.0/legalcode>

A short summary follows. The license text linked above is authoritative.

## You are free to

- **Share** - copy and redistribute the benchmark in any medium or format.
- **Adapt** - remix, transform, and build upon the benchmark, including
  for commercial use.

## Under the following terms

- **Attribution.** You must give appropriate credit, provide a link to
  the license, and indicate if changes were made. You may do so in any
  reasonable manner, but not in any way that suggests Synapta endorses
  you or your use.
- **ShareAlike.** If you remix, transform, or build upon the benchmark,
  you must distribute your contributions under the same license as the
  original.
- **No additional restrictions.** You may not apply legal terms or
  technological measures that legally restrict others from doing
  anything the license permits.

## Suggested citation

If you use this benchmark in published research, internal evaluation
reports, model cards, or product documentation, please cite it as:

```bibtex
@misc{synapta2026bfsi,
  title        = {Synapta Indian BFSI Benchmark v1: an extractive
                  question-answering evaluation set for Indian
                  financial regulation},
  author       = {Synapta},
  year         = {2026},
  howpublished = {\url{https://github.com/synapta/indian-bfsi-bench}},
  note         = {Released under CC-BY-SA-4.0.}
}
```

## Contamination notice

This is **v1**, intentionally frozen so that scores remain comparable
over time. Once a benchmark is published openly, it will eventually
appear in pretraining corpora; we accept that trade-off in exchange for
broad community use. Subsequent versions (v2, v3, ...) will refresh the
question set from documents not present in v1.

When reporting scores, please indicate whether your model's training
data is known to predate this release (2026-05) or to have included
public web crawls after that date, so readers can interpret the number
accordingly.

## No warranty

The benchmark is provided "as-is", without warranty of any kind, express
or implied. It is a research instrument, not legal or financial advice.
Gold answers reflect what the source RBI Master Direction or SEBI
Master Circular literally said at the time of corpus collection, not
what the regulator may have subsequently amended.

## Contact

Questions, contributions, errata: please open an issue or pull request
at <https://github.com/synapta/indian-bfsi-bench> (placeholder URL;
will be updated when the public repository is live).
