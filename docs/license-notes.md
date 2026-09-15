# License notes (decision deferred)

Findings as of 2026-09-16. This is not legal advice; confirm before choosing
the repository license.

## Upstream terms

| Material | License | Source |
|---|---|---|
| weathernext code and notebooks | Apache-2.0 | LICENSE in google-deepmind/weathernext |
| "All other materials" in that repo, which appears to cover the model weights | CC BY 4.0 | README of google-deepmind/weathernext |
| ERA5 / ERA5T | Copernicus C3S licence (free use with attribution) | CDS |
| JMA typhoon position table | JMA website terms of use (attribution required) | data.jma.go.jp |

## Implications for this repository

* Our own code can carry any OSI license; Apache-2.0 keeps it aligned with
  the upstream code.
* Forecast outputs derived from the weights should carry CC BY 4.0
  attribution to Google DeepMind if published.
* ERA5-derived files and JMA-derived tables need their respective
  attribution lines in the README.

## TODO

- [ ] Confirm the weights license text on the GCS bucket or in the model card.
- [ ] Check whether redistributing cropped forecast fields is acceptable.
- [ ] Pick the repository license and add LICENSE.
