# License notes

**Decision (2026-09-16): this repository is Apache-2.0.** See LICENSE and
NOTICE. What follows records why, and what obligations come from the model
weights and input data. This is not legal advice.

## Upstream terms

| Material | License | Source |
|---|---|---|
| weathernext code and notebooks | Apache-2.0 | LICENSE in google-deepmind/weathernext |
| "All other materials" in that repo, which covers the model weights | CC BY 4.0 | README of google-deepmind/weathernext |
| ERA5 / ERA5T | Copernicus C3S licence (free use with attribution) | CDS |
| JMA typhoon position table | JMA website terms of use (attribution required) | data.jma.go.jp |
| OpenStreetMap tiles (optional map background) | ODbL for the data, tile usage policy for the server | tile.openstreetmap.org |

## Why Apache-2.0 is safe here

The weights are never vendored into this repository. They are downloaded at
run time from `gs://dm_graphcast/` and used unmodified, so nothing under
CC BY 4.0 is redistributed by the repository itself. Our own source code is
therefore unencumbered, and Apache-2.0 keeps it aligned with the upstream
code we call into.

## Obligations that remain

* **Model attribution.** Any published forecast field, track file or figure
  derived from the weights is a derivative of a CC BY 4.0 work and must credit
  Google DeepMind and name the checkpoint. NOTICE and the README carry this;
  repeat it in figure captions and in any write-up.
* **ERA5 attribution.** "Contains modified Copernicus Climate Change Service
  information 2026", plus the disclaimer that neither the European Commission
  nor ECMWF is responsible for any use made of it.
* **JMA attribution.** Credit the Japan Meteorological Agency for best-track
  data.
* **OpenStreetMap attribution.** A figure drawn with `--basemap osm` must show
  "(c) OpenStreetMap contributors"; `analysis.plot.credit_line` adds it. The
  tile usage policy also asks for an identifying user agent and no bulk
  downloading, which the project user agent and the cache under `data/` cover.
* **NOTICE must be kept current.** If a different checkpoint is used, add it
  to the checkpoint list in NOTICE.

## Open item

- [ ] Confirm the weights license text on the GCS bucket or in a model card.
      The README wording ("all other materials") is the only statement found
      so far. This does not block Apache-2.0 for our code, only the exact
      wording of the attribution we give for outputs.
