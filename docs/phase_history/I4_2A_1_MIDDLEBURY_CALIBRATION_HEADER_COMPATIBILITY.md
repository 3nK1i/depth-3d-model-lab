# I4.2A.1 — Canonical Middlebury Calibration Header Compatibility

## Trigger

The first real I4.2A run against `templeR_par.txt` failed closed because the canonical Middlebury file contains a leading camera-count line (`47`) followed by 47 calibration rows. The initial synthetic fixtures were headerless and the parser therefore counted 48 non-empty lines.

A separate local dataset copy was also found to contain only 46 renamed images. That copy remains invalid. A fresh official TempleRing package was obtained and independently shown to contain canonical `templeR0001.png` through `templeR0047.png` plus the matching calibration files.

## Correction

The calibration parser now accepts exactly two strict forms:

1. canonical Middlebury form: integer header `47` + exactly 47 calibration rows;
2. legacy/headerless compatibility form: exactly 47 calibration rows.

When the integer header is present it must declare exactly 47. The parser also requires the exact canonical filename set `templeR0001.png..templeR0047.png`; a 47-row file with a shifted/missing/unexpected canonical name fails closed.

The preflight summary now records whether the count header was present, the declared count, and the number of validated camera rows.

## Preserved locks

- exactly 47 images remain mandatory;
- image dimensions remain 640×480;
- shared K validation is unchanged;
- benchmark per-view R/t and camera centers remain validation-only and cannot enter SfM or BA;
- no SfM, BA, MVS, DA3, dense filtering, mesh generation, metric-scale claim, measurement authority, or provider promotion is performed;
- I4.3 remains unauthorized.

## Validation

A3.1/I4.1.9 backward evidence remains unchanged. I4.2A.1 adds focused and stress cases for canonical headers, wrong count declarations, row-count mismatch, malformed headers, and canonical filename-set integrity while retaining the original I4.2A test suite.

## Next

Rerun the real I4.2A preflight against the clean official dataset directory. Only a passing 47/47 preflight authorizes I4.2B TempleRing SfM.
