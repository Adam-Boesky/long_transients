# Filtering review: fixes before the final dataset

Review of the extraction → filtering → combine pipeline, before producing the final candidate set.
Line numbers refer to the cluster copy of `Source_Analysis/filter_fields.py` (commit `4d7cc19` plus uncommitted edits) unless stated.
Status: `[ ]` to implement · `[~]` partly done · `[x]` implemented and verified.
Issue numbers are the ones used during the review.

---

## Implementation order

| Step | Issue | Change | Where |
|---|---|---|---|
| 1 | #12 | ✅ Rebuild every quadrant's PS1 list over the union of the g/r/i footprints; re-run association + merge (done, validated 10/22) | `rebuild_ps1_catalogs.py`; `cross_match -aq -ow`, `-mf -ow` |
| 2 | #4 | Remove the wide-association evaluation; keep the 1–3″ diversion | `filter_fields.py` |
| 3 | #5 | Support g+r-only fields | `filter_fields.py` |
| 4 | #8 | SNR routing tie-break on PSF SNR | `filter_fields.py` (`snr_filter`) |
| 5 | #9 | Per-band PS1 values; position from the first matched band | `Sources.py` (`Sources.data`) |
| 5b | #13 | Remove the Dec > −29.5 cut (applied by hand afterwards) | `filter_fields.py` |
| 5c | #14 | Annotate every detection with its Δmag sigma; filter on it | `filter_fields.py` |
| 6 | – | Move stale KDE caches aside; stage-3 filtering of all 485 fields into a new results directory | cluster run |
| 7 | #1, #11 | `group_id`; read only field folders | combine script |
| 8 | – | Enrichment, then website database update (annotations carried over by identity) | stage 5 |

---

## Fixes

### 12. [x] PS1 query box covers only the first band's image
- **Problem:**
  - Each quadrant has one `PSTARR.hdf5`, queried from the corners of the first band's image (usually g; `Extracting/Tile.py:61`) and shared by r and i.
  - The g/r/i deep stacks are offset from each other. Where an r or i image reaches past the g box, its detections have no PS1 objects to match, so they are flagged ZTF-only (flag 1).
- **Why the existing safeguards miss it:**
  - Padding: the CasJobs path adds 10.8″ (`Catalogs.py`, `query_buffer=0.003`). The local path adds none: `PS1_Local.query_tile`'s `query_buffer` only prunes parquet files.
  - The ~50–90 px NaN borders don't help, because stacks can be offset by more than a border width. Field 683 c06 q4: r/i sit ~84″ north of g, and the r detection is on valid pixels off the g image.
  - The cross-match flag-3 checks (PS1 more than 1′ away; NaN within 15 px) don't fire.
  - The 1–3″ diversion only considers flag-2 PS1 rows 1–3″ away. The missed PS1 object sits in the neighbouring quadrant's table, flagged 3, 0.1–0.6″ away.
- **Evidence:**
  - Header scan of 30,427 quadrants: the r/i image reaches beyond the g box by more than 10.8″ in 27,308, more than 60″ (about one border width) in 4,203, and more than 120″ in 132.
  - 9/21 run: 45 ZTF-only band detections lie outside their quadrant's PS1 coverage (15 CasJobs-era, up to 27.5″ out; 30 local-era, up to 7.1″). That gives ~31 false ZTF-only candidates of 10,935.
  - In-both detections in these strips are mislabelled ZTF-only, so ~0.1–0.3% of the footprint is effectively unsearched.
- **Fix:** rebuild PS1 and re-match; no ZTF re-extraction.
  - Code (local + cluster copies):
    - `Extracting/utils.py`: new `coord_range_from_wcs` (the old `get_coord_range` body) and `union_coord_range` (padded union of band boxes, RA=0 and pole safe). Padding: `PS1_QUERY_PAD_DEG` = 30″, which covers the 1″ match and the 1–3″ diversion.
    - `Source_Extractor.get_coord_range` calls `coord_range_from_wcs`.
    - `Tile.__init__` queries PS1 over the union of all bands' boxes, so new extractions are correct.
  - Rebuild script `Extracting/rebuild_ps1_catalogs.py`:
    - recomputes each quadrant's box from its saved per-band WCS and NaN-mask shape;
    - queries local PS1 and writes `PSTARR.hdf5` atomically;
    - keeps the old file as `PSTARR_pre_rebuild.hdf5` and marks the quadrant with `ps1_rebuilt.txt`, so runs can resume;
    - `--out-name` writes a side copy instead.
  - Validated (side copies, 5 quadrants: 2 local-era, 2 CasJobs-era, 1 at RA≈0; ~2–30 s each):
    - Local-era: identical objects and photometry inside the old box; only new objects outside it (+~10% rows).
    - CasJobs-era: same objects. Magnitudes are identical wherever both have one; 0.1–0.4% of objects have a magnitude in only one source. `psfLikelihood` and `primaryDetection` are dropped (unused). CasJobs `infoFlag`s on objects with no magnitude in a band become NaN.
    - Association on scratch copies of 683 c06 q4 and 466 c09 q2:
      - first-band (g) ZTF flags unchanged;
      - in r/i, 5–17 ZTF detections per band go 1 → 0 (false ZTF-only fixed) and 89–189 go 3 → 0 (strips more than 1′ from any PS1 object, previously excluded);
      - the field 683 and 466 test detections are now matched to the PS1 object found earlier;
      - the extra PS1 rows outside the images are flag 3, so they are ignored by filtering.
  - **Done (10/22):**
    - PS1 rebuilt for all 30,659 quadrants (0 failures). Spot checks of 80 random quadrants: no objects lost, every box covers every band image, local-era photometry identical. CasJobs era: 1–7 objects per quadrant (~0.01%) take a different PS1 stack measurement near skycell boundaries.
    - Re-associated and re-merged all 485 fields (`Extracting/cross_match_fields.sh`, 8 jobs, ~3.5 h); old merged tables are in `catalog_results/field_results_pre_ps1_rebuild/`.
    - Validated against the old tables:
      - identical ZTF detections (1.81B over g/r/i);
      - flag changes ZTF-only → matched: 3.0k g / 30.9k r / 21.6k i;
      - excluded strip (3) → matched: 3.9k / 76.6k / 65.5k;
      - 3 → 1: 106 / 452 / 99, now near PS1 coverage but unmatched;
      - 1 → 3: 31 in total;
      - no matched detection lost its match;
      - the field 683 test detections are now matched.
    - Found and fixed a pre-existing merge bug (`d09db85`): a quadrant lacking one band was dropped for all later bands (5 i-only quadrant-bands in fields 623 and 624); both fields re-merged.
    - Harmless: 1–3 PS1 rows per field at the RA=0 seam store RA as 360.0 or a tiny negative value (pre-existing; no ZTF detections).
- **Follow-ups after the rebuild:**
  - One quadrant has a broken WCS (offset ~1.3 million ″ in the header scan).
  - 4 of the 21 ZTF-only candidates carrying a PS1 object lie inside their quadrant's PS1 box; recheck them.
  - 280 ZTF-only candidates have a flag-2 g row with no PS1 values; recheck them.

### 4. [x] Remove the wide-association branch
- **Problem:** the branch that re-evaluates ZTF-only sources with a PS1 object 1–3″ away (as in-both) doesn't work:
  - `Sources` (`Sources.py` ~1898) rebuilds each source from the original ZTF-only row (flag 1), so `at_least_n_big_dmag_bands(catalog='in_both')` always fails.
  - `associate_in_btwn_distance` (~1245) overwrites entire ZTF columns.
  - The branch skips the PS1 saturation and quality cuts.
  - All 28 final `1_wide_association` rows duplicate in-both candidates (26 at 0.00″).
- **Experiment** (field 400, side copy of the code):
  - Fixed branch: 2 candidates, marginal 0.2–1 mag fading of ordinary point sources.
  - No diversion: 2,534 extra ZTF-only candidates (vs 11). Cutouts show stars and galaxies clearly visible in PS1, offset 1–2″ from the ZTF centroid.
  - So the branch only finds extended, offset or otherwise uninteresting objects.
- **Fix:**
  - Keep diverting ZTF-only sources that have a PS1 object 1–3″ away (they are not ZTF-only).
  - Delete the wide-association evaluation and its `1_wide_association` / `2_wide_association` outputs.
  - Drop the matching tables from the combine step.

### 5. [x] Support g+r-only fields
- **Problem:**
  - `at_least_n_big_dmag_bands` loops over g, r, i and looks up `envelopes['i']`, which raises `KeyError` for g+r fields.
  - `filter_fields()` only selects fields imaged in g, r *and* i.
  - There are 89 g+r-only fields; i is the only band any field lacks.
- **Fix:**
  - `at_least_n_big_dmag_bands`: `{band: data for band in BANDS if band in envelopes}`.
  - `filter_fields()`: select fields from the g and r lists.
- **Tested** on field 247 (64 quadrants, g+r) end to end:
  - same output files and statistics rows as a gri field;
  - i columns empty;
  - every candidate (88 in-both, 33 ZTF-only) significant in both g and r;
  - statistics combine and the table stacks with field 400.
  - Columns missing vs field 400 (`psfLikelihood`, `PSTARR_iinfoFlag`) come from CasJobs-era vs local-PS1 extraction, not from g+r.

### 8. [x] Rows routed to both one-sided branches
- **Problem:** `snr_filter(both_cat=True)` (763–768):
  - A row stays in-both only if both catalogues pass SNR > 5 in the same photometry type (PSF–PSF or Kron–Kron). Otherwise the ZTF-only and PS1-only tests are independent.
  - If each catalogue passes in a *different* type (ZTF PSF + PS1 Kron, or ZTF Kron + PS1 PSF), the row goes to **both** branches. It is also added twice, as flag 1 and flag 2, to `all_quality_source_tabs` (1677–1683).

  | ZTF PSF | ZTF Kron | PS1 PSF | PS1 Kron | Now | Fixed |
  |:-:|:-:|:-:|:-:|---|---|
  | ✅ | · | · | ✅ | ZTF-only **and** PS1-only | ZTF-only |
  | · | ✅ | ✅ | · | ZTF-only **and** PS1-only | PS1-only |
  | other combinations | | | | unchanged | unchanged |

- **Evidence** (24 fields, 72M in-both rows):
  - 6,063 rows are affected: 6,051 faint (ZTF Kron + PS1 PSF) and 12 compact-in-ZTF on a PS1 Kron source.
  - None survive. PS1-only branch: 99.8% are removed by `pstarr_mag_less_than`. ZTF-only branch: `only_big_dmag` 64%, `extended_source_artifact_filter` 31%, ≥2-band cut 2%.
  - 0 such rows in the 9/21 final tables.
- **Fix:** if a row passes both one-sided tests, route it only to the side with PSF SNR > 5. Δmag uses PSF magnitudes, and exactly one side always qualifies.
  - All other routing is unchanged, including the NaN rules and Kron-pair in-both rows.

### 9. [x] Candidate rows mix PS1 objects across bands
- **Problem:** `Sources.data` builds each candidate row in two steps:
  1. for each band, take the nearest row within 1.5″ (any flag);
  2. copy values in, **first non-empty wins, in g → r → i order**.

  The row has one set of `PSTARR_*` columns plus `ra`/`dec`. When g's nearest row is a PS1-only neighbour (flag 2) and r/i are matched to a different PS1 object, the row mixes objects. The ≥2-band check then compares ZTF r/i against the neighbour.
- **Example** (field 690): object A is matched in r and i; neighbour B (PS1-only) is 1.35″ away and picked in g.
  - Stored position, PS1 ID and PS1 r magnitude come from B.
  - r Δmag is −1.66 against B, vs −0.79 against its own match.
- **Fix** (fill order in `Sources.data`):
  1. `PSTARR_{b}*` (PSF/Kron magnitudes, errors, info flags) come from band b's own row; other bands are used only if it is empty.
  2. `ra`, `dec` and the shared `PSTARR_*` columns (ra/dec/ID/quality flags) come from the first band whose row is matched (flag 0); otherwise first non-empty, as now.
  3. Everything else unchanged.
- **Tested** on fields 690, 596, 400 and 466, current code vs fix on identical inputs. The current-code rerun is identical to 9/21.
  - ZTF-only tables: unchanged.
  - In-both candidates: −2 / +3 across the four fields, from corrected significance. A recomputation over all fields found 92 losses (0.05%) but did not count gains.
  - 7 rows change PS1 object. In 5, the old object was a neighbour 1.1–1.5″ from the ZTF detections; the new one is 0.03–0.43″.
  - 282 rows (~10%) change position. 276 had stored the PS1 position of a PS1-only g row.
    - Median distance to the ZTF detections drops from 0.26″ to 0.14″ (90th percentile 0.57″ → 0.33″).
    - 243 move closer; 26 move farther by more than 0.01″.
    - 1–2 per field move more than 0.5″: the neighbour cases.
  - No other column changes.
- **Website:** rows moving more than 0.5″ (~0.2% of in-both, the neighbour cases) get new identity matches.

### 13. [x] No declination cut in filtering
- **Change:** the `Dec > −29.5` cut (`dec_greater_than`) is commented out everywhere it ran:
  - the quality-filtered tables the KDE envelopes and `Sources` rows are built from;
  - the in-both branch;
  - the ZTF-only branch;
  - the PS1-only branch.
- **Why:** the declination limit will be applied by hand to the final candidates, so filtering keeps everything.
- **Effect:** candidates below −29.5° now appear in the outputs. Envelopes in the southernmost fields include those sources (they are rebuilt for stage 3 anyway).

### 14. [x] Δmag sigma for every detection
- **Change:** each detection gets a signed Δmag sigma from its field's KDE envelope, z = Φ⁻¹(CDF of Δmag at its magnitude), negative = brighter in ZTF. The envelope's cumulative distribution is interpolated bilinearly on the grid; |z| beyond what the grid resolves is reported as ±8 with a flag.
- **Columns:** `{band}_dmag`, `{band}_dmag_sigma`, `{band}_dmag_sigma_capped`.
  - Written onto the all-quality tables right after the envelopes are built, so candidate rows carry each band's own sigma.
  - Also written in `only_big_dmag` on each branch's tables, whose flags may differ.
- **Selection** reads the column: `only_big_dmag` keeps |z| > 3; `at_least_n_big_dmag_bands` counts bands with |z| > 3. Cost ~0.2 s per band per field.
- **Tested** side by side on fields 247 (g+r), 313 (gri) and 400 (mixed):
  - every candidate found by both versions is unchanged;
  - all 27 that differ are explained, 0 unexplained:
    - 20 are borderline (|z| within 0.03 of 3, where interpolating the boundaries and interpolating the CDF disagree);
    - 7 had only passed by comparing a neighbour row's ZTF limit with a PS1 magnitude borrowed from another band's row (e.g. field 400: the real ZTF g detection had failed the shape cut).
  - The committed code reproduces the tested patch exactly.

### 1. [x] Duplicate candidates across overlapping fields
- **Problem:** neighbouring fields overlap, and the combine step stacks per-field outputs with no deduplication, so an object in the overlap appears once per field.
- **Evidence** (9/21 `combined/`):
  - `0.hdf5` has 8,899 pairs within 1″ (17,182 rows); `1.hdf5` has 304 pairs.
  - All are across fields; there are none within a field.
- **Fix:** flag, don't collapse; done at the combine step, the first place all fields meet.
  - Add a `group_id` column. Rows within 1″ of each other, across fields *and* across tables (0, 1), share a value (chains linked); null for single detections. There is no primary; every row stays equal.
  - `group_id` is an integer: groups are numbered 1..N by mean RA (then Dec), independent of file order; -1 for single detections.
  - `source_uuid` is unchanged, so all labels stay attached.
- **Downstream:**
  - Website: group by `group_id`, show all members together, label them as one.
  - Counts and rates: count each distinct `group_id` once, plus every ungrouped row.

### 11. [x] Combine reads non-field folders
- **Problem:**
  - The combine step reads every folder in the results directory as a field, skipping only `combined`, `candidates` and `.DS_Store`.
  - A leftover debug copy such as `combined_pre_filtered_out/` (present in the 9/21 directory) would be read as a field. Every candidate would then be stacked twice, with no error.
  - Today's `combined/` is correct (186,970 rows).
- **Fix:** read only digit-named field folders, in all three loops.

### 15. [x] PS1 DR2 missing regions
- **Problem:** PS1 DR2 has no ObjectThin sources in seven boxes (~4.9 deg²) listed on MAST's [DR2 caveats page](https://outerspace.stsci.edu/display/PANSTARRS/PS1+DR2+caveats): a Dec 35.46–35.499 strip in three RA pieces, three polar boxes, and one box at (76.4–76.9, −27.18 to −26.72). ZTF sources there have no PS1 counterpart because the catalogue is missing, not because they are new.
  - Inside a big box most ZTF sources are more than 1′ from any PS1 object, so they get `Catalog_Flag` 3 and never become candidates; only a ~1′ rim inside each edge (and all of the 2.3′-tall strip) became ZTF-only.
  - `no_nearby_source_filter` removed most of those floods, but 13 ZTF-only candidates inside the boxes survived.
- **Fix:** `ps1_missing_region_filter` drops ZTF-only candidates inside the boxes, no padding, applied *after* `no_nearby_source_filter`.
  - Applied before it, removing the in-box floods first left their neighbours just outside the boxes (the real hole extends ~8″ below the strip's listed edge) with too few neighbours, and ~1,100 of them survived in 12 strip fields alone.
- **Rerun:** the 22 fields whose footprint touches a box (255, 648–654, 672–682, 858, 859, 868), then combine. Result: 13 ZTF-only rows removed (14,911 → 14,898), all inside the boxes; in-both and every other column unchanged.
- **Storage fault during the rerun:** holystore01 `OST0028` (index 40) hangs any read of a file stored on it while still reporting ACTIVE. Fields that needed one of its files were rerun with new output pinned to a healthy OST (`lfs setstripe -i 10`): 52 more fields whose filtered-out tables were on it (output identical to before), plus the Δmag envelopes of 7 fields rebuilt. Field 399's `0_i_filtered_out.ecsv` is unreadable and its merged catalogue is too, so its `filtered_out_in_both_i` values were restored from the previous combined run.

---

## Known limitations (not fixed)
Recorded for the paper.
- **PS1 NULL magnitudes.** In-both rows with no PS1 magnitude in a band (−999, PS1's NULL: no usable stack measurement) are never evaluated in that band.
  - 10–25% of in-both rows in g/r and 1–3% in i; over 99% are faint (median ZTF ~23).
  - −999 isn't an upper limit, so it can't stand in for one.
- **Blending.** ZTF's ~2″ PSF often includes several PS1 objects, but Δmag compares against one PS1 object. This is a limit of catalogue-level matching; a forced-photometry redesign was considered and rejected for this release.
- **Merged neighbours.** Detections within 1″ (in-both) or 3″ (ZTF-only) are merged without checking bands, so a close neighbour can be lost. Rare: band positions differ by more than 1.5″ in 10–40 rows per band pair.
- **Gaia veto sky loss.** The 5″ proper-motion/parallax veto removes ~3.5% of the footprint's sky (median field 2.2%, worst 36% in field 284). Removed sources are almost always the star itself (median 0.14″ / 0.23″).

---

## Cleanup (after each step's output is fully validated)
Done 2026-10-23, move-only: everything below sits in `_stale_2026-10-23/to_delete/` (data dir on the cluster) or `Data/_stale_2026-10-23/` (laptop), ready to `rm -rf` once you're happy.
1. [x] Pre-rebuild PS1 files, `ps1_rebuilt.txt` markers and `PSTARR_rebuilt.hdf5` side copies (61,323 files) -> `to_delete/catalog_results_ps1_rebuild_leftovers/`.
2. [x] Rebuild scripting: `Extracting/rebuild_ps1_catalogs.py` / `rebuild_ps1.sh` (laptop copies in `Data/_stale_2026-10-23/code/`; cluster copies archived on branch `archive/cluster-wip-2026-10-22`), `ps1_rebuild_quads_10_22.csv`.
3. [x] Validation scratch: `ps1_rebuild_test/`, `fill_fix_test/`, `gr_field_test/`, `wide_assoc_experiment/`, `fix{5,8,9,14}_test/`, `e2e_test/`, `reextract_fix_test/`.
4. [~] Cluster one-off scripts: archived on `archive/cluster-wip-2026-10-22`; the untracked copies are still in the cluster checkout (moving them was blocked by the permission check).
5. [~] Cluster switch-over backup: stash@{0} is archived on `archive/cluster-wip-2026-10-22`, so `git stash drop stash@{0}` and `~/backup_cluster_tree_10_22/` are safe to remove by hand (both still present).
6. [x] `catalog_results/field_results_pre_ps1_rebuild/` (2.2 TB) and the 77 October-2025 `.ecsv` leftovers -> `to_delete/`.
7. [x] `xmatch_lists/` -> `to_delete/`; `Extracting/cross_match_fields.sh` archived on the branch above.
8. [x] Superseded runs and pre-change backups -> `_stale_2026-10-23/old/` (not on the agreed list, so kept for you to decide): `filter_results_{7_30,9_13,kde,kde_sep_flag,kde_sep_flag_no_extended_artifact}`, `catalog_results_partial_backup/`, `gaia_cache_pre_reextract/`, `kde_envelopes_pre_ps1_rebuild/`, and one-off diagnostic CSVs/logs.
9. [ ] `lt_pre9` worktree (`git worktree remove ../lt_pre9`) and the `~/berger/*_tmp.py` scratch scripts: left for you (removal was blocked by the permission check).
10. [~] 2026-10-25: laptop `Data/_stale_2026-10-23/` deleted. The cluster `_stale_2026-10-23/` (now also holding the pre-#15 field folders and `combined`) is still there; deleting it was blocked by the permission check. Run by hand: `rm -rf /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/{_stale_2026-10-23,_ost40_replacements,filter_results_final_ps1gap_rerun}`.
