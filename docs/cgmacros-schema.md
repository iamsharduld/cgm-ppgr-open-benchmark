# CGMacros — verified data dictionary

**Source:** PhysioNet v1.0.0 (Open Access, CC BY-NC-SA 4.0). Field definitions below are transcribed
from the dataset's **own** data-dictionary files (`DataDictionary_*.csv`), which were downloaded and
read directly. Per-participant data files (`CGMacros-001.csv` … `CGMacros-045.csv`) live inside
`CGMacros_dateshifted365.zip`; their column headers are cross-checked against this dictionary on
extraction. **Nothing here is inferred** — every field is quoted from the official dictionary.

## File layout (verified from the directory listing)
```
physionet.org/files/cgmacros/1.0.0/
├── CGMacros_dateshifted365.zip      # per-participant CSVs + meal photos (the 627.9 MB bulk)
├── DataDictionary.pdf
├── DataDictionary_Bio.csv           # subject-level field defs
├── DataDictionary_CGMacros-00X.csv  # per-participant time-series field defs
├── DataDictionary_Gut_Health_Test.csv
├── DataDictionary_Microbes.csv
├── LICENSE.txt
└── SHA256SUMS.txt
```

## Per-participant time series — `CGMacros-0XX.csv` (one row per minute; meal rows carry macros)
| Field | Range / type | Meaning |
|---|---|---|
| `Timestamp` | Month/Day/Year HH:MM (date-shifted) | Incremental timestamp of the reading |
| `Libre GL` | 40–400 mg/dL | Abbott FreeStyle **Libre Pro** glucose |
| `Dexcom GL` | 40–400 mg/dL | **Dexcom G6 Pro** glucose ← **primary stream for this benchmark** |
| `HR` | 30–176 bpm | Fitbit Sense heart rate (last minute) |
| `Calories (Activity)` | 0–16.178 | Fitbit calories burned, last minute |
| `METs` | 10–176 | Fitbit Metabolic Equivalent ×10, last minute |
| `Meal Type` | Breakfast / Lunch / Dinner | **Non-null marks a meal start** (used to anchor PPGR windows) |
| `Calories` | 30–1180 | Estimated calories of the meal |
| `Carbs` | 0–176 g | Meal carbohydrate ← feature |
| `Protein` | 3–176 g | Meal protein ← feature |
| `Fat` | 0–176 g | Meal fat ← feature |
| `Fiber` | 0–176 g | Meal fibre ← feature |
| `Amount Consumed` | 0–100 % | Estimated % of meal eaten |
| `Image Path` | path | Location of the meal photo (not used in this benchmark) |

## Subject-level — `bio.csv` (from `DataDictionary_Bio.csv`)
Key fields (verified): `Age` (18–69), `Gender` (F/M), `BMI` (20.69–49.09), `Body weight` (lb),
`Height` (in), `Self-identify` (race/ethnicity), `A1c PDL (Lab)` (4.6–8.5, **mmol/mol**),
`Fasting GLU - PDL (Lab)` (79–218 mg/dL), `Insulin` (2.5–46.4 mcU/mL), `Triglycerides`,
`Cholesterol`, `HDL`, `Non HDL`, `LDL (Cal)` (note: 800 = calc error), `VLDL (Cal)` (note: 400 =
erroneous), `Cho/HDL Ratio`, three `Contour Fingerstick GLU` readings + times.

> ⚠️ **Diabetes-group labels:** the bio table carries A1c/fasting glucose but the published
> grouping into *15 healthy / 16 pre-diabetes / 14 T2D* (per the dataset description) must be
> confirmed against the actual bio file once extracted (whether the group is an explicit column or
> derived from A1c). Tracked in issue #8 — **not assumed here.**

## Microbiome & gut health (optional features)
- `DataDictionary_Microbes.csv`: **1,979** binary presence indicators (`0/1`) per bacterial taxon.
- `DataDictionary_Gut_Health_Test.csv`: **22** Viome gut-health scores, each an ordinal factor
  (1=Not Optimal, 2=Average, 3=Good) — e.g. *Metabolic Fitness*, *Active Microbial Diversity*,
  *Butyrate Production Pathways*, *Inflammatory Activity*.

## Implications for the benchmark
- **PPGR target** is computed from the **`Dexcom GL`** column over the 0–120 min window after each
  row where `Meal Type` is non-null (matches BIG IDEAs, which is Dexcom-only — see
  [`bigideas-schema.md`](./bigideas-schema.md)).
- **Meal features:** `Carbs/Protein/Fat/Fiber` (+`Calories`); **context:** time-of-day from
  `Timestamp`, recent `METs`/`Calories (Activity)`; **personal:** bio + (optional) microbiome.
- `Libre GL` provides the **CGM-brand sensitivity** analysis (issue #25).

## Verified against the extracted data (2026-06-18) — corrections to the official dictionary
Validated by reading the real `CGMacros-0XX.csv` (45 files) + `bio.csv` after extraction. Three
points where the dataset's own dictionary was incomplete/misleading:

1. **`Meal Type` has FOUR semantic values, not three:** Breakfast, Lunch, Dinner, **Snacks** (the
   `DataDictionary_CGMacros-00X.csv` listed only Breakfast/Lunch/Dinner). **And they appear in ~10 raw
   casing/spelling variants** across files (`Breakfast`/`breakfast`, `Snacks`/`snack`/`Snack`/`snack 1`,
   etc.) — the adapter anchors on *any* non-null `Meal Type` (so all eating events are kept) and
   normalizes case/spelling only for snack counting. Per prereg §3.3 snacks are treated as meals if
   they pass inclusion; a **snack-excluded sensitivity** is reported (whole-cohort snack rows = 343).
2. **`A1c PDL (Lab)` is in `%` (NGSP), not `mmol/mol`** as the dictionary states. Observed range
   4.6–8.5 (%); values like 5.4/6.5 are clearly NGSP %.
3. **No explicit diabetes-group column exists** in `bio.csv` (24 cols: subject, Age, Gender, BMI,
   labs, fingersticks…). The 15 healthy / 16 pre-DM / 14 T2D split is **derived from A1c via standard
   ADA thresholds**, which **reproduces the published counts EXACTLY**:
   - **healthy** A1c < 5.7 → **15**; **pre-DM** 5.7 ≤ A1c < 6.5 → **16**; **T2D** A1c ≥ 6.5 → **14**.
   - ⇒ Prereg §3.1 **primary non-diabetic population = healthy + pre-DM = 31 subjects** (derivation
     pre-specified here, ADA cutoffs, not chosen post-hoc).

**Column-name quirks in `CGMacros-0XX.csv`** (handle in the adapter): a leading `Unnamed: 0` index
column; `Amount Consumed ` has a trailing space; the photo column is `Image path` (lower-case `p`).
Meal macros are `Calories, Carbs, Protein, Fat, Fiber`; Fitbit activity is the separate
`Calories (Activity)`/`METs`/`HR`. Meal rows are those with non-null `Meal Type`, and a meal row's
`Timestamp` coincides with a CGM reading (per-minute rows), so meal↔CGM alignment is exact.
