# BIG IDEAs Lab Glycemic Variability — verified data dictionary

**Source:** PhysioNet v1.1.2 (Open Access, ODC-By 1.0). All fields below were verified by reading
the **actual downloaded files** (not the dataset description). Used as the **second cohort** for
cross-cohort transfer. Download status: **16/16 participants** have both `Dexcom_<ID>.csv` and
`Food_Log_<ID>.csv` (the large raw wearable signals ACC/BVP/EDA/HR/IBI/TEMP were deliberately **not**
downloaded — not needed, and ~34 GB).

## File layout (verified)
```
physionet.org/files/big-ideas-glycemic-wearable/1.1.2/
└── <ID>/                      # ID = 001 … 016
    ├── Dexcom_<ID>.csv        # CGM (used)
    ├── Food_Log_<ID>.csv      # meals + macros (used)
    ├── ACC_<ID>.csv  BVP_<ID>.csv  EDA_<ID>.csv
    ├── HR_<ID>.csv   IBI_<ID>.csv  TEMP_<ID>.csv   # raw wearable signals (NOT downloaded)
```
Population (verified from the dataset page): n=16, inclusion A1C **5.2–6.4%** (normal-to-prediabetic).

## CGM — `Dexcom_<ID>.csv` (Dexcom Clarity export format)
Header columns: `Index, Timestamp (YYYY-MM-DDThh:mm:ss), Event Type, Event Subtype, Patient Info,
Device Info, Source Device ID, Glucose Value (mg/dL), Insulin Value (u), Carb Value (grams),
Duration (hh:mm:ss), Glucose Rate of Change (mg/dL/min), Transmitter Time (Long Integer)`.
- **Parsing rule (verified):** the first ~12 rows are metadata (`Event Type` ∈ {FirstName, LastName,
  patient info, device}); **glucose readings are rows where `Event Type == "EGV"`**, with the value
  in **`Glucose Value (mg/dL)`** and time in **`Timestamp (YYYY-MM-DDThh:mm:ss)`**.
- Sampled at **5-min** cadence (confirmed: consecutive EGV rows at 17:23:32 → 17:28:32 → 17:33:32).

## Meals — `Food_Log_<ID>.csv`
Header columns (verified): `date, time, time_begin, time_end, logged_food, amount, unit,
searched_food, calorie, total_carb, dietary_fiber, sugar, protein, total_fat`.
- **Meal anchor:** `time_begin` (full datetime, e.g. `2020-02-13 18:00:00`).
- **Macro features (map to CGMacros):** `total_carb`→Carbs, `protein`→Protein, `total_fat`→Fat,
  `dietary_fiber`→Fiber, `calorie`→Calories (`sugar` is extra, no CGMacros equivalent).

## Harmonization with CGMacros (for transfer, issues #16/#23)
| Concept | CGMacros | BIG IDEAs |
|---|---|---|
| CGM glucose (primary) | `Dexcom GL` column | `Glucose Value (mg/dL)` (EGV rows) |
| CGM cadence | per-minute rows | 5-min |
| Meal time anchor | row with `Meal Type` non-null (`Timestamp`) | `Food_Log` `time_begin` |
| Carb / Protein / Fat / Fibre | `Carbs/Protein/Fat/Fiber` | `total_carb/protein/total_fat/dietary_fiber` |
| Calories | `Calories` | `calorie` |
| Population | 15 healthy / 16 pre-DM / 14 T2D | 16 @ A1C 5.2–6.4% |

Both reduce to the same task: **for each meal, take the 0–120 min post-meal Dexcom trace → iAUC**;
features = meal macros (+ optional context/personal). This is what makes the cross-cohort transfer
experiment well-posed.

## Verified data-quality findings (from the actual files, 2026-06-18)
- **CGM ↔ Food_Log date misalignment in 4/16 subjects (007, 013, 015, 016).** Their `Dexcom_<ID>.csv`
  and `Food_Log_<ID>.csv` timestamps fall in **entirely different date ranges** (months apart) — e.g.
  subj 007: Dexcom 2020-03-14…22 vs Food_Log 10/16/2019…10/24/2019. The de-identification
  date-shifting was applied **inconsistently between the two files** for these subjects, so no meal
  can be aligned to a glucose window. **Verified by inspecting raw timestamps** (not a loader bug);
  these 4 subjects are excluded → **12/16 evaluable**. Reported as a finding/limitation.
- **Mixed date formats** across subjects' food logs (ISO `YYYY-MM-DD` vs US `M/D/YYYY`); the loader
  must parse both.
- **Subject 003** `Food_Log_003.csv` is **headerless** (11 cols, no `protein`/`total_fat`); carbs +
  calorie present so its meals remain usable (missing macros median-imputed for models).
