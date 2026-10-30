# ZCTA civic participation table

**Not for decisions about food distribution, eligibility, pantry siting or any individual.**

This table does not choose where to run voter registration or GOTV; that is the food bank's decision with its counsel. Until then it supports a listening shortlist only.

Missing data is not missing civic life.

A low score is not low interest.

## Data card

| | |
|---|---|
| What it is | One row per ZIP Code Tabulation Area (ZCTA) that touches the food bank's 10-county service region: 240 rows. |
| Counties | Austin, Brazoria, Chambers, Fort Bend, Harris, Liberty, San Jacinto, Trinity, Walker, Waller. |
| What it measures | Voter turnout in the 2022 and 2024 general elections compared with the number of adult citizens (P1), plus a count of civic organizations and six neighbourhood conditions from the Census. |
| Unit | Areas, never people. No individual voter records, names or addresses were used or are included. |
| Nonpartisan | Turnout counts only. No party, candidate or vote-choice data. |
| Ranked | 175 of 240 ZCTAs. The other 65 are shown but not ranked (see "Why some rows are not ranked"). They hold about 4 percent of the region's registered voters. |
| Intended use | Planning where to listen first: focus groups and outreach conversations. |
| Not intended for | Food distribution, eligibility, pantry siting, decisions about any person, or choosing where to run voter registration or GOTV. |
| Built | 30 October 2026, from public files downloaded that day. |
| License | Texas Legislative Council data under CC BY; Census and IRS data are U.S. government works. See "Credits". |

## Files

| File | What it holds |
|---|---|
| `zcta_civic.xlsx` | Sheet `data`: the ZCTA table, with rows that are not ranked shaded grey and sorted last. Sheet `county`: the county table. Sheet `read_me`: the statements above and a definition for every column. |
| `county_civic.csv` | One row per county (10 rows). Same layout as the ZCTA CSV: use statement on row 1, column names on row 2. |
| `zcta_civic.csv` | The same table. Row 1 holds the use statement; the column names are on row 2. In Python use `pd.read_csv(path, skiprows=1, dtype={"ZCTA": str, "county_main": str})` so ZIP codes keep their leading digits. |
| `zcta_civic.geojson` | ZCTA shapes (EPSG:4326) with the same columns as properties, plus `use_limit` and `credits` at the top level. |
| `zcta_p1_map.png` | P1 in five equal-count classes over ranked ZCTAs; not-ranked ZCTAs in grey; county outlines. For the food bank's hand-off only; use county views in recorded talks. |

Column definitions live in the `read_me` sheet. This file explains how the table was built and where it is weak.

## How P1 is built

P1 = average of 2022 and 2024 general-election turnout divided by citizen voting-age population, less adults in correctional facilities. A reference measure, not a score of a community.

P1_rank 100 = highest measured participation among ranked ZCTAs.

Step by step:

1. **Turnout.** The Texas Legislative Council publishes registered voters and ballots cast for each voting district (VTD) in each general election. Voting districts and ZIP areas do not line up, so each district's counts are shared out to ZCTAs in proportion to the voting-age population of the 2020 Census blocks it holds. Each block is placed in one ZCTA using Census boundary files.
2. **Adult citizens.** Citizen voting-age population (`cvap`) comes from the Census American Community Survey (ACS) 2020-2024 five-year estimates, table B05003, with its margin of error.
3. **Correctional adjustment.** People held in adult correctional facilities in the 2020 Census (`corr_adult`) are subtracted from `cvap`. See "People in jails and detention".
4. **P1** = (to22 + to24) / (2 × cvap_adj).
5. **Rank.** P1 is turned into a percentile rank from 0 to 100 among ranked ZCTAs.
6. **Rank interval.** `cvap` is a survey estimate, so the rank is recomputed 1,000 times with `cvap` drawn from within its margin of error. `rank_p05` and `rank_p95` are the 5th and 95th percentiles of those ranks. The median width of this interval is about 15 rank points. Two ZCTAs whose intervals overlap should not be read as different.

P1 can come out above 1 in fast-growing areas, where 2024 voters outnumber the older survey estimate of adults. Those values are kept as measured and flagged `GROWTH`.

## Why some rows are not ranked

A row is not ranked when the measure would mislead for that area. `not_ranked_reason` holds one or more codes:

| Code | Rule | Rows |
|---|---|---|
| LAND | Less than half the ZCTA's land is in the region | 26 |
| PARTIAL | Less than 90 percent of the ZCTA's 2020 population lives in the region | 30 |
| POP | Fewer than 1,500 residents | 35 |
| NO_CVAP | No usable adult citizen estimate | 5 |
| GQ | More than half of residents live in group quarters (dorms, jails, nursing homes) | 2 |
| CORRECTIONAL | People in adult correctional facilities are 10 percent or more of voting-age residents | 4 |

Rows overlap, so the counts add to more than 65. P1 is left blank for LAND, PARTIAL and NO_CVAP, because the turnout or the adult count describes a different area from the ZCTA. For POP, GQ and CORRECTIONAL the measured P1 is kept for reference but not ranked.

**Partial ZCTAs.** A ZIP area can cross the region's edge. The table counts only the part inside the region, so a ZCTA that is mostly outside it would be compared on a fraction of its people. Those rows are PARTIAL or LAND.

**Not ranked is not low.** Grey rows are areas the data cannot describe well. They are not areas with low participation.

### Coverage by county

| County | ZCTAs ranked | Share of the county's registered voters in ranked ZCTAs |
|---|---|---|
| Harris | 127 of 135 | 98% |
| Fort Bend | 16 of 23 | 100% |
| Brazoria | 14 of 18 | 88% |
| Austin | 3 of 13 | 87% |
| Chambers | 3 of 7 | 94% |
| Trinity | 2 of 8 | 88% |
| Waller | 3 of 8 | 83% |
| San Jacinto | 3 of 6 | 64% |
| Liberty | 3 of 12 | 55% |
| Walker | 1 of 10 | 47% |

Counties are assigned by `county_main`, the county holding most of each ZCTA's land. In the rural counties most ZIP areas are small or cross county lines, so few are ranked. For Walker, Liberty and San Jacinto, a ZCTA ranking does not describe most of the county; use the county table there.

## County table

`county_civic.csv` and the `county` sheet give the same measure for each whole county. Where ZIP areas are too small or cross county lines, this is the fallback.

- Registered voters and ballots cast are the Texas Legislative Council county totals, summed straight from voting districts with no sharing out, so they are exact.
- `cvap` and the five neighbourhood conditions are the ACS county estimates, with margins of error. County estimates rest on much larger samples than ZIP estimates.
- `P1` uses the same formula as the ZCTA table. `P1_low` and `P1_high` show P1 with `cvap` moved to the top and bottom of its margin of error.
- `zctas_ranked`, `zctas_total` and `reg24_share_ranked` show how much of the county the ZCTA ranking covers.
- `civic_orgs_main` adds up `civic_orgs_n` over ZCTAs whose `county_main` is the county, so it carries the same mailing ZIP limits.
- Counties are not ranked. Ten rows are too few for a rank to mean much, and a ranking of counties invites a league table.

In Walker County about 15,000 people live in adult correctional facilities (2020 Census). They are subtracted before P1 is computed, which is why its `cvap_adj` is much lower than `cvap`.

## Caution flags

`caution` marks ranked rows to read with care:

| Code | Meaning | Rows |
|---|---|---|
| SPLIT | More than half of the ZCTA's 2024 registration comes from voting districts only partly inside it, so the share-out matters a lot | 57 |
| LOW_N | Fewer than 500 registered voters or fewer than 1,000 adult citizens | 48 |
| GROWTH | 2024 registration above 1.1 times the voting-age population | 20 |
| CORRECTIONAL | Adults in correctional facilities are 5 to 10 percent of voting-age residents | 6 |
| COLLEGE | College housing is 10 percent or more of residents | 5 |
| EDGE | 90 to 98 percent of the ZCTA's people are in the region | 5 |
| FACILITY_CHECK | The correctional count may be a detention facility; its type has not been checked | 1 |

## People in jails and detention

The Census counts people in adult correctional facilities where they are held. Most cannot vote there, so leaving them in the adult count would make the surrounding area look like it votes less. They are subtracted from `cvap`.

Two limits:

- Jails hold people awaiting trial, and many of them are eligible to vote. Subtracting them treats them as not eligible. In the few ZCTAs with a large jail this understates the eligible population.
- Immigration detention is counted with correctional facilities, and people held there are mostly not citizens. They may not be in `cvap` at all, so subtracting them can remove people twice. ZCTA 77032 (near the Houston airport) carries `FACILITY_CHECK` until the facility type is confirmed.

## Civic organizations

`civic_orgs_n` counts active organizations in the IRS Exempt Organizations Business Master File (Texas extract) whose activity code (NTEE) is one of: community and neighbourhood development (S20), community coalitions (S21), neighbourhood and block associations (S22), service clubs (S80, S81, S82) or citizen participation (W24). `civic_orgs_band` groups the count as 0, 1 to 2, 3 to 5 or 6 or more. `hoa_like_n` counts homeowner and property owner associations, found by activity code L5 or by name, and is kept separate.

Limits, in order of size:

- **Mailing ZIP, not where a group meets.** Each organization is counted in the ZIP of its IRS mailing address. A group that meets in one neighbourhood and mails from a treasurer's home or a downtown office is counted in the wrong place.
- **PO box ZIPs are lost.** 139 civic organizations in the region use a ZIP that belongs to no ZCTA (usually a PO box ZIP) and are not counted anywhere.
- **Only groups registered with the IRS.** Churches, mutual aid groups, tenant groups, sports leagues and informal neighbourhood groups often never file. They are not in this count.
- **No neighbourhood association list for the region.** Only the City of Houston keeps one, and it holds names and contact details, so it was not used.

77 ZCTAs show 0 civic organizations. That means none were found in this file, not that none exist.

## Neighbourhood conditions

Five context columns come from the ACS 2020-2024 five-year estimates, each with a 90 percent margin of error: same house as one year ago, owner-occupied housing, high school diploma or above, households that are not limited English speaking, and households with broadband. They are context only and are not used in P1 or the rank. Each is blank for 5 or 6 ZCTAs where the survey has no usable estimate.

## Language

Describe areas, not people. Do not call any area "disengaged", a "civic desert" or "underperforming". A low P1 describes a ballot count against a survey estimate. It says nothing about how much people care, organize or help each other.

## What this table has not shown

- **That turnout measures civic participation.** Voting is one form of civic life. There is no independent measure of the food bank's sense of participation to check P1 against.
- **How accurate the ZIP share-out is.** Totals match the Texas Legislative Council exactly at county level, but the split of each voting district between ZIP areas is an estimate, weakest where `SPLIT` is set and in rural areas.
- **That ranking by P1 leads to better listening choices.** That is for the food bank to judge.

## Checks run

- Registered voters and ballots cast for both years match the Texas Legislative Council's county totals exactly after the share-out.
- Rebuilding with a different weight (total population instead of voting-age population) gives almost the same ranking (Spearman correlation 0.998).
- A rebuild from a clean copy produces identical CSV, GeoJSON and map files.
- The ACS context columns were recomputed independently and matched.
- Tests check that no names, street addresses or individual records reach the outputs, that the use statement and required lines are present, and that not-ranked rows carry no rank.

## Credits

- Texas Legislative Council, Capitol Data Portal (CC BY, http://www.opendefinition.org/licenses/cc-by). Allocated to ZCTA by HAI; not a TLC product.
- U.S. Census Bureau: ACS 2020-2024 5-year, 2020 PL 94-171, TIGER/Line and cartographic boundaries.
- IRS Exempt Organizations Business Master File, Texas extract.
- Not a TLC, Census Bureau or IRS product.

## Rebuilding

Requires Python and [uv](https://docs.astral.sh/uv/).

```bash
uv run python -m hfb.build data/raw
```

This downloads every source file into `data/raw`, then writes `out/` and an internal QA report in `qa/`. To rebuild from files already downloaded, without network access:

```bash
uv run python -m hfb.build data/raw --seed path/to/saved/raw
```

Tests:

```bash
uv run pytest
```
