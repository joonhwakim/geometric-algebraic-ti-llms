# Sourced real-world value tables

Every familiar-item set the paper uses is built from one of these tables. Each item set's
`*_provenance.json` names its table (`source_table`), the selection rule (`spacing`, `step_ratio`)
and the value of each item (`items`, or `filler` in the six disjoint records); a record whose
`source_citation` is empty or absent cites its table's source below. Each table is an extraction
from one citable publication (the `*_familiarity.csv` tables and the area table add 2024 English
Wikipedia page views from the Wikimedia REST API, user traffic), never model-generated estimates.

| File | Source | Contents | License |
|---|---|---|---|
| `animals_pantheria_mass.csv` | PanTHERIA (Jones et al. 2009, *Ecology* 90:2648) | adult body mass (kg), 36 animals | CC0 1.0 |
| `cities_demographia_wua_2025.csv` | Demographia, *World Urban Areas*, 20th Edition, August 2025 (2025.11 revision), Schedule 2 (`http://www.demographia.com/db-worldua.pdf`) | built-up ("continuously urbanised") urban-area population, 2025 estimate; 941 of Schedule 2's 942 urban areas (not Guangzhou-Shenzhen, rank 1), 500,000–45,115,000 | © Demographia; copying and republishing permitted only without alteration |
| `cities_band_familiarity.csv` | the Demographia table above, with page views | 325 urban areas; `display` is a shortened name, `urban_area` is Demographia's | as above; page views CC0 |
| `countries_population_2026.csv` | Wikipedia, *List of countries and dependencies by population* | total population: the national census authority's most recent estimate, else the UN Population Division projection | CC BY-SA 4.0 |
| `countries_full_familiarity.csv` | the same Wikipedia list (full table), with page views | 139 countries | CC BY-SA 4.0; page views CC0 |
| `countries_band_familiarity.csv` | the same Wikipedia list, with page views | 61 countries, with a `display` name | CC BY-SA 4.0; page views CC0 |
| `countries_area_worldbank.csv` | World Bank, World Development Indicators, Surface area (sq. km), AG.SRF.TOTL.K2 (source: FAO), most recent value; with page views | 127 countries | CC BY 4.0; page views CC0 |
| `us_states_population_2024.csv` | US Census Bureau, Vintage 2024 Population Estimates (NST-EST2024-ALLDATA) | 50 states, estimates for 2024-07-01 | public domain (US government work) |

## Why Demographia rather than the UN for cities

The UN World Urbanization Prospects 2025 revision restructured around "degree of
urbanization" and no longer publishes the named-agglomeration file; the
agglomeration series that Wikipedia tabulates covers only the 85 cities above
5 million, a total span of just 8.3x. Spread over 11 items that is a 1.235x step,
which puts symbolic distance 6 at a 3.6x population ratio — below the point where
model accuracy saturates, so the top of the accuracy range is never reached.
Demographia's built-up-urban-area series uses a single definition across the whole
size range (500k and up), so an 11-item list can be spaced to span the model's
entire accuracy range. Populations differ from UN agglomeration figures because the
definitions differ; the two are never mixed.

**Mixed-vintage caveat (countries).** Unlike the city table, this list mixes national and UN
sources with differing reference dates. Inter-source and inter-year disagreement for these
countries is a few percent; the main set's neighbors are at least 1.46x apart (at least 1.16x in
its replicates), but the closest neighbors of the item-spacing sets are 4.5% apart (Brazil and
Nigeria in `countries_by_population_scale2`).
