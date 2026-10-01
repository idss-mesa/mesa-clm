# Dataset: DP1.00004.001 / BP_30min
Product: DP1.00004.001 — Barometric pressure. Barometric pressure is available as one- and thirty-minute averages for station pressure, which is determined from 0.1 Hz observations. Barometric pressure corrected to sea level and surface level (defined as water surface at aquatic sites and soil surface at terrestrial sites) is derived from station pressure averages and available at one- and thirty-minute increments. Observations are made by a single digital barometer located on the tower infrastructure and a single digital barometer located on the aquatic meteorological station.
Source: NEON (National Ecological Observatory Network) Data API, basic package, stacked across site-months.
Sites: SRER (Santa Rita Experimental Range NEON, AZ, domain D14 Desert Southwest; Desert shrubland dominated by Creosote bush).
Rows: 2928; months covered: 2025-06 to 2026-08 (2 distinct months).

## Columns (name | NEON description | type | unit | profile)
- startDateTime | Date and time at which a sampling is initiated | dateTime | - | datetime, n=2928, 2928 distinct, min=2025-06-01T00:00Z, max=2026-08-31T23:30Z
- endDateTime | Date and time at which a sampling is completed | dateTime | - | datetime, n=2928, 2928 distinct, min=2025-06-01T00:30Z, max=2026-09-01T00:00Z
- staPresMean | Arithmetic mean of station pressure | real | kilopascal | numeric, n=2903, 2697 distinct, min=89.5111, median=90.12, max=90.7108; 25 blank
- staPresMinimum | Minimum station pressure | real | kilopascal | numeric, n=2903, 194 distinct, min=89.4718, median=90.1019, max=90.6485; 25 blank
- staPresMaximum | Maximum station pressure | real | kilopascal | numeric, n=2903, 185 distinct, min=89.5318, median=90.1384, max=90.7585; 25 blank
- staPresVariance | Variance in station pressure | real | kilopascalsSquared | numeric, n=2903, 95 distinct, min=0, median=5e-05, max=0.0059; 25 blank
- staPresNumPts | Number of points used to calculate the arithmetic mean of station pressure | real | number | numeric, n=2903, 24 distinct, min=10, median=180, max=180; 25 blank
- staPresExpUncert | Expanded uncertainty for station pressure | real | kilopascal | numeric, n=2903, 48 distinct, min=0.04721, median=0.0473, max=0.04867; 25 blank
- staPresStdErMean | Standard error of the mean for station pressure | real | kilopascal | numeric, n=2903, 209 distinct, min=0, median=0.00054, max=0.00572; 25 blank
- staPresFinalQF | Station pressure quality flag indicating whether a data product has passed or failed an overall assessment of its quality, detailed in NEON.DOC.001113 (1=fail, 0=pass) | unsigned integer | - | numeric, n=2928, 2 distinct, min=0, median=0, max=1; top: 0 (2880), 1 (48)
- corPres | Mean station pressure corrected to sea level | real | kilopascal | numeric, n=2903, 2872 distinct, min=99.5998, median=100.75, max=101.653; 25 blank
- corPresExpUncert | Expanded uncertainty for pressure corrected to sea level | real | kilopascal | numeric, n=2903, 1227 distinct, min=0.07167, median=0.07813, max=0.08957; 25 blank
- corPresFinalQF | Pressure corrected to sea level quality flag indicating whether a data product has passed or failed an overall assessment of its quality, detailed in NEON.DOC.000653 and NEON.DOC.001113 (1=fail, 0=pass) | unsigned integer | - | numeric, n=2928, 2 distinct, min=0, median=0, max=1; top: 0 (2878), 1 (50)
- siteID | NEON site code | string | - | 1 distinct; top: SRER (2928)
- domainID | Unique identifier of the NEON domain | string | - | 1 distinct; top: D14 (2928)
- month | Month (YYYY-MM) of the source file; added by neon-ducklake | string | - | 2 distinct; top: 2026-08 (1488), 2025-06 (1440)
- release | NEON release tag of the source file (RELEASE-YYYY or PROVISIONAL); added by neon-ducklake | string | - | 2 distinct; top: PROVISIONAL (1488), RELEASE-2026 (1440)
- publicationDate | Publication timestamp of the source file; added by neon-ducklake | dateTime | - | datetime, n=2928, 2 distinct, min=2025-07-10T22:49:05Z, max=2026-09-07T18:43:32Z
- sourceFile | NEON file name the row was read from; added by neon-ducklake | string | - | 2 distinct; top: NEON.D14.SRER.DP1.00004.001.000.035.030.BP_30min.2026-08.ba… (1488), NEON.D14.SRER.DP1.00004.001.000.035.030.BP_30min.2025-06.ba… (1440)
- horizontalPosition | Horizontal (HOR) index of the sensor position, from the file name | string | - | 1 distinct; top: 000 (2928)
- verticalPosition | Vertical (VER) index of the sensor position, from the file name | string | - | 1 distinct; top: 035 (2928)
- tmi | Temporal index (averaging interval code), from the file name | string | - | 1 distinct; top: 030 (2928)
