# Dataset: DP1.10003.001 / brd_perpoint
Product: DP1.10003.001 — Breeding landbird point counts. Count, distance from observer, and taxonomic identification of breeding landbirds observed during point counts
Source: NEON (National Ecological Observatory Network) Data API, basic package, stacked across site-months.
Sites: HARV (Harvard Forest & Quabbin Watershed NEON, Massachusetts, domain D01 Northeast; temperate deciduous/mixed forest) and SRER (Santa Rita Experimental Range NEON, Arizona, domain D14 Desert Southwest; semi-arid desert grassland/shrubland).
Rows: 900; months covered: 2020-04 to 2024-06 (12 distinct months).

## Columns (name | NEON description | type | unit | profile)
- uid | Unique ID within NEON database; an identifier for the record | string | - | (identifier; not profiled)
- namedLocation | Name of the measurement location in the NEON database | string | - | 20 distinct; top: HARV_008.birdGrid.brd (45), HARV_006.birdGrid.brd (45), HARV_001.birdGrid.brd (45), HARV_004.birdGrid.brd (45), HARV_016.birdGrid.brd (45), HARV_012.birdGrid.brd (45), HARV_021.birdGrid.brd (45), HARV_023.birdGrid.brd (45)
- domainID | Unique identifier of the NEON domain | string | - | 2 distinct; top: D01 (450), D14 (450)
- siteID | NEON site code | string | - | 2 distinct; top: HARV (450), SRER (450)
- plotID | Plot identifier (NEON site code_XXX) | string | - | 20 distinct; top: HARV_008 (45), HARV_006 (45), HARV_001 (45), HARV_004 (45), HARV_016 (45), HARV_012 (45), HARV_021 (45), HARV_023 (45)
- plotType | NEON plot type in which sampling occurred: tower, distributed or gradient | string | - | 1 distinct; top: distributed (900)
- pointID | Identifier for a point location | string | - | 9 distinct; top: 7 (100), 8 (100), 9 (100), 6 (100), 3 (100), 5 (100), 2 (100), 1 (100)
- nlcdClass | National Land Cover Database Vegetation Type Name | string | - | 5 distinct; top: shrubScrub (450), deciduousForest (180), evergreenForest (135), mixedForest (90), woodyWetlands (45)
- decimalLatitude | The geographic latitude (in decimal degrees, WGS84) of the geographic center of the reference area | real | decimalDegree | 20 distinct; top: 42.443535 (45), 42.401149 (45), 42.422491 (45), 42.427892 (45), 42.458224 (45), 42.429715 (45), 42.451400 (45), 42.414919 (45)
- decimalLongitude | The geographic longitude (in decimal degrees, WGS84) of the geographic center of the reference area | real | decimalDegree | 20 distinct; top: -72.224036 (45), -72.253238 (45), -72.257323 (45), -72.228126 (45), -72.231982 (45), -72.239787 (45), -72.250100 (45), -72.228418 (45)
- geodeticDatum | Model used to measure horizontal position on the earth | string | - | 1 distinct; top: WGS84 (900)
- coordinateUncertainty | The horizontal distance (in meters) from the given decimalLatitude and decimalLongitude describing the smallest circle containing the whole of the Location. Zero is not a valid value for this term | real | meter | 3 distinct; top: 250.1 (540), 250.2 (270), 250.3 (90)
- elevation | Elevation (in meters) above sea level | real | meter | 20 distinct; top: 246.7 (45), 196.5 (45), 183.8 (45), 185.7 (45), 232.8 (45), 234.0 (45), 175.1 (45), 180.5 (45)
- elevationUncertainty | Uncertainty in elevation values (in meters) | real | meter | 4 distinct; top: 0.1 (495), 0.3 (180), 0.2 (135), 0.4 (90)
- startDate | The start date-time or interval during which an event occurred | dateTime | - | 899 distinct; top: 2024-06-12T13:15Z (2), 2020-06-02T10:14Z (1), 2020-06-02T10:41Z (1), 2020-06-02T11:17Z (1), 2020-06-02T11:50Z (1), 2020-06-02T12:22Z (1), 2020-06-02T12:51Z (1), 2020-06-02T13:14Z (1)
- boutNumber | Number of the sampling bout within a calendar year, beginning with 1 | unsigned integer | number | 1 distinct; top: 1 (900)
- samplingImpracticalRemarks | Technician notes; free text comments accompanying the sampling impractical record | string | - | 1 distinct; top: Terrain Unsafe (could not safely approach to within 25 m of  (2)
- samplingImpractical | Samples and/or measurements were not collected due to the indicated circumstance | string | - | 2 distinct; top: OK (898), logistical (2)
- eventID | An identifier for the set of information associated with the event, which includes information about the place and time of the event | string | - | 10 distinct; top: BRD.HARV.2020.1 (90), BRD.HARV.2021.1 (90), BRD.HARV.2022.1 (90), BRD.HARV.2023.1 (90), BRD.HARV.2024.1 (90), BRD.SRER.2020.1 (90), BRD.SRER.2021.1 (90), BRD.SRER.2022.1 (90)
- startCloudCoverPercentage | Observer estimate of percent cloud cover at start of sampling | unsigned integer | percent | 21 distinct; top: 0 (455), 100 (152), 25 (38), 5 (33), 30 (27), 80 (24), 50 (23), 2 (19)
- endCloudCoverPercentage | Observer estimate of percent cloud cover at end of sampling | unsigned integer | percent | 24 distinct; top: 0 (499), 10 (54), 70 (48), 100 (40), 20 (39), 25 (24), 40 (23), 98 (18)
- startRH | Relative humidity as measured by handheld weather meter at the start of sampling | unsigned integer | percent | numeric, n=898, min=20, median=60, max=100
- endRH | Relative humidity as measured by handheld weather meter at the end of sampling | unsigned integer | percent | numeric, n=889, min=15, median=53, max=98
- observedHabitat | Observer assessment of dominant habitat at the sampling point at sampling time | string | - | 8 distinct; top: shrubland (450), deciduous forest (204), mixed deciduous/evergreen forest (202), evergreen forest (32), wetland (5), riparian (2), grassland (1), other (1)
- observedAirTemp | The air temperature measured with a handheld weather meter | real | celsius | 24 distinct; top: 18 (86), 17 (81), 16 (80), 15 (78), 13 (66), 14 (64), 12 (62), 19 (60)
- kmPerHourObservedWindSpeed | The average wind speed measured with a handheld weather meter, in kilometers per hour | real | kilometersPerHour | 8 distinct; top: 0 (460), 1 (252), 2 (118), 3 (47), 4 (12), 5 (4), 10 (1), 6 (1)
- laboratoryName | Name of the laboratory or facility that is processing the sample | string | - | 1 distinct; top: Bird Conservancy of the Rockies (900)
- samplingProtocolVersion | The NEON document number and version where detailed information regarding the sampling method used is available; format NEON.DOC.######vX | string | - | 3 distinct; top: NEON.DOC.014041vK (540), NEON.DOC.014041vJ (180), NEON.DOC.014041vL (180)
- remarks | Technician notes; free text comments accompanying the record | string | - | 31 distinct; top: BBCU b/f count at pt. 07; PRAW & CSWA heard on way to 3 from (9), OVEN nest w/eggs after pt. 5 near it female flushed; OVEN ne (9), HAWO at nest w/flg. prior to count 001; 2 duck like FO prior (9), Flushed a woodcock between points 9 and 6, alerted by wing t (9), Found an Eastern Whip-poor-will nest near point 3, not far f (9), Saw recent moose scat at point 4. Light drizzle starting at  (9), Light rain starting at 0820 (9), GWWA x BWWA singing a GWWA-type song ('zee ZAA ZAA ZAA'). Ap (9)
- measuredBy | An identifier for the technician who measured or collected the data | string | - | 4 distinct; top: RADIN (450), KKLAP (270), WFREE (90), JGLAG (90)
