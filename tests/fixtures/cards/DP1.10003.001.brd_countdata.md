# Dataset: DP1.10003.001 / brd_countdata
Product: DP1.10003.001 — Breeding landbird point counts. Count, distance from observer, and taxonomic identification of breeding landbirds observed during point counts
Source: NEON (National Ecological Observatory Network) Data API, basic package, stacked across site-months.
Sites: HARV (Harvard Forest & Quabbin Watershed NEON, Massachusetts, domain D01 Northeast; temperate deciduous/mixed forest) and SRER (Santa Rita Experimental Range NEON, Arizona, domain D14 Desert Southwest; semi-arid desert grassland/shrubland).
Rows: 15484; months covered: 2020-04 to 2024-06 (12 distinct months).

## Columns (name | NEON description | type | unit | profile)
- uid | Unique ID within NEON database; an identifier for the record | string | - | (identifier; not profiled)
- namedLocation | Name of the measurement location in the NEON database | string | - | 20 distinct; top: SRER_003.birdGrid.brd (1055), SRER_002.birdGrid.brd (1022), SRER_007.birdGrid.brd (1019), SRER_005.birdGrid.brd (1017), SRER_001.birdGrid.brd (994), SRER_010.birdGrid.brd (979), SRER_009.birdGrid.brd (969), SRER_004.birdGrid.brd (885)
- domainID | Unique identifier of the NEON domain | string | - | 2 distinct; top: D14 (9416), D01 (6068)
- siteID | NEON site code | string | - | 2 distinct; top: SRER (9416), HARV (6068)
- plotID | Plot identifier (NEON site code_XXX) | string | - | 20 distinct; top: SRER_003 (1055), SRER_002 (1022), SRER_007 (1019), SRER_005 (1017), SRER_001 (994), SRER_010 (979), SRER_009 (969), SRER_004 (885)
- plotType | NEON plot type in which sampling occurred: tower, distributed or gradient | string | - | 1 distinct; top: distributed (15484)
- pointID | Identifier for a point location | string | - | 9 distinct; top: 1 (1768), 2 (1737), 7 (1735), 5 (1730), 8 (1729), 4 (1725), 3 (1722), 6 (1691)
- startDate | The start date-time or interval during which an event occurred | dateTime | - | 898 distinct; top: 2024-05-03T12:19Z (36), 2024-05-03T12:50Z (34), 2024-05-03T13:21Z (32), 2020-04-22T13:15Z (31), 2023-04-25T12:33Z (31), 2023-05-02T12:24Z (31), 2024-05-02T12:50Z (31), 2024-05-03T12:35Z (31)
- boutNumber | Number of the sampling bout within a calendar year, beginning with 1 | unsigned integer | number | 1 distinct; top: 1 (15484)
- eventID | An identifier for the set of information associated with the event, which includes information about the place and time of the event | string | - | 10 distinct; top: BRD.SRER.2024.1 (2094), BRD.SRER.2020.1 (2027), BRD.SRER.2023.1 (1907), BRD.SRER.2022.1 (1753), BRD.HARV.2021.1 (1688), BRD.SRER.2021.1 (1635), BRD.HARV.2020.1 (1168), BRD.HARV.2022.1 (1100)
- pointCountMinute | The minute of sampling within the point count period | unsigned integer | - | 7 distinct; top: 1 (7485), 2 (2121), 3 (1592), 4 (1445), 5 (1369), 6 (1316), 88 (156)
- targetTaxaPresent | Indicator of whether the sample contained individuals of the target taxa | string | - | 2 distinct; top: Y (14833), N (651)
- taxonID | Species code, based on one or more sources | string | - | 179 distinct; top: CACW (1098), MODO (1010), OVEN (1000), GAQU (995), WWDO (911), REVI (717), VEER (471), CBTH (464)
- scientificName | Scientific name, associated with the taxonID. This is the name of the lowest level taxonomic rank that can be determined | string | - | 179 distinct; top: Campylorhynchus brunneicapillus (1098), Zenaida macroura (1010), Seiurus aurocapilla (1000), Callipepla gambelii (995), Zenaida asiatica (911), Vireo olivaceus (717), Catharus fuscescens (471), Toxostoma curvirostre (464)
- taxonRank | The lowest level taxonomic rank that can be determined for the individual or specimen | string | - | 5 distinct; top: species (14690), family (67), class (40), genus (31), subspecies (5)
- vernacularName | A common or vernacular name | string | - | 170 distinct; top: Cactus Wren (1098), Mourning Dove (1010), Ovenbird (1000), Gambel's Quail (995), White-winged Dove (911), Red-eyed Vireo (717), Veery (471), Curve-billed Thrasher (464)
- observerDistance | Radial distance between the observer and the individual(s) being observed | real | meter | numeric, n=14670, min=1, median=59, max=999
- detectionMethod | How the individual(s) was (were) first detected by the observer | string | - | 10 distinct; top: singing (7644), calling (5600), visual (958), drumming (219), other aural (e.g. wing beats) (170), calling and singing (145), visual and singing (42), unknown (32)
- visualConfirmation | Whether the individual(s) was (were) seen after the initial detection | string | - | 2 distinct; top: No (12949), Yes (1884)
- sexOrAge | Sex of individual if detectable, age of individual if individual can not be sexed | string | - | 4 distinct; top: Unknown (14304), Male (375), Female (143), Juvenile (11)
- clusterSize | Number of individuals in a cluster (a group of individuals of the same species) | unsigned integer | number | 13 distinct; top: 1 (14413), 2 (316), 3 (57), 4 (26), 5 (7), 7 (4), 10 (3), 21 (2)
- clusterCode | Alphabetic code (A-Z) linked to clusters (groups of individuals of the same species) spanning multiple records | string | - | 2 distinct; top: A (159), B (10)
- identifiedBy | An identifier for the technician who identified the specimen | string | - | 4 distinct; top: RADIN (9416), KKLAP (3212), JGLAG (1688), WFREE (1168)
- identificationHistoryID | Identifier for linking records related to this identification history | string | - | all blank
