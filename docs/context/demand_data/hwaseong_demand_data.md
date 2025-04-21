## 🧩 **1. Overview of Living Lab Demand Patterns (기반 수요 패턴)**

- The Living Lab area centers around coordinates `[37.206462, 126.827929]` and uses **Uber’s H3 Resolution 9** hexagonal grid system.
- Zones are grouped into **9 groups**, and **daily OD (Origin-Destination) trip volumes** are allocated based on:
  - Number of households per zone
  - Number of commercial facilities
- Assumes operation hours from **7 AM to 10 PM**.
- OD volumes provided follow the format of `base.csv`.

---

## 🗂 **2. OD Data Sample Table**

- **Group, Zone, Rate (%)** format. Example from Group 1:
  ```
  Group  Zone  Rate
  1      26    37.0%
  1      25    0.3%
  ...
  ```

- **Column types in OD table**:
  - `O`: Origin Zone (integer)
  - `D`: Destination Zone (integer)

---

## 🏘 **3. Zone Population & Attributes Table**

Includes columns:
- `Group`, `Zone`, `CB`, `LA`, `Pop`, `Rate`
  - `CB`: Commercial Buildings
  - `LA`: Leisure/Attraction Facilities
  - `Pop`: Population
  - `Rate`: Percentage of group trips

Example:
```
Group  Zone  CB  LA   Pop   Rate
1      26    0   1131 3393  37.0%
1      25    20  0    30    0.3%
...
```

This continues across all 9 groups, defining characteristics of each zone.

---

## 📡 **4. Real-Time Demand Generation (실시간 수요)**

- Based on OD volumes, **random coordinate points** are generated within each zone.
- Each OD demand is assigned a `call ID` based on departure time order.
- Format in `base_real_time.csv`:
  ```
  Idx  O   D
  0    71  34
  ```

---

## 🔄 **5. Demand-Based Dispatch Comparison (수요 변화에 따른 배차 결과 비교)**

- Compares **dispatch outcomes** for:
  - Time-based extreme fluctuations in demand
  - Spatial concentration
  - Differences between predicted and actual demand

---

## 🕒 **6. Time-Based OD Redistribution**

- Total OD volume reallocated based on hourly public transit peak ratios (7 AM to 10 PM).
- Generates real-time demand consistent with scenarios.

**Example data:**
- `Base.csv` vs. `S1.csv` and `S1_real_time.csv`
- Ratio of demand change: (S1/Base)

| Base | S1  | Ratio |
|------|-----|-------|
| 126  | 156 | 1.24  |
| 126  | 160 | 1.27  |
| ...  | ... | ...   |
| 126  | 17  | 0.13  |

---

## 📍 **7. Scenario: OD Concentration Based on POI (관광지 기반 집중)**

- POIs like leisure, commercial, medical are **concentrated in Zones 4 and 6**.
- OD constructed to reflect high POI density.

Data files:
- OD: `S2.csv`
- Real-time: `S2_real_time.csv`

---

## 🧬 **8. Combined Scenario (첨두율 + 공간 집중)**

- Combines time-based peaks (S1) + POI concentration (S2)
- File: `S3.csv` and `S3_real_time.csv`
- Also uses card data time-distribution to reflect **spatiotemporal variance**
  - File: `S4.csv`, `S4_real_time.csv`
  - Highlights **difference in time-based OD direction** vs card-based data.

---

## 🔮 **9. Scenario-Based Randomized Demand Adjustments**

- Generates **variations of OD volumes** from base scenario using random rules.
- Applies 3 ratios: **0.5x, 1.0x, 1.5x**
- Each has 3 rules for variability.

### 📊 **MAPE (Mean Absolute Percentage Error) Results**:

| File                       | Ratio | Rule | MAPE (%) |
|---------------------------|-------|------|----------|
| base_trip0.5_rule1.csv    | 0.5   | 1    | 57.28    |
| base_trip1_rule2.csv      | 1.0   | 2    | 48.74    |
| base_trip1.5_rule3.csv    | 1.5   | 3    | 78.61    |
| ...                       | ...   | ...  | ...      |

- MAPE excludes OD pairs with 0 volume.

---

## 🧮 **10. Formula: Random Demand Generation Logic**

Defines how variable demand is calculated:

```
vʰ,o,d_s′ = vʰ,o,d_s ± vʰ,o,d_s × xᵢ if vʰ,o,d_s ≥ 0
```

Where:
- `vʰ,o,d_s′`: varied OD demand at hour h
- `xᵢ`: random variation factor for each rule r ∈ {1,2,3}
- `y`: value drawn from a conditional random distribution

---

## 📁 **11. Data Provided**

| Type                  | File Name                     |
|-----------------------|-------------------------------|
| Zone shapefile        | `OD_Zone.shp`                 |
| Zone centroids        | `Zone_centroid.shp`           |
| Base OD volume        | `base.csv`                    |
| Scenario OD volumes   | `S1.csv`, `S2.csv`, `S3.csv`, `S4.csv` |
| Base real-time demand | `base_real_time.csv`          |
| Scenario real-time    | `S1_real_time.csv`, etc.      |
| Randomized OD files   | `base_trip*.csv` (x9 total)    |
| Table schema          | `테이블 정의서.xlsx`          |
