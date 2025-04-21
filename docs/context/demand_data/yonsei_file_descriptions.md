## 🔗 **Overall Architecture (Big Picture)**

Imagine this process in stages:

### 1. **Define Spatial Units (Zones)**
### 2. **Generate Base OD Demand**
### 3. **Generate Real-Time Requests**
### 4. **Create Scenarios (Time / Space / Randomness)**
### 5. **Compare Predictions with Reality**
### 6. **Validate Dispatch Performance**

Each stage uses specific files as input/output. Let’s now explore **each file category** and how it connects to the whole.

---

## 🗺️ **1. Spatial Files: Define the Analysis Zones**

| File Name               | Function |
|------------------------|----------|
| `OD_Zone.shp`          | Shapefile representing polygon boundaries of each hexagonal H3 zone |
| `Zone_centroid.shp`    | Centroid coordinates of each zone (used to simulate precise locations for ride origin/destination points) |

🧠 **Used in**: All subsequent steps — OD creation, trip generation, mapping.

---

## 📊 **2. Base OD Demand Files: Core Data Foundation**

| File Name          | Function |
|-------------------|----------|
| `base.csv`         | The **main origin-destination matrix** for the Living Lab zones |
| `base_real_time.csv` | Generated real-time trips based on `base.csv`. Assigns timestamps and coordinates per OD pair |

### 🔄 Relationship:

- `base.csv` (OD matrix of how many trips move from zone A to zone B)
  → used to generate random **trip coordinates & times**
  → saved in `base_real_time.csv` (ride requests over time)

---

## 🧪 **3. Scenario-Based OD Demand**

These simulate various **realistic or extreme demand situations**.

| Scenario | OD File       | Real-Time File         | Description |
|----------|---------------|------------------------|-------------|
| S1       | `S1.csv`      | `S1_real_time.csv`     | Time-based variation (peak-hour weighting, 7 AM–10 PM) |
| S2       | `S2.csv`      | `S2_real_time.csv`     | Space-based concentration in Zones 4 & 6 with high POI density |
| S3       | `S3.csv`      | `S3_real_time.csv`     | S1 + S2 combined (peak hour *and* spatial concentration) |
| S4       | `S4.csv`      | `S4_real_time.csv`     | Card-data based distribution considering actual hourly trends |

🧠 These simulate how transit demand would differ under:
- Congested hours
- Mall/clinic hotspots
- Combined effects
- Real-world data-informed modeling

Each CSV contains a new OD matrix, and its real-time version is coordinate-rich request data used for simulation or dispatch planning.

---

## 🧪 **4. Synthetic Variance Files (Random Demand Rules)**

These simulate **errors, noise, or variation** in real-world predictions.

### 🔀 File Name Format:  
`base_trip<ratio>_rule<rule>.csv`  
(e.g., `base_trip0.5_rule2.csv`)

| Ratio | Purpose |
|-------|---------|
| 0.5   | Decreased total demand (50%) |
| 1.0   | No change in total demand |
| 1.5   | Increased total demand (150%) |

### 🧩 Rules:
- Each rule defines a different **randomized way to distort** the original OD values
- Uniform or skewed distribution
- Used to **test robustness** of the system when reality doesn’t match prediction

### 🔎 Evaluation:
- Compares against `base.csv`
- Error measured via **MAPE (Mean Absolute Percentage Error)**

---

## 📄 **5. Supporting File: Table Schema**

| File Name              | Function |
|------------------------|----------|
| `테이블 정의서.xlsx`  | Defines column names, types, units for all CSV tables |

Essential for anyone reproducing, parsing, or validating the data formats used across files.

---

## 🔄 **6. Interrelationships Diagram (Conceptual)**

Here’s a visual-style relationship summary:

```
[ OD_Zone.shp + Zone_centroid.shp ]
                  ↓
         Spatial Definition (Zoning)

          + base.csv
            ↓
    ----------------------
    | Real-time Trip Gen |
    | → base_real_time   |
    ----------------------

Scenarios (S1-S4):
→ [S1.csv → S1_real_time]
→ [S2.csv → S2_real_time]
→ ...

Synthetic Variants:
→ [base_trip0.5_rule1.csv]
→ [base_trip1.5_rule3.csv]
   ↓
Compare to base.csv using MAPE

All used in:
→ Dispatch simulations
→ Error analysis
→ Performance benchmarking
```

---

## ✅ **Usage Summary by Role**

| Use Case                   | Files Used |
|----------------------------|------------|
| Build base demand matrix   | `base.csv` |
| Assign trip coordinates    | `Zone_centroid.shp`, `base_real_time.csv` |
| Simulate time-based peaks  | `S1.csv`, `S1_real_time.csv` |
| Simulate spatial hotspots  | `S2.csv`, `S2_real_time.csv` |
| Combined effect            | `S3.csv`, `S3_real_time.csv` |
| Apply real card trends     | `S4.csv`, `S4_real_time.csv` |
| Analyze prediction error   | `base_trip*.csv`, MAPE values |
| Shape visualizations       | `OD_Zone.shp`, `Zone_centroid.shp` |
| Format checking            | `테이블 정의서.xlsx` |
