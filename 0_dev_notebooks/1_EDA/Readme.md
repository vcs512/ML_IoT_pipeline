# Exploratory Data Analysis (EDA)

Perform EDA:
- Analyse dataset
- View trends and stratification in data
- Verify pre-processing necessities
- Define metrics that will be of interest

## Observations
### Complete set
| **Class**     | **Samples** | **total (%)** |
|---------------|:-----------:|:-----------:|
| 0 (usable)    |     517     |    39.86    |
| 1 (defective) |     780     |    60.14    |
| **TOTAL**     |   **1297**  |  **100.00** |

### Sets division
| **Set**   | **Samples** | **total (%)** |
|-----------|:-----------:|:-----------:|
| Train     |     1114    |    85.89    |
| Test      |     183     |    14.11    |
| **TOTAL** |   **1297**  |  **100.00** |

### Data
Unbalance between usable (40%) and defective (60%).
Important metrics to check unbalance efects:
- Recall
- Precision
- Confusion metrix

Useful data augmentation:
- Vertical/horizontal flip.

Data already has different zoom, background colors and angulation.