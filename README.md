# Machine-Learning-XGBoost-Classification-Visualization

This project uses an XGBoost model to perform wildfire prediction and visualize the results along with spatial and temporal and distribution of wildfires through interactive Tableau dashboards.

The code (`XGBoost.py`) and (`xgbf.py`) handles data loading, preprocessing, training, and evaluation of the model. 

(Tableau file is not included in this repository, but snapshots of the dashboards can be viewed in the poster.)

FILES
-----
```plaintext
.
├── README.txt 							               # This user guide
├── CODE/
│   ├── CA_Env_Data_Cleansing.py				         # Preprocessing script for CA Environmental Conditions dataset
│   ├── CA_Environmental_Profiling_Report.html		# EDA report on California environmental data
│   ├── xgbf.py							               # XGBoost model training script
│   ├── XGBoost.py							            # Output file with model predictions
│   ├── timeseries.py						            # Time series decomposition of NASA FRP data
│   └── Wildfire_Analysis.twbx					      # Tableau workbook for wildfire analysis
│
└── DOC/
    ├── team006poster.pdf						         # Final project poster
    └── team006report.pdf						         # Final project report
