# Machine-Learning-XGBoost-Classification-Visualization
DESCRIPTION
-----------
(Tableau files are not stored in this repository)

This project uses an XGBoost model to perform wildfire prediction and visualize the results along with spatial and temporal and distribution of wildfires through interactive Tableau dashboards.

The code (`XGBoost.py`) and (`xgbf.py`) handles data loading, preprocessing, training, and evaluation of the model. 

The Tableau file (`Wildfire_Analysis.twbx`) use the model’s output and multiple analytic insights for advanced visualization dashboards.

INSTALLATION
------------
1. Ensure to have Python 3.x installed.
2. Install required Python packages:

   pip install xgboost pandas scikit-learn numpy

3. Tableau is required to open `.twbx` files (Tableau Desktop or Tableau Public).

EXECUTION
---------
1. Run the model training and output script:

   python xgbf.py

   - This script will read input data (`Cleaned_CA_Env_Data`), train the model, and get the best combination of parameters for xgboost

   python XGBoost.py

   - This script will use the model with the best combination to evaluate our test data.

2. Open Tableau:
   - Load `Wildfire_Analysis.twbx` using Tableau.

3. Explore the dashboard to visualize model prediction, performance and other analytical insights.

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
