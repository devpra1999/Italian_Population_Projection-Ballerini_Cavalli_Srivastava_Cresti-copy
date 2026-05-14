# Italian Population Projection

This repository contains the code and input files used for the Italian population projection project based on a cohort-component approach.

## Main Streamlit Applications

- `final_streamlit.py`  
  Final complete Streamlit application used for the interactive population projections.

- `streamlit_simpl.py`  
  Simplified Streamlit version implementing the same projection logic in a lighter and more straightforward structure.

## Non-Streamlit Versions

- `final_nostreamlit.py`  
  Standard Python version of the projection model without Streamlit. It reproduces the same logic implemented in `streamlit_simpl.py`, but as a regular Python script.

- `final_notebook.ipynb`  
  Jupyter Notebook version of the same workflow implemented in `streamlit_simpl.py`, designed for step-by-step execution and easier inspection of intermediate outputs.

## General Logic

The projection framework is based on a cohort-component approach:
- population ageing through survival probabilities,
- fertility estimation and newborn generation,
- migration flows,
- iterative yearly population updates by age and sex.

The repository also includes separate modules and input datasets for:
- fertility forecasting,
- mortality forecasting,
- migration forecasting,
- uncertainty simulations,
- comparison with external projections (ISTAT, UN, Eurostat).
