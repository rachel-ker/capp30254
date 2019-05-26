# Final Pipeline  
  
This repository contains files for the improved machine learning pipeline to be applied to DonorChoose Data.  
  
## To Run  
run 'python3 run.py' on command line  
results will appear in csv file  
  
## To Modify Parameters  
modifications of parameters can be done in config.py file.
  
## Files:  
etl.py - generic code to read, explore, transform data  
pipeline.py - generic code for train-test split, evaluation, building the different models  
config.py - configurations and constants for this code  
run.py - putting everything together to apply to DonorsChoose data (produces a csv file of results)  
results.ipynb - exploring and understanding results to justify model selection  
  
requirements.txt - required libraries and versions  
script.sbatch - to run job on rcc  
Data folder include the data used  