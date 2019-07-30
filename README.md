# gdpr_tools_production

### A typical top-level directory layout

Module expects calling file to have the following path structure:

    Project
    ├── calling_file.py                   # Python file calling gdpr module
    ├── config.json                       # Config file, will be generated if not present
    ├── data
        ├── example_training_data.txt     # Data file for clf model
        ├── LeipzigCorpora                # All leipzig data
            ├──...
    ├── models
        ├──...                            # Contains all models (clf, we, dv, temp etc)
    ├── predictions         
        ├── data_to_predict               # Data you want to predict
        ├── data_predicted                # Predictions for data specified in folder above 
