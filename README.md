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
        
Calling file example
```python
import gdpr_tools_production

model = gdpr_tools_production.NameRecognizer()


# Only train CLF. Load WE and DV. Get predictions on the go, to follow how the predictions change over time
model.train(clf_data_path='labeled_data_copy_negative_list_kommune_fix.txt', we_model_path='we_model',
            dv_model_path='dict_vec_new_only_pre&suf3', running_prediction=True)

# Train all models
model.train(clf_data_path='labeled_data_copy_negative_list_kommune_fix.txt')

# Load all models and predict
model.predict(clf_model_path = 'clf_epoch_0_iter_10_loss_0.0189585592597723', we_model_path='we_model',
            dv_model_path='dict_vec_new_only_pre&suf3')
```
