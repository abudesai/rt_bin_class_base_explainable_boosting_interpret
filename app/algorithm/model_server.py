import numpy as np, pandas as pd
import os
import sys
import json
import pprint

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.data_schema = data_schema
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 5
        self.id_field_name = self.data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]  
        
        
    
    def _get_preprocessor(self): 
        try: 
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
            return self.preprocessor
        except: 
            print(f'No preprocessor found to load from {self.model_path}. Did you train the model first?')
        return None
    
    
    def _get_model(self):
        try: 
            self.model = classifier.load_model(self.model_path)
            return self.model
        except: 
            print(f'No model found to load from {self.model_path}. Did you train the model first?')
        return None
    
        
    
    def predict(self, data):  
        
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                    
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)          
        # Grab input features for prediction
        pred_X = proc_data['X'].astype(np.float)        
        # make predictions
        preds = model.predict( pred_X )
        # inverse transform the predictions to original scale
        preds = pipeline.get_inverse_transform_on_preds(preprocessor, model_cfg, preds)        
        # get the names for the id and prediction fields
        id_field_name = self.data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]  
        # return the prediction df with the id and class probability fields
        preds_df = data[[id_field_name]].copy()
        preds_df['prediction'] = preds   
        
        return preds_df

    def predict_proba(self, data):
        preds = self._get_predictions(data)
        # get class names (labels)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        # get the name for the id field
        id_field_name = self.data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]
        # return te prediction df with the id and class probability fields
        preds_df = data[[id_field_name]].copy()
        preds_df[class_names[0]] = 1 - preds
        preds_df[class_names[-1]] = preds
        return preds_df
    
    
    def explain_local(self, data):
        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f'''Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations.'''
            print(msg)
        
        data = data.head(self.MAX_LOCAL_EXPLANATIONS)
        print(f"Now generating local explanations for {data.shape[0]} sample(s).")     
        # ------------------------------------------------------------------------------
        preprocessor = self._get_preprocessor()        
        proc_data = preprocessor.transform(data)  
        pred_X, ids = proc_data['X'].astype(np.float), proc_data['ids']  
        
        model = self._get_model()
        
        local_explanations = model.explain_local(pred_X)        
        
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        all_explanations = []        
        for i in range(pred_X.shape[0]):
            local_expl_data = local_explanations.data(i)            
            sample_expl_dict = {}
            sample_expl_dict[self.id_field_name] = ids[i]
            sample_expl_dict['predicted_class'] = class_names[int(local_expl_data["perf"]["predicted"])] 
            sample_expl_dict['predicted_class_prob'] = np.round(local_expl_data["perf"]["predicted_score"],4)
            sample_expl_dict['Intercept'] = local_expl_data['extra']['scores'][0]            
            feature_impacts = {}
            for f_name, f_impact in zip(local_expl_data["names"], local_expl_data["scores"]):
                feature_impacts[f_name] = round(f_impact,4)
            
            sample_expl_dict["feature_impacts"] = feature_impacts
            all_explanations.append(sample_expl_dict)            
            # pprint.pprint(sample_expl_dict)
        # ------------------------------------------------------  
        all_explanations = json.dumps(all_explanations, cls=utils.NpEncoder, indent=2)
        return all_explanations
        
        

    def _get_predictions(self, data):
        preprocessor = self._get_preprocessor()
        model = self._get_model()
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X = proc_data['X'].astype(np.float)
        # make predictions
        preds = model.predict( pred_X )
        return preds
    
        
        

