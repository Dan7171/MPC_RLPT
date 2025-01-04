import os
import time
def make_model_path(models_dst_dir):
    training_starttime = time.strftime('%Y:%m:%d(%a)%H:%M:%S')
    model_dir = os.path.join('BGU/Rlpt/trained_models') 
    model_file_path = os.path.join(models_dst_dir, training_starttime, 'model.pth')  
    return model_file_path