import os
import torch

# General Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_WORKERS_SUPERVISED = 16
NUM_WORKERS_SIMCLR = 8
dir_data_labeled = '/media/databases/tiputini/white-lipped-peccary-vs-collared-peccary/roboflow-yolov9-format-no-data-augmentation/'

CHECKPOINT_PATH = "/home/dvillacreses/simCLR_tiputini"
PATH_SUPERVISED_MODELS = '/home/dvillacreses/code/outputs'
PATH_OUTPUTS = '/home/dvillacreses/code/outputs'
PATH_UNLABELED_METADATA = '/media/databases/tiputini/original_db'
PATH_OUTPUT_GRAPHS = '/home/dvillacreses/code/outputs/graphs'
PATH_OUTPUT_TABLES = '/home/dvillacreses/code/outputs/tables'

# General training configuration
MAX_EPOCHS_SUPERVISED = 200

TARGET_SIZE = (224,224)
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 10**-5
NUM_WORKERS = 20
CLASES = 2
PATIENCE = 10
RANDOM_STATE = 0