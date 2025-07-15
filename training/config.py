import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 5
LR = 1e-3
NUM_CLASSES = 29

TRAIN_DIR = "asl_classifier/data/asl_alphabet_train/asl_alphabet_train"
VAL_DIR = "asl_classifier/data/asl_alphabet_val/asl_alphabet_val"
TEST_DIR = "asl_classifier/data/asl_alphabet_test/asl_alphabet_test"

MODEL_PATH = "asl_classifier/models/best_model.pth"
PLOT_PATH = "asl_classifier/plots/training_plot.png"