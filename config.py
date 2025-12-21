# Audio
SAMPLE_RATE = 22050
DURATION = 30            
NUM_SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 25
NUM_WORKERS = 0            
SEED = 42

# Classes (GTZAN)
GENRES = [
    "Hip-Hop", "Pop", "Rock", "Experimental", "Jazz", "Folk", "Electronic", "International"
]
NUM_CLASSES = len(GENRES)

# Paths
DATA_DIR = "data/training_data"
METADATA_DIR = "data/metadata"
