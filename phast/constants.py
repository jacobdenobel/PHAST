import os

DATA_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), "data")
I_DET = os.path.join(DATA_DIR, "idet.npy")
SOUND_DIR = os.path.join(DATA_DIR, "sounds")
SOUNDS = {
    x[:-4]: os.path.join(SOUND_DIR, x)
    for x in os.listdir(SOUND_DIR)
    if x.endswith(".wav")
}
