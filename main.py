import sys

from scripts.evaluate import evaluate_model_on_fgnet
from scripts.train import train
# from scripts.evaluate import evaluate  # Puedes descomentar cuando implementes esto

if __name__ == "__main__":
    mode = "eval"  # Cambia a "eval" cuando tengas evaluate.py listo

    if mode == "train":
        train()
    elif mode == "eval":
        weights = "checkpoints/best_model.pt"
        fgnet_data_dir = "data/FGNET/images"
        evaluate_model_on_fgnet(weights_path=weights, fgnet_dir=fgnet_data_dir)
    else:
        print("Modo no reconocido")
        sys.exit(1)
