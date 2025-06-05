import numpy as np
import json

def load_data(file_path):
    """Carga datos desde un archivo .npy."""
    return np.load(file_path)

def load_metadata(file_path="metadata.json"):
    """Carga metadatos desde un archivo JSON."""
    with open(file_path, "r") as f:
        meta = json.load(f)
    return meta

# Aquí se podrían agregar más funciones útiles como normalización, etc.