import os
import pandas as pd
import yaml

# Carga config.yaml para obtener rutas
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

csv_path = config["csv_path"]
root_dir = config["root_dir"]

# Paso 1: Crear índice filename → carpeta
print("[INFO] Generando índice de imágenes válidas desde disco...")
image_map = {}

for celeb in os.listdir(root_dir):
    celeb_folder = os.path.join(root_dir, celeb)
    if not os.path.isdir(celeb_folder):
        continue

    for fname in os.listdir(celeb_folder):
        if fname.endswith(".jpg"):
            image_map[fname] = celeb

print(f"[INFO] Imágenes encontradas en disco: {len(image_map)}")

# Paso 2: Cargar CSV original y filtrar solo los que están en disco
df = pd.read_csv("data/cacd/CACD_features_sex.csv")  # archivo CSV originalcsv_path)
df_filtered = df[df["name"].isin(image_map.keys())].copy()

print(f"[INFO] Entradas válidas en CSV tras filtrado: {len(df_filtered)}")

# Paso 3: Añadir columna con nombre de carpeta
# Esto elimina ambigüedad al acceder desde CACDDataset
print("[INFO] Añadiendo columna 'folder' al CSV limpio...")
df_filtered["folder"] = df_filtered["name"].map(image_map)

# Paso 4: Guardar CSV limpio
df_filtered.to_csv("data/cacd/CACD_filtered.csv", index=False)
print("[DONE] Archivo guardado como data/cacd/CACD_filtered.csv")
