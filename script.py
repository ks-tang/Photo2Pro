
import torch
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os

# -------------------------------------------------------
# 1. Charger la photo d'entrée
# -------------------------------------------------------
input_image_path = "photo.jpg"   
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Charger l'image
image_bgr = cv2.imread(input_image_path)
h, w, _ = image_bgr.shape


# -------------------------------------------------------
# 2. Mask et Détection du visage (Mediapipe)
# -------------------------------------------------------

# Charger le mask créé dans Paint
mask_image_path = "mask_perso.png"
if os.path.exists(mask_image_path):
    mask_pil = Image.open(mask_image_path).convert("L")  # grayscale

    # Redimensionner pour le pipeline
    mask_image = mask_pil.resize((1024, 1024))  

else: # sinon on crée le mask
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    results = face_detection.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Créer un masque vide
    mask = np.ones((h, w), dtype=np.uint8) * 255  # blanc = zone à modifier

    if results.detections:
        for detection in results.detections:
            # Récupérer bounding box du visage
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # Étendre un peu la bbox (cheveux, front)
            pad = int(0.2 * (y2 - y1))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            # Garder visage en noir = zone à conserver
            mask[y1:y2, x1:x2] = 0  

    # Sauvegarder masque
    cv2.imwrite("mask.png", mask)

# -------------------------------------------------------
# 3. Préparer images pour Diffusers
# -------------------------------------------------------
init_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).resize((1024, 1024))
#mask_image = Image.fromarray(mask).resize((1024, 1024))

# -------------------------------------------------------
# 4. Charger modèle Stable Diffusion Inpainting
# -------------------------------------------------------

if torch.cuda.is_available():
    torch.device('cuda')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
else:
    torch.device('cpu')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32   # <--- float32 pour CPU
    )
    pipe = pipe.to("cpu")

# -------------------------------------------------------
# 5. Génération photo pro
# -------------------------------------------------------
prompt = (
    "Modern professional headshot, subtle smile, black blazer, studio lighting, plain neutral background, ultra realistic, 8k photography"
)

result = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    guidance_scale=7.5,
    num_inference_steps=50
)

# -------------------------------------------------------
# 6. Sauvegarde résultat
# -------------------------------------------------------

# Chercher le prochain numéro disponible
i = 1
while True:
    output_image_path = os.path.join(results_dir, f"portrait_pro{i}.jpg")
    if not os.path.exists(output_image_path):
        break
    i += 1


result.images[0].save(output_image_path)
print(f"✅ Portrait pro généré : {output_image_path}")
