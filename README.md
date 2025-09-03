# 📸 Photo2Pro – Transformez vos photos en portraits professionnels 

## 🚀 Description

Photo2Pro est un projet basé sur l’IA qui permet de transformer une photo simple en portrait professionnel adapté pour LinkedIn ou d’autres plateformes.
L’application utilise un pipeline de diffusion (StableDiffusionInpaintPipeline) pour remplacer automatiquement les vêtements ou améliorer le style de la photo, tout en gardant le visage intact.

## ✨ Fonctionnalités

🧑‍💼 Transformation d’une photo simple en portrait professionnel

👔 Modification automatique de la tenue (costume, chemise, polo, etc.)

🎨 Utilisation de masks personnalisés (automatiques ou créés manuellement)

⚡ Support GPU (CUDA) ou CPU

## ▶️ Utilisation

1. Placer une photo dans le dossier du projet (photo.jpg).
2. Lancer le script : python script.py
3. L'image résultat se trouvera dans le dossier results.

## ⚙️ Paramétrage

1. Utiliser un mask manuel (mask_perso.png) pour contrôler les zones modifiées.
Avec un éditeur d'image, coloriez en noir la partie que vous ne voulez pas modifier et en blanc la partie à modifier avec le prompt.

2. Modifier le prompt dans script.py pour changer le style de la tenue.

## Exemples

<table align="center">
  <tr>
    <td><img src="photo.jpg" width="300"></td>
    <td><img src="results/portrait_pro33.jpg" width="300"></td>
  </tr>
  <tr>
    <td align="center">Image originale</td>
    <td align="center">Image générée</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td><img src="photo.jpg" width="300"></td>
    <td><img src="results/portrait_pro24.jpg" width="300"></td>
  </tr>
  <tr>
    <td align="center">Image originale</td>
    <td align="center">Image générée</td>
  </tr>
</table>