# app.py - Interface Streamlit pour modèle MNIST pré-entraîné
import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import pandas as pd
from io import BytesIO
from tensorflow import keras
# Les options `invert` et `crop` sont définies dans la sidebar ci-dessous
# Configuration de la page
st.set_page_config(
    page_title="Classificateur d'images",
    page_icon="🔢",
    layout="wide"
)

# Titre principal
st.title("🔢 Classificateur d'images")
st.markdown("### Interface de prédiction")

# Sidebar pour les paramètres
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIpqAKKAFNK90T4Q2WATX58YoCHtCRqdnc8A&s", 
             width=256,
             caption="Exemples du dataset CIFAR-10")
    
    st.markdown("### ⚙️ Configuration")
    
    # Options de prétraitement
    st.markdown("### 🔧 Prétraitement")
    invert_option = st.radio(
        "Inverser les couleurs:",
        ("Auto (recommandé)", "Oui", "Non"),
        index=0,
        help="Auto: détecte et inverse si nécessaire pour que le chiffre soit blanc sur fond noir (format MNIST)."
    )
    auto_crop = st.checkbox(
        "Rogner automatiquement",
        value=True,
        help="Supprime les marges de fond autour du chiffre."
    )
    
    # Informations sur le modèle
    st.markdown("### 📊 Informations")
    if st.button("Afficher les infos du modèle"):
        try:
            metadata = joblib.load('model3/model_metadata.pkl')
            st.json(metadata, expanded=False)
        except:
            st.warning("Métadonnées non trouvées")
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.info("""
    1. Chargez une image de chiffre
    2. L'image est automatiquement prétraitée
    3. Visualisez la prédiction et les probabilités
    """)

# Charger le modèle et les métadonnées
@st.cache_resource
def load_model():
    """Charger le modèle sauvegardé"""
    try:
        # Charger le modèle
        model = joblib.load('model3/mnist_model.joblib')
        st.success("✅ Modèle chargé avec succès!")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

@st.cache_resource
def load_metadata():
    """Charger les métadonnées"""
    try:
        metadata = joblib.load('model3/model_metadata.pkl')
        return metadata
    except:
        return {
            'class_names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'input_shape': (28, 28, 1)
        }

# Fonctions de prétraitement
def preprocess_image(image, invert='auto', crop=True):
    """
    Prétraiter une image pour le modèle MNIST.
    - image: PIL Image (grayscale or RGB)
    - invert: 'auto'|'Oui'|'Non' or True/False
    - crop: bool, rognage automatique
    Retourne: (img_array_ready, pil_image_resized_28x28)
    """
    # Convertir en niveaux de gris
    if image.mode != 'L':
        image = image.convert('L')

    # Convertir en array numpy (uint8)
    img_array = np.array(image)

    # Rogner automatiquement (optionnel) - robuste à fond blanc ou noir
    if crop:
        # Déterminer couleur de fond à partir des coins
        corners = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
        bg_mean = np.mean(corners)
        # Si fond clair (proche de 255), considérer les pixels < 250 comme non-fond
        if bg_mean > 127:
            mask = img_array < 250
        else:
            mask = img_array > 5
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            image = image.crop((cmin, rmin, cmax+1, rmax+1))
            img_array = np.array(image)

    # Redimensionner à 28x28
    image = image.resize((28, 28))
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Déterminer l'inversion des couleurs
    # Normalize invert parameter values
    invert_flag = None
    if isinstance(invert, str):
        if invert.lower().startswith('a'):
            invert_flag = 'auto'
        elif invert.lower().startswith('o') or invert.lower().startswith('y'):
            invert_flag = True
        else:
            invert_flag = False
    else:
        invert_flag = bool(invert)

    if invert_flag == 'auto':
        # Vérifier les coins pour estimer couleur de fond (après normalisation)
        corner_vals = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
        if np.mean(corner_vals) > 0.5:
            img_array = 1.0 - img_array
    elif invert_flag:
        img_array = 1.0 - img_array

    # Ajouter les dimensions batch et channel
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array, image

# Fonction pour créer des visualisations
def create_visualization(original_img, processed_img, predictions, class_names):
    """Créer des visualisations pour les résultats"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Image originale
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Image prétraitée
    axes[1].imshow(processed_img, cmap='gray')
    axes[1].set_title('Image prétraitée (28x28)')
    axes[1].axis('off')
    
    # Graphique des probabilités
    y_pos = np.arange(len(class_names))
    colors = ['lightblue'] * len(class_names)
    predicted_class = np.argmax(predictions)
    colors[predicted_class] = 'orange'
    
    axes[2].barh(y_pos, predictions[0], color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(class_names)
    axes[2].set_xlabel('Probabilité')
    axes[2].set_title('Probabilités de prédiction')
    axes[2].set_xlim([0, 1])
    
    # Ajouter les valeurs
    for i, v in enumerate(predictions[0]):
        axes[2].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig

# Charger le modèle et les métadonnées
model = load_model()
metadata = load_metadata()
class_names = metadata.get('class_names', [str(i) for i in range(10)])

# Interface principale avec onglets
tab1, tab2, tab3 = st.tabs(["📤 Chargement d'image", "📊 Analyse", "ℹ️ À propos"])

with tab1:
    st.header("Charger une image")
    
    # Option pour upload ou exemples
    input_method = st.radio(
        "Choisissez une méthode d'entrée:",
        ["📤 Charger une image", "🎨 Utiliser un exemple"]
    )
    
    if input_method == "📤 Charger une image":
        uploaded_file = st.file_uploader(
            "Choisissez une image de chiffre manuscrit",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Format supporté: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file is not None:
            # Afficher l'image uploadée
            original_image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(original_image, 
                        caption=f"Image originale - {uploaded_file.name}",
                        width=256)
            
            # Bouton de prédiction
            if st.button("🔍 Analyser l'image", type="primary", use_container_width=True):
                with st.spinner("Prétraitement et analyse en cours..."):
                    # Prétraiter l'image
                    img_array, processed_image = preprocess_image(
                        original_image, 
                        invert=invert_option,
                        crop=auto_crop
                    )
                    
                    # Faire la prédiction
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    # Afficher les résultats
                    with col2:
                        st.image(processed_image, 
                                caption="Image prétraitée",
                                width=256)
                    
                    # Afficher la prédiction principale
                    st.markdown("---")
                    st.subheader("📈 Résultats de la prédiction")
                    
                    # Métriques
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("Chiffre prédit", 
                                 f"{class_names[predicted_class]}",
                                 delta=None)
                    
                    with col_metric2:
                        st.metric("Confiance", 
                                 f"{confidence:.2%}",
                                 delta=None)
                    
                    with col_metric3:
                        # Top 3 prédictions
                        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                        top_3_text = ", ".join([class_names[i] for i in top_3_idx])
                        st.metric("Top 3", top_3_text)
                    
                    # Visualisation
                    fig = create_visualization(
                        original_image, 
                        processed_image, 
                        predictions, 
                        class_names
                    )
                    st.pyplot(fig)
                    
                    # Télécharger les résultats
                    st.markdown("---")
                    st.subheader("📥 Exporter les résultats")
                    
                    # Créer un DataFrame avec les probabilités
                    results_df = pd.DataFrame({
                        'Chiffre': class_names,
                        'Probabilité': predictions[0],
                        'Est_prédiction': [i == predicted_class for i in range(10)]
                    })
                    
                    # Afficher le tableau
                    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Probabilité']),
                               use_container_width=True)
                    
                    # Boutons de téléchargement
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Télécharger le graphique
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
                        st.download_button(
                            label="📥 Télécharger le graphique",
                            data=buf.getvalue(),
                            file_name="prediction_visualization.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        # Télécharger les données
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Télécharger les données (CSV)",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
    
    else:  # Utiliser un exemple
        st.subheader("🎨 Exemples prédéfinis")
        
        # Générer des exemples simples
        examples = st.selectbox(
            "Choisissez un exemple:",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Cas difficile"]
        )
        
        # Créer une image d'exemple (simple dessin en noir et blanc)
        def create_example_digit(digit):
            # Créer une image 28x28 avec le chiffre dessiné simplement
            from PIL import ImageDraw, ImageFont
            img = Image.new('L', (28, 28), color=0)  # Fond noir
            
            # Pour un vrai exemple, on pourrait charger des images prédéfinies
            # Ici, on va créer un texte simple
            try:
                draw = ImageDraw.Draw(img)
                # Essayer d'utiliser une police, sinon dessiner un cercle/rectangle
                font = ImageFont.load_default()
                # Position approximative
                draw.text((8, 0), str(digit), fill=255, font=font)
            except:
                # Dessiner un cercle pour les cas difficiles
                if digit == "Cas difficile":
                    draw.ellipse([5, 5, 23, 23], outline=255, fill=0)
                else:
                    draw.text((10, 5), str(digit), fill=255)
            
            return img
        
        example_image = create_example_digit(examples)
        st.image(example_image, caption=f"Exemple: {examples}", width=200)
        
        if st.button("Tester cet exemple", type="secondary"):
            with st.spinner("Analyse en cours..."):
                img_array, processed_image = preprocess_image(
                    example_image,
                    invert=invert_option,
                    crop=auto_crop
                )
                
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                
                st.success(f"**Prédiction : {class_names[predicted_class]}**")
                st.write(f"Confiance : {np.max(predictions[0]):.2%}")

with tab2:
    st.header("📊 Analyse détaillée")
    
    if model is not None:
        # Informations sur le modèle
        st.subheader("Caractéristiques du modèle")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.metric("Nombre de classes", len(class_names))
            st.metric("Taille d'entrée", "28x28 pixels")
        
        with col_info2:
            # Type de modèle
            model_type = type(model).__name__
            st.metric("Type de modèle", "Réseau de neurones convolutif (CNN)")
        
        # Matrice de confusion (exemple statique)
        st.subheader("Performance du modèle")
        
        # Pour une vraie matrice, vous devriez la sauvegarder pendant l'entraînement
        st.info("""
        **Note:** Pour afficher la matrice de confusion réelle, 
        sauvegardez-la pendant l'entraînement et chargez-la ici.
        """)
        
        # Exemple de matrice de confusion (à remplacer par vos données)
        st.markdown("**Exemple de matrice de confusion:**")
        
        # Créer une matrice exemple
        np.random.seed(42)
        example_cm = np.random.randint(0, 100, (10, 10))
        np.fill_diagonal(example_cm, np.random.randint(800, 1000, 10))
        
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        im = ax_cm.imshow(example_cm, cmap='Blues')
        ax_cm.set_xticks(range(10))
        ax_cm.set_yticks(range(10))
        ax_cm.set_xticklabels(class_names)
        ax_cm.set_yticklabels(class_names)
        ax_cm.set_xlabel('Prédiction')
        ax_cm.set_ylabel('Vérité')
        ax_cm.set_title('Matrice de confusion (exemple)')
        
        # Ajouter les valeurs
        for i in range(10):
            for j in range(10):
                ax_cm.text(j, i, example_cm[i, j],
                          ha="center", va="center", 
                          color="white" if example_cm[i, j] > 500 else "black")
        
        plt.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)
    
    else:
        st.warning("Modèle non chargé. Veuillez vérifier que le fichier model.joblib existe.")

with tab3:
    st.header("ℹ️ À propos de cette application")
    
    st.markdown("""
    ### 📋 Description
    Cette application permet de classifier des chiffres manuscrits à l'aide d'un modèle 
    d'intelligence artificielle pré-entraîné sur le dataset MNIST.
    
    ### 🎯 Fonctionnalités
    - **Prédiction en temps réel** des chiffres manuscrits
    - **Prétraitement automatique** des images
    - **Visualisation interactive** des résultats
    - **Export des données** (graphiques et CSV)
    
    ### 🏗️ Architecture technique
    - **Backend**: Modèle de classification sauvegardé avec Joblib
    - **Interface**: Streamlit pour une interaction simple
    - **Prétraitement**: PIL (Python Imaging Library) pour le traitement d'images
    
    ### 🔧 Configuration requise
    - Python 3.8+
    - Streamlit
    - Joblib
    - NumPy, Pandas, Matplotlib
    - PIL (Pillow)
    
    ### 🚀 Comment exécuter
    1. Installer les dépendances: `pip install -r requirements.txt`
    2. Lancer l'application: `streamlit run app.py`
    3. Ouvrir http://localhost:8501 dans votre navigateur
    """)
    
    # Afficher les versions des bibliothèques
    with st.expander("📦 Versions des bibliothèques"):
        import sys
        st.write(f"Python: {sys.version}")
        st.write(f"Streamlit: {st.__version__}")
        st.write(f"NumPy: {np.__version__}")
        try:
            st.write(f"Joblib: {joblib.__version__}")
        except:
            st.write("Joblib: Non disponible")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Classificateur de chiffres manuscrits (MNIST) • Développé avec Streamlit • Modèle CNN</p>
        <p style='font-size: 0.8rem;'>
            Le modèle a été entraîné sur le dataset MNIST contenant 60,000 images de chiffres manuscrits
        </p>
    </div>
    """,
    unsafe_allow_html=True
)