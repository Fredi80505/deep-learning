# app.py — Application Streamlit professionnelle pour la classification CIFAR-10
# Version améliorée avec structure modulaire et design percutant

import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import io
import sys
import traceback

# ============================
# CONFIGURATION DE LA PAGE
# ============================
st.set_page_config(
    page_title="Classificateur CIFAR-10 - Vision par ordinateur",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
#        STYLE CSS 
# ============================
st.markdown("""
<style>
    [data-testid="stApp"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #1e293b;
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    h1 {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Cards */
    .professional-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 20px;
    }
    
    .professional-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.025em;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #3b82f6 !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 -4px 12px rgba(59, 130, 246, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
        font-size: 14px;
    }
    
    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
        font-size: 28px;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 32px !important;
        background: rgba(255, 255, 255, 0.7) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6 !important;
        background: rgba(59, 130, 246, 0.05) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 4px;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Success/Error/Warning messages */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Custom spacing */
    .spacer {
        height: 20px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        padding: 20px;
        margin-top: 40px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# CONSTANTES ET CONFIGURATION
# ============================
CLASS_NAMES = [
    "✈️ Avion", "🚗 Voiture", "🐦 Oiseau", "🐱 Chat", "🦌 Cerf",
    "🐶 Chien", "🐸 Grenouille", "🐴 Cheval", "🚢 Navire", "🚚 Camion"
]

CLASS_NAMES_SIMPLE = [
    "avion", "voiture", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "navire", "camion"
]

MODEL_PATH = "model4/cifar_model.joblib"

# ============================
# FONCTIONS UTILITAIRES
# ============================
@st.cache_resource
def load_model():
    """Chargement du modèle pré-entraîné"""
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Modèle chargé avec succès!")
        return model
    except FileNotFoundError:
        st.error(f"❌ Fichier modèle non trouvé à : {MODEL_PATH}")
        st.info("Veuillez vérifier que le fichier modèle existe bien à cet emplacement.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        st.code(traceback.format_exc())
        return None

def preprocess_image_cifar(image: Image.Image):
    """Prétraite l'image pour le modèle CIFAR-10"""
    try:
        image = image.convert("RGB")
        image = image.resize((32, 32))
        img_array = np.array(image).astype("float32")
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Erreur lors du prétraitement : {str(e)}")
        return None

def plot_probabilities(predictions):
    """Crée un graphique de probabilités professionnel"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Style du graphique
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('#f8fafc')
    
    # Données
    y_pos = np.arange(len(CLASS_NAMES_SIMPLE))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(CLASS_NAMES_SIMPLE)))
    
    bars = ax.barh(y_pos, predictions[0], color=colors, edgecolor='white', linewidth=1.5)
    
    # Personnalisation
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES_SIMPLE, fontsize=11, fontweight='medium')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité", fontsize=12, fontweight='bold', color='#475569')
    ax.set_title("Distribution des Probabilités", fontsize=16, fontweight='bold', color='#0f172a', pad=20)
    
    # Grid et bordures
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('#cbd5e1')
    ax.spines["bottom"].set_color('#cbd5e1')
    
    # Couleur des ticks
    ax.tick_params(colors='#475569')
    
    # Ajout des valeurs
    for i, (bar, prob) in enumerate(zip(bars, predictions[0])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{prob:.2%}", 
                va='center', 
                fontsize=10,
                fontweight='bold',
                color='#475569')
    
    # Ajout d'une ligne pour la prédiction maximale
    max_prob = np.max(predictions[0])
    ax.axvline(x=max_prob, color='#ef4444', linestyle='--', alpha=0.5, label=f'Max: {max_prob:.2%}')
    ax.legend(loc='lower right', frameon=False)
    
    fig.tight_layout()
    return fig

def create_comparison_image(original_image, heatmap_image=None):
    """Crée une visualisation comparative de l'image originale"""
    fig, axes = plt.subplots(1, 2 if heatmap_image is None else 3, figsize=(12, 4))
    
    # Image originale
    axes[0].imshow(original_image)
    axes[0].set_title("Image Originale", fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Image redimensionnée (32x32)
    resized_img = original_image.resize((32, 32))
    axes[1].imshow(resized_img)
    axes[1].set_title("Image Redimensionnée (32×32)", fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # Heatmap si fournie
    if heatmap_image is not None and len(axes) > 2:
        axes[2].imshow(heatmap_image)
        axes[2].set_title("Carte d'Activation", fontweight='bold', pad=10)
        axes[2].axis('off')
    
    fig.patch.set_facecolor('#f8fafc')
    plt.tight_layout()
    return fig

# ============================
# INTERFACE UTILISATEUR
# ============================

# Header principal
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
st.markdown("# 🤖 Vision par ordinateur - Classificateur CIFAR-10")
st.markdown("### Classification d'images par Intelligence Artificielle")
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    
    # Logo/Image
    st.image("https://datasets-dev.10web.me/wp-content/uploads/2022/09/CIFAR-10-dataset-Activeloop-Platform-visualization-image-1.webp", 
             caption="Exemples CIFAR-10", 
             width=300)
    
    st.markdown("### 📊 Configuration")
    
    # Sélecteur de mode
    mode = st.radio(
        "Mode de fonctionnement:",
        ["🚀 Production", "🔍 Développement"],
        index=0
    )
    
    # Options avancées
    with st.expander("⚙️ Paramètres avancés"):
        confidence_threshold = st.slider(
            "Seuil de confiance",
            min_value=0.5,
            max_value=0.99,
            value=0.7,
            step=0.05,
            help="Seuil minimum de confiance pour accepter la prédiction"
        )
        
        show_details = st.checkbox("Afficher les détails techniques", value=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Informations dataset
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    st.markdown("### 🗃️ Jeu de données CIFAR-10")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classes", "10")
        st.metric("Images", "60K")
    
    with col2:
        st.metric("Résolution", "32×32")
        st.metric("Couleurs", "RGB")
    
    st.markdown("""
    **Catégories:**
    - Transport: ✈️ 🚗 🚢 🚚
    - Animaux: 🐦 🐱 🦌 🐶 🐸 🐴
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Aide rapide
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    st.markdown("### 🆘 Aide Rapide")
    st.info("""
    1. **Téléversez** une image
    2. **Analysez** la prédiction
    3. **Explorez** les détails
    4. **Exportez** les résultats
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Chargement du modèle (une seule fois)
model = load_model()

# Onglets principaux
tab1, tab2, tab3 = st.tabs(["📤 Prédiction", "📊 Analyse", "📚 Documentation"])

# Tab 1 - Prédiction
with tab1:
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    st.markdown("## 🚀 Prédiction en Temps Réel")
    
    # Zone de téléversement
    col_upload, col_examples = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "### 📤 Téléversez votre image",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Formats supportés : PNG, JPG, JPEG, BMP"
        )
    
    with col_examples:
        st.markdown("### 🎯 Exemples")
        example_images = {
            "Avion": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpENry9yuEEuZR2afkKF8MCb_PNvNY0L2jSQ&s",
            "Voiture": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRp0wjdE0l7ihIgn-OvZOkjBSXGRbwHDYPsUQ&s",
            "Chat": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSAHrVB6TlHMIAKCJL_rZdrmc6WQCs68oRNoA&s"
        }
        
        selected_example = st.selectbox(
            "Voir un exemple :",
            list(example_images.keys())
        )
        
        if selected_example:
            st.image(example_images[selected_example], 
                    width=150, 
                    caption=f"Exemple: {selected_example}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section de prédiction
    if uploaded_file and model is not None:
        st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
        
        # Affichage de l'image
        image = Image.open(uploaded_file)
        
        col_img, col_stats = st.columns([2, 1])
        
        with col_img:
            st.markdown("### 📷 Image Originale")
            st.image(image, use_container_width=True)
            
            # Statistiques de l'image
            img_array = np.array(image)
            st.caption(f"Dimensions: {image.size[0]}×{image.size[1]} | "
                      f"Couleurs: {len(np.unique(img_array)) if len(img_array.shape) > 2 else 'Grayscale'}")
        
        with col_stats:
            st.markdown("### 📋 Informations")
            st.metric("Format", uploaded_file.type.split('/')[-1].upper())
            st.metric("Taille", f"{uploaded_file.size / 1024:.1f} KB")
            st.metric("Mode", image.mode)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Bouton de prédiction
        col_btn, col_space = st.columns([1, 3])
        with col_btn:
            predict_btn = st.button("🎯 Lancer la Prédiction", 
                                   type="primary", 
                                   use_container_width=True)
        
        if predict_btn:
            with st.spinner("🔍 Analyse en cours..."):
                # Prétraitement
                img_array_processed = preprocess_image_cifar(image)
                
                if img_array_processed is not None:
                    # Prédiction
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    predictions = model.predict(img_array_processed, verbose=0)
                    
                    # Résultats
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = np.max(predictions[0])
                    
                    # Affichage des résultats
                    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                    
                    col_result, col_confidence = st.columns([2, 1])
                    
                    with col_result:
                        st.markdown(f"### 🎯 **Résultat : {predicted_class}**")
                        
                        # Indicateur de confiance
                        if confidence > confidence_threshold:
                            st.success(f"Confiance élevée ({confidence:.2%})")
                        else:
                            st.warning(f"Confiance modérée ({confidence:.2%})")
                    
                    with col_confidence:
                        st.metric("Niveau de Confiance", 
                                 f"{confidence:.2%}",
                                 delta=f"{(confidence - confidence_threshold):+.2%}" if confidence > confidence_threshold else None)
                    
                    # Graphique des probabilités
                    st.markdown("### 📊 Distribution des Probabilités")
                    fig = plot_probabilities(predictions)
                    st.pyplot(fig)
                    
                    # Tableau des résultats
                    st.markdown("### 📈 Détails par Classe")
                    df_results = pd.DataFrame({
                        "Classe": CLASS_NAMES_SIMPLE,
                        "Probabilité": predictions[0],
                        "Emoji": ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚚"]
                    }).sort_values("Probabilité", ascending=False)
                    
                    # Formatage du dataframe
                    df_display = df_results.copy()
                    df_display["Probabilité"] = df_display["Probabilité"].apply(lambda x: f"{x:.2%}")
                    df_display = df_display.reset_index(drop=True)
                    df_display.index = df_display.index + 1
                    
                    st.dataframe(df_display, 
                                use_container_width=True,
                                column_config={
                                    "Emoji": st.column_config.TextColumn(""),
                                    "Classe": st.column_config.TextColumn("Classe"),
                                    "Probabilité": st.column_config.TextColumn("Probabilité")
                                })
                    
                    # Options d'export
                    st.markdown("### 📥 Export des Résultats")
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        # Export graphique
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        st.download_button(
                            "💾 Télécharger le graphique",
                            buf.getvalue(),
                            "prediction_graph.png",
                            "image/png",
                            use_container_width=True
                        )
                    
                    with export_col2:
                        # Export données
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            "📊 Télécharger les données",
                            csv,
                            "prediction_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2 - Analyse
with tab2:
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    st.markdown("## 📊 Analyse des Performances")
    
    if model is not None:
        # Métriques du modèle
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy Test", "79,11%", "±1,5%")
        
        with col2:
            st.metric("Précision Moyenne", "81,41%")
        
        with col3:
            st.metric("Temps d'Inférence", "< 50ms")
        
        # Informations techniques
        st.markdown("### 🏗️ Architecture du Modèle")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            **Spécifications:**
            - Type: CNN (Convolutional Neural Network)
            - Couches: 4 couches convolutionnelles et Pooling + 2 couches denses
            - Paramètres: 241 928
            - Fonction d'activation: ReLU et Softmax
            - Optimiseur: Adam
            """)
        
        with tech_col2:
            st.markdown("""
            **Entraînement:**
            - Époques: 100
            - Batch size: 50
            - Jeu de données: CIFAR-10
            - Split: 90/10
            """)
        
        # Matrice de confusion (exemple)
        st.markdown("### 📈 Performances par Classe")
        
        # Données simulées pour l'exemple
        class_performance = pd.DataFrame({
            "Classe": CLASS_NAMES_SIMPLE,
            "Précision": [0.85, 0.92, 0.77, 0.61, 0.68, 0.72, 0.80, 0.85, 0.85, 0.88],
            "Rappel": [0.79, 0.91, 0.65, 0.64, 0.80, 0.68, 0.86, 0.79, 0.92, 0.88]
        })
        
        fig_perf, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(class_performance))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, class_performance["Précision"], width, label='Précision', color='#3b82f6')
        bars2 = ax.bar(x + width/2, class_performance["Rappel"], width, label='Rappel', color='#8b5cf6')
        
        ax.set_xlabel('Classe', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Précision et Rappel par Classe', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES_SIMPLE, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig_perf)
        
    else:
        st.warning("⚠️ Modèle non chargé. Veuillez vérifier le fichier modèle.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3 - Documentation
with tab3:
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    st.markdown("## 📚 Documentation Technique")
    
    st.markdown("""
    ### 🎯 Objectif de l'Application
    
    Cette application démontre l'utilisation d'un réseau de neurones convolutifs (CNN)
    entraîné sur le dataset CIFAR-10 pour la classification d'images en 10 catégories.
    
    ### 🧠 Pipeline Technique
    
    1. **Prétraitement d'image**
       - Conversion RGB
       - Redimensionnement 32×32
       - Normalisation [0, 1]
    
    2. **Inférence du modèle**
       - Passage forward dans le CNN
       - Calcul des probabilités
       - Sélection de la classe
    
    3. **Post-traitement**
       - Visualisation des résultats
       - Analyse des probabilités
       - Export des données
    
        
    ### 🚀 Déploiement
    
    ```bash
    # Installation
    pip install -r requirements.txt
    
    # Lancement
    streamlit run app_stream4.py
    
    # Accès
    Local: http://localhost:8501
    Réseau: http://<IP>:8501
    ```
    
    ### 🛠️ Technologies Utilisées
    
    - **Streamlit** : Interface web
    - **TensorFlow/Keras** : Deep Learning
    - **Joblib** : Sérialisation modèle
    - **PIL/Pillow** : Traitement image
    - **Matplotlib** : Visualisation
    - **Pandas/Numpy** : Data processing
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section FAQ
    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
    st.markdown("### ❓ Foire Aux Questions")
    
    with st.expander("📝 Comment améliorer la précision?"):
        st.markdown("""
        - Utiliser des images de meilleure qualité
        - Recadrer l'objet principal
        - Éviter les images floues
        - Utiliser un éclairage uniforme
        """)
    
    with st.expander("⚠️ Le modèle ne fonctionne pas bien sur certaines images"):
        st.markdown("""
        Le modèle CIFAR-10 a été entraîné sur des images 32×32, ce qui limite:
        - La détection de petits détails
        - Les images complexes
        - Les objets partiellement visibles
        """)
    
    with st.expander("🔧 Problèmes techniques courants"):
        st.markdown("""
        **Problème:** Modèle non chargé
        **Solution:** Vérifier le chemin `model4/cifar_model.joblib`
        
        **Problème:** Erreur de prétraitement
        **Solution:** Utiliser des formats PNG/JPG standard
        
        **Problème:** Application lente
        **Solution:** Réduire la taille des images Chargées
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
---
**Vision par ordinateur - Classificateur CIFAR-10** • Développé avec Streamlit • © 2026 Alfred BARGO""")
st.markdown("</div>", unsafe_allow_html=True)