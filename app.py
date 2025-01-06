import cv2
import easyocr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np

# Extraction de texte
def extract_text(image_path):
    reader = easyocr.Reader(['fr'])
    result = reader.readtext(image_path)
    text = " ".join([detection[1] for detection in result])
    return text

# Fonction pour déterminer le type de document
def detect_document_type(text, reference_texts):
    # Convertir les textes en vecteurs TF-IDF
    vectorizer = TfidfVectorizer()
    all_texts = reference_texts + [text]  # Ajouter le texte extrait aux textes de référence
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculer la similarité cosinus entre le texte extrait et les références
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    doc_types = ["carte nationale d'identite", "permis de conduire", "facture", "carte vitale"]
    
    # Trouver le type avec la plus grande similarité
    max_index = similarities.argmax()
    return doc_types[max_index]

# Streamlit App
def main():
    st.title("Document Type Detection")
    st.write("Upload an image (CNI, Permis, Facture, etc.) to classify it.")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        # Convert the uploaded image to an OpenCV format
        image_bytes = uploaded_image.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
     
        # Afficher l'image prétraitée
        st.image(image, caption="Preprocessed Image", channels="GRAY", use_column_width=True)

        # Extraire le texte
        text = extract_text(image)
        
        # Références pour le calcul de la similarité
        reference_texts = [
            "république française carte nationale d'identité nom prénom sexe date de naissance lieu de naissance",  # Exemple CNI
            "permis de conduire catégorie conducteur numéro date d'émission validité",  # Exemple Permis
            "facture montant date fournisseur produit total",  # Exemple Facture
            "carte vitale"
        ]

        # Détecter le type de document
        document_type = detect_document_type(text, reference_texts)
        
        # Afficher le texte extrait et le type détecté
        st.write(f"### Extracted Text:")
        st.text(text)
        st.write(f"### Detected Document Type: {document_type}")
        
        # Créer un DataFrame pour afficher
        df = pd.DataFrame({"image": [uploaded_image.name], "text": [text], "type": [document_type]})

        # Afficher le DataFrame
        st.write("### Results:")
        st.dataframe(df)

        # Convertir en CSV et offrir l'option de téléchargement
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="document_result.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
