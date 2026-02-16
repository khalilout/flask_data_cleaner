import os
import numpy as np  # ajouté
from flask import Flask, request, send_file, make_response, jsonify
import pandas as pd
import xmltodict
import io
import json
from io import BytesIO

app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    return "Service de nettoyage de données est opérationnel."

#  Fusion intelligente des valeurs
def fusion_valeurs(series):
    valeurs = series.dropna().astype(str)
    valeurs = valeurs[valeurs.str.strip() != ""]
    uniques = valeurs.unique()

    if len(uniques) == 0:
        return np.nan
    elif len(uniques) == 1:
        return uniques[0]
    else:
        return " ; ".join(uniques)
    
#  Détection automatique des colonnes clés
def detecter_colonnes_cles(df):
    colonnes_obj = df.select_dtypes(include=["object"]).columns
    colonnes_cles = []

    for col in colonnes_obj:
        if df[col].nunique() > 1:
            colonnes_cles.append(col)

    return colonnes_cles


@app.route('/clean', methods=['POST'])
@app.route('/api/clean', methods=['POST'])
def import_file():
    file = request.files['file']
    ext = file.filename.split('.')[-1].lower()

    # Lecture du fichier selon son format
    if ext in ['csv', 'txt']:
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    elif ext == 'json':
        df = pd.read_json(file)
    elif ext == 'xml':
        df = pd.read_xml(BytesIO(file))
    # elif ext == 'xml':
    #         # ✅ CORRECTION XML : Parsing amélioré
    #         try:
    #             content = file.read()
    #             data = xmltodict.parse(content)
                
    #             print(f"🔍 Clés XML trouvées: {list(data.keys())}")
                
    #             # Stratégie de recherche des records
    #             records = None
                
    #             # 1. Chercher dans les clés communes
    #             if 'root' in data:
    #                 records = data['root']
    #             elif 'data' in data:
    #                 records = data['data']
    #             elif 'records' in data:
    #                 records = data['records']
    #             elif 'items' in data:
    #                 records = data['items']
    #             else:
    #                 # Prendre la première clé
    #                 root_key = list(data.keys())[0]
    #                 records = data[root_key]
    #                 print(f"✅ Utilisation de la clé racine: {root_key}")
                
    #             # 2. Si records est un dict, chercher une liste dedans
    #             if isinstance(records, dict):
    #                 print(f"🔍 Records est un dict, recherche de liste...")
    #                 found_list = False
                    
    #                 for key, value in records.items():
    #                     if isinstance(value, list):
    #                         records = value
    #                         found_list = True
    #                         print(f"✅ Liste trouvée dans la clé: {key}")
    #                         break
                    
    #                 # Si pas de liste trouvée, c'est un seul record
    #                 if not found_list:
    #                     records = [records]
    #                     print("⚠️ Aucune liste trouvée, conversion en liste unique")
                
    #             # 3. Si records est déjà une liste, c'est bon
    #             elif isinstance(records, list):
    #                 print(f"✅ Records est déjà une liste de {len(records)} éléments")
    #             else:
    #                 # Cas imprévu, le mettre en liste
    #                 records = [records]
    #                 print("⚠️ Type inattendu, conversion en liste")
                
    #             df = pd.DataFrame(records)
    #             print(f"✅ DataFrame créé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                
    #         except Exception as e:
    #             error_msg = f"Erreur lors de la lecture du fichier XML: {str(e)}"
    #             print(f"❌ {error_msg}")
    #             return jsonify({"error": error_msg}), 400
    # else:
    #     return "Format non supporté", 400
    # elif ext == 'xml':
    #     # Lire le contenu du fichier XML
    #     content = file.read()
    #     data = xmltodict.parse(content)
        
    #     # Trouver les données (première clé du dictionnaire)
    #     root_key = list(data.keys())[0]
    #     root = data[root_key]
        
    #     # Si c'est une liste, c'est bon
    #     if isinstance(root, list):
    #         records = root
    #     # Si c'est un dictionnaire
    #     elif isinstance(root, dict):
    #         # Chercher la première liste dans le dictionnaire
    #         records = None
    #         for key, value in root.items():
    #             if isinstance(value, list):
    #                 records = value
    #                 break
    #         # Si pas de liste trouvée, c'est un seul enregistrement
    #         if records is None:
    #             records = [root]
    #     else:
    #         # Si ce n'est ni liste ni dict, mettre dans une liste
    #         records = [root]
        
    #     # Créer le DataFrame
    #     df = pd.DataFrame(records) 
    else:
        return "Format non supporté", 400

    df_original = df.copy()
    print(f"💾 Données originales sauvegardées: {df_original.shape}")

        # Nettoyage initial ("--" → NaN)

    #  Nettoyage initial ("--" → NaN)
    VALEURS_MANQUANTES = [
        "", " ", "  ", "   ", "\t", "\n", "\r",

        "--", "---", "—", "–", "-", "_",
        "?", "??", "***", "*",

        "NA", "N/A", "n/a", "na", "Na",
        "NULL", "null",
        "None", "none",
        "Nil", "nil",

        "Missing", "missing",
        "Unknown", "unknown",
        "Undefined", "undefined",

        "Not Available", "not available",
        "Not Applicable", "not applicable",

        "#N/A", "#NA", "#VALUE!", "#DIV/0!",
        "#REF!", "#NAME?", "#NUM!",

        "vide", "inconnu",
        "non renseigné", "non disponible",
        "aucun",

        "empty", "not provided", "no data"
    ]

    df.replace(VALEURS_MANQUANTES, np.nan, inplace=True)
    

    #  FUSION DES DOUBLONS (IMPORTANT)
    colonnes_cles = detecter_colonnes_cles(df)
    if colonnes_cles:
        df = df.groupby(colonnes_cles, as_index=False).agg(fusion_valeurs)

    # Conversion numérique
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')



    # Valeurs manquantes
    colonnes_numeriques = df.select_dtypes(include=["int64", "float64"]).columns #si la colonne est numérique
    colonnes_categorielles = df.select_dtypes(include=["object"]).columns #si la colonne est catégorielle

    for col in colonnes_numeriques:
        if df[col].isnull().sum() > 0:
            skewness = df[col].skew()
            if abs(skewness) < 0.5:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())

    for col in colonnes_categorielles:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Valeur Aberante
    outlier_method = request.form.get("outlier_method", "none")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if outlier_method == "iqr":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
    elif outlier_method == "median":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            med = df[col].median()
            df.loc[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR), col] = med
    elif outlier_method == "log":
        for col in num_cols:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])
    elif outlier_method == "delete":
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

            

    # ✅ CORRECTION STATISTIQUES : Calculer sur l'ORIGINAL !
    stats = {}
        
    # Calculer le nombre total de doublons sur l'ORIGINAL
    nb_doublons_total = int(df_original.duplicated().sum())
    print(f"📊 Doublons totaux trouvés: {nb_doublons_total}")

    for col in df.columns:
        stats[col] = {}
            
        # ✅ Valeurs manquantes calculées sur l'ORIGINAL
        if col in df_original.columns:
            stats[col]["missing"] = int(df_original[col].isnull().sum())
            print(f"📊 {col}: {stats[col]['missing']} valeurs manquantes")
        else:
            stats[col]["missing"] = 0
            
        # ✅ Doublons calculés sur l'ORIGINAL
        if col in df_original.columns:
            stats[col]["duplicates"] = int(df_original[col].duplicated().sum())
        else:
            stats[col]["duplicates"] = 0
            
        if col in num_cols:
            # Pour les colonnes numériques
            # ✅ Calculer les outliers sur l'ORIGINAL
            if col in df_original.columns:
                # Convertir en numérique pour le calcul
                col_original_numeric = pd.to_numeric(df_original[col], errors='coerce')
                    
                Q1 = col_original_numeric.quantile(0.25)
                Q3 = col_original_numeric.quantile(0.75)
                IQR = Q3 - Q1
                    
                # Compter les outliers
                outliers_mask = (col_original_numeric < Q1 - 1.5*IQR) | (col_original_numeric > Q3 + 1.5*IQR)
                stats[col]["outliers"] = int(outliers_mask.sum())
                print(f"📊 {col}: {stats[col]['outliers']} valeurs aberrantes")
            else:
                stats[col]["outliers"] = 0
                
            # Statistiques descriptives (sur le nettoyé)
            stats[col]["mean"] = float(df[col].mean())
            stats[col]["median"] = float(df[col].median())
            stats[col]["min"] = float(df[col].min())
            stats[col]["max"] = float(df[col].max())
            stats[col]["std"] = float(df[col].std())
        else:
            # Pour les colonnes catégorielles
            if len(df[col].mode()) > 0:
                stats[col]["mode"] = str(df[col].mode()[0])
            else:
                stats[col]["mode"] = "N/A"
                stats[col]["unique_count"] = int(df[col].nunique())
                stats[col]["outliers"] = 0  # Pas d'outliers pour catégorielles

        print(f"✅ Statistiques calculées pour {len(stats)} colonnes")


    # Export CSV
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Retour CSV avec stats dans headers (Laravel pourra les lire)
    response = send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='donnees_nettoyees.csv'
    )

    # Ajouter stats dans headers (JSON stringifiée)
    import json
    response.headers["X-Data-Stats"] = json.dumps(stats)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
    # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
    # app.run(debug=True, port=5000)