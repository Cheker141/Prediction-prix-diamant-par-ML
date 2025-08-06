# Diamond Price Prediction 💎

Un modèle de machine learning pour prédire le prix des diamants avec une précision de **98.02%**, incluant une interface graphique interactive.

## 🎯 Résultats clés

| Métrique | Valeur |
|----------|--------|
| **R² Score** | 98.02% |
| **RMSE** | 561$ |
| **Variables** | 5 features optimisées |
| **Modèle** | Random Forest Regressor |

## 💎 Interface utilisateur

🖥️ **Interface Streamlit interactive** permettant de :
- Saisir les caractéristiques d'un diamant (dimensions, qualité)
- Obtenir une estimation de prix instantanée
- Visualiser les détails techniques (volume, profondeur)

![Interface Preview](https://via.placeholder.com/600x400/1f1f23/ffffff?text=Diamond+Price+Predictor+Interface)

## 🚀 Utilisation rapide

### Méthode 1 : Interface graphique (Recommandée)
```bash
streamlit run diamond_app.py
```
Ouvre une interface web interactive dans votre navigateur.

### Méthode 2 : Code Python
```python
import pandas as pd
import pickle

# Charger le modèle
with open('models/diamond_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prédire le prix d'un diamant
# Variables : [depth, table, clarity_num, cut_num, color_num, Volume]
diamond_features = pd.DataFrame({
    'depth': [62.0], 'table': [57.0], 'clarity_num': [4], 
    'cut_num': [4], 'color_num': [6], 'Volume': [150.0]
})
predicted_price = model.predict(diamond_features)
print(f"Prix prédit : {predicted_price[0]:.0f}$")
```

## 🛠️ Installation

```bash
git clone https://github.com/[votre-username]/diamond-price-prediction
cd diamond-price-prediction
pip install -r requirements.txt
```

## 📈 Méthodologie

### 1. Analyse exploratoire
- Matrice de corrélation des variables
- Analyse par segments de taille (effet masqué des variables qualitatives)
- Création de la variable Volume (x × y × z)

### 2. Modélisation comparative
- **Régression linéaire** : 90.47% R² → Baseline solide
- **Random Forest** : 98.02% R² → Capture les non-linéarités et interactions

### 3. Optimisation
- Élimination des variables redondantes (x, y, z, carat)
- Volume seul capture 99% de l'information géométrique

## 🏗️ Structure du projet

```
├── diamond_app.py          # Interface Streamlit interactive
├── notebooks/              # Analyse complète en Jupyter
├── models/                 # Modèle Random Forest sauvegardé  
├── src/                    # Fonctions Python (predict_diamond.py)
├── reports/                # Rapport détaillé (PDF)
└── requirements.txt        # Dépendances Python
```

## 🔧 Guide d'utilisation de l'interface

### Paramètres d'entrée :
- **Dimensions (mm)** : Longueur (x), Largeur (y), Hauteur (z)
- **Qualité** : Cut (Fair → Ideal), Color (D → J), Clarity (I1 → IF)
- **Technique** : Table (40-80%), Profondeur (50-80%)

### Résultats affichés :
- **Prix estimé** en dollars
- **Volume calculé** (mm³)  
- **Résumé complet** des caractéristiques
- **Évaluation qualitative** du diamant

## 📊 Variable la plus importante
**Volume** (89.13% d'importance) - La géométrie du diamant est le facteur prédictif principal, plus que le poids en carats.

## 🔍 Insights découverts

- **Paradoxe de la pureté** : À taille constante, les diamants de pureté exceptionnelle peuvent coûter moins cher (segmentation de marché)
- **Dominance géométrique** : Volume > Carat pour la prédiction de prix
- **Non-linéarités importantes** : Random Forest surpasse nettement la régression linéaire (+7.5% R²)
- **Interactions complexes** : Les effets des variables qualitatives varient selon la taille

## ⚡ Technologies utilisées

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## 📄 Rapport détaillé

Le rapport académique complet avec méthodologie détaillée, analyses statistiques et conclusions est disponible sur [LinkedIn](www.linkedin.com/in/cheker-neffati).

## 📊 Dataset

Utilise le dataset `diamonds` de Seaborn (~54k observations) avec les caractéristiques :
- **Physiques** : carat, dimensions (x,y,z), depth, table
- **Qualitatives** : cut, color, clarity  
- **Cible** : price

---

⭐ **N'hésitez pas à star le projet si il vous a été utile !**