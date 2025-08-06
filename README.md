# Diamond Price Prediction

Modèle de machine learning pour estimer le prix des diamants basé sur leurs caractéristiques physiques et qualitatives.

## Démo en ligne

**[🔗 Essayer l'application](https://prediction-prix-diamant-par-ml-cheker.streamlit.app/)**

## Performance

- **R² Score** : 98.02%
- **RMSE** : 561$
- **Algorithme** : Random Forest
- **Dataset** : 54,000 diamants

## Fonctionnalités

L'application web permet de :
- Saisir les dimensions d'un diamant (longueur, largeur, hauteur)
- Spécifier la qualité (taille, couleur, pureté)
- Ajuster les paramètres techniques (table, profondeur)
- Obtenir une estimation de prix instantanée

## Installation locale

```bash
git clone https://github.com/Cheker141/Prediction-prix-diamant-par-ML.git
cd Prediction-prix-diamant-par-ML
pip install -r requirements.txt
streamlit run diamond_app.py
```

## Méthodologie

### Analyse des données
- Étude de corrélation entre les variables
- Création de la variable "Volume" (x × y × z)
- Analyse de l'importance des features

### Modélisation
- Comparaison de différents algorithmes
- Optimisation des hyperparamètres
- Validation croisée

### Variable principale
Le **volume** représente 89% de l'importance dans la prédiction, surpassant le poids en carats traditionnellement utilisé.

## Technologies

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy

## Structure

```
├── diamond_app.py          # Application web
├── diamond_price_model.pkl # Modèle entraîné
├── diamonds_analysis.ipynb # Analyse des données
├── requirements.txt        # Dépendances
└── README.md              # Documentation
```

## Utilisation programmatique

```python
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load('diamond_price_model.pkl')

# Prédiction
data = pd.DataFrame({
    'depth': [62.0],
    'table': [57.0], 
    'clarity_num': [4],
    'cut_num': [4],
    'color_num': [6],
    'Volume': [150.0]
})

price = model.predict(data)[0]
print(f"Prix estimé: ${price:,.0f}")
```

## Contact

[LinkedIn](https://www.linkedin.com/in/cheker-neffati)