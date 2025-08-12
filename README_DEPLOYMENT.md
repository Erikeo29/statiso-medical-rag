# Guide de Déploiement - StatISO-Medical RAG

## 📋 Fichiers nécessaires pour déployer sur un autre PC

### Structure minimale requise :
```
StatISO-Medical/
├── streamlit_app_RAG.py           # Application principale
├── requirements.txt                # Dépendances Python
├── streamlit_modules/              # Modules obligatoires
│   ├── __init__.py
│   ├── iso_2602.py                # Calculateur IC
│   ├── iso_16269_6.py             # Calculateur IT
│   ├── data_handler.py            # Gestion données
│   ├── report_generator.py        # Rapports
│   ├── training_module.py         # Formation
│   └── rag_module.py              # Moteur RAG
└── data/                          # Créé automatiquement
    ├── training/                  # Base quiz
    └── vector_store.pkl           # Index RAG
```

## 🚀 Instructions d'installation

### 1. Copier les fichiers
```bash
# Copier tout le dossier du projet
cp -r "1- Normes stat - Claude_v1/" /destination/
```

### 2. Installer Python (3.9 minimum)
```bash
# Vérifier la version
python3 --version

# Si besoin, installer Python 3.9+
sudo apt update
sudo apt install python3.9 python3-pip
```

### 3. Créer un environnement virtuel (recommandé)
```bash
cd /destination/StatISO-Medical/
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 5. Lancer l'application
```bash
streamlit run streamlit_app_RAG.py
```

L'application sera accessible sur `http://localhost:8501`

## 🔧 Configuration optionnelle

### Changer le port
```bash
streamlit run streamlit_app_RAG.py --server.port 8080
```

### Mode production
```bash
streamlit run streamlit_app_RAG.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501
```

## ⚠️ Notes importantes

1. **Première exécution** : Les dossiers `data/` seront créés automatiquement
2. **Base de données** : La base SQLite de progression sera initialisée au premier lancement
3. **Index RAG** : L'index vectoriel sera construit automatiquement
4. **Mémoire** : Prévoir ~500 MB de RAM pour l'application
5. **Navigateur** : Compatible Chrome, Firefox, Edge, Safari

## 🐳 Option Docker (alternative)

Créer un `Dockerfile` :
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app_RAG.py", "--server.headless", "true"]
```

Puis :
```bash
docker build -t statiso-medical .
docker run -p 8501:8501 statiso-medical
```

## 📦 Archive complète

Pour créer une archive prête à déployer :
```bash
tar -czf StatISO-Medical-RAG.tar.gz \
  streamlit_app_RAG.py \
  requirements.txt \
  streamlit_modules/ \
  README_DEPLOYMENT.md
```

## 🆘 Dépannage

### Erreur "Module not found"
→ Vérifier que tous les fichiers `streamlit_modules/*.py` sont présents

### Port 8501 déjà utilisé
→ Utiliser `--server.port 8502` ou tuer le processus existant

### Erreur SQLite
→ Supprimer `data/training/user_progress.db` et relancer

### Performance lente
→ Augmenter la RAM disponible ou réduire le cache Streamlit

## 📞 Support

Pour toute question sur le déploiement, consulter la documentation principale ou contacter l'équipe R&D.