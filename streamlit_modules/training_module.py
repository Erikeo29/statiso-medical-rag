"""
Module de formation interactive intégré - Phase 2
Combine quiz, parcours d'apprentissage et simulations
"""
import json
import random
import sqlite3
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass

@dataclass
class LearningModule:
    """Structure d'un module d'apprentissage"""
    id: str
    title: str
    description: str
    duration_minutes: int
    difficulty: int  # 1-3
    prerequisites: List[str]
    learning_objectives: List[str]
    content_type: str  # "theory", "practice", "quiz", "simulation"
    tags: List[str]

class TrainingSystem:
    """Système de formation complet avec quiz, parcours et simulations"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Créer le dossier s'il n'existe pas
            Path("data/training").mkdir(parents=True, exist_ok=True)
            db_path = "data/training/user_progress.db"
        
        self.db_path = db_path
        self.init_database()
        self.quiz_database = self.load_quiz_database()
        self.learning_paths = self._initialize_learning_paths()
        self.modules = self._initialize_modules()
    
    def init_database(self):
        """Initialise la base de données de progression"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table utilisateurs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                department TEXT,
                experience_level TEXT
            )
        ''')
        
        # Table progression
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                module_id TEXT,
                score REAL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_spent INTEGER,
                attempts INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Table réponses quiz
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question_id TEXT,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct BOOLEAN,
                answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_taken INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Table badges/certifications
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS badges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                badge_type TEXT,
                badge_name TEXT,
                earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                certificate_hash TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_quiz_database(self) -> Dict:
        """Charge la base de questions depuis JSON"""
        quiz_path = Path("data/training/quiz_database.json")
        
        if not quiz_path.exists():
            # Créer une base de questions par défaut
            default_quiz = self.create_default_quiz_database()
            quiz_path.parent.mkdir(parents=True, exist_ok=True)
            with open(quiz_path, 'w', encoding='utf-8') as f:
                json.dump(default_quiz, f, ensure_ascii=False, indent=2)
            return default_quiz
        
        with open(quiz_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_default_quiz_database(self) -> Dict:
        """Crée une base de questions par défaut pour dispositifs médicaux"""
        return {
            "basics": {
                "title": "Concepts de base",
                "questions": [
                    {
                        "id": "B001",
                        "question": "Quelle est la différence principale entre un intervalle de confiance et un intervalle de tolérance ?",
                        "type": "multiple_choice",
                        "options": [
                            "L'intervalle de confiance est toujours plus large",
                            "L'intervalle de confiance estime la position de la moyenne, l'intervalle de tolérance encadre une proportion de la population",
                            "Il n'y a pas de différence, ce sont des synonymes",
                            "L'intervalle de tolérance nécessite moins d'échantillons"
                        ],
                        "correct_answer": 1,
                        "explanation": "L'IC (ISO 2602) estime où se trouve la vraie moyenne, tandis que l'IT (ISO 16269-6) garantit qu'une proportion donnée de la population sera dans ces limites.",
                        "difficulty": 1,
                        "tags": ["concepts", "fondamentaux"]
                    },
                    {
                        "id": "B002",
                        "question": "Pour un procédé de coating avec une cible de 12.0 ± 0.5 μm, quel type d'analyse utiliseriez-vous pour vérifier que 99% de la production sera conforme ?",
                        "type": "multiple_choice",
                        "options": [
                            "Intervalle de confiance bilatéral",
                            "Test t de Student",
                            "Intervalle de tolérance bilatéral",
                            "Carte de contrôle Shewhart"
                        ],
                        "correct_answer": 2,
                        "explanation": "Un intervalle de tolérance bilatéral (ISO 16269-6) permet de garantir qu'une proportion donnée (99%) de la production sera dans les limites spécifiées.",
                        "difficulty": 2,
                        "tags": ["application", "tolérance", "biosenseur"]
                    },
                    {
                        "id": "B003",
                        "question": "Vrai ou Faux : Un Cpk > 1.33 indique un processus capable pour les dispositifs médicaux.",
                        "type": "true_false",
                        "correct_answer": True,
                        "explanation": "VRAI. Dans l'industrie des dispositifs médicaux, un Cpk ≥ 1.33 est généralement requis pour considérer un processus comme capable.",
                        "difficulty": 1,
                        "tags": ["capabilité", "qualité"]
                    }
                ]
            },
            "biosensor_specific": {
                "title": "Applications Biosenseurs",
                "questions": [
                    {
                        "id": "BIO001",
                        "question": "Pour des électrodes avec résistance cible 100Ω ± 5Ω, vous mesurez x̄=99.8Ω et s=1.2Ω (n=30). Le processus est-il centré ?",
                        "type": "multiple_choice",
                        "options": [
                            "Oui, car 99.8 est dans [95, 105]",
                            "Non, il faut faire un test d'hypothèse",
                            "Impossible à dire sans plus de données",
                            "Oui, car l'écart est < 1%"
                        ],
                        "correct_answer": 1,
                        "explanation": "Il faut réaliser un test d'hypothèse (H₀: μ=100) pour déterminer statistiquement si l'écart observé est significatif.",
                        "difficulty": 2,
                        "tags": ["électrodes", "centrage", "test"]
                    },
                    {
                        "id": "BIO002",
                        "question": "L'épaisseur d'un coating Ag/AgCl doit être 12±0.3 μm. Avec n=25, quelle marge d'erreur maximale accepteriez-vous pour l'IC à 95% ?",
                        "type": "numerical",
                        "correct_answer": 0.1,
                        "tolerance": 0.02,
                        "unit": "μm",
                        "explanation": "Pour être confiant que la moyenne reste dans les specs, la marge d'erreur devrait être ≤ 0.1 μm (environ 1/3 de la tolérance).",
                        "difficulty": 3,
                        "tags": ["coating", "IC", "calcul"]
                    }
                ]
            }
        }
    
    def _initialize_learning_paths(self) -> Dict:
        """Définit les parcours d'apprentissage disponibles"""
        return {
            "beginner": {
                "title": "🎓 Parcours Débutant - Fondamentaux ISO",
                "description": "Introduction aux analyses statistiques ISO pour dispositifs médicaux",
                "total_duration": 240,
                "modules": [
                    "intro_statistics",
                    "intro_iso2602",
                    "intro_iso16269",
                    "basic_calculations",
                    "quiz_beginner"
                ],
                "certification": "Certificat Fondamentaux ISO - Dispositifs Médicaux"
            },
            "intermediate": {
                "title": "📊 Parcours Intermédiaire - Applications Pratiques",
                "description": "Applications aux procédés de fabrication biosenseurs",
                "total_duration": 360,
                "modules": [
                    "review_basics",
                    "advanced_iso2602",
                    "advanced_iso16269",
                    "capability_analysis",
                    "biosensor_cases",
                    "quiz_intermediate"
                ],
                "certification": "Certificat Praticien ISO - Biosenseurs"
            },
            "expert": {
                "title": "🏆 Parcours Expert - Maîtrise Complète",
                "description": "Expertise en validation et qualification process",
                "total_duration": 480,
                "modules": [
                    "review_intermediate",
                    "validation_methods",
                    "process_optimization",
                    "regulatory_compliance",
                    "advanced_simulations",
                    "quiz_expert"
                ],
                "certification": "Certificat Expert ISO - Qualification Process"
            }
        }
    
    def _initialize_modules(self) -> Dict[str, LearningModule]:
        """Initialise les modules d'apprentissage"""
        return {
            "intro_statistics": LearningModule(
                id="intro_statistics",
                title="Introduction aux statistiques",
                description="Concepts de base : moyenne, écart-type, distributions",
                duration_minutes=30,
                difficulty=1,
                prerequisites=[],
                learning_objectives=[
                    "Comprendre moyenne et écart-type",
                    "Distinguer population et échantillon",
                    "Reconnaître une distribution normale"
                ],
                content_type="theory",
                tags=["statistiques", "fondamentaux"]
            ),
            "intro_iso2602": LearningModule(
                id="intro_iso2602",
                title="ISO 2602 - Intervalles de confiance",
                description="Estimation de la moyenne avec incertitude",
                duration_minutes=45,
                difficulty=1,
                prerequisites=["intro_statistics"],
                learning_objectives=[
                    "Calculer un IC pour la moyenne",
                    "Interpréter le niveau de confiance",
                    "Choisir entre distribution Z et t"
                ],
                content_type="theory",
                tags=["ISO2602", "IC"]
            ),
            "intro_iso16269": LearningModule(
                id="intro_iso16269",
                title="ISO 16269-6 - Intervalles de tolérance",
                description="Encadrer une proportion de la population",
                duration_minutes=45,
                difficulty=1,
                prerequisites=["intro_statistics"],
                learning_objectives=[
                    "Comprendre le concept d'IT",
                    "Utiliser les facteurs k",
                    "Distinguer IT bilatéral et unilatéral"
                ],
                content_type="theory",
                tags=["ISO16269", "IT"]
            ),
            "biosensor_cases": LearningModule(
                id="biosensor_cases",
                title="Cas pratiques biosenseurs",
                description="Applications réelles en production d'électrodes",
                duration_minutes=60,
                difficulty=2,
                prerequisites=["intro_iso2602", "intro_iso16269"],
                learning_objectives=[
                    "Analyser des données de résistance",
                    "Valider un process de coating",
                    "Calculer la capabilité Cpk"
                ],
                content_type="practice",
                tags=["biosenseur", "pratique", "électrodes"]
            )
        }
    
    def get_adaptive_quiz(self, user_id: str, module: str = "all", difficulty: int = 1, num_questions: int = 5) -> List[Dict]:
        """Génère un quiz adaptatif basé sur l'historique de l'utilisateur"""
        available_questions = []
        
        # Collecter les questions appropriées
        for category in self.quiz_database.values():
            for question in category.get("questions", []):
                if module == "all" or any(tag in question.get("tags", []) for tag in [module]):
                    if question.get("difficulty", 1) <= difficulty + 1:
                        available_questions.append(question)
        
        # Sélection aléatoire pondérée
        if len(available_questions) > num_questions:
            selected = random.sample(available_questions, num_questions)
        else:
            selected = available_questions
        
        return selected
    
    def evaluate_response(self, question: Dict, user_answer) -> Dict:
        """Évalue une réponse utilisateur"""
        is_correct = False
        
        if question["type"] == "multiple_choice":
            is_correct = (user_answer == question["correct_answer"])
        elif question["type"] == "true_false":
            is_correct = (user_answer == question["correct_answer"])
        elif question["type"] == "numerical":
            try:
                user_value = float(user_answer)
                correct_value = float(question["correct_answer"])
                tolerance = question.get("tolerance", 0.01)
                is_correct = abs(user_value - correct_value) <= tolerance
            except:
                is_correct = False
        
        return {
            "is_correct": is_correct,
            "correct_answer": question.get("correct_answer"),
            "explanation": question.get("explanation", ""),
            "difficulty": question.get("difficulty", 1)
        }
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """Récupère les statistiques simplifiées d'un utilisateur"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as total, SUM(is_correct) as correct
                FROM quiz_responses
                WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                return {
                    "total_questions": result[0],
                    "correct_answers": result[1] or 0,
                    "success_rate": (result[1] or 0) / result[0] * 100
                }
        except:
            pass
        
        return {
            "total_questions": 0,
            "correct_answers": 0,
            "success_rate": 0
        }
    
    def save_quiz_response(self, user_id: str, question_id: str, user_answer, is_correct: bool):
        """Enregistre une réponse de quiz"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quiz_responses 
                (user_id, question_id, user_answer, is_correct, time_taken)
                VALUES (?, ?, ?, ?, 0)
            ''', (user_id, question_id, str(user_answer), is_correct))
            
            conn.commit()
            conn.close()
        except:
            pass  # Ignorer les erreurs pour simplifier
    
    def create_or_get_user(self, user_id: str, name: str = None) -> str:
        """Crée ou récupère un utilisateur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vérifier si l'utilisateur existe
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        if cursor.fetchone():
            conn.close()
            return user_id
        
        # Créer nouvel utilisateur
        cursor.execute('''
            INSERT INTO users (user_id, name, email, department, experience_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, name or user_id, f"{user_id}@example.com", "R&D", "Débutant"))
        
        conn.commit()
        conn.close()
        return user_id


class SimulationEngine:
    """Moteur de simulations interactives pour l'apprentissage"""
    
    def __init__(self):
        self.random_state = 42
    
    def simulate_ic_vs_it_comparison(self, n: int = 30, confidence: float = 0.95, proportion: float = 0.95) -> go.Figure:
        """Simule et compare IC vs IT visuellement"""
        np.random.seed(self.random_state)
        
        # Générer des données exemple
        true_mean = 100
        true_std = 2
        data = np.random.normal(true_mean, true_std, n)
        
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        # Calcul IC
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        ic_margin = t_critical * sample_std / np.sqrt(n)
        ic_lower = sample_mean - ic_margin
        ic_upper = sample_mean + ic_margin
        
        # Calcul IT (approximation)
        k_factor = 2.5  # Approximation pour 95/95
        it_margin = k_factor * sample_std
        it_lower = sample_mean - it_margin
        it_upper = sample_mean + it_margin
        
        # Créer la figure
        fig = go.Figure()
        
        # Histogramme des données
        fig.add_trace(go.Histogram(
            x=data,
            name='Données',
            nbinsx=15,
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # IC
        fig.add_vrect(x0=ic_lower, x1=ic_upper, 
                      fillcolor="green", opacity=0.2,
                      annotation_text="IC", annotation_position="top")
        
        # IT
        fig.add_vrect(x0=it_lower, x1=it_upper,
                      fillcolor="red", opacity=0.1,
                      annotation_text="IT", annotation_position="bottom")
        
        # Moyenne
        fig.add_vline(x=sample_mean, line_dash="dash", 
                      annotation_text=f"x̄={sample_mean:.1f}")
        
        fig.update_layout(
            title=f"Comparaison IC vs IT (n={n})<br>" +
                  f"IC: [{ic_lower:.1f}, {ic_upper:.1f}] - Largeur: {2*ic_margin:.1f}<br>" +
                  f"IT: [{it_lower:.1f}, {it_upper:.1f}] - Largeur: {2*it_margin:.1f}",
            xaxis_title="Valeur",
            yaxis_title="Fréquence",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def simulate_sample_size_effect(self) -> go.Figure:
        """Montre l'effet de n sur la largeur des intervalles"""
        sample_sizes = [10, 20, 30, 50, 100, 200]
        ic_widths = []
        it_widths = []
        
        for n in sample_sizes:
            # IC : largeur ∝ 1/√n
            t_critical = stats.t.ppf(0.975, df=n-1)
            ic_width = 2 * t_critical / np.sqrt(n)
            ic_widths.append(ic_width)
            
            # IT : largeur diminue lentement
            k_approx = 2.0 + 10/n  # Approximation
            it_width = 2 * k_approx
            it_widths.append(it_width)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sample_sizes, y=ic_widths,
            mode='lines+markers',
            name='IC (relatif)',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=sample_sizes, y=it_widths,
            mode='lines+markers',
            name='IT (relatif)',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Impact de la taille d'échantillon sur la précision",
            xaxis_title="Taille d'échantillon (n)",
            yaxis_title="Largeur relative de l'intervalle",
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    def simulate_process_capability(self, mean: float = 100, std: float = 1.5, 
                                  lsl: float = 95, usl: float = 105) -> Dict:
        """Simule et visualise la capabilité d'un processus"""
        np.random.seed(self.random_state)
        
        # Générer données process
        n = 1000
        data = np.random.normal(mean, std, n)
        
        # Calculer Cp et Cpk
        spec_range = usl - lsl
        cp = spec_range / (6 * std)
        cpk = min((mean - lsl) / (3 * std), (usl - mean) / (3 * std))
        
        # Calculer % hors specs
        below_lsl = np.sum(data < lsl) / n * 100
        above_usl = np.sum(data > usl) / n * 100
        out_of_spec = below_lsl + above_usl
        
        # Créer visualisation
        fig = go.Figure()
        
        # Histogramme
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name='Production',
            marker_color='lightblue'
        ))
        
        # Courbe normale théorique
        x_range = np.linspace(data.min(), data.max(), 100)
        y_normal = stats.norm.pdf(x_range, mean, std) * n * (data.max() - data.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x_range, y=y_normal,
            mode='lines',
            name='Distribution théorique',
            line=dict(color='red', width=2)
        ))
        
        # Limites de spécification
        fig.add_vline(x=lsl, line_dash="dash", line_color="red",
                      annotation_text=f"LSL={lsl}")
        fig.add_vline(x=usl, line_dash="dash", line_color="red",
                      annotation_text=f"USL={usl}")
        fig.add_vline(x=mean, line_dash="solid", line_color="green",
                      annotation_text=f"μ={mean}")
        
        # Zones de non-conformité
        fig.add_vrect(x0=data.min(), x1=lsl,
                      fillcolor="red", opacity=0.1)
        fig.add_vrect(x0=usl, x1=data.max(),
                      fillcolor="red", opacity=0.1)
        
        fig.update_layout(
            title=f"Analyse de Capabilité<br>" +
                  f"Cp={cp:.2f}, Cpk={cpk:.2f}<br>" +
                  f"Hors specs: {out_of_spec:.2f}% ({below_lsl:.2f}% < LSL, {above_usl:.2f}% > USL)",
            xaxis_title="Valeur",
            yaxis_title="Fréquence",
            template="plotly_white"
        )
        
        return {
            "figure": fig,
            "cp": cp,
            "cpk": cpk,
            "out_of_spec_percentage": out_of_spec,
            "process_capable": cpk >= 1.33
        }