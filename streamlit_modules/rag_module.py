"""
Module RAG simplifié pour StatISO-Medical
Version allégée sans dépendances lourdes pour démonstration
"""
import re
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle

@dataclass
class Document:
    """Structure d'un document"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None

@dataclass
class RAGResponse:
    """Structure d'une réponse RAG"""
    answer: str
    sources: List[Dict]
    confidence: float
    suggestions: List[str]

class SimpleVectorStore:
    """Store vectoriel simplifié utilisant numpy (sans ChromaDB)"""
    
    def __init__(self, persist_path: str = "data/vector_store.pkl"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents = []
        self.embeddings = []
        self.load()
    
    def add_documents(self, documents: List[Document]):
        """Ajoute des documents au store"""
        for doc in documents:
            # Créer un embedding simple basé sur TF-IDF
            embedding = self._create_simple_embedding(doc.content)
            doc.embedding = embedding
            self.documents.append(doc)
            self.embeddings.append(embedding)
        self.save()
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Recherche les documents les plus similaires"""
        if not self.documents:
            return []
        
        # Créer l'embedding de la requête
        query_embedding = self._create_simple_embedding(query)
        
        # Calculer les similarités cosinus
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les n meilleurs résultats
        results = []
        for idx, sim in similarities[:n_results]:
            doc = self.documents[idx]
            results.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata,
                'similarity': sim
            })
        
        return results
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Crée un embedding simple basé sur les fréquences de mots"""
        # Vocabulaire simplifié pour les termes statistiques
        vocab = [
            'intervalle', 'confiance', 'tolérance', 'moyenne', 'écart-type',
            'population', 'échantillon', 'variance', 'distribution', 'normale',
            'facteur', 'k', 't', 'student', 'formule', 'calcul', 'iso',
            '2602', '16269', 'bilatéral', 'unilatéral', 'proportion', 'niveau',
            'spécification', 'limite', 'lsl', 'usl', 'capabilité', 'cp', 'cpk',
            'processus', 'qualité', 'contrôle', 'biosenseur', 'électrode',
            'coating', 'résistance', 'épaisseur', 'mesure', 'analyse'
        ]
        
        # Normaliser le texte
        text_lower = text.lower()
        
        # Créer le vecteur
        embedding = []
        for word in vocab:
            count = text_lower.count(word)
            # TF normalisé
            tf = count / (len(text_lower.split()) + 1)
            embedding.append(tf)
        
        # Normaliser le vecteur
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux vecteurs"""
        return np.dot(vec1, vec2)
    
    def save(self):
        """Sauvegarde le store"""
        try:
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f)
        except:
            pass  # Ignorer les erreurs de sauvegarde
    
    def load(self):
        """Charge le store"""
        try:
            if self.persist_path.exists():
                with open(self.persist_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings = data['embeddings']
        except:
            pass  # Ignorer les erreurs de chargement

class ISOKnowledgeBase:
    """Base de connaissances des normes ISO"""
    
    def __init__(self):
        self.iso_2602_content = self._load_iso_2602()
        self.iso_16269_content = self._load_iso_16269()
        self.medical_device_content = self._load_medical_device_knowledge()
    
    def _load_iso_2602(self) -> Dict:
        """Charge le contenu structuré de ISO 2602"""
        return {
            "title": "ISO 2602:1980 - Intervalles de confiance",
            "sections": {
                "definition": {
                    "title": "Définition",
                    "content": """Un intervalle de confiance (IC) est une estimation par intervalle 
                    d'un paramètre de population. Pour un niveau de confiance de 95%, nous sommes 
                    confiants à 95% que l'intervalle contient la vraie valeur du paramètre.""",
                    "keywords": ["intervalle confiance", "IC", "estimation", "paramètre"]
                },
                "formulas": {
                    "title": "Formules",
                    "content": """Variance inconnue: IC = x̄ ± t(n-1,α/2) × s/√n
                    où x̄ est la moyenne, s l'écart-type échantillon, n la taille, 
                    et t la valeur critique de Student.""",
                    "keywords": ["formule", "calcul", "student", "variance"]
                },
                "usage": {
                    "title": "Utilisation",
                    "content": """L'IC est utilisé pour estimer la position de la moyenne 
                    de la population. Plus n augmente, plus l'intervalle se rétrécit.""",
                    "keywords": ["usage", "application", "moyenne"]
                }
            }
        }
    
    def _load_iso_16269(self) -> Dict:
        """Charge le contenu structuré de ISO 16269-6"""
        return {
            "title": "ISO 16269-6:2014 - Intervalles de tolérance",
            "sections": {
                "definition": {
                    "title": "Définition",
                    "content": """Un intervalle de tolérance (IT) est conçu pour contenir 
                    une proportion spécifiée de la population avec un certain niveau de confiance. 
                    Par exemple, un IT 95/95 contient 95% de la population avec 95% de confiance.""",
                    "keywords": ["intervalle tolérance", "IT", "proportion", "population"]
                },
                "formulas": {
                    "title": "Formules",
                    "content": """Intervalle bilatéral: IT = x̄ ± k(n,p,1-α) × s
                    où k est le facteur de tolérance qui dépend de n, p (proportion) 
                    et 1-α (niveau de confiance).""",
                    "keywords": ["formule", "facteur k", "bilatéral"]
                },
                "difference_ic": {
                    "title": "Différence avec IC",
                    "content": """L'IT est toujours plus large que l'IC car il vise à contenir 
                    les valeurs individuelles, pas seulement estimer la moyenne. 
                    Pour n=30, l'IT est environ 6 fois plus large que l'IC.""",
                    "keywords": ["différence", "IC vs IT", "comparaison"]
                }
            }
        }
    
    def _load_medical_device_knowledge(self) -> Dict:
        """Charge les connaissances spécifiques aux dispositifs médicaux"""
        return {
            "title": "Applications Dispositifs Médicaux",
            "sections": {
                "biosensors": {
                    "title": "Biosenseurs",
                    "content": """Pour les biosenseurs, les paramètres critiques incluent 
                    la résistance (cible 100Ω ± 5Ω), l'épaisseur de coating (12μm ± 0.5μm), 
                    et la conductivité. Un Cpk > 1.33 est requis pour la conformité.""",
                    "keywords": ["biosenseur", "résistance", "coating", "Cpk"]
                },
                "validation": {
                    "title": "Validation Process",
                    "content": """La validation selon ISO 13485 nécessite des intervalles 
                    de tolérance pour garantir que 99% de la production sera conforme. 
                    Utiliser IT bilatéral avec niveau 95/99.""",
                    "keywords": ["validation", "ISO 13485", "conformité"]
                }
            }
        }
    
    def get_all_sections(self) -> List[Dict]:
        """Retourne toutes les sections comme documents"""
        sections = []
        
        # ISO 2602
        for section_id, section in self.iso_2602_content["sections"].items():
            sections.append({
                "id": f"iso2602_{section_id}",
                "content": section["content"],
                "metadata": {
                    "source": "ISO 2602:1980",
                    "section": section["title"],
                    "keywords": section["keywords"]
                }
            })
        
        # ISO 16269-6
        for section_id, section in self.iso_16269_content["sections"].items():
            sections.append({
                "id": f"iso16269_{section_id}",
                "content": section["content"],
                "metadata": {
                    "source": "ISO 16269-6:2014",
                    "section": section["title"],
                    "keywords": section["keywords"]
                }
            })
        
        # Connaissances médicales
        for section_id, section in self.medical_device_content["sections"].items():
            sections.append({
                "id": f"medical_{section_id}",
                "content": section["content"],
                "metadata": {
                    "source": "Dispositifs Médicaux",
                    "section": section["title"],
                    "keywords": section["keywords"]
                }
            })
        
        return sections

class SimpleRAGEngine:
    """Moteur RAG simplifié"""
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.knowledge_base = ISOKnowledgeBase()
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Initialise la base de connaissances"""
        # Charger toutes les sections
        sections = self.knowledge_base.get_all_sections()
        
        # Créer des documents
        documents = []
        for section in sections:
            doc = Document(
                id=section["id"],
                content=section["content"],
                metadata=section["metadata"]
            )
            documents.append(doc)
        
        # Ajouter au vector store si pas déjà fait
        if not self.vector_store.documents:
            self.vector_store.add_documents(documents)
    
    def query(self, question: str, n_sources: int = 3) -> RAGResponse:
        """Traite une question et génère une réponse"""
        # Rechercher les documents pertinents
        relevant_docs = self.vector_store.search(question, n_results=n_sources)
        
        if not relevant_docs:
            return RAGResponse(
                answer="Je n'ai pas trouvé d'information pertinente.",
                sources=[],
                confidence=0.0,
                suggestions=["Reformulez votre question", "Consultez la documentation"]
            )
        
        # Générer la réponse
        answer = self._generate_answer(question, relevant_docs)
        
        # Calculer la confiance
        confidence = relevant_docs[0]['similarity'] if relevant_docs else 0.0
        
        # Générer des suggestions
        suggestions = self._generate_suggestions(question, relevant_docs)
        
        return RAGResponse(
            answer=answer,
            sources=self._format_sources(relevant_docs),
            confidence=confidence,
            suggestions=suggestions
        )
    
    def _generate_answer(self, question: str, documents: List[Dict]) -> str:
        """Génère une réponse basée sur les documents"""
        # Identifier le type de question
        question_lower = question.lower()
        
        # Construire le contexte
        context = "\n\n".join([doc['content'] for doc in documents[:2]])
        
        # Générer selon le type
        if "différence" in question_lower or "vs" in question_lower:
            answer = self._generate_comparison_answer(context)
        elif "calculer" in question_lower or "formule" in question_lower:
            answer = self._generate_calculation_answer(context)
        elif "qu'est-ce" in question_lower or "définition" in question_lower:
            answer = self._generate_definition_answer(context)
        else:
            answer = self._generate_general_answer(context, question)
        
        # Ajouter les sources
        sources = [f"{doc['metadata']['source']} - {doc['metadata']['section']}" 
                  for doc in documents[:2]]
        answer += f"\n\n📚 Sources: {', '.join(sources)}"
        
        return answer
    
    def _generate_comparison_answer(self, context: str) -> str:
        """Génère une réponse de comparaison"""
        answer = "**Comparaison IC vs IT:**\n\n"
        
        if "IC" in context and "IT" in context:
            answer += "• **Intervalle de Confiance (IC)**: Estime la position de la moyenne\n"
            answer += "• **Intervalle de Tolérance (IT)**: Contient une proportion de la population\n\n"
            answer += "L'IT est toujours plus large que l'IC (environ 6× pour n=30)."
        else:
            answer += context[:300] + "..."
        
        return answer
    
    def _generate_calculation_answer(self, context: str) -> str:
        """Génère une réponse de calcul"""
        answer = "**Méthode de calcul:**\n\n"
        
        # Chercher les formules
        if "IC = " in context or "IT = " in context:
            formulas = re.findall(r"(I[CT]\s*=\s*[^.]+)", context)
            if formulas:
                for formula in formulas[:2]:
                    answer += f"• {formula}\n"
        else:
            answer += context[:300] + "..."
        
        return answer
    
    def _generate_definition_answer(self, context: str) -> str:
        """Génère une réponse de définition"""
        # Extraire la première phrase pertinente
        sentences = context.split('.')
        for sentence in sentences:
            if "est" in sentence or "définition" in sentence.lower():
                return sentence.strip() + "."
        
        return context[:200] + "..."
    
    def _generate_general_answer(self, context: str, question: str) -> str:
        """Génère une réponse générale"""
        # Retourner les parties les plus pertinentes du contexte
        return context[:400] + "..."
    
    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """Formate les sources pour l'affichage"""
        return [
            {
                "source": doc['metadata']['source'],
                "section": doc['metadata']['section'],
                "confidence": doc['similarity']
            }
            for doc in documents
        ]
    
    def _generate_suggestions(self, question: str, documents: List[Dict]) -> List[str]:
        """Génère des questions suggérées"""
        suggestions = []
        
        # Basé sur les mots-clés trouvés
        keywords = set()
        for doc in documents:
            keywords.update(doc['metadata'].get('keywords', []))
        
        # Suggestions génériques basées sur les mots-clés
        if "IC" in str(keywords) or "intervalle confiance" in str(keywords):
            suggestions.append("Comment calculer un intervalle de confiance ?")
        if "IT" in str(keywords) or "intervalle tolérance" in str(keywords):
            suggestions.append("Quand utiliser un intervalle de tolérance ?")
        if "facteur k" in str(keywords):
            suggestions.append("Comment déterminer le facteur k ?")
        if "biosenseur" in str(keywords):
            suggestions.append("Quelles sont les spécifications pour les biosenseurs ?")
        
        return suggestions[:3]  # Limiter à 3 suggestions

class DecisionTreeAssistant:
    """Assistant basé sur des arbres de décision"""
    
    def __init__(self):
        self.decision_trees = self._create_decision_trees()
    
    def _create_decision_trees(self) -> Dict:
        """Crée les arbres de décision"""
        return {
            "analysis_type": {
                "question": "Quel est votre objectif principal ?",
                "options": {
                    "Estimer la moyenne": {
                        "recommendation": "Utilisez un Intervalle de Confiance (ISO 2602)",
                        "next": "confidence_level"
                    },
                    "Garantir la conformité de production": {
                        "recommendation": "Utilisez un Intervalle de Tolérance (ISO 16269-6)",
                        "next": "tolerance_params"
                    },
                    "Évaluer la capabilité": {
                        "recommendation": "Calculez Cp et Cpk avec IT",
                        "next": "capability_specs"
                    },
                    "Comparer deux lots": {
                        "recommendation": "Test d'hypothèse avec IC",
                        "next": "comparison_type"
                    }
                }
            },
            "confidence_level": {
                "question": "Quel niveau de confiance souhaitez-vous ?",
                "options": {
                    "90%": {"info": "Risque de 10% - Usage exploratoire"},
                    "95%": {"info": "Standard industriel - Recommandé"},
                    "99%": {"info": "Haute confiance - Analyses critiques"}
                }
            },
            "tolerance_params": {
                "question": "Quelle proportion de la population à couvrir ?",
                "options": {
                    "90%": {"info": "Tolérance large - Process standard"},
                    "95%": {"info": "Tolérance normale - Recommandé"},
                    "99%": {"info": "Tolérance stricte - Dispositifs critiques"}
                }
            }
        }
    
    def get_recommendation(self, context: str) -> Dict:
        """Retourne une recommandation basée sur le contexte"""
        context_lower = context.lower()
        
        if "moyenne" in context_lower:
            return {
                "method": "Intervalle de Confiance",
                "reason": "Pour estimer la position de la moyenne",
                "norm": "ISO 2602:1980"
            }
        elif "production" in context_lower or "conformité" in context_lower:
            return {
                "method": "Intervalle de Tolérance",
                "reason": "Pour garantir la conformité de la production",
                "norm": "ISO 16269-6:2014"
            }
        elif "capabilité" in context_lower or "cpk" in context_lower:
            return {
                "method": "Analyse de Capabilité",
                "reason": "Pour évaluer la performance du processus",
                "norm": "ISO 16269-6 avec calcul Cpk"
            }
        else:
            return {
                "method": "Analyse exploratoire",
                "reason": "Commencez par visualiser vos données",
                "norm": "Statistiques descriptives"
            }
    
    def get_next_step(self, current_step: str, choice: str) -> Dict:
        """Retourne la prochaine étape dans l'arbre de décision"""
        if current_step in self.decision_trees:
            tree = self.decision_trees[current_step]
            if choice in tree.get("options", {}):
                option = tree["options"][choice]
                return {
                    "recommendation": option.get("recommendation", ""),
                    "info": option.get("info", ""),
                    "next_step": option.get("next", None)
                }
        return {"recommendation": "Fin de l'analyse", "next_step": None}