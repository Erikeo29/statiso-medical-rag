#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatISO-Medical MVP avec RAG - Application Streamlit complète
Interface web pour analyses statistiques ISO avec système RAG intégré
Phase 1: Analyses statistiques
Phase 2: Formation interactive
Phase 3: RAG et Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Ajouter le chemin vers les modules
sys.path.append(str(Path(__file__).parent))

# Import des modules MVP
from streamlit_modules.iso_2602 import ISO2602Calculator
from streamlit_modules.iso_16269_6 import ISO16269_6Calculator
from streamlit_modules.data_handler import DataHandler
from streamlit_modules.report_generator import ReportGenerator
from streamlit_modules.training_module import TrainingSystem, SimulationEngine
from streamlit_modules.rag_module import SimpleRAGEngine, DecisionTreeAssistant

# Configuration de la page Streamlit
st.set_page_config(
    page_title="StatISO-Medical RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stAlert {
        background-color: #f0f8ff;
        border: 1px solid #1f77b4;
    }
    .result-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .rag-response {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .source-card {
        background-color: #fff;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des calculateurs et systèmes
@st.cache_resource
def initialize_systems():
    """Initialise tous les systèmes (mise en cache pour performance)"""
    return {
        'iso2602': ISO2602Calculator(),
        'iso16269': ISO16269_6Calculator(),
        'data_handler': DataHandler(),
        'report_gen': ReportGenerator(),
        'training': TrainingSystem(),
        'simulation': SimulationEngine(),
        'rag': SimpleRAGEngine(),
        'decision_tree': DecisionTreeAssistant()
    }

systems = initialize_systems()

# État de session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'user_id' not in st.session_state:
    st.session_state.user_id = "user_" + str(hash(str(datetime.now())))[:8]
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'decision_path' not in st.session_state:
    st.session_state.decision_path = []

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🔬 StatISO-Medical RAG</h1>
    <p>Système Intelligent d'Analyse Statistique avec IA - ISO 2602 & ISO 16269-6</p>
    <p style="font-size: 0.9em;">✨ Nouveau: Assistant IA avec RAG pour répondre à vos questions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour configuration
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=StatISO+RAG", use_column_width=True)
    
    st.markdown("### ⚙️ Configuration")
    
    # Mode d'utilisation
    usage_mode = st.selectbox(
        "Mode d'utilisation",
        ["🤖 Assistant IA", "📊 Analyse Guidée", "🎓 Formation", "🔬 Mode Expert"]
    )
    
    if usage_mode == "📊 Analyse Guidée":
        # Sélection du type d'analyse
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Intervalle de Confiance (ISO 2602)", 
             "Intervalle de Tolérance (ISO 16269-6)",
             "Analyse Comparative",
             "Aide à la décision"]
        )
        
        # Paramètres statistiques
        st.markdown("### 📊 Paramètres")
        confidence_level = st.select_slider(
            "Niveau de confiance",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%"
        )
        
        if "Tolérance" in analysis_type:
            proportion = st.select_slider(
                "Proportion de population",
                options=[0.90, 0.95, 0.99],
                value=0.95,
                format_func=lambda x: f"{int(x*100)}%"
            )
            bilateral = st.checkbox("Intervalle bilatéral", value=True)
            method = st.radio("Méthode", ["Paramétrique (Normal)", "Non-paramétrique"])
    
    # Section données exemple
    st.markdown("### 📁 Données exemple")
    if st.button("🔬 Charger données biosenseur"):
        np.random.seed(42)
        sample_data = np.random.normal(100, 1.5, 30)
        st.session_state.data = sample_data
        st.success("✅ Données biosenseur chargées!")

# Zone principale avec tabs
if usage_mode == "🤖 Assistant IA":
    # Mode Assistant IA avec RAG
    st.header("🤖 Assistant IA Intelligent")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Je suis votre assistant expert en analyses statistiques ISO. 
        Posez-moi vos questions sur les intervalles de confiance, de tolérance, 
        la capabilité process, ou les applications aux dispositifs médicaux.
        """)
        
        # Zone de chat
        st.markdown("### 💬 Discussion")
        
        # Input utilisateur
        user_question = st.text_input(
            "Votre question:",
            placeholder="Ex: Quelle est la différence entre IC et IT ?",
            key="user_input"
        )
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_button = st.button("📤 Envoyer", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Effacer historique", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_button and user_question:
            # Ajouter à l'historique
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Obtenir la réponse RAG
            with st.spinner("🤔 Je réfléchis..."):
                rag_response = systems['rag'].query(user_question)
            
            # Ajouter la réponse à l'historique
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": rag_response.answer,
                "sources": rag_response.sources,
                "confidence": rag_response.confidence,
                "suggestions": rag_response.suggestions
            })
        
        # Afficher l'historique du chat
        if st.session_state.chat_history:
            st.markdown("### 📜 Historique de la conversation")
            
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**👤 Vous:** {message['content']}")
                else:
                    # Réponse de l'assistant
                    st.markdown(f"**🤖 Assistant:**")
                    st.markdown(f"<div class='rag-response'>{message['content']}</div>", 
                               unsafe_allow_html=True)
                    
                    # Afficher les sources si disponibles
                    if message.get("sources"):
                        with st.expander("📚 Sources consultées"):
                            for source in message["sources"]:
                                st.markdown(f"""
                                <div class='source-card'>
                                <b>{source['source']}</b> - {source['section']}<br>
                                Confiance: {source['confidence']:.2%}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Afficher le niveau de confiance
                    confidence = message.get("confidence", 0)
                    if confidence > 0.7:
                        st.success(f"✅ Confiance: {confidence:.1%}")
                    elif confidence > 0.4:
                        st.warning(f"⚠️ Confiance: {confidence:.1%}")
                    else:
                        st.error(f"❌ Confiance faible: {confidence:.1%}")
                    
                    # Suggestions de questions
                    if message.get("suggestions"):
                        st.markdown("**💡 Questions suggérées:**")
                        for suggestion in message["suggestions"]:
                            if st.button(f"→ {suggestion}", key=f"suggest_{i}_{suggestion[:20]}"):
                                st.session_state.user_input = suggestion
                                st.rerun()
    
    with col2:
        st.markdown("### 🎯 Questions fréquentes")
        
        faq_questions = [
            "Qu'est-ce qu'un intervalle de confiance ?",
            "Différence entre IC et IT ?",
            "Comment calculer un Cpk ?",
            "Quand utiliser ISO 2602 vs ISO 16269-6 ?",
            "Quelle taille d'échantillon minimum ?",
            "Comment interpréter un IT 95/95 ?",
            "Validation process biosenseur ?"
        ]
        
        for q in faq_questions:
            if st.button(q, key=f"faq_{q[:20]}", use_container_width=True):
                st.session_state.user_input = q
                st.rerun()
        
        st.markdown("### 📊 Aide contextuelle")
        st.info("""
        **Conseils:**
        - Soyez précis dans vos questions
        - Mentionnez le contexte (biosenseur, électrode, etc.)
        - Indiquez les paramètres si connus (n, confidence, etc.)
        """)

elif usage_mode == "📊 Analyse Guidée":
    # Mode analyse guidée avec arbre de décision
    st.header("📊 Analyse Guidée par IA")
    
    # Arbre de décision interactif
    st.markdown("### 🌳 Assistant de Décision")
    
    # Étape 1: Objectif
    objective = st.selectbox(
        "Quel est votre objectif principal ?",
        [
            "Estimer la moyenne d'un processus",
            "Garantir la conformité de production",
            "Évaluer la capabilité (Cp/Cpk)",
            "Comparer deux lots",
            "Déterminer la taille d'échantillon",
            "Je ne sais pas"
        ]
    )
    
    # Obtenir la recommandation
    recommendation = systems['decision_tree'].get_recommendation(objective)
    
    # Afficher la recommandation
    st.markdown("### 🎯 Recommandation")
    st.success(f"""
    **Méthode recommandée:** {recommendation['method']}
    
    **Raison:** {recommendation['reason']}
    
    **Norme applicable:** {recommendation['norm']}
    """)
    
    # Paramètres supplémentaires selon le choix
    if "Intervalle de Confiance" in recommendation['method']:
        st.markdown("### ⚙️ Paramètres pour IC")
        
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("Taille d'échantillon", min_value=2, value=30)
            conf_level = st.select_slider("Niveau de confiance", [0.90, 0.95, 0.99], 0.95)
        with col2:
            sigma_known = st.checkbox("Écart-type population connu ?")
            if sigma_known:
                sigma = st.number_input("Valeur de σ", value=1.0, min_value=0.01)
        
        if st.button("📊 Calculer l'exemple", type="primary"):
            # Générer des données exemple
            np.random.seed(42)
            data = np.random.normal(100, 2, n_samples)
            
            # Calculer IC
            result = systems['iso2602'].calculate_confidence_interval(
                data, conf_level, sigma if sigma_known else None
            )
            
            # Afficher résultats
            st.markdown("### 📈 Résultats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Limite inférieure", f"{result['lower_bound']:.3f}")
            with col2:
                st.metric("Moyenne", f"{result['mean']:.3f}")
            with col3:
                st.metric("Limite supérieure", f"{result['upper_bound']:.3f}")
            
            st.info(f"""
            💡 **Interprétation:** Nous sommes confiants à {int(conf_level*100)}% 
            que la vraie moyenne se situe entre {result['lower_bound']:.3f} 
            et {result['upper_bound']:.3f}.
            """)
    
    elif "Intervalle de Tolérance" in recommendation['method']:
        st.markdown("### ⚙️ Paramètres pour IT")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_samples = st.number_input("Taille d'échantillon", min_value=3, value=30)
        with col2:
            conf_level = st.select_slider("Niveau de confiance", [0.90, 0.95, 0.99], 0.95)
        with col3:
            proportion = st.select_slider("Proportion à couvrir", [0.90, 0.95, 0.99], 0.95)
        
        if st.button("📊 Calculer l'exemple", type="primary"):
            # Générer des données exemple
            np.random.seed(42)
            data = np.random.normal(100, 2, n_samples)
            
            # Calculer IT
            result = systems['iso16269'].calculate_tolerance_interval(
                data, conf_level, proportion
            )
            
            # Afficher résultats
            st.markdown("### 📈 Résultats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Limite inférieure", f"{result['lower_bound']:.3f}")
            with col2:
                st.metric("Centre", f"{result['mean']:.3f}")
            with col3:
                st.metric("Limite supérieure", f"{result['upper_bound']:.3f}")
            
            st.metric("Facteur k", f"{result['k_factor']:.3f}")
            
            st.info(f"""
            💡 **Interprétation:** Nous sommes confiants à {int(conf_level*100)}% 
            que {int(proportion*100)}% de la population se situe entre 
            {result['lower_bound']:.3f} et {result['upper_bound']:.3f}.
            """)

elif usage_mode == "🎓 Formation":
    # Mode Formation avec quiz et simulations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Cours Interactif", "❓ Quiz", "🔬 Simulations", "📊 Progression"
    ])
    
    with tab1:
        st.header("📚 Cours Interactif avec IA")
        
        # Sélection du module
        module = st.selectbox(
            "Choisir un module",
            [
                "Introduction aux statistiques ISO",
                "Intervalles de Confiance (ISO 2602)",
                "Intervalles de Tolérance (ISO 16269-6)",
                "Différences IC vs IT",
                "Capabilité Process (Cp/Cpk)",
                "Applications Biosenseurs"
            ]
        )
        
        # Contenu adaptatif basé sur le module
        if "Introduction" in module:
            st.markdown("""
            ## Introduction aux Statistiques ISO
            
            ### Objectifs d'apprentissage
            - Comprendre les concepts de base
            - Distinguer population et échantillon
            - Maîtriser moyenne et écart-type
            
            ### Concepts clés
            
            **Population vs Échantillon:**
            - **Population** : Ensemble complet des valeurs possibles
            - **Échantillon** : Sous-ensemble mesuré de la population
            
            **Paramètres importants:**
            - **Moyenne (μ ou x̄)** : Tendance centrale
            - **Écart-type (σ ou s)** : Dispersion des valeurs
            - **Taille (N ou n)** : Nombre d'observations
            """)
            
            # Question interactive
            if st.button("💬 Poser une question sur ce module"):
                question = f"Expliquez les concepts de base du module {module}"
                response = systems['rag'].query(question)
                st.markdown("### 🤖 Réponse de l'Assistant")
                st.info(response.answer)
        
        elif "ISO 2602" in module:
            st.markdown("""
            ## ISO 2602 - Intervalles de Confiance
            
            ### Principe
            L'intervalle de confiance estime la position de la moyenne de la population.
            
            ### Formule principale
            **IC = x̄ ± t(n-1, α/2) × s/√n**
            
            Où:
            - x̄ : moyenne échantillon
            - t : valeur critique de Student
            - s : écart-type échantillon
            - n : taille échantillon
            """)
            
            # Simulation interactive
            if st.checkbox("Voir simulation"):
                n = st.slider("Taille d'échantillon", 10, 100, 30)
                
                np.random.seed(42)
                data = np.random.normal(100, 2, n)
                result = systems['iso2602'].calculate_confidence_interval(data, 0.95)
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=data, name="Données"))
                fig.add_vline(x=result['mean'], line_dash="solid", 
                             annotation_text="Moyenne")
                fig.add_vline(x=result['lower_bound'], line_dash="dash",
                             annotation_text="IC inf")
                fig.add_vline(x=result['upper_bound'], line_dash="dash",
                             annotation_text="IC sup")
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("❓ Quiz Adaptatif")
        
        # Configuration du quiz
        col1, col2 = st.columns(2)
        with col1:
            quiz_topic = st.selectbox(
                "Sujet",
                ["Concepts de base", "ISO 2602", "ISO 16269-6", "Biosenseurs"]
            )
        with col2:
            difficulty = st.select_slider("Difficulté", [1, 2, 3], 2,
                                        format_func=lambda x: ["Facile", "Moyen", "Difficile"][x-1])
        
        if st.button("🚀 Démarrer le quiz", type="primary"):
            # Générer des questions
            questions = systems['training'].get_adaptive_quiz(
                st.session_state.user_id, quiz_topic, difficulty, 5
            )
            
            if questions:
                st.session_state.current_quiz = questions
                st.session_state.quiz_index = 0
        
        # Afficher les questions si quiz actif
        if 'current_quiz' in st.session_state and st.session_state.current_quiz:
            question = st.session_state.current_quiz[st.session_state.quiz_index]
            
            st.markdown(f"### Question {st.session_state.quiz_index + 1}")
            st.markdown(f"**{question['question']}**")
            
            if question['type'] == 'multiple_choice':
                answer = st.radio("Réponse:", question['options'])
            elif question['type'] == 'true_false':
                answer = st.radio("Réponse:", ["Vrai", "Faux"])
            
            if st.button("Valider"):
                # Évaluation
                is_correct = (answer == question.get('correct_answer'))
                if is_correct:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect")
                st.info(question.get('explanation', ''))
    
    with tab3:
        st.header("🔬 Simulations Interactives")
        
        simulation_type = st.selectbox(
            "Type de simulation",
            ["IC vs IT", "Impact de n", "Capabilité", "Couverture IC"]
        )
        
        if simulation_type == "IC vs IT":
            st.subheader("Comparaison IC vs IT")
            
            n = st.slider("Taille échantillon", 10, 200, 30)
            
            if st.button("Simuler"):
                fig = systems['simulation'].simulate_ic_vs_it_comparison(n, 0.95, 0.95)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                💡 L'IT est toujours plus large que l'IC car il vise à contenir 
                les valeurs individuelles, pas seulement estimer la moyenne.
                """)
    
    with tab4:
        st.header("📊 Votre Progression")
        
        user_stats = systems['training'].get_user_statistics(st.session_state.user_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Questions répondues", user_stats['total_questions'])
        with col2:
            st.metric("Taux de réussite", f"{user_stats['success_rate']:.1f}%")
        with col3:
            level = "Expert" if user_stats['success_rate'] > 80 else "En progression"
            st.metric("Niveau", level)

else:  # Mode Expert
    # Mode Expert avec toutes les fonctionnalités
    tabs = st.tabs([
        "📤 Import", "📈 Analyse", "📊 Visualisation", 
        "🤖 IA/RAG", "📚 Documentation", "📥 Export"
    ])
    
    with tabs[0]:  # Import
        st.header("Import des données")
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV ou Excel",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file:
            data = systems['data_handler'].load_data(uploaded_file)
            st.session_state.data = data
            st.success(f"✅ {len(data)} valeurs importées")
            
            # Aperçu
            st.dataframe(pd.DataFrame(data[:10], columns=["Valeur"]))
    
    with tabs[1]:  # Analyse
        st.header("Analyses Statistiques")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Statistiques descriptives
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Moyenne", f"{np.mean(data):.3f}")
            with col2:
                st.metric("Écart-type", f"{np.std(data, ddof=1):.3f}")
            with col3:
                st.metric("Min", f"{np.min(data):.3f}")
            with col4:
                st.metric("Max", f"{np.max(data):.3f}")
            
            # Analyses disponibles
            analysis = st.selectbox(
                "Type d'analyse",
                ["IC (ISO 2602)", "IT (ISO 16269-6)", "Capabilité", "Comparaison IC vs IT"]
            )
            
            if st.button("Analyser", type="primary"):
                if "IC" in analysis:
                    result = systems['iso2602'].calculate_confidence_interval(data)
                    st.success(f"IC 95%: [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]")
                elif "IT" in analysis:
                    result = systems['iso16269'].calculate_tolerance_interval(data)
                    st.success(f"IT 95/95: [{result['lower_bound']:.3f}, {result['upper_bound']:.3f}]")
        else:
            st.warning("⚠️ Veuillez d'abord importer des données")
    
    with tabs[3]:  # IA/RAG
        st.header("🤖 Assistant IA Expert")
        
        question = st.text_area(
            "Question technique",
            placeholder="Posez une question complexe sur vos analyses..."
        )
        
        if st.button("Obtenir une réponse"):
            response = systems['rag'].query(question)
            
            st.markdown("### Réponse")
            st.markdown(f"<div class='rag-response'>{response.answer}</div>",
                       unsafe_allow_html=True)
            
            if response.sources:
                st.markdown("### Sources")
                for source in response.sources:
                    st.markdown(f"- {source['source']} ({source['confidence']:.1%})")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>StatISO-Medical RAG v3.0 - Linxens France</p>
    <p>Système intelligent d'analyse statistique avec IA pour dispositifs médicaux</p>
    <p>ISO 2602:1980 | ISO 16269-6:2014 | ISO 13485</p>
</div>
""", unsafe_allow_html=True)