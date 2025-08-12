"""
Module de génération de rapports - PDF et visualisations
"""
import io
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import conditionnel pour plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import conditionnel pour reportlab (PDF)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class ReportGenerator:
    """Générateur de rapports et visualisations"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        
    def create_distribution_plot(
        self, 
        data: np.ndarray, 
        results: Dict = None,
        interactive: bool = True
    ) -> Union['go.Figure', 'plt.Figure']:
        """
        Crée un graphique de distribution des données
        
        Args:
            data: Données à visualiser
            results: Résultats d'analyse pour superposer les limites
            interactive: Si True, utilise plotly, sinon matplotlib
            
        Returns:
            Figure plotly ou matplotlib
        """
        if interactive and PLOTLY_AVAILABLE:
            # Version Plotly interactive
            fig = go.Figure()
            
            # Histogramme
            fig.add_trace(go.Histogram(
                x=data,
                name='Données',
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7,
                histnorm='probability density'
            ))
            
            # Courbe de distribution normale théorique
            mean = results.get('mean', np.mean(data)) if results else np.mean(data)
            std = results.get('std', np.std(data, ddof=1)) if results else np.std(data, ddof=1)
            
            x_range = np.linspace(data.min() - 2*std, data.max() + 2*std, 100)
            from scipy import stats
            y_normal = stats.norm.pdf(x_range, mean, std)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Distribution normale',
                line=dict(color='red', width=2)
            ))
            
            # Ajouter les limites si disponibles
            if results:
                if 'lower_bound' in results and results['lower_bound'] != -np.inf:
                    fig.add_vline(x=results['lower_bound'], 
                                line_dash="dash", 
                                line_color="green",
                                annotation_text="Limite inf.")
                
                if 'upper_bound' in results and results['upper_bound'] != np.inf:
                    fig.add_vline(x=results['upper_bound'], 
                                line_dash="dash", 
                                line_color="green",
                                annotation_text="Limite sup.")
                
                fig.add_vline(x=mean, 
                            line_dash="solid", 
                            line_color="blue",
                            annotation_text="Moyenne")
            
            # Mise en forme
            fig.update_layout(
                title=f"Distribution des données - {results.get('interval_type', 'Analyse') if results else 'Analyse'}",
                xaxis_title="Valeur",
                yaxis_title="Densité",
                showlegend=True,
                template="plotly_white"
            )
            
            return fig
        
        else:
            # Version matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histogramme
            n, bins, patches = ax.hist(data, bins=20, density=True, 
                                      alpha=0.7, color='lightblue', 
                                      edgecolor='black', label='Données')
            
            # Courbe normale
            mean = results.get('mean', np.mean(data)) if results else np.mean(data)
            std = results.get('std', np.std(data, ddof=1)) if results else np.std(data, ddof=1)
            x_range = np.linspace(data.min() - 2*std, data.max() + 2*std, 100)
            
            from scipy import stats
            y_normal = stats.norm.pdf(x_range, mean, std)
            ax.plot(x_range, y_normal, 'r-', linewidth=2, label='Distribution normale')
            
            # Lignes verticales
            ax.axvline(mean, color='blue', linestyle='-', linewidth=2, label='Moyenne')
            
            if results:
                if 'lower_bound' in results and results['lower_bound'] != -np.inf:
                    ax.axvline(results['lower_bound'], color='green', 
                              linestyle='--', linewidth=2, label='Limite inf.')
                
                if 'upper_bound' in results and results['upper_bound'] != np.inf:
                    ax.axvline(results['upper_bound'], color='green', 
                              linestyle='--', linewidth=2, label='Limite sup.')
            
            ax.set_xlabel('Valeur')
            ax.set_ylabel('Densité')
            ax.set_title(f"Distribution des données")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def create_control_chart(self, data: np.ndarray) -> Union['go.Figure', 'plt.Figure']:
        """
        Crée une carte de contrôle
        
        Args:
            data: Données temporelles
            
        Returns:
            Figure plotly ou matplotlib
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Limites de contrôle
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        uwl = mean + 2 * std
        lwl = mean - 2 * std
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Données
            fig.add_trace(go.Scatter(
                x=list(range(1, len(data) + 1)),
                y=data,
                mode='lines+markers',
                name='Mesures',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Ligne moyenne
            fig.add_hline(y=mean, line_dash="solid", line_color="green", 
                         annotation_text=f"Moyenne: {mean:.3f}")
            
            # Limites de contrôle
            fig.add_hline(y=ucl, line_dash="dash", line_color="red", 
                         annotation_text=f"UCL: {ucl:.3f}")
            fig.add_hline(y=lcl, line_dash="dash", line_color="red", 
                         annotation_text=f"LCL: {lcl:.3f}")
            
            # Limites d'avertissement
            fig.add_hline(y=uwl, line_dash="dot", line_color="orange", 
                         annotation_text=f"UWL: {uwl:.3f}")
            fig.add_hline(y=lwl, line_dash="dot", line_color="orange", 
                         annotation_text=f"LWL: {lwl:.3f}")
            
            fig.update_layout(
                title="Carte de Contrôle de Shewhart",
                xaxis_title="Numéro d'échantillon",
                yaxis_title="Valeur mesurée",
                template="plotly_white",
                showlegend=True
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Données
            x = range(1, len(data) + 1)
            ax.plot(x, data, 'b-o', label='Mesures', markersize=6)
            
            # Lignes de contrôle
            ax.axhline(y=mean, color='green', linestyle='-', label=f'Moyenne: {mean:.3f}')
            ax.axhline(y=ucl, color='red', linestyle='--', label=f'UCL: {ucl:.3f}')
            ax.axhline(y=lcl, color='red', linestyle='--', label=f'LCL: {lcl:.3f}')
            ax.axhline(y=uwl, color='orange', linestyle=':', label=f'UWL: {uwl:.3f}')
            ax.axhline(y=lwl, color='orange', linestyle=':', label=f'LWL: {lwl:.3f}')
            
            ax.set_xlabel('Numéro d\'échantillon')
            ax.set_ylabel('Valeur mesurée')
            ax.set_title('Carte de Contrôle de Shewhart')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def generate_pdf_report(
        self, 
        results: Dict, 
        data: np.ndarray,
        company_name: str = "Linxens France"
    ) -> io.BytesIO:
        """
        Génère un rapport PDF complet
        
        Args:
            results: Résultats des calculs
            data: Données analysées
            company_name: Nom de l'entreprise
            
        Returns:
            BytesIO object contenant le PDF
        """
        if not REPORTLAB_AVAILABLE:
            # Fallback si reportlab n'est pas disponible
            buffer = io.BytesIO()
            report_text = self._generate_text_report(results, data, company_name)
            buffer.write(report_text.encode('utf-8'))
            buffer.seek(0)
            return buffer
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Style personnalisé pour titre
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # En-tête
        story.append(Paragraph("RAPPORT D'ANALYSE STATISTIQUE", title_style))
        story.append(Paragraph(f"{company_name}", styles['Heading2']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Résumé des données
        story.append(Paragraph("1. RÉSUMÉ DES DONNÉES", styles['Heading2']))
        
        data_summary = [
            ['Paramètre', 'Valeur'],
            ['Nombre d\'échantillons', str(len(data))],
            ['Moyenne', f"{np.mean(data):.4f}"],
            ['Écart-type', f"{np.std(data, ddof=1):.4f}"],
            ['Minimum', f"{np.min(data):.4f}"],
            ['Maximum', f"{np.max(data):.4f}"],
            ['Médiane', f"{np.median(data):.4f}"]
        ]
        
        t = Table(data_summary, colWidths=[3*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Résultats de l'analyse
        story.append(Paragraph("2. RÉSULTATS DE L'ANALYSE", styles['Heading2']))
        
        # Adapter selon le type de résultat
        if 'confidence_level' in results:
            analysis_type = "Intervalle de Confiance (ISO 2602)" if 'margin_error' in results else "Intervalle de Tolérance (ISO 16269-6)"
            story.append(Paragraph(f"Type d'analyse: {analysis_type}", styles['Heading3']))
            
            results_data = [
                ['Paramètre', 'Valeur'],
                ['Niveau de confiance', f"{results['confidence_level']*100:.0f}%"]
            ]
            
            if 'proportion' in results:
                results_data.append(['Proportion couverte', f"{results['proportion']*100:.0f}%"])
            
            if 'lower_bound' in results and results['lower_bound'] != -np.inf:
                results_data.append(['Limite inférieure', f"{results['lower_bound']:.4f}"])
            
            if 'upper_bound' in results and results['upper_bound'] != np.inf:
                results_data.append(['Limite supérieure', f"{results['upper_bound']:.4f}"])
            
            if 'k_factor' in results:
                results_data.append(['Facteur k', f"{results['k_factor']:.4f}"])
            
            if 'margin_error' in results:
                results_data.append(['Marge d\'erreur', f"{results['margin_error']:.4f}"])
            
            t2 = Table(results_data, colWidths=[3*inch, 2*inch])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(t2)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Interprétation
        story.append(Paragraph("3. INTERPRÉTATION", styles['Heading2']))
        
        if 'margin_error' in results:
            interpretation = f"""
            L'intervalle de confiance à {results['confidence_level']*100:.0f}% pour la moyenne est 
            [{results['lower_bound']:.3f}, {results['upper_bound']:.3f}]. 
            Cela signifie que nous sommes confiants à {results['confidence_level']*100:.0f}% 
            que la vraie moyenne de la population se situe dans cet intervalle.
            """
        elif 'k_factor' in results:
            interpretation = f"""
            L'intervalle de tolérance calculé est [{results.get('lower_bound', 'N/A'):.3f}, {results.get('upper_bound', 'N/A'):.3f}].
            Nous sommes confiants à {results['confidence_level']*100:.0f}% que {results['proportion']*100:.0f}% 
            de la population se situe dans cet intervalle.
            """
        else:
            interpretation = "Analyse statistique complète selon les normes ISO."
        
        story.append(Paragraph(interpretation, styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Conformité et recommandations
        story.append(Paragraph("4. CONFORMITÉ ET RECOMMANDATIONS", styles['Heading2']))
        
        if 'conformity_analysis' in results:
            conf = results['conformity_analysis']
            if 'process_capable' in conf:
                if conf['process_capable']:
                    story.append(Paragraph("✅ Le processus est CAPABLE", styles['Heading3']))
                    story.append(Paragraph(f"Cpk = {conf['Cpk']:.3f} (> 1.33)", styles['Normal']))
                else:
                    story.append(Paragraph("⚠️ Le processus NÉCESSITE UNE AMÉLIORATION", styles['Heading3']))
                    story.append(Paragraph(f"Cpk = {conf['Cpk']:.3f} (< 1.33)", styles['Normal']))
        
        recommendations = [
            "Continuer la surveillance du processus",
            "Vérifier régulièrement la normalité des données",
            "Maintenir un échantillonnage représentatif",
            "Documenter tout changement de processus"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Paragraph("--- Fin du rapport ---", styles['Normal']))
        story.append(Paragraph("Généré avec StatISO-Medical v1.0", styles['Normal']))
        
        # Construire le PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    def _generate_text_report(
        self, 
        results: Dict, 
        data: np.ndarray,
        company_name: str
    ) -> str:
        """
        Génère un rapport texte simple (fallback)
        
        Returns:
            Rapport au format texte
        """
        report = []
        report.append("="*60)
        report.append("RAPPORT D'ANALYSE STATISTIQUE")
        report.append(f"{company_name}")
        report.append(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        report.append("="*60)
        report.append("")
        
        report.append("1. RÉSUMÉ DES DONNÉES")
        report.append("-"*40)
        report.append(f"Nombre d'échantillons: {len(data)}")
        report.append(f"Moyenne: {np.mean(data):.4f}")
        report.append(f"Écart-type: {np.std(data, ddof=1):.4f}")
        report.append(f"Minimum: {np.min(data):.4f}")
        report.append(f"Maximum: {np.max(data):.4f}")
        report.append(f"Médiane: {np.median(data):.4f}")
        report.append("")
        
        report.append("2. RÉSULTATS DE L'ANALYSE")
        report.append("-"*40)
        
        for key, value in results.items():
            if not isinstance(value, (dict, list, np.ndarray)):
                report.append(f"{key}: {value}")
        
        report.append("")
        report.append("="*60)
        report.append("Fin du rapport")
        
        return "\n".join(report)
    
    def create_comparison_plot(
        self,
        ic_results: Dict,
        it_results: Dict,
        data: np.ndarray
    ) -> Union['go.Figure', 'plt.Figure']:
        """
        Crée un graphique comparant IC et IT
        
        Args:
            ic_results: Résultats de l'intervalle de confiance
            it_results: Résultats de l'intervalle de tolérance
            data: Données originales
            
        Returns:
            Figure de comparaison
        """
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Boîte pour IC
            fig.add_trace(go.Box(
                y=[ic_results['lower_bound'], ic_results['mean'], ic_results['upper_bound']],
                name='Intervalle de Confiance',
                marker_color='lightblue',
                boxmean='sd'
            ))
            
            # Boîte pour IT
            fig.add_trace(go.Box(
                y=[it_results['lower_bound'], it_results['mean'], it_results['upper_bound']],
                name='Intervalle de Tolérance',
                marker_color='lightgreen',
                boxmean='sd'
            ))
            
            # Points de données
            fig.add_trace(go.Scatter(
                x=['Données']*len(data),
                y=data,
                mode='markers',
                name='Données',
                marker=dict(color='grey', size=5, opacity=0.5)
            ))
            
            fig.update_layout(
                title="Comparaison IC vs IT",
                yaxis_title="Valeur",
                showlegend=True,
                template="plotly_white"
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Préparer les données pour le boxplot
            positions = [1, 2]
            widths = [0.6, 0.6]
            
            # IC
            ic_data = [ic_results['lower_bound'], ic_results['mean'], ic_results['upper_bound']]
            # IT
            it_data = [it_results['lower_bound'], it_results['mean'], it_results['upper_bound']]
            
            bp = ax.boxplot([ic_data, it_data], 
                           positions=positions,
                           widths=widths,
                           labels=['IC', 'IT'],
                           patch_artist=True)
            
            # Couleurs
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightgreen')
            
            # Ajouter les données
            ax.scatter([3]*len(data), data, alpha=0.5, color='grey', label='Données')
            
            ax.set_ylabel('Valeur')
            ax.set_title('Comparaison Intervalle de Confiance vs Intervalle de Tolérance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig