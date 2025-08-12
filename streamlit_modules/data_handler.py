"""
Module de gestion des données - Import/Export
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Union
import io

class DataHandler:
    """Gestionnaire pour import/export de données"""
    
    @staticmethod
    def load_data(file, column_name: Optional[str] = None) -> np.ndarray:
        """
        Charge les données depuis un fichier CSV ou Excel
        
        Args:
            file: Fichier uploadé via Streamlit
            column_name: Nom de la colonne à extraire
            
        Returns:
            np.ndarray avec les données
        """
        # Déterminer le type de fichier
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Format de fichier non supporté. Utilisez CSV ou Excel.")
        
        # Si une colonne spécifique est demandée
        if column_name:
            if column_name not in df.columns:
                raise ValueError(f"Colonne '{column_name}' non trouvée dans le fichier")
            data = df[column_name].dropna().values
        else:
            # Essayer de trouver la première colonne numérique
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("Aucune colonne numérique trouvée dans le fichier")
            data = df[numeric_cols[0]].dropna().values
        
        if len(data) == 0:
            raise ValueError("Aucune donnée valide trouvée")
        
        return data.astype(float)
    
    @staticmethod
    def create_sample_data(
        distribution: str = 'normal',
        n: int = 30,
        params: Optional[dict] = None
    ) -> np.ndarray:
        """
        Génère des données d'exemple
        
        Args:
            distribution: Type de distribution ('normal', 'uniform', 'exponential')
            n: Nombre d'échantillons
            params: Paramètres de la distribution
            
        Returns:
            np.ndarray avec données générées
        """
        np.random.seed(42)  # Pour reproductibilité
        
        if distribution == 'normal':
            params = params or {'mean': 100, 'std': 2}
            return np.random.normal(params['mean'], params['std'], n)
        
        elif distribution == 'uniform':
            params = params or {'low': 95, 'high': 105}
            return np.random.uniform(params['low'], params['high'], n)
        
        elif distribution == 'exponential':
            params = params or {'scale': 100}
            return np.random.exponential(params['scale'], n)
        
        else:
            raise ValueError(f"Distribution '{distribution}' non supportée")
    
    @staticmethod
    def export_results_to_excel(results: dict) -> io.BytesIO:
        """
        Exporte les résultats vers un fichier Excel
        
        Args:
            results: Dictionnaire contenant les résultats d'analyse
            
        Returns:
            BytesIO object contenant le fichier Excel
        """
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Format pour les en-têtes
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1f77b4',
                'font_color': 'white',
                'border': 1
            })
            
            # Format pour les cellules
            cell_format = workbook.add_format({
                'border': 1,
                'align': 'center'
            })
            
            # Format pour les nombres
            number_format = workbook.add_format({
                'border': 1,
                'align': 'center',
                'num_format': '0.0000'
            })
            
            # Feuille de résumé
            summary_data = []
            for analysis_type, analysis_results in results.items():
                if isinstance(analysis_results, dict):
                    for key, value in analysis_results.items():
                        if not isinstance(value, (dict, list, np.ndarray)):
                            summary_data.append({
                                'Analyse': analysis_type,
                                'Paramètre': key,
                                'Valeur': value
                            })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Résumé', index=False)
                
                # Appliquer le formatage
                worksheet = writer.sheets['Résumé']
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
            
            # Feuilles détaillées pour chaque analyse
            for analysis_type, analysis_results in results.items():
                if isinstance(analysis_results, dict):
                    # Créer un DataFrame pour cette analyse
                    detail_data = []
                    for key, value in analysis_results.items():
                        if not isinstance(value, (dict, list, np.ndarray)):
                            detail_data.append({'Paramètre': key, 'Valeur': value})
                    
                    if detail_data:
                        detail_df = pd.DataFrame(detail_data)
                        sheet_name = analysis_type[:30]  # Excel limite à 31 caractères
                        detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Formatage
                        worksheet = writer.sheets[sheet_name]
                        for col_num, value in enumerate(detail_df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
            
            # Ajouter les données brutes si disponibles
            if 'data' in results:
                data_df = pd.DataFrame({'Valeurs': results['data']})
                data_df.to_excel(writer, sheet_name='Données', index=False)
                
                # Statistiques descriptives
                stats_df = pd.DataFrame({
                    'Statistique': ['Moyenne', 'Écart-type', 'Minimum', 'Q1', 'Médiane', 'Q3', 'Maximum'],
                    'Valeur': [
                        np.mean(results['data']),
                        np.std(results['data'], ddof=1),
                        np.min(results['data']),
                        np.percentile(results['data'], 25),
                        np.median(results['data']),
                        np.percentile(results['data'], 75),
                        np.max(results['data'])
                    ]
                })
                stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
        
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def validate_data(data: np.ndarray) -> dict:
        """
        Valide les données d'entrée
        
        Args:
            data: Données à valider
            
        Returns:
            Dict avec résultats de validation
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Vérifications de base
        n = len(data)
        validation['statistics']['n'] = n
        
        if n < 2:
            validation['errors'].append("Au moins 2 observations requises")
            validation['is_valid'] = False
            return validation
        
        if n < 10:
            validation['warnings'].append(f"Taille d'échantillon faible (n={n}). Recommandé: n≥30")
        
        # Vérifier les valeurs manquantes
        if np.any(np.isnan(data)):
            validation['errors'].append("Valeurs manquantes détectées")
            validation['is_valid'] = False
            return validation
        
        # Vérifier les valeurs infinies
        if np.any(np.isinf(data)):
            validation['errors'].append("Valeurs infinies détectées")
            validation['is_valid'] = False
            return validation
        
        # Statistiques descriptives
        validation['statistics']['mean'] = np.mean(data)
        validation['statistics']['std'] = np.std(data, ddof=1)
        validation['statistics']['min'] = np.min(data)
        validation['statistics']['max'] = np.max(data)
        
        # Vérifier la variabilité
        cv = validation['statistics']['std'] / abs(validation['statistics']['mean']) * 100
        validation['statistics']['cv'] = cv
        
        if cv < 0.1:
            validation['warnings'].append("Très faible variabilité détectée (CV < 0.1%)")
        elif cv > 50:
            validation['warnings'].append("Forte variabilité détectée (CV > 50%)")
        
        # Détection d'outliers potentiels
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        n_outliers = np.sum((data < lower_fence) | (data > upper_fence))
        if n_outliers > 0:
            validation['warnings'].append(f"{n_outliers} valeur(s) aberrante(s) potentielle(s) détectée(s)")
            validation['statistics']['n_outliers'] = n_outliers
        
        return validation
    
    @staticmethod
    def format_results_for_display(results: dict, precision: int = 4) -> pd.DataFrame:
        """
        Formate les résultats pour affichage
        
        Args:
            results: Résultats à formater
            precision: Nombre de décimales
            
        Returns:
            DataFrame formaté pour affichage
        """
        formatted_data = []
        
        for key, value in results.items():
            if isinstance(value, (int, float, np.number)):
                if isinstance(value, float):
                    formatted_value = f"{value:.{precision}f}"
                else:
                    formatted_value = str(value)
                
                # Formater le nom du paramètre
                formatted_key = key.replace('_', ' ').title()
                formatted_data.append({
                    'Paramètre': formatted_key,
                    'Valeur': formatted_value
                })
        
        return pd.DataFrame(formatted_data)
    
    @staticmethod
    def save_session_data(data: np.ndarray, results: dict, filename: str) -> io.BytesIO:
        """
        Sauvegarde les données et résultats de session
        
        Args:
            data: Données analysées
            results: Résultats d'analyse
            filename: Nom du fichier
            
        Returns:
            BytesIO object avec les données sauvegardées
        """
        buffer = io.BytesIO()
        
        # Créer un dictionnaire de session
        session = {
            'data': data.tolist(),
            'results': results,
            'metadata': {
                'date': pd.Timestamp.now().isoformat(),
                'n_samples': len(data),
                'analyses_performed': list(results.keys())
            }
        }
        
        # Sauvegarder en JSON
        import json
        json_str = json.dumps(session, indent=2, default=str)
        buffer.write(json_str.encode())
        buffer.seek(0)
        
        return buffer