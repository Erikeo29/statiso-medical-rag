"""
Module ISO 16269-6:2014 - Intervalles statistiques de tolérance/dispersion
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional

class ISO16269_6Calculator:
    """Calculateur pour intervalles de tolérance selon ISO 16269-6:2014"""
    
    # Facteurs k pour intervalles de tolérance (tables simplifiées)
    # Structure: k_factors[confidence_level][proportion][sample_size]
    K_FACTORS_BILATERAL = {
        0.95: {
            0.90: {10: 2.355, 15: 2.068, 20: 1.926, 30: 1.777, 50: 1.646, 100: 1.527, 200: 1.450},
            0.95: {10: 2.911, 15: 2.566, 20: 2.396, 30: 2.220, 50: 2.065, 100: 1.927, 200: 1.838},
            0.99: {10: 3.981, 15: 3.520, 20: 3.295, 30: 3.064, 50: 2.863, 100: 2.684, 200: 2.570}
        },
        0.99: {
            0.90: {10: 3.617, 15: 2.967, 20: 2.675, 30: 2.394, 50: 2.166, 100: 1.978, 200: 1.866},
            0.95: {10: 4.294, 15: 3.529, 20: 3.184, 30: 2.850, 50: 2.581, 100: 2.355, 200: 2.213},
            0.99: {10: 5.610, 15: 4.621, 20: 4.175, 30: 3.742, 50: 3.390, 100: 3.098, 200: 2.921}
        }
    }
    
    K_FACTORS_UNILATERAL = {
        0.95: {
            0.90: {10: 1.855, 15: 1.868, 20: 1.926, 30: 1.777, 50: 1.646, 100: 1.527, 200: 1.450},
            0.95: {10: 2.355, 15: 2.068, 20: 1.926, 30: 1.777, 50: 1.646, 100: 1.527, 200: 1.411},
            0.99: {10: 3.379, 15: 2.965, 20: 2.760, 30: 2.555, 50: 2.355, 100: 2.173, 200: 2.041}
        },
        0.99: {
            0.90: {10: 2.713, 15: 2.480, 20: 2.355, 30: 2.140, 50: 1.965, 100: 1.805, 200: 1.712},
            0.95: {10: 3.370, 15: 2.954, 20: 2.752, 30: 2.549, 50: 2.379, 100: 2.228, 200: 2.143},
            0.99: {10: 4.728, 15: 4.010, 20: 3.638, 30: 3.271, 50: 2.974, 100: 2.737, 200: 2.605}
        }
    }
    
    def __init__(self):
        self.norm_name = "ISO 16269-6:2014"
        self.description = "Détermination des intervalles statistiques de dispersion"
    
    def _interpolate_k_factor(
        self, 
        n: int, 
        confidence: float, 
        proportion: float,
        bilateral: bool = True
    ) -> float:
        """Interpole le facteur k pour une taille d'échantillon donnée"""
        
        k_table = self.K_FACTORS_BILATERAL if bilateral else self.K_FACTORS_UNILATERAL
        
        # Vérifier si les niveaux existent
        if confidence not in k_table:
            # Utiliser 0.95 par défaut si non trouvé
            confidence = 0.95
        if proportion not in k_table[confidence]:
            # Utiliser 0.95 par défaut si non trouvé
            proportion = 0.95
        
        k_values = k_table[confidence][proportion]
        sizes = sorted(k_values.keys())
        
        # Si n est dans la table, retourner directement
        if n in k_values:
            return k_values[n]
        
        # Si n < min ou n > max, utiliser la valeur limite
        if n <= sizes[0]:
            return k_values[sizes[0]]
        if n >= sizes[-1]:
            return k_values[sizes[-1]]
        
        # Interpolation linéaire
        for i in range(len(sizes) - 1):
            if sizes[i] < n < sizes[i + 1]:
                x1, x2 = sizes[i], sizes[i + 1]
                y1, y2 = k_values[x1], k_values[x2]
                return y1 + (y2 - y1) * (n - x1) / (x2 - x1)
        
        return k_values[sizes[-1]]
    
    def calculate_tolerance_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        proportion: float = 0.95,
        bilateral: bool = True,
        method: str = "normal"
    ) -> Dict:
        """
        Calcule l'intervalle de tolérance
        
        Args:
            data: Données d'échantillon
            confidence_level: Niveau de confiance (0.95 ou 0.99)
            proportion: Proportion de la population à couvrir (0.90, 0.95, 0.99)
            bilateral: True pour bilatéral, False pour unilatéral
            method: "normal" pour paramétrique, "nonparametric" pour non-paramétrique
            
        Returns:
            Dict contenant les résultats
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if n < 3:
            raise ValueError("Au moins 3 observations requises pour un intervalle de tolérance")
        
        if method == "normal":
            # Méthode paramétrique (distribution normale)
            # Arrondir les niveaux aux valeurs supportées
            if confidence_level not in [0.95, 0.99]:
                confidence_level = 0.95 if confidence_level < 0.97 else 0.99
            if proportion not in [0.90, 0.95, 0.99]:
                if proportion < 0.925:
                    proportion = 0.90
                elif proportion < 0.97:
                    proportion = 0.95
                else:
                    proportion = 0.99
            
            k = self._interpolate_k_factor(n, confidence_level, proportion, bilateral)
            
            if bilateral:
                lower_bound = mean - k * std
                upper_bound = mean + k * std
            else:
                # Pour unilatéral, on calcule généralement la limite supérieure
                lower_bound = -np.inf
                upper_bound = mean + k * std
            
            return {
                'mean': mean,
                'std': std,
                'sample_size': n,
                'confidence_level': confidence_level,
                'proportion': proportion,
                'k_factor': k,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_type': 'Bilatéral' if bilateral else 'Unilatéral',
                'method': 'Paramétrique (Normal)',
                'interval_width': upper_bound - lower_bound if bilateral else None
            }
        
        else:
            # Méthode non-paramétrique
            sorted_data = np.sort(data)
            
            # Calcul des indices pour méthode non-paramétrique
            # Formule simplifiée - en production, utiliser les tables exactes
            if bilateral:
                r = int(np.floor((n + 1) * (1 - proportion) / 2))
                s = n - r + 1
                if r < 1 or s > n:
                    raise ValueError(f"Taille d'échantillon insuffisante pour ces paramètres")
                lower_bound = sorted_data[max(0, r - 1)]
                upper_bound = sorted_data[min(n - 1, s - 1)]
            else:
                r = int(np.floor((n + 1) * (1 - proportion)))
                lower_bound = -np.inf
                upper_bound = sorted_data[min(n - 1, n - r)]
            
            return {
                'mean': mean,
                'std': std,
                'sample_size': n,
                'confidence_level': confidence_level,
                'proportion': proportion,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_type': 'Bilatéral' if bilateral else 'Unilatéral',
                'method': 'Non-paramétrique',
                'order_statistics_used': {'lower': r, 'upper': s if bilateral else n - r}
            }
    
    def compare_with_specifications(
        self,
        data: np.ndarray,
        lower_spec: Optional[float] = None,
        upper_spec: Optional[float] = None,
        confidence_level: float = 0.95,
        proportion: float = 0.95
    ) -> Dict:
        """
        Compare l'intervalle de tolérance avec les spécifications
        
        Returns:
            Dict avec analyse de capabilité
        """
        tolerance_result = self.calculate_tolerance_interval(
            data, confidence_level, proportion, 
            bilateral=(lower_spec is not None and upper_spec is not None)
        )
        
        result = tolerance_result.copy()
        
        # Analyse de conformité
        conformity_analysis = {}
        
        if lower_spec is not None:
            conformity_analysis['lower_spec'] = lower_spec
            conformity_analysis['lower_margin'] = tolerance_result['mean'] - tolerance_result['k_factor'] * tolerance_result['std'] - lower_spec
            conformity_analysis['lower_spec_ok'] = conformity_analysis['lower_margin'] > 0
        
        if upper_spec is not None:
            conformity_analysis['upper_spec'] = upper_spec
            conformity_analysis['upper_margin'] = upper_spec - (tolerance_result['mean'] + tolerance_result['k_factor'] * tolerance_result['std'])
            conformity_analysis['upper_spec_ok'] = conformity_analysis['upper_margin'] > 0
        
        if lower_spec is not None and upper_spec is not None:
            # Calcul des indices de capabilité
            spec_range = upper_spec - lower_spec
            process_range = 6 * tolerance_result['std']
            
            conformity_analysis['Cp'] = spec_range / process_range
            conformity_analysis['Cpk'] = min(
                (tolerance_result['mean'] - lower_spec) / (3 * tolerance_result['std']),
                (upper_spec - tolerance_result['mean']) / (3 * tolerance_result['std'])
            )
            conformity_analysis['process_capable'] = conformity_analysis['Cpk'] > 1.33
        
        result['conformity_analysis'] = conformity_analysis
        
        return result
    
    def determine_sample_size(
        self,
        confidence_level: float = 0.95,
        proportion: float = 0.95,
        bilateral: bool = True,
        relative_precision: float = 0.10
    ) -> Dict:
        """
        Détermine la taille d'échantillon recommandée
        
        Args:
            confidence_level: Niveau de confiance
            proportion: Proportion de population à couvrir
            bilateral: Type d'intervalle
            relative_precision: Précision relative souhaitée
            
        Returns:
            Dict avec recommandations de taille d'échantillon
        """
        # Tables de recommandations basées sur ISO 16269-6
        recommendations = {
            (0.95, 0.95): {'minimum': 30, 'recommended': 50, 'optimal': 100},
            (0.95, 0.99): {'minimum': 50, 'recommended': 100, 'optimal': 200},
            (0.99, 0.95): {'minimum': 50, 'recommended': 100, 'optimal': 200},
            (0.99, 0.99): {'minimum': 100, 'recommended': 200, 'optimal': 500}
        }
        
        key = (confidence_level, proportion)
        if key not in recommendations:
            # Valeurs par défaut
            rec = {'minimum': 30, 'recommended': 50, 'optimal': 100}
        else:
            rec = recommendations[key]
        
        # Ajustement pour intervalle unilatéral (peut être légèrement plus petit)
        if not bilateral:
            rec = {k: int(v * 0.8) for k, v in rec.items()}
        
        return {
            'confidence_level': confidence_level,
            'proportion': proportion,
            'bilateral': bilateral,
            'minimum_n': rec['minimum'],
            'recommended_n': rec['recommended'],
            'optimal_n': rec['optimal'],
            'note': "Tailles basées sur les recommandations ISO 16269-6"
        }
    
    def test_normality(self, data: np.ndarray) -> Dict:
        """
        Tests de normalité multiples
        
        Args:
            data: Données à tester
            
        Returns:
            Dict avec résultats des tests
        """
        n = len(data)
        
        results = {
            'sample_size': n,
            'tests': {}
        }
        
        # Test de Shapiro-Wilk (recommandé pour n < 50)
        if n <= 5000:
            stat_sw, p_sw = stats.shapiro(data)
            results['tests']['shapiro_wilk'] = {
                'statistic': stat_sw,
                'p_value': p_sw,
                'normal': p_sw > 0.05,
                'recommended_for': 'n < 50'
            }
        
        # Test d'Anderson-Darling
        ad_result = stats.anderson(data, dist='norm')
        results['tests']['anderson_darling'] = {
            'statistic': ad_result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], ad_result.critical_values)),
            'normal': ad_result.statistic < ad_result.critical_values[2],  # 5% level
            'recommended_for': 'Général'
        }
        
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        results['tests']['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p > 0.05,
            'recommended_for': 'n > 50'
        }
        
        # Conclusion générale
        normal_tests = [
            results['tests'].get('shapiro_wilk', {}).get('normal', True),
            results['tests']['anderson_darling']['normal'],
            results['tests']['kolmogorov_smirnov']['normal']
        ]
        
        results['conclusion'] = {
            'is_normal': sum(normal_tests) >= 2,  # Majorité des tests
            'confidence': sum(normal_tests) / len(normal_tests),
            'recommendation': 'Données normales' if sum(normal_tests) >= 2 else 'Considérer méthodes non-paramétriques'
        }
        
        return results
    
    def detect_outliers(self, data: np.ndarray, method: str = 'iqr') -> Dict:
        """
        Détection des valeurs aberrantes
        
        Args:
            data: Données à analyser
            method: 'iqr', 'zscore', ou 'grubbs'
            
        Returns:
            Dict avec outliers détectés
        """
        n = len(data)
        outliers = []
        outlier_indices = []
        
        if method == 'iqr':
            # Méthode IQR (Interquartile Range)
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR
            
            for i, value in enumerate(data):
                if value < lower_fence or value > upper_fence:
                    outliers.append(value)
                    outlier_indices.append(i)
            
            details = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_fence': lower_fence,
                'upper_fence': upper_fence
            }
        
        elif method == 'zscore':
            # Méthode Z-score
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            threshold = 3
            
            for i, value in enumerate(data):
                z_score = abs((value - mean) / std)
                if z_score > threshold:
                    outliers.append(value)
                    outlier_indices.append(i)
            
            details = {
                'mean': mean,
                'std': std,
                'threshold': threshold
            }
        
        else:  # grubbs
            # Test de Grubbs
            from scipy.stats import t
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            # Valeur critique pour test de Grubbs
            alpha = 0.05
            t_critical = t.ppf(1 - alpha/(2*n), n-2)
            grubbs_critical = ((n-1)/np.sqrt(n)) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))
            
            for i, value in enumerate(data):
                g_statistic = abs(value - mean) / std
                if g_statistic > grubbs_critical:
                    outliers.append(value)
                    outlier_indices.append(i)
            
            details = {
                'grubbs_critical': grubbs_critical,
                'alpha': alpha
            }
        
        return {
            'method': method,
            'n_outliers': len(outliers),
            'outliers': outliers,
            'outlier_indices': outlier_indices,
            'percentage': len(outliers) / n * 100,
            'details': details,
            'clean_data': np.delete(data, outlier_indices) if outliers else data
        }