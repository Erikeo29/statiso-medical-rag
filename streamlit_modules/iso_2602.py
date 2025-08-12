"""
Module ISO 2602:1980 - Intervalles de confiance pour la moyenne
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional

class ISO2602Calculator:
    """Calculateur pour intervalles de confiance selon ISO 2602:1980"""
    
    def __init__(self):
        self.norm_name = "ISO 2602:1980"
        self.description = "Interprétation statistique - Estimation de la moyenne - Intervalle de confiance"
    
    def calculate_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        sigma_known: Optional[float] = None
    ) -> Dict:
        """
        Calcule l'intervalle de confiance pour la moyenne
        
        Args:
            data: Données d'échantillon
            confidence_level: Niveau de confiance (0.90, 0.95, 0.99)
            sigma_known: Écart-type de la population si connu
            
        Returns:
            Dict contenant les résultats du calcul
        """
        n = len(data)
        mean = np.mean(data)
        
        if n < 2:
            raise ValueError("Au moins 2 observations requises")
        
        alpha = 1 - confidence_level
        
        if sigma_known is not None:
            # Cas 1: σ connu - Distribution normale
            std_error = sigma_known / np.sqrt(n)
            z_critical = stats.norm.ppf(1 - alpha/2)
            margin_error = z_critical * std_error
            distribution_used = "Normale (Z)"
            critical_value = z_critical
        else:
            # Cas 2: σ inconnu - Distribution t de Student
            sample_std = np.std(data, ddof=1)
            std_error = sample_std / np.sqrt(n)
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * std_error
            distribution_used = "t-Student"
            critical_value = t_critical
        
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error
        
        return {
            'mean': mean,
            'sample_size': n,
            'confidence_level': confidence_level,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin_error': margin_error,
            'std_error': std_error,
            'sample_std': np.std(data, ddof=1) if sigma_known is None else None,
            'population_std': sigma_known,
            'distribution': distribution_used,
            'critical_value': critical_value,
            'degrees_freedom': n-1 if sigma_known is None else None,
            'interval_width': upper_bound - lower_bound
        }
    
    def determine_sample_size(
        self,
        margin_error: float,
        std_dev: float,
        confidence_level: float = 0.95,
        finite_population: Optional[int] = None
    ) -> Dict:
        """
        Détermine la taille d'échantillon nécessaire
        
        Args:
            margin_error: Marge d'erreur souhaitée
            std_dev: Écart-type estimé
            confidence_level: Niveau de confiance
            finite_population: Taille de la population si finie
            
        Returns:
            Dict avec taille d'échantillon et détails
        """
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha/2)
        
        # Formule de base pour population infinie
        n = (z * std_dev / margin_error) ** 2
        n_infinite = int(np.ceil(n))
        
        # Correction pour population finie si applicable
        if finite_population:
            n_finite = n_infinite / (1 + (n_infinite - 1) / finite_population)
            n_final = int(np.ceil(n_finite))
        else:
            n_final = n_infinite
        
        return {
            'required_sample_size': n_final,
            'margin_error': margin_error,
            'std_dev': std_dev,
            'confidence_level': confidence_level,
            'z_value': z,
            'finite_correction_applied': finite_population is not None,
            'population_size': finite_population
        }
    
    def hypothesis_test(
        self,
        data: np.ndarray,
        null_value: float,
        alternative: str = 'two-sided',
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Test d'hypothèse sur la moyenne
        
        Args:
            data: Données d'échantillon
            null_value: Valeur de l'hypothèse nulle (μ₀)
            alternative: 'two-sided', 'less', ou 'greater'
            confidence_level: Niveau de confiance
            
        Returns:
            Dict avec résultats du test
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        std_error = std / np.sqrt(n)
        
        # Calcul de la statistique t
        t_stat = (mean - null_value) / std_error
        
        # Calcul de la p-value selon l'alternative
        df = n - 1
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        elif alternative == 'less':
            p_value = stats.t.cdf(t_stat, df)
        else:  # greater
            p_value = 1 - stats.t.cdf(t_stat, df)
        
        # Décision
        alpha = 1 - confidence_level
        reject_null = p_value < alpha
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'null_value': null_value,
            'sample_mean': mean,
            'alternative': alternative,
            'confidence_level': confidence_level,
            'reject_null': reject_null,
            'conclusion': f"{'Rejet' if reject_null else 'Non-rejet'} de H₀ au niveau {confidence_level*100:.0f}%"
        }
    
    def prediction_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        n_future: int = 1
    ) -> Dict:
        """
        Calcule l'intervalle de prédiction pour de futures observations
        
        Args:
            data: Données historiques
            confidence_level: Niveau de confiance
            n_future: Nombre d'observations futures à prédire (moyenne de n_future)
            
        Returns:
            Dict avec l'intervalle de prédiction
        """
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        # Erreur de prédiction
        pred_error = std * np.sqrt(1 + 1/n + 1/n_future)
        margin = t_critical * pred_error
        
        return {
            'predicted_mean': mean,
            'lower_bound': mean - margin,
            'upper_bound': mean + margin,
            'confidence_level': confidence_level,
            'n_future': n_future,
            'prediction_error': pred_error,
            'margin': margin,
            'interval_width': 2 * margin
        }
    
    def compare_means(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        confidence_level: float = 0.95,
        equal_variance: bool = True
    ) -> Dict:
        """
        Compare deux moyennes (test t à deux échantillons)
        
        Args:
            data1: Premier échantillon
            data2: Deuxième échantillon
            confidence_level: Niveau de confiance
            equal_variance: Supposer variances égales (test de Student) ou non (test de Welch)
            
        Returns:
            Dict avec résultats de la comparaison
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        mean_diff = mean1 - mean2
        
        if equal_variance:
            # Test de Student (variances égales)
            sp = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            std_error = sp * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # Test de Welch (variances inégales)
            std_error = np.sqrt(std1**2/n1 + std2**2/n2)
            df = (std1**2/n1 + std2**2/n2)**2 / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
        
        t_stat = mean_diff / std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Intervalle de confiance pour la différence
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_critical * std_error
        ci_upper = mean_diff + t_critical * std_error
        
        return {
            'mean1': mean1,
            'mean2': mean2,
            'mean_difference': mean_diff,
            'std_error': std_error,
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'equal_variance_assumed': equal_variance,
            'significant_difference': p_value < (1 - confidence_level)
        }
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        statistic='mean'
    ) -> Dict:
        """
        Calcule un intervalle de confiance par bootstrap
        
        Args:
            data: Données d'échantillon
            confidence_level: Niveau de confiance
            n_bootstrap: Nombre d'échantillons bootstrap
            statistic: 'mean', 'median', ou callable
            
        Returns:
            Dict avec l'intervalle bootstrap
        """
        n = len(data)
        
        # Fonction statistique
        if statistic == 'mean':
            stat_func = np.mean
        elif statistic == 'median':
            stat_func = np.median
        else:
            stat_func = statistic
        
        # Bootstrap
        bootstrap_stats = []
        np.random.seed(42)  # Pour reproductibilité
        
        for _ in range(n_bootstrap):
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(stat_func(resample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Percentiles pour l'intervalle
        alpha = 1 - confidence_level
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'point_estimate': stat_func(data),
            'lower_bound': ci_lower,
            'upper_bound': ci_upper,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'statistic': statistic if isinstance(statistic, str) else 'custom',
            'interval_width': ci_upper - ci_lower
        }