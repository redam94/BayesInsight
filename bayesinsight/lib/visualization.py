"""
Functions for visualizing and diagnosing Bayesian priors in marketing mix models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class PriorVisualization:
    """Helper class for visualizing priors"""
    
    def plot_media_prior(
        self,
        prior: "PositiveHierarchicalPrior",
        groups: List[str],
        n_samples: int = 1000,
        figsize: tuple = (15, 10)
    ):
        """
        Visualize media prior distributions including hierarchical structure.
        
        Parameters
        ----------
        prior : PositiveHierarchicalPrior
            Prior specification for media effects
        groups : List[str]
            List of groups (e.g., geographies)
        n_samples : int
            Number of samples to draw
        figsize : tuple
            Figure size for plots
        """
        
        with pm.Model() as model:
            # Build prior structure
            effects = prior.build()
            
            # Sample from prior
            prior_samples = pm.sample_prior_predictive(samples=n_samples)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Prior Distribution Analysis: {prior.name}", fontsize=14)
        
        # Plot 1: Global distribution
        ax = axes[0,0]
        global_samples = prior_samples[f"{prior.name}_global_mean"]
        sns.histplot(global_samples, ax=ax)
        ax.set_title("Global Effect Distribution")
        ax.set_xlabel("Effect Size (log scale)")
        
        # Plot 2: Group-specific distributions
        ax = axes[0,1]
        for group in groups:
            group_samples = prior_samples[f"{prior.name}_effect_{group}"]
            sns.kdeplot(group_samples, ax=ax, label=group)
        ax.set_title("Group-Specific Effect Distributions")
        ax.set_xlabel("Effect Size")
        ax.legend()
        
        # Plot 3: Between-group variation
        ax = axes[1,0]
        group_std_samples = prior_samples[f"{prior.name}_group_std"]
        sns.histplot(group_std_samples, ax=ax)
        ax.set_title("Between-Group Variation")
        ax.set_xlabel("Standard Deviation")
        
        # Plot 4: Box plot comparison
        ax = axes[1,1]
        plot_data = []
        for group in groups:
            samples = prior_samples[f"{prior.name}_effect_{group}"]
            plot_data.extend([(group, s) for s in samples])
        plot_df = pd.DataFrame(plot_data, columns=['Group', 'Effect'])
        sns.boxplot(data=plot_df, x='Group', y='Effect', ax=ax)
        ax.set_title("Effect Size by Group")
        ax.set_ylabel("Effect Size")
        
        plt.tight_layout()
        return fig, axes

    def plot_adstock_transform(
        self,
        prior: "MediaPrior",
        max_periods: int = 12,
        n_samples: int = 100,
        figsize: tuple = (12, 6)
    ):
        """
        Visualize adstock transformation distributions.
        
        Parameters
        ----------
        prior : MediaPrior
            Prior specification for media effects
        max_periods : int
            Number of periods to show
        n_samples : int
            Number of samples to draw
        figsize : tuple
            Figure size for plots
        """
        
        with pm.Model() as model:
            # Build adstock parameters
            retention, lag = prior.build_adstock_prior()
            
            # Sample parameters
            trace = pm.sample_prior_predictive(samples=n_samples)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Adstock Transform Analysis: {prior.name}", fontsize=14)
        
        # Plot 1: Retention rate distribution
        ax = axes[0]
        retention_samples = trace[f"{prior.name}_retention_rate_log"]
        sns.histplot(retention_samples, ax=ax)
        ax.set_title("Retention Rate Distribution")
        ax.set_xlabel("Retention Rate")
        
        # Plot 2: Example decay curves
        ax = axes[1]
        t = np.arange(max_periods)
        for i in range(min(n_samples, 20)):  # Plot first 20 samples
            retention_rate = retention_samples[i]
            decay = retention_rate ** t
            ax.plot(t, decay, alpha=0.3, color='blue')
        
        ax.set_title("Example Decay Curves")
        ax.set_xlabel("Periods")
        ax.set_ylabel("Effect Size")
        
        plt.tight_layout()
        return fig, axes

    def plot_hill_transform(
        self,
        prior: "MediaPrior",
        n_samples: int = 100,
        n_points: int = 100,
        figsize: tuple = (12, 6)
    ):
        """
        Visualize Hill transform (saturation curve) distributions.
        
        Parameters
        ----------
        prior : MediaPrior
            Prior specification for media effects
        n_samples : int
            Number of samples to draw
        n_points : int
            Number of points for curve plotting
        figsize : tuple
            Figure size for plots
        """
        
        with pm.Model() as model:
            # Build Hill transform parameters
            K, n = prior.build_media_priors()
            
            # Sample parameters
            trace = pm.sample_prior_predictive(samples=n_samples)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Hill Transform Analysis: {prior.name}", fontsize=14)
        
        # Plot 1: Parameter distributions
        ax = axes[0]
        K_samples = trace[f"{prior.name}_K"]
        n_samples = trace[f"{prior.name}_n"]
        
        sns.scatterplot(x=K_samples, y=n_samples, alpha=0.5, ax=ax)
        ax.set_title("Parameter Joint Distribution")
        ax.set_xlabel("Saturation Point (K)")
        ax.set_ylabel("Shape Parameter (n)")
        
        # Plot 2: Example response curves
        ax = axes[1]
        x = np.linspace(0, 2, n_points)
        
        for i in range(min(n_samples, 20)):  # Plot first 20 samples
            K, n = K_samples[i], n_samples[i]
            y = x**n / (K**n + x**n)
            ax.plot(x, y, alpha=0.3, color='blue')
        
        ax.set_title("Example Response Curves")
        ax.set_xlabel("Normalized Spend")
        ax.set_ylabel("Response")
        ax.grid(True)
        
        plt.tight_layout()
        return fig, axes

    def plot_seasonal_prior(
        self,
        prior: "SeasonalPrior",
        n_samples: int = 100,
        n_periods: int = 52,
        figsize: tuple = (12, 6)
    ):
        """
        Visualize seasonal prior distributions.
        
        Parameters
        ----------
        prior : SeasonalPrior
            Prior specification for seasonal effects
        n_samples : int
            Number of samples to draw
        n_periods : int
            Number of periods to show
        figsize : tuple
            Figure size for plots
        """
        
        with pm.Model() as model:
            # Build seasonal components
            components = prior.build_coeff_prior()
            
            # Sample parameters
            trace = pm.sample_prior_predictive(samples=n_samples)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Seasonal Pattern Analysis: {prior.name}", fontsize=14)
        
        # Plot 1: Component amplitudes
        ax = axes[0]
        component_samples = trace[f"{prior.name}_amplitudes"]
        
        for i in range(prior.n_fourier):
            sns.kdeplot(
                component_samples[:, i*2], 
                label=f'Sin {i+1}',
                ax=ax
            )
            sns.kdeplot(
                component_samples[:, i*2+1],
                label=f'Cos {i+1}',
                ax=ax
            )
        
        ax.set_title("Component Amplitude Distributions")
        ax.set_xlabel("Amplitude")
        ax.legend()
        
        # Plot 2: Example seasonal patterns
        ax = axes[1]
        t = np.linspace(0, 2*np.pi, n_periods)
        
        for i in range(min(n_samples, 20)):  # Plot first 20 samples
            pattern = np.zeros(n_periods)
            for j in range(prior.n_fourier):
                pattern += (
                    component_samples[i, j*2] * np.sin((j+1)*t) +
                    component_samples[i, j*2+1] * np.cos((j+1)*t)
                )
            ax.plot(t, pattern, alpha=0.3, color='blue')
        
        ax.set_title("Example Seasonal Patterns")
        ax.set_xlabel("Time")
        ax.set_ylabel("Effect Size")
        
        plt.tight_layout()
        return fig, axes

def display_all_priors(model: "BayesInsightModel"):
    """
    Display visualizations for all priors in a model.
    
    Parameters
    ----------
    model : BayesInsightModel
        Model containing priors to visualize
    """
    viz = PriorVisualization()
    
    # Get unique groups
    groups = list(model.data.metadata.allowed_geos)
    
    # Plot media priors
    for var in model.return_media_variables():
        print(f"\nAnalyzing priors for {var.variable_name}:")
        
        # Plot media effect prior
        viz.plot_media_prior(var.coeff_prior, groups)
        plt.show()
        
        # Plot transformations
        viz.plot_adstock_transform(var)
        plt.show()
        
        viz.plot_hill_transform(var)
        plt.show()
    
    # Plot seasonal priors if present
    for var in model.return_season_variables():
        print(f"\nAnalyzing seasonal prior for {var.variable_name}:")
        viz.plot_seasonal_prior(var)
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create sample model
    from bayesinsight.models.dataloading import MFF, MetaData
    from bayesinsight import BayesInsightModel
    from bayesinsight.models.variablemodels import MediaVariableDetails
    
    # Generate sample data
    data = pd.DataFrame({
        "Period": pd.date_range("2024-01-01", periods=52, freq="W"),
        "Geography": ["US"] * 52,
        "Campaign": ["Total"]*52,
        "Outlet": ["Total"] * 52,
        "Creative": ["Total"] * 52,
        "Product": ["Total"] * 52,
        "VariableName": ["Sales"] * 52,
        "VariableValue": np.random.lognormal(0, 0.5, 52)
    })
    
    
    mff = MFF.from_mff_df(data)
    
    # Create model with sample priors
    model = BayesInsightModel(
        data=mff,
        variable_details=[
            MediaVariableDetails(
                variable_name="Sales",
            )
        ]
    )
    
    # Display all priors
    display_all_priors(model)