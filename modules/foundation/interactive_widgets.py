#!/usr/bin/env python3
"""
Interactive Learning Widgets for Python for Semiconductors Learning Series

This module provides interactive Jupyter widgets for hands-on parameter tuning,
visualization, and real-time feedback in the learning environment.

Usage in Jupyter notebooks:
    from modules.foundation.interactive_widgets import *
    
    # Create ML parameter tuning widget
    ml_widget = create_ml_parameter_widget()
    display(ml_widget)
    
    # Create 3D wafer visualization
    wafer_viz = create_wafer_visualization_widget()
    display(wafer_viz)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json

# Check for optional dependencies
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("ipywidgets not available. Install with: pip install ipywidgets")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available. Install with: pip install plotly")

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification, make_regression
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("scikit-learn not available. Install with: pip install scikit-learn")

RANDOM_SEED = 42

class InteractiveMLTuner:
    """Interactive widget for ML hyperparameter tuning."""
    
    def __init__(self, problem_type: str = 'classification'):
        if not HAS_WIDGETS or not HAS_SKLEARN:
            raise ImportError("Required dependencies not available")
            
        self.problem_type = problem_type
        self.X, self.y = self._generate_sample_data()
        self.results_history = []
        
    def _generate_sample_data(self):
        """Generate sample semiconductor-like data."""
        np.random.seed(RANDOM_SEED)
        
        if self.problem_type == 'classification':
            # Simulate wafer pass/fail classification
            X, y = make_classification(
                n_samples=1000, n_features=10, n_informative=7,
                n_redundant=2, n_clusters_per_class=2, random_state=RANDOM_SEED
            )
            feature_names = [f'Process_Param_{i+1}' for i in range(10)]
            return pd.DataFrame(X, columns=feature_names), y
        else:
            # Simulate yield regression
            X, y = make_regression(
                n_samples=1000, n_features=8, noise=10, random_state=RANDOM_SEED
            )
            feature_names = [f'Process_Param_{i+1}' for i in range(8)]
            return pd.DataFrame(X, columns=feature_names), y
    
    def create_widget(self):
        """Create the interactive ML tuning widget."""
        # Parameter controls
        n_estimators = widgets.IntSlider(
            value=100, min=10, max=500, step=10,
            description='N Estimators:', style={'description_width': 'initial'}
        )
        
        max_depth = widgets.IntSlider(
            value=10, min=1, max=50, step=1,
            description='Max Depth:', style={'description_width': 'initial'}
        )
        
        min_samples_split = widgets.IntSlider(
            value=2, min=2, max=20, step=1,
            description='Min Samples Split:', style={'description_width': 'initial'}
        )
        
        min_samples_leaf = widgets.IntSlider(
            value=1, min=1, max=10, step=1,
            description='Min Samples Leaf:', style={'description_width': 'initial'}
        )
        
        # Control buttons
        train_button = widgets.Button(
            description="Train Model", button_style='primary',
            icon='play', layout=widgets.Layout(width='150px')
        )
        
        reset_button = widgets.Button(
            description="Reset", button_style='warning',
            icon='refresh', layout=widgets.Layout(width='100px')
        )
        
        # Output area
        output = widgets.Output()
        
        # Results display
        results_area = widgets.HTML(value="<h4>Model Performance</h4><p>Click 'Train Model' to see results</p>")
        
        def train_model(b):
            """Train model with current parameters."""
            with output:
                clear_output(wait=True)
                
                # Get current parameter values
                params = {
                    'n_estimators': n_estimators.value,
                    'max_depth': max_depth.value,
                    'min_samples_split': min_samples_split.value,
                    'min_samples_leaf': min_samples_leaf.value,
                    'random_state': RANDOM_SEED
                }
                
                print(f"Training {self.problem_type} model with parameters:")
                for key, value in params.items():
                    if key != 'random_state':
                        print(f"  {key}: {value}")
                
                # Train model
                if self.problem_type == 'classification':
                    model = RandomForestClassifier(**params)
                    scoring = 'accuracy'
                else:
                    model = RandomForestRegressor(**params)
                    scoring = 'r2'
                
                # Cross-validation
                scores = cross_val_score(model, self.X, self.y, cv=5, scoring=scoring)
                mean_score = scores.mean()
                std_score = scores.std()
                
                print(f"\nCross-validation results:")
                print(f"  Mean {scoring}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
                
                # Store results
                result = {
                    'params': params.copy(),
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
                }
                self.results_history.append(result)
                
                # Update results display
                self._update_results_display(results_area)
                
                # Feature importance
                model.fit(self.X, self.y)
                importance_df = pd.DataFrame({
                    'Feature': self.X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print(f"\nTop 5 Feature Importances:")
                for _, row in importance_df.head().iterrows():
                    print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        def reset_history(b):
            """Reset results history."""
            self.results_history = []
            results_area.value = "<h4>Model Performance</h4><p>History cleared. Train a model to see results.</p>"
            with output:
                clear_output()
        
        # Connect event handlers
        train_button.on_click(train_model)
        reset_button.on_click(reset_history)
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML("<h3>🔧 ML Parameter Tuning Lab</h3>"),
            n_estimators, max_depth, min_samples_split, min_samples_leaf,
            widgets.HBox([train_button, reset_button])
        ])
        
        results_panel = widgets.VBox([results_area, output])
        
        return widgets.HBox([controls, results_panel])
    
    def _update_results_display(self, results_widget):
        """Update the results display with history."""
        if not self.results_history:
            return
            
        html = "<h4>Model Performance History</h4>"
        html += "<table style='border-collapse: collapse; width: 100%;'>"
        html += "<tr style='background-color: #f2f2f2;'>"
        html += "<th style='border: 1px solid #ddd; padding: 8px;'>Time</th>"
        html += "<th style='border: 1px solid #ddd; padding: 8px;'>N Est.</th>"
        html += "<th style='border: 1px solid #ddd; padding: 8px;'>Depth</th>"
        html += "<th style='border: 1px solid #ddd; padding: 8px;'>Score</th>"
        html += "</tr>"
        
        for result in self.results_history[-10:]:  # Show last 10 results
            score_color = "green" if result['mean_score'] > 0.8 else "orange" if result['mean_score'] > 0.7 else "red"
            html += f"<tr>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{result['timestamp']}</td>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{result['params']['n_estimators']}</td>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{result['params']['max_depth']}</td>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px; color: {score_color};'>{result['mean_score']:.4f}</td>"
            html += f"</tr>"
        
        html += "</table>"
        
        if len(self.results_history) > 1:
            best_result = max(self.results_history, key=lambda x: x['mean_score'])
            html += f"<p><strong>🏆 Best Score:</strong> {best_result['mean_score']:.4f} "
            html += f"(N Est: {best_result['params']['n_estimators']}, "
            html += f"Depth: {best_result['params']['max_depth']})</p>"
        
        results_widget.value = html

class WaferVisualization3D:
    """3D visualization widget for wafer defect patterns."""
    
    def __init__(self):
        if not HAS_PLOTLY or not HAS_WIDGETS:
            raise ImportError("Required dependencies not available")
        
        self.wafer_data = self._generate_wafer_data()
    
    def _generate_wafer_data(self):
        """Generate synthetic wafer defect data."""
        np.random.seed(RANDOM_SEED)
        
        # Create circular wafer layout
        n_points = 1000
        r = np.random.uniform(0, 100, n_points)  # Radius
        theta = np.random.uniform(0, 2*np.pi, n_points)  # Angle
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Add defect probability based on position
        defect_prob = 0.1 + 0.05 * (r / 100) + 0.03 * np.random.random(n_points)
        
        # Create defects
        defects = np.random.random(n_points) < defect_prob
        
        # Add process parameters
        temperature = 250 + 50 * np.random.random(n_points)
        pressure = 1.0 + 0.5 * np.random.random(n_points)
        flow_rate = 100 + 20 * np.random.random(n_points)
        
        return pd.DataFrame({
            'x': x, 'y': y, 'r': r, 'theta': theta,
            'defect': defects,
            'temperature': temperature,
            'pressure': pressure,
            'flow_rate': flow_rate
        })
    
    def create_widget(self):
        """Create 3D wafer visualization widget."""
        # Parameter controls
        color_by = widgets.Dropdown(
            options=['defect', 'temperature', 'pressure', 'flow_rate'],
            value='defect',
            description='Color by:',
            style={'description_width': 'initial'}
        )
        
        show_defects_only = widgets.Checkbox(
            value=False,
            description='Show defects only',
            style={'description_width': 'initial'}
        )
        
        opacity = widgets.FloatSlider(
            value=0.7, min=0.1, max=1.0, step=0.1,
            description='Opacity:', style={'description_width': 'initial'}
        )
        
        # Output for plot
        plot_output = widgets.Output()
        
        def update_plot(*args):
            """Update the 3D visualization."""
            with plot_output:
                clear_output(wait=True)
                
                data = self.wafer_data.copy()
                if show_defects_only.value:
                    data = data[data['defect']]
                
                if color_by.value == 'defect':
                    colors = data['defect'].astype(int)
                    colorscale = 'Reds'
                    title_suffix = "Defect Status"
                else:
                    colors = data[color_by.value]
                    colorscale = 'Viridis'
                    title_suffix = color_by.value.replace('_', ' ').title()
                
                # Create 3D scatter plot
                fig = go.Figure(data=go.Scatter3d(
                    x=data['x'], y=data['y'], z=data['r'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors,
                        colorscale=colorscale,
                        opacity=opacity.value,
                        colorbar=dict(title=title_suffix)
                    ),
                    text=[f"Position: ({x:.1f}, {y:.1f})<br>"
                          f"Radius: {r:.1f}<br>"
                          f"Defect: {'Yes' if d else 'No'}<br>"
                          f"Temp: {t:.1f}°C<br>"
                          f"Pressure: {p:.2f} atm"
                          for x, y, r, d, t, p in zip(
                              data['x'], data['y'], data['r'], 
                              data['defect'], data['temperature'], data['pressure']
                          )],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'3D Wafer Analysis - Colored by {title_suffix}',
                    scene=dict(
                        xaxis_title='X Position (mm)',
                        yaxis_title='Y Position (mm)',
                        zaxis_title='Radial Distance (mm)',
                        bgcolor='white'
                    ),
                    width=700, height=500
                )
                
                fig.show()
        
        # Connect event handlers
        color_by.observe(update_plot, names='value')
        show_defects_only.observe(update_plot, names='value')
        opacity.observe(update_plot, names='value')
        
        # Initial plot
        update_plot()
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML("<h3>🔬 3D Wafer Defect Visualization</h3>"),
            color_by, show_defects_only, opacity
        ])
        
        return widgets.VBox([controls, plot_output])

class ProcessParameterExplorer:
    """Interactive widget for exploring semiconductor process parameters."""
    
    def __init__(self):
        if not HAS_WIDGETS:
            raise ImportError("ipywidgets not available")
        
        self.process_data = self._generate_process_data()
    
    def _generate_process_data(self):
        """Generate synthetic process parameter data."""
        np.random.seed(RANDOM_SEED)
        
        n_samples = 500
        temperature = np.random.normal(250, 20, n_samples)
        pressure = np.random.normal(1.0, 0.1, n_samples)
        flow_rate = np.random.normal(100, 10, n_samples)
        
        # Yield depends on parameters
        yield_base = 85
        temp_effect = -0.1 * (temperature - 250) ** 2 / 100
        pressure_effect = -5 * (pressure - 1.0) ** 2
        flow_effect = -0.01 * (flow_rate - 100) ** 2
        
        yield_pct = yield_base + temp_effect + pressure_effect + flow_effect
        yield_pct += np.random.normal(0, 2, n_samples)  # Add noise
        yield_pct = np.clip(yield_pct, 0, 100)
        
        return pd.DataFrame({
            'temperature': temperature,
            'pressure': pressure,
            'flow_rate': flow_rate,
            'yield': yield_pct
        })
    
    def create_widget(self):
        """Create process parameter exploration widget."""
        # Parameter range controls
        temp_range = widgets.FloatRangeSlider(
            value=[200, 300], min=150, max=350, step=5,
            description='Temperature (°C):',
            style={'description_width': 'initial'}
        )
        
        pressure_range = widgets.FloatRangeSlider(
            value=[0.5, 1.5], min=0.3, max=2.0, step=0.1,
            description='Pressure (atm):',
            style={'description_width': 'initial'}
        )
        
        flow_range = widgets.FloatRangeSlider(
            value=[80, 120], min=50, max=150, step=5,
            description='Flow Rate (sccm):',
            style={'description_width': 'initial'}
        )
        
        # Analysis type
        analysis_type = widgets.Dropdown(
            options=['Yield vs Temperature', 'Yield vs Pressure', 'Yield vs Flow Rate', 'Parameter Correlation'],
            value='Yield vs Temperature',
            description='Analysis:',
            style={'description_width': 'initial'}
        )
        
        # Output area
        plot_output = widgets.Output()
        stats_output = widgets.Output()
        
        def update_analysis(*args):
            """Update the analysis based on current settings."""
            # Filter data based on ranges
            filtered_data = self.process_data[
                (self.process_data['temperature'] >= temp_range.value[0]) &
                (self.process_data['temperature'] <= temp_range.value[1]) &
                (self.process_data['pressure'] >= pressure_range.value[0]) &
                (self.process_data['pressure'] <= pressure_range.value[1]) &
                (self.process_data['flow_rate'] >= flow_range.value[0]) &
                (self.process_data['flow_rate'] <= flow_range.value[1])
            ].copy()
            
            with plot_output:
                clear_output(wait=True)
                
                plt.figure(figsize=(10, 6))
                
                if analysis_type.value == 'Yield vs Temperature':
                    plt.scatter(filtered_data['temperature'], filtered_data['yield'], alpha=0.6)
                    plt.xlabel('Temperature (°C)')
                    plt.ylabel('Yield (%)')
                    plt.title('Yield vs Temperature')
                elif analysis_type.value == 'Yield vs Pressure':
                    plt.scatter(filtered_data['pressure'], filtered_data['yield'], alpha=0.6)
                    plt.xlabel('Pressure (atm)')
                    plt.ylabel('Yield (%)')
                    plt.title('Yield vs Pressure')
                elif analysis_type.value == 'Yield vs Flow Rate':
                    plt.scatter(filtered_data['flow_rate'], filtered_data['yield'], alpha=0.6)
                    plt.xlabel('Flow Rate (sccm)')
                    plt.ylabel('Yield (%)')
                    plt.title('Yield vs Flow Rate')
                else:  # Parameter Correlation
                    corr_matrix = filtered_data[['temperature', 'pressure', 'flow_rate', 'yield']].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                    plt.title('Parameter Correlation Matrix')
                
                plt.tight_layout()
                plt.show()
            
            with stats_output:
                clear_output(wait=True)
                
                print(f"📊 Analysis Results ({len(filtered_data)} samples)")
                print("=" * 40)
                print(f"Average Yield: {filtered_data['yield'].mean():.2f}%")
                print(f"Yield Std Dev: {filtered_data['yield'].std():.2f}%")
                print(f"Min Yield: {filtered_data['yield'].min():.2f}%")
                print(f"Max Yield: {filtered_data['yield'].max():.2f}%")
                
                # Optimal conditions
                best_idx = filtered_data['yield'].idxmax()
                best_sample = filtered_data.loc[best_idx]
                print(f"\n🎯 Best Conditions in Current Range:")
                print(f"  Temperature: {best_sample['temperature']:.1f}°C")
                print(f"  Pressure: {best_sample['pressure']:.2f} atm")
                print(f"  Flow Rate: {best_sample['flow_rate']:.1f} sccm")
                print(f"  Yield: {best_sample['yield']:.2f}%")
        
        # Connect event handlers
        temp_range.observe(update_analysis, names='value')
        pressure_range.observe(update_analysis, names='value')
        flow_range.observe(update_analysis, names='value')
        analysis_type.observe(update_analysis, names='value')
        
        # Initial analysis
        update_analysis()
        
        # Layout
        controls = widgets.VBox([
            widgets.HTML("<h3>⚙️ Process Parameter Explorer</h3>"),
            temp_range, pressure_range, flow_range, analysis_type
        ])
        
        return widgets.VBox([controls, plot_output, stats_output])

# Convenience functions for easy use
def create_ml_parameter_widget(problem_type: str = 'classification'):
    """Create ML parameter tuning widget."""
    if not HAS_WIDGETS:
        print("❌ ipywidgets not available. Please install: pip install ipywidgets")
        return None
        
    tuner = InteractiveMLTuner(problem_type)
    return tuner.create_widget()

def create_wafer_visualization_widget():
    """Create 3D wafer visualization widget."""
    if not HAS_PLOTLY or not HAS_WIDGETS:
        print("❌ Required dependencies not available.")
        print("Please install: pip install plotly ipywidgets")
        return None
        
    viz = WaferVisualization3D()
    return viz.create_widget()

def create_process_parameter_widget():
    """Create process parameter exploration widget."""
    if not HAS_WIDGETS:
        print("❌ ipywidgets not available. Please install: pip install ipywidgets")
        return None
        
    explorer = ProcessParameterExplorer()
    return explorer.create_widget()

def display_widget_gallery():
    """Display a gallery of all available widgets."""
    if not HAS_WIDGETS:
        print("❌ ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    gallery_html = """
    <div style="border: 2px solid #4CAF50; padding: 20px; margin: 10px; border-radius: 10px;">
        <h2>🎮 Interactive Learning Widget Gallery</h2>
        <p>Choose from the following interactive widgets to enhance your learning experience:</p>
        
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; width: 300px;">
                <h4>🔧 ML Parameter Tuning</h4>
                <p>Experiment with hyperparameters and see real-time model performance.</p>
                <code>create_ml_parameter_widget()</code>
            </div>
            
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; width: 300px;">
                <h4>🔬 3D Wafer Visualization</h4>
                <p>Explore wafer defect patterns in interactive 3D space.</p>
                <code>create_wafer_visualization_widget()</code>
            </div>
            
            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; width: 300px;">
                <h4>⚙️ Process Parameter Explorer</h4>
                <p>Analyze the relationship between process parameters and yield.</p>
                <code>create_process_parameter_widget()</code>
            </div>
        </div>
        
        <h3>📋 Assessment Integration</h3>
        <p>These widgets integrate with the assessment system to track your learning progress:</p>
        <ul>
            <li>Parameter exploration counts toward practical skills</li>
            <li>Widget interactions are logged for progress tracking</li>
            <li>Achievement badges unlock based on widget usage</li>
        </ul>
    </div>
    """
    
    display(HTML(gallery_html))

# Example usage for testing
if __name__ == "__main__":
    print("Interactive Widgets Module")
    print("=" * 40)
    print("Available widgets:")
    print("1. create_ml_parameter_widget()")
    print("2. create_wafer_visualization_widget()")
    print("3. create_process_parameter_widget()")
    print("4. display_widget_gallery()")
    
    if HAS_WIDGETS:
        print("\n✅ ipywidgets available")
    else:
        print("\n❌ ipywidgets not available")
        
    if HAS_PLOTLY:
        print("✅ plotly available")
    else:
        print("❌ plotly not available")
        
    if HAS_SKLEARN:
        print("✅ scikit-learn available")
    else:
        print("❌ scikit-learn not available")