"""Synthetic dataset generators for semiconductor manufacturing scenarios.

This module provides synthetic data generators for various semiconductor manufacturing
datasets that may not be publicly available but are commonly needed for ML education
and prototyping.

Generators include:
- Time series equipment sensor data
- Process recipe databases with outcome labels  
- High-resolution microscopic defect patterns
- Wafer pattern generators with configurable defect types

All generators follow the same API pattern and produce realistic data distributions
based on domain knowledge from semiconductor manufacturing.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Suppress scientific notation warnings for large datasets
warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_SEED = 42


@dataclass
class DatasetMetadata:
    """Metadata structure for synthetic datasets."""
    name: str
    description: str
    size: int
    features: int
    target_type: str
    domain: str
    created_by: str = "synthetic_generator"
    version: str = "1.0"


class TimeSensorDataGenerator:
    """Generate time series equipment sensor data for semiconductor manufacturing."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def generate(self, 
                 n_samples: int = 10000,
                 n_sensors: int = 50,
                 sequence_length: int = 100,
                 anomaly_rate: float = 0.05,
                 output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate time series sensor data with realistic patterns.
        
        Args:
            n_samples: Number of time series sequences
            n_sensors: Number of sensors per sequence
            sequence_length: Length of each time series
            anomaly_rate: Fraction of sequences with anomalies
            output_dir: Directory to save generated data
            
        Returns:
            Dictionary containing sensor data, labels, and metadata
        """
        print(f"Generating time series sensor data: {n_samples} sequences, {n_sensors} sensors...")
        
        # Generate base sensor patterns (temperature, pressure, flow, etc.)
        sensor_types = ['temperature', 'pressure', 'flow_rate', 'voltage', 'current', 
                       'vibration', 'chemical_conc', 'gas_flow', 'vacuum_level', 'power']
        
        # Create sensor data with realistic patterns
        data = []
        labels = []
        timestamps = []
        
        for seq_idx in range(n_samples):
            # Generate timestamp sequence
            start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(hours=seq_idx * 24)
            time_seq = pd.date_range(start=start_time, periods=sequence_length, freq='1min')
            timestamps.append(time_seq)
            
            # Determine if this sequence has an anomaly
            has_anomaly = self.rng.random() < anomaly_rate
            # Ensure we have enough sequence length for anomaly placement
            min_start = max(5, sequence_length // 10)
            max_start = max(min_start + 1, sequence_length - max(5, sequence_length // 10))
            anomaly_start = self.rng.randint(min_start, max_start) if has_anomaly and max_start > min_start else None
            
            sequence_data = np.zeros((sequence_length, n_sensors))
            
            for sensor_idx in range(n_sensors):
                sensor_type = sensor_types[sensor_idx % len(sensor_types)]
                
                # Generate base signal based on sensor type
                base_signal = self._generate_sensor_signal(sensor_type, sequence_length)
                
                # Add normal process variation
                noise = self.rng.normal(0, 0.1, sequence_length)
                signal = base_signal + noise
                
                # Add anomaly if applicable
                if has_anomaly and anomaly_start is not None:
                    anomaly_length = min(self.rng.randint(2, 10), sequence_length - anomaly_start - 1)
                    anomaly_end = min(anomaly_start + anomaly_length, sequence_length)
                    
                    # Different anomaly types
                    anomaly_type = self.rng.choice(['spike', 'drift', 'stuck', 'noise'])
                    signal = self._inject_anomaly(signal, anomaly_start, anomaly_end, anomaly_type)
                
                sequence_data[:, sensor_idx] = signal
            
            data.append(sequence_data)
            labels.append(1 if has_anomaly else 0)
        
        # Convert to numpy arrays
        data_array = np.array(data)  # Shape: (n_samples, sequence_length, n_sensors)
        labels_array = np.array(labels)
        
        # Create metadata
        metadata = DatasetMetadata(
            name="synthetic_time_series_sensors",
            description="Synthetic semiconductor equipment sensor time series data",
            size=n_samples,
            features=n_sensors * sequence_length,
            target_type="binary_classification",
            domain="semiconductor_manufacturing"
        )
        
        result = {
            'sensor_data': data_array,
            'labels': labels_array,
            'timestamps': timestamps,
            'metadata': metadata,
            'sensor_names': [f"sensor_{i:02d}_{sensor_types[i % len(sensor_types)]}" 
                           for i in range(n_sensors)]
        }
        
        # Save to files if output directory specified
        if output_dir:
            self._save_time_series_data(result, output_dir)
            
        return result
    
    def _generate_sensor_signal(self, sensor_type: str, length: int) -> np.ndarray:
        """Generate realistic sensor signal based on type."""
        t = np.linspace(0, length, length)
        
        if sensor_type == 'temperature':
            # Temperature with slow drift and cyclic component
            base = 25.0 + 10 * np.sin(2 * np.pi * t / 50) + 0.01 * t
            return base + self.rng.normal(0, 0.5, length)
        elif sensor_type == 'pressure':
            # Pressure with step changes
            base = np.ones(length) * 760.0
            for step in range(0, length, length // 5):
                base[step:] += self.rng.normal(0, 5)
            return base + self.rng.normal(0, 1, length)
        elif sensor_type == 'flow_rate':
            # Flow rate with periodic maintenance shutdowns
            base = 50.0 * np.ones(length)
            # Random maintenance windows
            for _ in range(self.rng.randint(0, 3)):
                start = self.rng.randint(0, length - 10)
                base[start:start+10] = 0
            return base + self.rng.normal(0, 2, length)
        else:
            # Generic sensor with trend and noise
            trend = 0.001 * t
            cyclic = 0.5 * np.sin(2 * np.pi * t / 25)
            return trend + cyclic + self.rng.normal(0, 0.2, length)
    
    def _inject_anomaly(self, signal: np.ndarray, start: int, end: int, anomaly_type: str) -> np.ndarray:
        """Inject different types of anomalies into sensor signal."""
        signal = signal.copy()
        
        if anomaly_type == 'spike':
            # Sharp spike in signal
            spike_magnitude = self.rng.uniform(3, 8) * np.std(signal)
            signal[start:end] += spike_magnitude
        elif anomaly_type == 'drift':
            # Gradual drift away from normal
            drift_amount = self.rng.uniform(2, 5) * np.std(signal)
            drift_pattern = np.linspace(0, drift_amount, end - start)
            signal[start:end] += drift_pattern
        elif anomaly_type == 'stuck':
            # Sensor stuck at constant value
            stuck_value = signal[start]
            signal[start:end] = stuck_value
        elif anomaly_type == 'noise':
            # Increased noise level
            noise_factor = self.rng.uniform(3, 10)
            noise = self.rng.normal(0, noise_factor * np.std(signal), end - start)
            signal[start:end] += noise
            
        return signal
    
    def _save_time_series_data(self, data: Dict[str, Any], output_dir: Path):
        """Save time series data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sensor data as compressed numpy array
        np.savez_compressed(
            output_dir / "sensor_data.npz",
            sensor_data=data['sensor_data'],
            labels=data['labels'],
            sensor_names=data['sensor_names']
        )
        
        # Save metadata as JSON
        metadata_dict = {
            'name': data['metadata'].name,
            'description': data['metadata'].description,
            'size': data['metadata'].size,
            'features': data['metadata'].features,
            'target_type': data['metadata'].target_type,
            'domain': data['metadata'].domain,
            'created_by': data['metadata'].created_by,
            'version': data['metadata'].version,
            'sensor_names': data['sensor_names'],
            'anomaly_rate': 0.05,  # Default rate
            'sequence_length': data['sensor_data'].shape[1],
            'n_sensors': data['sensor_data'].shape[2]
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"  [saved] Time series sensor data -> {output_dir}")


class ProcessRecipeGenerator:
    """Generate process recipe databases with outcome labels."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate(self,
                 n_recipes: int = 5000,
                 n_parameters: int = 25,
                 output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate process recipe database with realistic manufacturing parameters.
        
        Args:
            n_recipes: Number of process recipes to generate
            n_parameters: Number of process parameters per recipe
            output_dir: Directory to save generated data
            
        Returns:
            Dictionary containing recipe data, outcomes, and metadata
        """
        print(f"Generating process recipe database: {n_recipes} recipes, {n_parameters} parameters...")
        
        # Define process parameter types commonly used in semiconductor manufacturing
        param_types = [
            'temperature_deg_c', 'pressure_torr', 'gas_flow_sccm', 'rf_power_w',
            'time_seconds', 'etch_rate_nm_min', 'deposition_rate_ang_s', 
            'gas_ratio_percent', 'chamber_spacing_mm', 'substrate_bias_v',
            'plasma_density', 'ion_energy_ev', 'precursor_flow', 'carrier_gas_flow',
            'substrate_temp', 'cooling_rate', 'anneal_temp', 'ramp_rate',
            'vacuum_level', 'contamination_ppm', 'humidity_percent', 'clean_time',
            'seasoning_cycles', 'maintenance_hours', 'operator_experience'
        ]
        
        # Generate parameter names
        param_names = [param_types[i % len(param_types)] + f"_{i}" for i in range(n_parameters)]
        
        # Generate recipe data
        recipes = []
        outcomes = []
        
        for recipe_idx in range(n_recipes):
            recipe = {}
            
            # Generate parameters with realistic ranges based on type
            for i, param_name in enumerate(param_names):
                param_type = param_types[i % len(param_types)]
                recipe[param_name] = self._generate_parameter_value(param_type)
            
            recipes.append(recipe)
            
            # Generate outcome based on parameter quality
            outcome = self._calculate_recipe_outcome(recipe, param_names)
            outcomes.append(outcome)
        
        # Convert to DataFrame
        df_recipes = pd.DataFrame(recipes)
        df_outcomes = pd.DataFrame({
            'yield_percent': [out['yield'] for out in outcomes],
            'defect_density_per_cm2': [out['defect_density'] for out in outcomes],
            'process_time_minutes': [out['process_time'] for out in outcomes],
            'quality_score': [out['quality_score'] for out in outcomes],
            'cost_per_wafer_usd': [out['cost'] for out in outcomes]
        })
        
        # Create metadata
        metadata = DatasetMetadata(
            name="synthetic_process_recipes",
            description="Synthetic semiconductor process recipe database with outcomes",
            size=n_recipes,
            features=n_parameters,
            target_type="multi_target_regression",
            domain="semiconductor_manufacturing"
        )
        
        result = {
            'recipes': df_recipes,
            'outcomes': df_outcomes,
            'metadata': metadata,
            'parameter_descriptions': self._get_parameter_descriptions(param_names)
        }
        
        # Save to files if output directory specified
        if output_dir:
            self._save_recipe_data(result, output_dir)
            
        return result
    
    def _generate_parameter_value(self, param_type: str) -> float:
        """Generate realistic parameter value based on type."""
        if 'temperature' in param_type:
            return self.rng.uniform(20, 800)  # °C
        elif 'pressure' in param_type:
            return self.rng.uniform(0.001, 760)  # Torr
        elif 'flow' in param_type:
            return self.rng.uniform(1, 500)  # sccm
        elif 'power' in param_type:
            return self.rng.uniform(10, 2000)  # Watts
        elif 'time' in param_type:
            return self.rng.uniform(10, 7200)  # seconds
        elif 'rate' in param_type:
            return self.rng.uniform(0.1, 100)  # nm/min or Å/s
        elif 'percent' in param_type or 'ratio' in param_type:
            return self.rng.uniform(0, 100)  # percent
        elif 'voltage' in param_type or 'bias' in param_type:
            return self.rng.uniform(-500, 500)  # volts
        elif 'energy' in param_type:
            return self.rng.uniform(1, 1000)  # eV
        elif 'ppm' in param_type:
            return self.rng.uniform(0.1, 1000)  # ppm
        elif 'experience' in param_type:
            return self.rng.uniform(0, 20)  # years
        else:
            return self.rng.uniform(0, 100)  # generic parameter
    
    def _calculate_recipe_outcome(self, recipe: Dict[str, float], param_names: List[str]) -> Dict[str, float]:
        """Calculate realistic outcomes based on recipe parameters."""
        # Simple model relating parameter quality to outcomes
        param_values = np.array(list(recipe.values()))
        
        # Normalize parameters (rough approximation)
        normalized_params = (param_values - np.mean(param_values)) / (np.std(param_values) + 1e-8)
        
        # Base yield calculation with some parameter interactions
        base_yield = 85.0  # Base yield percentage
        temp_effect = -0.1 * abs(normalized_params[0])  # Temperature sensitivity
        pressure_effect = -0.05 * abs(normalized_params[1])  # Pressure sensitivity
        time_effect = -0.02 * abs(normalized_params[4]) if len(normalized_params) > 4 else 0
        
        # Random process variation
        random_effect = self.rng.normal(0, 2)
        
        yield_pct = max(70.0, min(99.5, 
            base_yield + temp_effect + pressure_effect + time_effect + random_effect))
        
        # Defect density (inversely related to yield)
        defect_density = max(0.1, 10.0 * (100 - yield_pct) / 30 + self.rng.exponential(0.5))
        
        # Process time (based on relevant parameters)
        process_time = max(30, 120 + 0.1 * param_values[4] if len(param_values) > 4 else 120) + self.rng.normal(0, 10)
        
        # Quality score (composite metric)
        quality_score = (yield_pct / 100) * (100 / (defect_density + 1)) * 0.01
        quality_score = max(0.1, min(1.0, quality_score + self.rng.normal(0, 0.05)))
        
        # Cost per wafer (based on complexity and time)
        base_cost = 50.0  # USD
        complexity_cost = np.sum(np.abs(normalized_params)) * 2
        time_cost = process_time * 0.2
        cost = base_cost + complexity_cost + time_cost + self.rng.normal(0, 5)
        
        return {
            'yield': yield_pct,
            'defect_density': defect_density,
            'process_time': process_time,
            'quality_score': quality_score,
            'cost': max(20.0, cost)
        }
    
    def _get_parameter_descriptions(self, param_names: List[str]) -> Dict[str, str]:
        """Generate descriptions for process parameters."""
        descriptions = {}
        for name in param_names:
            if 'temperature' in name:
                descriptions[name] = "Process temperature in degrees Celsius"
            elif 'pressure' in name:
                descriptions[name] = "Chamber pressure in Torr"
            elif 'flow' in name:
                descriptions[name] = "Gas flow rate in sccm (standard cubic centimeters per minute)"
            elif 'power' in name:
                descriptions[name] = "RF power in Watts"
            elif 'time' in name:
                descriptions[name] = "Process time in seconds"
            else:
                descriptions[name] = f"Process parameter: {name}"
        return descriptions
    
    def _save_recipe_data(self, data: Dict[str, Any], output_dir: Path):
        """Save recipe data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save recipes and outcomes as CSV
        data['recipes'].to_csv(output_dir / "recipes.csv", index=False)
        data['outcomes'].to_csv(output_dir / "outcomes.csv", index=False)
        
        # Save metadata and descriptions as JSON
        metadata_dict = {
            'name': data['metadata'].name,
            'description': data['metadata'].description,
            'size': data['metadata'].size,
            'features': data['metadata'].features,
            'target_type': data['metadata'].target_type,
            'domain': data['metadata'].domain,
            'created_by': data['metadata'].created_by,
            'version': data['metadata'].version,
            'parameter_descriptions': data['parameter_descriptions'],
            'outcome_variables': [
                'yield_percent',
                'defect_density_per_cm2', 
                'process_time_minutes',
                'quality_score',
                'cost_per_wafer_usd'
            ]
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"  [saved] Process recipe database -> {output_dir}")


class WaferDefectPatternGenerator:
    """Generate synthetic wafer defect patterns for computer vision tasks."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate(self,
                 n_wafers: int = 1000,
                 map_size: int = 64,
                 defect_types: Optional[List[str]] = None,
                 output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate synthetic wafer defect patterns.
        
        Args:
            n_wafers: Number of wafer maps to generate
            map_size: Size of wafer maps (map_size x map_size)
            defect_types: List of defect types to generate
            output_dir: Directory to save generated data
            
        Returns:
            Dictionary containing wafer maps, labels, and metadata
        """
        if defect_types is None:
            defect_types = ['None', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 
                          'Loc', 'Random', 'Scratch', 'Near-full']
        
        print(f"Generating wafer defect patterns: {n_wafers} wafers, {map_size}x{map_size} maps...")
        
        wafer_maps = []
        labels = []
        
        for i in range(n_wafers):
            # Select defect type
            defect_type = self.rng.choice(defect_types)
            
            # Generate wafer map
            wafer_map = self._generate_wafer_pattern(map_size, defect_type)
            
            wafer_maps.append(wafer_map)
            labels.append(defect_type)
        
        wafer_maps = np.array(wafer_maps)
        
        # Create metadata
        metadata = DatasetMetadata(
            name="synthetic_wafer_defect_patterns",
            description="Synthetic wafer defect patterns for computer vision",
            size=n_wafers,
            features=map_size * map_size,
            target_type="multi_class_classification",
            domain="semiconductor_manufacturing"
        )
        
        result = {
            'wafer_maps': wafer_maps,
            'labels': labels,
            'defect_types': defect_types,
            'map_size': map_size,
            'metadata': metadata
        }
        
        # Save to files if output directory specified
        if output_dir:
            self._save_wafer_data(result, output_dir)
            
        return result
    
    def _generate_wafer_pattern(self, size: int, defect_type: str) -> np.ndarray:
        """Generate a single wafer defect pattern."""
        # Create base wafer (circular)
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= (center - 1) ** 2
        
        # Base wafer (good dies = 1, no die = 0)
        wafer = mask.astype(float)
        
        if defect_type == 'None':
            # No defects, just add some random good/bad dies
            noise_mask = self.rng.random((size, size)) < 0.05
            wafer[mask & noise_mask] = 0
            
        elif defect_type == 'Center':
            # Central defect cluster
            center_radius = size // 8
            center_mask = (x - center) ** 2 + (y - center) ** 2 <= center_radius ** 2
            wafer[center_mask] = 0
            
        elif defect_type == 'Donut':
            # Ring-shaped defect
            inner_radius = size // 4
            outer_radius = size // 3
            ring_mask = ((x - center) ** 2 + (y - center) ** 2 >= inner_radius ** 2) & \
                       ((x - center) ** 2 + (y - center) ** 2 <= outer_radius ** 2)
            wafer[ring_mask] = 0
            
        elif defect_type == 'Edge-Loc':
            # Edge localized defect
            edge_angle = self.rng.uniform(0, 2 * np.pi)
            edge_x = center + int((center - 5) * np.cos(edge_angle))
            edge_y = center + int((center - 5) * np.sin(edge_angle))
            edge_radius = size // 10
            edge_mask = (x - edge_x) ** 2 + (y - edge_y) ** 2 <= edge_radius ** 2
            wafer[edge_mask & mask] = 0
            
        elif defect_type == 'Edge-Ring':
            # Ring near edge
            ring_radius = center - size // 8
            ring_width = size // 16
            ring_mask = ((x - center) ** 2 + (y - center) ** 2 >= (ring_radius - ring_width) ** 2) & \
                       ((x - center) ** 2 + (y - center) ** 2 <= (ring_radius + ring_width) ** 2)
            wafer[ring_mask] = 0
            
        elif defect_type == 'Loc':
            # Localized defect cluster
            loc_x = self.rng.randint(size // 4, 3 * size // 4)
            loc_y = self.rng.randint(size // 4, 3 * size // 4)
            loc_radius = self.rng.randint(size // 12, size // 6)
            loc_mask = (x - loc_x) ** 2 + (y - loc_y) ** 2 <= loc_radius ** 2
            wafer[loc_mask & mask] = 0
            
        elif defect_type == 'Random':
            # Random scattered defects
            random_defects = self.rng.random((size, size)) < 0.3
            wafer[random_defects & mask] = 0
            
        elif defect_type == 'Scratch':
            # Linear scratch defect
            scratch_angle = self.rng.uniform(0, np.pi)
            scratch_width = 2
            # Create line through center
            cos_a, sin_a = np.cos(scratch_angle), np.sin(scratch_angle)
            dist_to_line = np.abs((x - center) * (-sin_a) + (y - center) * cos_a)
            scratch_mask = dist_to_line <= scratch_width
            wafer[scratch_mask & mask] = 0
            
        elif defect_type == 'Near-full':
            # Nearly full wafer defect (most dies bad)
            good_dies = self.rng.random((size, size)) < 0.2
            wafer[mask & ~good_dies] = 0
        
        return wafer
    
    def _save_wafer_data(self, data: Dict[str, Any], output_dir: Path):
        """Save wafer data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save wafer maps as compressed numpy array
        np.savez_compressed(
            output_dir / "wafer_maps.npz",
            wafer_maps=data['wafer_maps'],
            labels=data['labels']
        )
        
        # Save metadata as JSON
        metadata_dict = {
            'name': data['metadata'].name,
            'description': data['metadata'].description,
            'size': data['metadata'].size,
            'features': data['metadata'].features,
            'target_type': data['metadata'].target_type,
            'domain': data['metadata'].domain,
            'created_by': data['metadata'].created_by,
            'version': data['metadata'].version,
            'defect_types': data['defect_types'],
            'map_size': data['map_size'],
            'class_distribution': {defect_type: data['labels'].count(defect_type) 
                                 for defect_type in data['defect_types']}
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"  [saved] Wafer defect patterns -> {output_dir}")


def generate_all_synthetic_datasets(output_base_dir: Path = Path("datasets/synthetic")) -> None:
    """Generate all synthetic datasets for the repository."""
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating all synthetic semiconductor datasets...")
    
    # Generate time series sensor data
    time_gen = TimeSensorDataGenerator()
    time_gen.generate(n_samples=1000, output_dir=output_base_dir / "time_series_sensors")
    
    # Generate process recipes
    recipe_gen = ProcessRecipeGenerator()
    recipe_gen.generate(n_recipes=2000, output_dir=output_base_dir / "process_recipes")
    
    # Generate wafer defect patterns
    wafer_gen = WaferDefectPatternGenerator()
    wafer_gen.generate(n_wafers=1500, output_dir=output_base_dir / "wafer_defect_patterns")
    
    print("All synthetic datasets generated successfully!")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic semiconductor datasets")
    parser.add_argument("--output-dir", default="datasets/synthetic", 
                       help="Base output directory for synthetic datasets")
    parser.add_argument("--dataset-type", choices=["time_series", "recipes", "wafer_patterns", "all"],
                       default="all", help="Type of dataset to generate")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.dataset_type == "time_series":
        gen = TimeSensorDataGenerator()
        gen.generate(output_dir=output_dir / "time_series_sensors")
    elif args.dataset_type == "recipes":
        gen = ProcessRecipeGenerator()
        gen.generate(output_dir=output_dir / "process_recipes")
    elif args.dataset_type == "wafer_patterns":
        gen = WaferDefectPatternGenerator()
        gen.generate(output_dir=output_dir / "wafer_defect_patterns")
    elif args.dataset_type == "all":
        generate_all_synthetic_datasets(output_dir)