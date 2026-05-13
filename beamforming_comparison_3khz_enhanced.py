#!/usr/bin/env python3
"""
Enhanced Beamforming Comparison for 3kHz Signal
Compare beamforming performance with dB scale and Gain/SNR calculation

For UMA-16 array with 3kHz signal source at center
"""

import numpy as np
from acoular import *
import matplotlib.pyplot as plt
import os

class EnhancedBeamformingComparison:
    """
    Enhanced beamforming comparison with dB scale and Gain/SNR analysis
    """
    
    def __init__(self, xml_file, audio_file):
        """
        Initialize comparison
        
        Args:
            xml_file: UMA-16 array XML file
            audio_file: Audio file with 3kHz signal
        """
        self.xml_file = xml_file
        self.audio_file = audio_file
        
        # Acoular components
        self.mic_geom = None
        self.time_samples = None
        
        # Results storage
        self.results = {}
        self.reference_noise_level = None
        
        # Setup
        self.setup_acoular()
    
    def setup_acoular(self):
        """Setup Acoular components"""
        print("Setting up Acoular components...")
        
        # Load microphone geometry
        self.mic_geom = MicGeom(file=self.xml_file)
        print(f"Loaded {self.mic_geom.pos.shape[1]} microphones")
        
        # Load audio data
        self.time_samples = TimeSamples(file=self.audio_file)
        print(f"Audio loaded: {self.time_samples.sample_freq} Hz")
        
        # Estimate noise level from a quiet region (if available)
        self.estimate_noise_level()
    
    def estimate_noise_level(self):
        """
        Estimate background noise level for SNR calculation
        """
        try:
            # Get a sample of the audio data
            sample_gen = self.time_samples.result()
            sample_block = next(sample_gen)
            
            # Use the minimum energy across channels as noise estimate
            channel_energies = np.mean(sample_block**2, axis=1)
            self.reference_noise_level = np.min(channel_energies)
            
            print(f"Estimated noise level: {self.reference_noise_level:.2e}")
            
        except Exception as e:
            print(f"Could not estimate noise level: {e}")
            self.reference_noise_level = 1e-6  # Default small value
    
    def create_test_points(self):
        """
        Create test points for comparison
        
        Returns:
            Dictionary of test points
        """
        test_points = {
            'center': (0.0, 0.0),           # True source location
            'front': (0.0, 0.2),           # Front of array
            'back': (0.0, -0.2),           # Back of array
            'right': (0.2, 0.0),           # Right side
            'left': (-0.2, 0.0),           # Left side
            'front_right': (0.2, 0.2),    # Front-right diagonal
            'front_left': (-0.2, 0.2),    # Front-left diagonal
            'back_right': (0.2, -0.2),    # Back-right diagonal
            'back_left': (-0.2, -0.2),    # Back-left diagonal
            'custom': (0.108, 0.195),    # Back-left diagonal
        }
        
        print(f"Created {len(test_points)} test points")
        return test_points
    
    def beamform_at_point(self, focus_point, z_distance=0.6, target_freq=3000):
        """
        Perform enhanced beamforming at specific point
        
        Args:
            focus_point: (x, y) coordinates
            z_distance: Distance from array
            target_freq: Target frequency (3000 Hz)
            
        Returns:
            Dictionary with enhanced results including dB and SNR
        """
        x, y = focus_point
        
        # Create small grid around focus point
        margin = 0.005  # 5mm margin
        grid = RectGrid(
            x_min=x-margin, x_max=x+margin,
            y_min=y-margin, y_max=y+margin,
            z=z_distance,
            increment=0.01
        )
        
        # Create steering vector
        steer = SteeringVector(grid=grid, mics=self.mic_geom)
        
        # Power spectra for frequency analysis
        power_spectra = PowerSpectra(
            source=self.time_samples,
            block_size=1024,
            window='Hanning'
        )
        
        # Test multiple beamformers
        beamformers = {}
        
        # BeamformerBase
        try:
            bf_base = BeamformerBase(freq_data=power_spectra, steer=steer)
            power_base = bf_base.synthetic(target_freq, 1)
            beamformers['base'] = {
                'power': np.max(power_base),
                'power_array': power_base
            }
        except Exception as e:
            print(f"BeamformerBase error: {e}")
            beamformers['base'] = None
        
        # BeamformerCapon
        try:
            bf_capon = BeamformerCapon(freq_data=power_spectra, steer=steer)
            power_capon = bf_capon.synthetic(target_freq, 1)
            beamformers['capon'] = {
                'power': np.max(power_capon),
                'power_array': power_capon
            }
        except Exception as e:
            print(f"BeamformerCapon error: {e}")
            beamformers['capon'] = None
        
        # BeamformerTime for time signal analysis
        try:
            bf_time = BeamformerTime(source=self.time_samples, steer=steer)
            time_gen = bf_time.result()
            time_blocks = []
            for i, block in enumerate(time_gen):
                time_blocks.append(block)
                if i >= 5:  # Limit to 5 blocks
                    break
            
            if time_blocks:
                time_signal = np.vstack(time_blocks)
                signal_energy = np.mean(time_signal**2)
                
                # Frequency analysis of time signal
                fft_signal = np.fft.fft(time_signal.flatten())
                freqs = np.fft.fftfreq(len(fft_signal), 1/self.time_samples.sample_freq)
                
                # Find peak around target frequency
                freq_mask = (freqs >= target_freq-200) & (freqs <= target_freq+200)
                if np.any(freq_mask):
                    freq_power_target = np.mean(np.abs(fft_signal[freq_mask])**2)
                else:
                    freq_power_target = 0
                
                beamformers['time'] = {
                    'power': signal_energy,
                    'freq_power': freq_power_target,
                    'time_signal': time_signal
                }
            else:
                beamformers['time'] = None
                
        except Exception as e:
            print(f"BeamformerTime error: {e}")
            beamformers['time'] = None
        
        # Calculate enhanced metrics
        result = {
            'focus_point': focus_point,
            'grid_size': grid.size,
            'beamformers': beamformers
        }
        
        # Add dB and SNR calculations
        for bf_name, bf_result in beamformers.items():
            if bf_result is not None:
                power = bf_result['power']
                
                # Convert to dB
                power_db = 10 * np.log10(power + 1e-12)  # Add small value to avoid log(0)
                
                # Calculate SNR (Signal to Noise Ratio)
                if self.reference_noise_level > 0:
                    snr_linear = power / self.reference_noise_level
                    snr_db = 10 * np.log10(snr_linear + 1e-12)
                else:
                    snr_linear = float('inf')
                    snr_db = float('inf')
                
                # Add to result
                result[f'{bf_name}_power'] = power
                result[f'{bf_name}_power_db'] = power_db
                result[f'{bf_name}_snr_linear'] = snr_linear
                result[f'{bf_name}_snr_db'] = snr_db
            else:
                # Set default values for failed beamformers
                result[f'{bf_name}_power'] = 0
                result[f'{bf_name}_power_db'] = -np.inf
                result[f'{bf_name}_snr_linear'] = 0
                result[f'{bf_name}_snr_db'] = -np.inf
        
        return result
    
    def run_comparison(self, z_distance=0.6):
        """
        Run enhanced beamforming comparison at all test points
        
        Args:
            z_distance: Distance from array
        """
        print(f"\nRunning enhanced beamforming comparison at z={z_distance}m...")
        
        test_points = self.create_test_points()
        
        for name, point in test_points.items():
            print(f"\nTesting point '{name}' at {point}")
            
            try:
                result = self.beamform_at_point(point, z_distance)
                self.results[name] = result
                
                # Print enhanced results
                for bf_type in ['base', 'capon', 'time']:
                    power = result.get(f'{bf_type}_power', 0)
                    power_db = result.get(f'{bf_type}_power_db', -np.inf)
                    snr_db = result.get(f'{bf_type}_snr_db', -np.inf)
                    
                    if power > 0:
                        print(f"  {bf_type.upper()}: Power={power:.2e}, dB={power_db:.1f}, SNR={snr_db:.1f}dB")
                    else:
                        print(f"  {bf_type.upper()}: Failed")
                
            except Exception as e:
                print(f"  Error: {e}")
                self.results[name] = None
    
    def calculate_gain_factors(self):
        """
        Calculate gain factors relative to center (true source)
        """
        if 'center' not in self.results or self.results['center'] is None:
            print("Cannot calculate gain factors - no center reference!")
            return
        
        center_result = self.results['center']
        
        print(f"\n" + "="*80)
        print("GAIN FACTOR ANALYSIS (Relative to Center)")
        print("="*80)
        
        for bf_type in ['base', 'capon', 'time']:
            center_power = center_result.get(f'{bf_type}_power', 0)
            
            if center_power > 0:
                print(f"\n{bf_type.upper()} Beamformer Gain Factors:")
                print(f"{'Point':<12} {'Power':<12} {'Power(dB)':<10} {'Gain':<8} {'Gain(dB)':<10}")
                print("-" * 60)
                
                for point_name, result in self.results.items():
                    if result is not None:
                        point_power = result.get(f'{bf_type}_power', 0)
                        point_power_db = result.get(f'{bf_type}_power_db', -np.inf)
                        
                        if point_power > 0:
                            gain_linear = point_power / center_power
                            gain_db = 10 * np.log10(gain_linear)
                            
                            print(f"{point_name:<12} {point_power:<12.2e} {point_power_db:<10.1f} {gain_linear:<8.3f} {gain_db:<10.1f}")
                        else:
                            print(f"{point_name:<12} {'Failed':<12} {'-inf':<10} {'0.000':<8} {'-inf':<10}")
            else:
                print(f"\n{bf_type.upper()} Beamformer: Failed at center - cannot calculate gains")
    
    def analyze_results(self):
        """
        Enhanced analysis with dB scale and SNR
        """
        print("\n" + "="*80)
        print("ENHANCED BEAMFORMING COMPARISON ANALYSIS")
        print("="*80)
        
        # Filter valid results
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            print("No valid results to analyze!")
            return
        
        # Analyze each beamformer type
        for bf_type in ['base', 'capon', 'time']:
            print(f"\n{bf_type.upper()} BEAMFORMER ANALYSIS:")
            print("-" * 50)
            
            # Extract metrics for this beamformer
            points = []
            powers = []
            powers_db = []
            snrs_db = []
            
            for point_name, result in valid_results.items():
                power = result.get(f'{bf_type}_power', 0)
                if power > 0:
                    points.append(point_name)
                    powers.append(power)
                    powers_db.append(result.get(f'{bf_type}_power_db', -np.inf))
                    snrs_db.append(result.get(f'{bf_type}_snr_db', -np.inf))
            
            if points:
                # Find best performance
                best_idx = np.argmax(powers)
                best_point = points[best_idx]
                
                print(f"Best performance: {best_point}")
                print(f"  Power: {powers[best_idx]:.2e}")
                print(f"  Power (dB): {powers_db[best_idx]:.1f} dB")
                print(f"  SNR: {snrs_db[best_idx]:.1f} dB")
                
                # Summary table
                print(f"\nDetailed Results:")
                print(f"{'Point':<12} {'Power':<12} {'Power(dB)':<10} {'SNR(dB)':<8}")
                print("-" * 50)
                
                for i, point in enumerate(points):
                    print(f"{point:<12} {powers[i]:<12.2e} {powers_db[i]:<10.1f} {snrs_db[i]:<8.1f}")
            else:
                print(f"No valid results for {bf_type.upper()} beamformer")
        
        # Calculate gain factors
        self.calculate_gain_factors()
    
    def save_enhanced_results(self, output_file="enhanced_beamforming_results.txt"):
        """
        Save enhanced results with dB and SNR to file
        """
        print(f"\nSaving enhanced results to {output_file}...")
        
        with open(output_file, 'w') as f:
            f.write("ENHANCED BEAMFORMING COMPARISON RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Test Configuration:\n")
            f.write(f"- Array: UMA-16 ({self.mic_geom.pos.shape[1]} microphones)\n")
            f.write(f"- Sample rate: {self.time_samples.sample_freq} Hz\n")
            f.write(f"- Target frequency: 3000 Hz\n")
            f.write(f"- Source location: Center (0.0, 0.0)\n")
            f.write(f"- Noise level estimate: {self.reference_noise_level:.2e}\n\n")
            
            # Results for each beamformer
            valid_results = {k: v for k, v in self.results.items() if v is not None}
            
            for bf_type in ['base', 'capon', 'time']:
                f.write(f"{bf_type.upper()} BEAMFORMER RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Point':<12} {'Power':<12} {'dB':<8} {'SNR(dB)':<8} {'Gain':<8}\n")
                f.write("-" * 60 + "\n")
                
                # Get center reference for gain calculation
                center_power = 0
                if 'center' in valid_results:
                    center_power = valid_results['center'].get(f'{bf_type}_power', 0)
                
                for point_name, result in valid_results.items():
                    power = result.get(f'{bf_type}_power', 0)
                    power_db = result.get(f'{bf_type}_power_db', -np.inf)
                    snr_db = result.get(f'{bf_type}_snr_db', -np.inf)
                    
                    if power > 0 and center_power > 0:
                        gain = power / center_power
                    else:
                        gain = 0
                    
                    f.write(f"{point_name:<12} {power:<12.2e} {power_db:<8.1f} {snr_db:<8.1f} {gain:<8.3f}\n")
                
                f.write("\n")
        
        print(f"Enhanced results saved to {output_file}")
    
    def create_enhanced_visualization(self, output_file="enhanced_beamforming_comparison.png"):
        """
        Create enhanced visualization with dB scale
        """
        try:
            valid_results = {k: v for k, v in self.results.items() if v is not None}
            
            if not valid_results:
                print("No valid results to visualize!")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Enhanced Beamforming Comparison - 3kHz Signal (dB Scale)', fontsize=16)
            
            beamformer_types = ['base', 'capon', 'time']
            
            for i, bf_type in enumerate(beamformer_types):
                # Extract data for this beamformer
                points = []
                powers_db = []
                snrs_db = []
                
                for point_name, result in valid_results.items():
                    power = result.get(f'{bf_type}_power', 0)
                    if power > 0:
                        points.append(point_name)
                        powers_db.append(result.get(f'{bf_type}_power_db', -np.inf))
                        snrs_db.append(result.get(f'{bf_type}_snr_db', -np.inf))
                
                if points:
                    # Power in dB
                    axes[0, i].bar(points, powers_db)
                    axes[0, i].set_title(f'{bf_type.upper()} - Power (dB)')
                    axes[0, i].set_ylabel('Power (dB)')
                    axes[0, i].tick_params(axis='x', rotation=45)
                    axes[0, i].grid(True, alpha=0.3)
                    
                    # SNR in dB
                    axes[1, i].bar(points, snrs_db)
                    axes[1, i].set_title(f'{bf_type.upper()} - SNR (dB)')
                    axes[1, i].set_ylabel('SNR (dB)')
                    axes[1, i].tick_params(axis='x', rotation=45)
                    axes[1, i].grid(True, alpha=0.3)
                    
                    # Highlight center (true source)
                    if 'center' in points:
                        center_idx = points.index('center')
                        axes[0, i].bar(center_idx, powers_db[center_idx], color='red', alpha=0.7)
                        axes[1, i].bar(center_idx, snrs_db[center_idx], color='red', alpha=0.7)
                else:
                    axes[0, i].text(0.5, 0.5, f'No valid data\nfor {bf_type.upper()}', 
                                   ha='center', va='center', transform=axes[0, i].transAxes)
                    axes[1, i].text(0.5, 0.5, f'No valid data\nfor {bf_type.upper()}', 
                                   ha='center', va='center', transform=axes[1, i].transAxes)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Enhanced visualization saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating enhanced visualization: {e}")


def main():
    """
    Main function
    """
    print("="*80)
    print("ENHANCED BEAMFORMING COMPARISON FOR 3KHZ SIGNAL")
    print("With dB Scale and Gain/SNR Analysis")
    print("="*80)
    
    # File paths
    xml_file = "upload/uma16_array.xml"
    audio_file = "12k.h5"
    
    # Check files
    if not os.path.exists(xml_file):
        print(f"XML file not found: {xml_file}")
        return
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return
    
    # Create enhanced comparison object
    comparison = EnhancedBeamformingComparison(xml_file, audio_file)
    
    # Run comparison
    comparison.run_comparison(z_distance=0.6)
    
    # Analyze results
    comparison.analyze_results()
    
    # Save results
    comparison.save_enhanced_results("enhanced_beamforming_3khz_results.txt")
    
    # Create visualization
    comparison.create_enhanced_visualization("enhanced_beamforming_3khz.png")
    
    print("\n" + "="*80)
    print("ENHANCED COMPARISON COMPLETED!")
    print("="*80)
    print("Files generated:")
    print("- enhanced_beamforming_3khz_results.txt")
    print("- enhanced_beamforming_3khz.png")


if __name__ == "__main__":
    main()

