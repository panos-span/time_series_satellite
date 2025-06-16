import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show, plotting_extent
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore')

class KastoriaRasterAnalyzer:
    """
    Kastoria raster analysis using PROPERLY implemented rasterio.mask functionality.
    This version demonstrates the correct way to use rasterio.mask.
    """
    
    def __init__(self, sentinel_path="Kastoria.tif", study_area_path="kastoria_study_area.geojson"):
        """
        Initialize the raster analyzer.
        """
        self.sentinel_path = sentinel_path
        self.study_area_path = study_area_path
        self.sentinel_data = None
        self.study_area = None
        self.time_series_data = {}
        
        # Standard Sentinel-2 band configuration (assuming 10 bands per time step)
        self.sentinel_bands = {
            'Blue': 0,      # Band 2 (490nm)
            'Green': 1,     # Band 3 (560nm) 
            'Red': 2,       # Band 4 (665nm)
            'RedEdge1': 3,  # Band 5 (705nm)
            'RedEdge2': 4,  # Band 6 (740nm)
            'RedEdge3': 5,  # Band 7 (783nm)
            'NIR': 6,       # Band 8 (842nm)
            'RedEdge4': 7,  # Band 8A (865nm)
            'SWIR1': 8,     # Band 11 (1610nm)
            'SWIR2': 9      # Band 12 (2190nm)
        }
        
    def load_data(self):
        """
        Load Sentinel-2 raster data and study area polygon.
        """
        print("🔄 Loading Sentinel-2 data and study area...")
        
        # Load raster data
        with rasterio.open(self.sentinel_path) as src:
            self.sentinel_data = {
                'data': src.read(),
                'meta': src.meta.copy(),
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'shape': src.shape,
                'count': src.count
            }
        
        # Load study area polygon
        self.study_area = gpd.read_file(self.study_area_path)
        
        # Determine band structure (240 bands total)
        total_bands = self.sentinel_data['count']
        bands_per_timestep = 10  # Assuming standard Sentinel-2 subset
        self.num_timesteps = total_bands // bands_per_timestep
        
        self.band_structure = {
            'total_bands': total_bands,
            'bands_per_timestep': bands_per_timestep,
            'num_timesteps': self.num_timesteps
        }
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Total bands: {total_bands}")
        print(f"   - Time steps: {self.num_timesteps}")
        print(f"   - Bands per time step: {bands_per_timestep}")
        print(f"   - Raster shape: {self.sentinel_data['shape']}")
        print(f"   - Study area CRS: {self.study_area.crs}")
        print(f"   - Raster CRS: {self.sentinel_data['crs']}")
        
        return self.sentinel_data, self.study_area
    
    def get_band_index(self, timestep, band_name):
        """
        Get the absolute band index for a specific timestep and band name.
        """
        if band_name not in self.sentinel_bands:
            raise ValueError(f"Band '{band_name}' not recognized. Available: {list(self.sentinel_bands.keys())}")
        
        relative_band_idx = self.sentinel_bands[band_name]
        absolute_band_idx = timestep * self.band_structure['bands_per_timestep'] + relative_band_idx
        
        return absolute_band_idx
    
    def prepare_geometries(self):
        """
        Prepare geometries for rasterio.mask - ensuring correct format and CRS.
        """
        print("🔧 Preparing geometries for masking...")
        
        # Reproject study area to match raster CRS
        study_area_proj = self.study_area.to_crs(self.sentinel_data['crs'])
        
        # Extract geometries in the correct format for rasterio.mask
        # rasterio.mask expects an iterable of geometries
        geometries = [geom for geom in study_area_proj.geometry]
        
        print(f"   ✅ Geometries prepared:")
        print(f"      - Number of geometries: {len(geometries)}")
        print(f"      - Geometry type: {geometries[0].geom_type}")
        print(f"      - Geometry CRS: {study_area_proj.crs}")
        print(f"      - Raster CRS: {self.sentinel_data['crs']}")
        print(f"      - CRS match: {study_area_proj.crs == self.sentinel_data['crs']}")
        
        return geometries
    
    def mask_band_correctly(self, band_index, geometries):
        """
        Mask a single band using rasterio.mask correctly.
        
        Parameters:
        -----------
        band_index : int
            1-based band index for rasterio
        geometries : list
            List of geometries for masking
        """
        with rasterio.open(self.sentinel_path) as src:
            try:
                # Use rasterio.mask correctly with proper parameters
                masked_array, masked_transform = mask(
                    dataset=src,
                    shapes=geometries,  # Must be iterable of geometries
                    crop=True,         # Crop to the extent of geometries
                    indexes=[band_index],  # Specify which band(s) to mask (1-based)
                    nodata=np.nan,     # Set nodata value
                    filled=True,       # Return filled array (not masked array)
                    all_touched=False  # Only pixels whose center is in polygon
                )
                
                # masked_array has shape (1, height, width) for single band
                # Extract the 2D array
                masked_2d = masked_array[0]
                
                print(f"      Band {band_index}: {masked_2d.shape} (successful)")
                
                return masked_2d, masked_transform
                
            except Exception as e:
                print(f"      Band {band_index}: Failed - {e}")
                return None, None
    
    def extract_bands_for_timestep(self, timestep, required_bands=['Red', 'Green', 'Blue', 'NIR', 'SWIR1']):
        """
        Extract and mask required bands for a specific timestep using CORRECT rasterio.mask.
        """
        print(f"📦 Extracting bands for timestep {timestep}...")
        
        # Prepare geometries once
        geometries = self.prepare_geometries()
        
        bands_data = {}
        masked_transform = None
        
        for band_name in required_bands:
            try:
                if band_name not in self.sentinel_bands:
                    print(f"   ⚠️ {band_name} not in band mapping, skipping...")
                    continue
                    
                # Get 0-based band index and convert to 1-based for rasterio
                band_idx_0based = self.get_band_index(timestep, band_name)
                band_idx_1based = band_idx_0based + 1
                
                print(f"   Processing {band_name} (index {band_idx_0based} -> {band_idx_1based})...")
                
                # Use correct rasterio.mask implementation
                masked_data, transform = self.mask_band_correctly(band_idx_1based, geometries)
                
                if masked_data is not None:
                    bands_data[band_name] = masked_data.astype(float)
                    if masked_transform is None:
                        masked_transform = transform
                        
            except Exception as e:
                print(f"   ❌ Error extracting {band_name}: {e}")
        
        print(f"   ✅ Successfully extracted {len(bands_data)} bands")
        if bands_data:
            sample_shape = next(iter(bands_data.values())).shape
            print(f"   ✅ Masked data shape: {sample_shape}")
        
        return bands_data, masked_transform
    
    def calculate_spectral_indices(self, timestep=0, indices=['NDVI', 'NDWI', 'BSI']):
        """
        Calculate spectral indices for a specific timestep using properly masked data.
        """
        print(f"📊 Calculating spectral indices for timestep {timestep}...")
        
        # Extract required bands using correct masking
        bands_data, transform = self.extract_bands_for_timestep(timestep)
        
        if not bands_data:
            print(f"   ❌ No bands extracted for timestep {timestep}")
            return {}
        
        print(f"   ✅ Available bands: {list(bands_data.keys())}")
        
        calculated_indices = {}
        
        for index_name in indices:
            try:
                if index_name == 'NDVI' and 'NIR' in bands_data and 'Red' in bands_data:
                    nir = bands_data['NIR']
                    red = bands_data['Red']
                    
                    # Calculate NDVI with proper handling of division by zero
                    denominator = nir + red
                    ndvi = np.where(
                        denominator != 0,
                        (nir - red) / denominator,
                        np.nan
                    )
                    calculated_indices['NDVI'] = ndvi
                    
                elif index_name == 'NDWI' and 'Green' in bands_data and 'NIR' in bands_data:
                    green = bands_data['Green']
                    nir = bands_data['NIR']
                    
                    # Calculate NDWI
                    denominator = green + nir
                    ndwi = np.where(
                        denominator != 0,
                        (green - nir) / denominator,
                        np.nan
                    )
                    calculated_indices['NDWI'] = ndwi
                    
                elif index_name == 'BSI' and all(b in bands_data for b in ['Red', 'SWIR1', 'NIR', 'Blue']):
                    red = bands_data['Red']
                    swir1 = bands_data['SWIR1']
                    nir = bands_data['NIR']
                    blue = bands_data['Blue']
                    
                    # Calculate BSI (Bare Soil Index)
                    numerator = (red + swir1) - (nir + blue)
                    denominator = (red + swir1) + (nir + blue)
                    
                    bsi = np.where(
                        denominator != 0,
                        numerator / denominator,
                        np.nan
                    )
                    calculated_indices['BSI'] = bsi
                    
                else:
                    print(f"   ⚠️ Cannot calculate {index_name} - missing required bands")
                    continue
                
                # Calculate statistics for the index
                valid_pixels = ~np.isnan(calculated_indices[index_name])
                if np.any(valid_pixels):
                    mean_val = np.nanmean(calculated_indices[index_name])
                    print(f"   ✅ {index_name} calculated successfully (mean: {mean_val:.3f})")
                else:
                    print(f"   ⚠️ {index_name} calculated but contains only NaN values")
                
            except Exception as e:
                print(f"   ❌ Error calculating {index_name}: {e}")
        
        return calculated_indices
    
    def visualize_rgb_bands(self, timestep=0, figsize=(15, 5)):
        """
        Visualize Red, Green, and Blue bands separately for a specific timestep.
        """
        if self.sentinel_data is None:
            self.load_data()
        
        print(f"🎨 Visualizing RGB bands for timestep {timestep}...")
        
        # Extract RGB bands using correct masking
        bands_data, transform = self.extract_bands_for_timestep(timestep, ['Red', 'Green', 'Blue'])
        
        if not bands_data or len(bands_data) < 3:
            print("❌ Could not extract RGB bands")
            return None
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Individual bands
        band_names = ['Red', 'Green', 'Blue']
        colors = ['Reds', 'Greens', 'Blues']
        
        for i, (band_name, cmap) in enumerate(zip(band_names, colors)):
            if band_name in bands_data:
                band_data = bands_data[band_name]
                
                # Normalize for better visualization (handle NaN values)
                valid_data = band_data[~np.isnan(band_data)]
                if len(valid_data) > 0:
                    p98 = np.percentile(valid_data[valid_data > 0], 98) if np.any(valid_data > 0) else np.max(valid_data)
                    band_normalized = np.clip(band_data / p98 * 255, 0, 255)
                else:
                    band_normalized = band_data
                
                im = axes[i].imshow(band_normalized, cmap=cmap)
                axes[i].set_title(f'{band_name} Band\n(Timestep {timestep})')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], shrink=0.6)
        
        # RGB composite
        if all(b in bands_data for b in ['Red', 'Green', 'Blue']):
            red_band = bands_data['Red']
            green_band = bands_data['Green']
            blue_band = bands_data['Blue']
            
            # Normalize RGB for composite (handle NaN values)
            def normalize_band(band):
                valid_data = band[~np.isnan(band)]
                if len(valid_data) > 0 and np.any(valid_data > 0):
                    p98 = np.percentile(valid_data[valid_data > 0], 98)
                    return np.clip(band / p98, 0, 1)
                else:
                    return np.zeros_like(band)
            
            red_norm = normalize_band(red_band)
            green_norm = normalize_band(green_band)
            blue_norm = normalize_band(blue_band)
            
            # Handle NaN values in composite
            red_norm = np.nan_to_num(red_norm, nan=0)
            green_norm = np.nan_to_num(green_norm, nan=0)
            blue_norm = np.nan_to_num(blue_norm, nan=0)
            
            rgb_composite = np.dstack([red_norm, green_norm, blue_norm])
            
            axes[3].imshow(rgb_composite)
            axes[3].set_title(f'RGB Composite\n(Timestep {timestep})')
            axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'kastoria_rgb.png', dpi=300)
        plt.show()
        
        return fig
    
    def visualize_spectral_indices(self, timestep=0, indices=['NDVI', 'NDWI', 'BSI'], figsize=(15, 5)):
        """
        Visualize calculated spectral indices.
        """
        # Calculate indices using the correct masking method
        calculated_indices = self.calculate_spectral_indices(timestep, indices)
        
        print(f"🎨 Visualizing spectral indices for timestep {timestep}...")
        
        if not calculated_indices:
            print("❌ No indices calculated to visualize")
            return None
        
        fig, axes = plt.subplots(1, len(indices), figsize=figsize)
        if len(indices) == 1:
            axes = [axes]
        
        colormaps = {'NDVI': 'RdYlGn', 'NDWI': 'Blues', 'BSI': 'YlOrBr'}
        
        for i, index_name in enumerate(indices):
            if index_name in calculated_indices:
                index_data = calculated_indices[index_name]
                
                print(f"   Plotting {index_name}: shape = {index_data.shape}")
                
                # Handle visualization with proper NaN handling
                valid_data = index_data[~np.isnan(index_data)]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                    
                    im = axes[i].imshow(index_data, 
                                      cmap=colormaps.get(index_name, 'viridis'),
                                      vmin=vmin, vmax=vmax)
                    axes[i].set_title(f'{index_name} Index\n(Timestep {timestep})')
                    axes[i].axis('off')
                    plt.colorbar(im, ax=axes[i], shrink=0.6)
                    
                    # Add statistics
                    mean_val = np.nanmean(index_data)
                    std_val = np.nanstd(index_data)
                    valid_pixels = len(valid_data)
                    total_pixels = index_data.size
                    
                    stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nValid: {valid_pixels}/{total_pixels}'
                    axes[i].text(0.02, 0.98, stats_text, 
                               transform=axes[i].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    axes[i].text(0.5, 0.5, f'{index_name}\nNo Valid Data', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'{index_name}\nNot Available', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'kastoria_spectral_indices.png', dpi=300)
        plt.show()
        
        return fig
    
    def calculate_time_series_statistics(self, indices=['NDVI', 'NDWI', 'BSI'], max_timesteps=None):
        """
        Calculate mean values of spectral indices for all time steps.
        """
        print(f"📈 Calculating time series statistics...")
        
        if max_timesteps is None:
            max_timesteps = self.num_timesteps
        
        time_series_results = []
        
        for timestep in range(min(max_timesteps, self.num_timesteps)):
            print(f"   Processing timestep {timestep + 1}/{max_timesteps}...")
            
            try:
                # Use the correct calculation method
                calculated_indices = self.calculate_spectral_indices(timestep, indices)
                
                # Calculate statistics
                timestep_stats = {'timestep': timestep}
                
                for index_name, index_data in calculated_indices.items():
                    # Remove NaN values
                    valid_data = index_data[~np.isnan(index_data)]
                    
                    if len(valid_data) > 0:
                        timestep_stats[f'{index_name}_mean'] = np.mean(valid_data)
                        timestep_stats[f'{index_name}_std'] = np.std(valid_data)
                        timestep_stats[f'{index_name}_median'] = np.median(valid_data)
                        timestep_stats[f'{index_name}_min'] = np.min(valid_data)
                        timestep_stats[f'{index_name}_max'] = np.max(valid_data)
                        timestep_stats[f'{index_name}_valid_pixels'] = len(valid_data)
                        timestep_stats[f'{index_name}_total_pixels'] = index_data.size
                    else:
                        for stat in ['mean', 'std', 'median', 'min', 'max']:
                            timestep_stats[f'{index_name}_{stat}'] = np.nan
                        timestep_stats[f'{index_name}_valid_pixels'] = 0
                        timestep_stats[f'{index_name}_total_pixels'] = index_data.size if index_name in calculated_indices else 0
                
                time_series_results.append(timestep_stats)
                
                # Print progress summary
                if calculated_indices:
                    sample_index = list(calculated_indices.keys())[0]
                    sample_mean = timestep_stats.get(f'{sample_index}_mean', 'N/A')
                    print(f"     ✅ Completed - {sample_index} mean: {sample_mean}")
                
            except Exception as e:
                print(f"   ❌ Error processing timestep {timestep}: {e}")
        
        # Convert to DataFrame
        self.time_series_df = pd.DataFrame(time_series_results)
        
        print(f"✅ Time series analysis complete!")
        print(f"   - Processed timesteps: {len(time_series_results)}")
        print(f"   - Indices analyzed: {indices}")
        
        # Save to CSV
        output_file = 'kastoria_spectral_indices_timeseries_correct.csv'
        self.time_series_df.to_csv(output_file, index=False)
        print(f"   - Results saved to: {output_file}")
        
        # Print sample statistics
        if not self.time_series_df.empty:
            print(f"\n📊 SAMPLE STATISTICS:")
            for index_name in indices:
                mean_col = f'{index_name}_mean'
                if mean_col in self.time_series_df.columns:
                    overall_mean = self.time_series_df[mean_col].mean()
                    overall_std = self.time_series_df[mean_col].std()
                    print(f"   - {index_name}: μ={overall_mean:.3f}, σ={overall_std:.3f}")
        
        return self.time_series_df
    
    def plot_time_series(self, indices=['NDVI'], figsize=(12, 8)):
        """
        Plot time series evolution of spectral indices.
        """
        if not hasattr(self, 'time_series_df') or self.time_series_df is None:
            print("⚠️ No time series data available. Run calculate_time_series_statistics() first.")
            return None
        
        print(f"📊 Plotting time series for indices: {', '.join(indices)}")
        
        # Create subplots
        n_plots = len(indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        colors = ['green', 'blue', 'brown', 'red', 'purple']
        
        for i, index_name in enumerate(indices):
            mean_col = f'{index_name}_mean'
            std_col = f'{index_name}_std'
            
            if mean_col in self.time_series_df.columns:
                x = self.time_series_df['timestep']
                y_mean = self.time_series_df[mean_col]
                y_std = self.time_series_df[std_col] if std_col in self.time_series_df.columns else None
                
                # Plot mean line
                axes[i].plot(x, y_mean, 'o-', color=colors[i % len(colors)], 
                           linewidth=2, markersize=6, label=f'{index_name} Mean')
                
                # Add error bars if std is available
                if y_std is not None:
                    axes[i].fill_between(x, y_mean - y_std, y_mean + y_std, 
                                       color=colors[i % len(colors)], alpha=0.2, 
                                       label=f'{index_name} ± 1σ')
                
                axes[i].set_ylabel(f'{index_name} Value')
                axes[i].set_title(f'{index_name} Time Series Evolution - Kastoria Study Area (Correct Masking)')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                
                # Add statistics text
                mean_overall = np.nanmean(y_mean)
                trend = 'Increasing' if y_mean.iloc[-1] > y_mean.iloc[0] else 'Decreasing'
                axes[i].text(0.02, 0.98, f'Overall Mean: {mean_overall:.3f}\nTrend: {trend}', 
                           transform=axes[i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        plt.savefig('kastoria_time_series_correct.png', dpi=300)
        plt.show()
        
        return fig

# Initialize the analyzer
print("🚀 KASTORIA RASTER ANALYSIS - CORRECT RASTERIO.MASK IMPLEMENTATION")
print("="*70)

analyzer = KastoriaRasterAnalyzer()

# Step 1: Load data
print("\nSTEP 1: Loading raster data and study area...")
sentinel_data, study_area = analyzer.load_data()

# Step 2: Visualize RGB bands for timestep 0
print("\nSTEP 2: Visualizing RGB bands using correct masking...")
rgb_fig = analyzer.visualize_rgb_bands(timestep=0)

# Step 3: Test the masking process explicitly
print("\nSTEP 3: Testing correct rasterio.mask implementation...")
test_bands, test_transform = analyzer.extract_bands_for_timestep(0, ['Red', 'Green', 'Blue', 'NIR', 'SWIR1'])
if test_bands:
    print(f"   ✅ Successfully extracted {len(test_bands)} bands with correct masking")
    for band_name, band_data in test_bands.items():
        valid_pixels = np.sum(~np.isnan(band_data))
        total_pixels = band_data.size
        print(f"   ✅ {band_name}: {band_data.shape}, {valid_pixels}/{total_pixels} valid pixels")
else:
    print("   ❌ Band extraction failed")

# Step 4: Calculate and visualize spectral indices
print("\nSTEP 4: Calculating and visualizing spectral indices...")
indices_fig = analyzer.visualize_spectral_indices(timestep=0, indices=['NDVI', 'NDWI', 'BSI'])

# Step 5: Calculate time series statistics (process first 6 timesteps for demonstration)
print("\nSTEP 5: Calculating time series statistics...")
time_series_df = analyzer.calculate_time_series_statistics(
    indices=['NDVI', 'NDWI', 'BSI'], 
    max_timesteps=6  # Process first 6 timesteps for demonstration
)

# Step 6: Plot time series evolution
print("\nSTEP 6: Plotting time series evolution...")
ts_fig = analyzer.plot_time_series(indices=['NDVI', 'NDWI', 'BSI'])

# Display summary
print(f"\n{'='*70}")
print("✅ RASTER ANALYSIS COMPLETE WITH CORRECT RASTERIO.MASK!")
print("="*70)
print("📁 FILES CREATED:")
print("   - kastoria_spectral_indices_timeseries_correct.csv")
print("\n🔧 TECHNICAL APPROACH:")
print("   - PROPER rasterio.mask implementation with correct parameters")
print("   - Geometries passed as iterable list: [geometry]")
print("   - Correct CRS alignment between raster and vector data")
print("   - Proper band indexing with 1-based indexes parameter")
print("   - Appropriate handling of crop=True and nodata values")
print("   - Robust NaN handling in visualization and statistics")
print("\n📊 ANALYSIS SUMMARY:")
if hasattr(analyzer, 'time_series_df') and not analyzer.time_series_df.empty:
    print(f"   - Time steps processed: {len(analyzer.time_series_df)}")
    print(f"   - Spectral indices: NDVI, NDWI, BSI")
    if 'NDVI_mean' in analyzer.time_series_df.columns:
        ndvi_mean = analyzer.time_series_df['NDVI_mean'].mean()
        ndvi_trend = 'Increasing' if analyzer.time_series_df['NDVI_mean'].iloc[-1] > analyzer.time_series_df['NDVI_mean'].iloc[0] else 'Decreasing'
        print(f"   - NDVI: μ={ndvi_mean:.3f}, trend={ndvi_trend}")
    if 'NDWI_mean' in analyzer.time_series_df.columns:
        ndwi_mean = analyzer.time_series_df['NDWI_mean'].mean()
        print(f"   - NDWI: μ={ndwi_mean:.3f}")
    if 'BSI_mean' in analyzer.time_series_df.columns:
        bsi_mean = analyzer.time_series_df['BSI_mean'].mean()
        print(f"   - BSI: μ={bsi_mean:.3f}")
print("\n📋 NEXT STEPS:")
print("   - Combine with meteorological data")
print("   - Create interactive visualizations")
print("   - Integrate with OGC web services")