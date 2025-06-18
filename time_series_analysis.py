import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore')

# Additional imports for web services
from owslib.wfs import WebFeatureService
import contextily as ctx

class KastoriaTimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for Kastoria including:
    - Static maps with multiple data sources
    - Meteorological time series visualization
    - Integration with geodata.gov.gr services
    - Complete 24-timestep spectral indices integration
    """
    
    def __init__(self, study_area_path="kastoria_study_area.geojson", 
                 meteorological_data_path="kastoria_meteorological_data.csv",
                 spectral_indices_path="kastoria_spectral_indices_timeseries_complete.csv"):  # Updated filename
        """
        Initialize the time series analyzer with complete spectral analysis integration.
        """
        self.study_area_path = study_area_path
        self.meteorological_data_path = meteorological_data_path
        self.spectral_indices_path = spectral_indices_path  # Added spectral indices path
        self.study_area = None
        self.meteorological_data = None
        self.spectral_indices_data = None  # Added spectral indices data
        self.meteorological_point = None
        self.administrative_data = {}
        self.transportation_data = {}
        
        # geodata.gov.gr WFS endpoints
        self.geodata_wfs_url = "http://geodata.gov.gr/geoserver/ows"
        self.geodata_wms_url = "http://geodata.gov.gr/geoserver/ows"
        
    def load_data(self):
        """
        Load study area, meteorological data, and complete spectral indices analysis.
        """
        print("🔄 Loading study area, meteorological data, and spectral indices...")
        
        # Load study area polygon
        try:
            self.study_area = gpd.read_file(self.study_area_path)
            print(f"   ✅ Study area loaded: {len(self.study_area)} polygon(s)")
        except Exception as e:
            print(f"   ❌ Error loading study area: {e}")
            return False
        
        # Load meteorological data
        try:
            self.meteorological_data = pd.read_csv(self.meteorological_data_path)
            self.meteorological_data['date'] = pd.to_datetime(self.meteorological_data['date'])
            
            # Create meteorological point from the data
            if 'latitude' in self.meteorological_data.columns and 'longitude' in self.meteorological_data.columns:
                lat = self.meteorological_data['latitude'].iloc[0]
                lon = self.meteorological_data['longitude'].iloc[0]
                self.meteorological_point = gpd.GeoDataFrame(
                    [{'id': 1, 'type': 'meteorological_station'}],
                    geometry=[Point(lon, lat)],
                    crs='EPSG:4326'
                )
            
            print(f"   ✅ Meteorological data loaded: {len(self.meteorological_data)} records")
            print(f"   ✅ Date range: {self.meteorological_data['date'].min()} to {self.meteorological_data['date'].max()}")
            print(f"   ✅ Variables: {[col for col in self.meteorological_data.columns if col not in ['date', 'latitude', 'longitude']]}")
            
        except Exception as e:
            print(f"   ❌ Error loading meteorological data: {e}")
            return False
        
        # Load complete spectral indices analysis
        try:
            self.spectral_indices_data = pd.read_csv(self.spectral_indices_path)
            print(f"   ✅ Spectral indices data loaded: {len(self.spectral_indices_data)} timesteps (COMPLETE ANALYSIS)")
            
            # Display available spectral indices
            spectral_columns = [col for col in self.spectral_indices_data.columns if any(idx in col for idx in ['NDVI', 'NDWI', 'BSI'])]
            unique_indices = list(set([col.split('_')[0] for col in spectral_columns if '_' in col]))
            print(f"   ✅ Available spectral indices: {unique_indices}")
            print(f"   ✅ Complete temporal coverage: {len(self.spectral_indices_data)} timesteps analyzed")
            
        except Exception as e:
            print(f"   ⚠️ Error loading spectral indices data: {e}")
            print(f"   ⚠️ Continuing without spectral indices integration")
            self.spectral_indices_data = None
        
        return True
    
    def fetch_administrative_boundaries(self):
        """
        Fetch administrative boundaries from geodata.gov.gr WFS services.
        """
        print("🗺️ Fetching administrative boundaries from geodata.gov.gr...")
        
        try:
            # Try to access the WFS service
            wfs = WebFeatureService(url=self.geodata_wfs_url, version='2.0.0')
            
            # List available layers
            available_layers = list(wfs.contents.keys())
            print(f"   Available layers: {len(available_layers)}")
            
            # Look for region/administrative layers
            region_layers = [layer for layer in available_layers if any(keyword in layer.lower() 
                           for keyword in ['region', 'periphery', 'admin', 'municipality', 'περιφ'])]
            
            if region_layers:
                print(f"   Found administrative layers: {region_layers[:3]}...")
                
                # Try to fetch the first administrative layer
                layer_name = region_layers[0]
                try:
                    response = wfs.getfeature(typename=layer_name, maxfeatures=50, outputFormat='application/json')
                    
                    # Convert to GeoDataFrame
                    import json
                    from io import StringIO
                    
                    geojson_data = json.loads(response.read().decode('utf-8'))
                    self.administrative_data['boundaries'] = gpd.GeoDataFrame.from_features(
                        geojson_data['features'], crs='EPSG:4326'
                    )
                    
                    print(f"   ✅ Administrative boundaries loaded: {len(self.administrative_data['boundaries'])} features")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not fetch administrative data: {e}")
                    
            else:
                print("   ⚠️ No administrative layers found")
                
        except Exception as e:
            print(f"   ⚠️ Could not connect to geodata.gov.gr WFS: {e}")
            print("   Using alternative approach...")
            
            # Alternative: Create simplified administrative context
            self.create_alternative_administrative_data()
    
    def create_alternative_administrative_data(self):
        """
        Create alternative administrative context when WFS is not available.
        """
        print("   🔄 Creating alternative administrative context...")
        
        # Get study area bounds
        bounds = self.study_area.bounds
        minx, miny, maxx, maxy = bounds.iloc[0]
        
        # Create a simplified regional boundary around Kastoria
        # West Macedonia region approximate boundaries
        west_macedonia_coords = [
            [21.0, 40.2],  # Southwest
            [22.2, 40.2],  # Southeast  
            [22.2, 40.8],  # Northeast
            [21.0, 40.8],  # Northwest
            [21.0, 40.2]   # Close polygon
        ]
        
        west_macedonia_polygon = Polygon(west_macedonia_coords)
        
        self.administrative_data['boundaries'] = gpd.GeoDataFrame([{
            'name': 'West Macedonia Region (Approximate)',
            'type': 'region',
            'admin_level': 1
        }], geometry=[west_macedonia_polygon], crs='EPSG:4326')
        
        print("   ✅ Alternative administrative boundaries created")
    
    def fetch_transportation_networks(self):
        """
        Fetch transportation networks or create representative data.
        """
        print("🛣️ Creating transportation network context...")
        
        # Create representative transportation features for the Kastoria area
        # Based on known major roads around Kastoria
        
        # Major road connecting Kastoria to Florina (National Road)
        kastoria_florina_road = [
            [21.2685, 40.5167],  # Kastoria center
            [21.3500, 40.6500],  # Towards Florina
            [21.4000, 40.7000]   # Further north
        ]
        
        # Road to Kozani (southward)
        kastoria_kozani_road = [
            [21.2685, 40.5167],  # Kastoria center
            [21.2000, 40.4000],  # Southwest
            [21.1500, 40.3000]   # Towards Kozani
        ]
        
        # Local roads around the lake
        lake_road_north = [
            [21.2600, 40.5200],  # North side of lake
            [21.2800, 40.5150],  # Around lake
            [21.2900, 40.5100]   # Continue around
        ]
        
        from shapely.geometry import LineString
        
        roads_data = [
            {
                'name': 'National Road (Kastoria-Florina)',
                'road_type': 'national',
                'geometry': LineString(kastoria_florina_road)
            },
            {
                'name': 'Regional Road (Kastoria-Kozani)',
                'road_type': 'regional', 
                'geometry': LineString(kastoria_kozani_road)
            },
            {
                'name': 'Local Road (Lake Circuit)',
                'road_type': 'local',
                'geometry': LineString(lake_road_north)
            }
        ]
        
        self.transportation_data['roads'] = gpd.GeoDataFrame(
            roads_data, crs='EPSG:4326'
        )
        
        print(f"   ✅ Transportation network created: {len(self.transportation_data['roads'])} road segments")
    
    def create_static_map(self, figsize=(15, 10)):
        """
        Create comprehensive static map showing all data layers.
        """
        print("🗺️ Creating comprehensive static map...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Map 1: Study area with administrative boundaries
        ax1 = axes[0]
        
        # Plot administrative boundaries first (background)
        if 'boundaries' in self.administrative_data:
            self.administrative_data['boundaries'].plot(
                ax=ax1, color='lightgray', edgecolor='gray', alpha=0.7, linewidth=1
            )
        
        # Plot study area
        self.study_area.plot(ax=ax1, color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
        
        # Add meteorological point
        if self.meteorological_point is not None:
            self.meteorological_point.plot(ax=ax1, color='blue', markersize=100, marker='*', 
                                          edgecolor='darkblue', linewidth=2)
        
        ax1.set_title('Study Area and Administrative Context')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Study Area'),
            Patch(facecolor='lightgray', alpha=0.7, label='Administrative Region')
        ]
        if self.meteorological_point is not None:
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', 
                                        markersize=10, label='Meteorological Station'))
        ax1.legend(handles=legend_elements)
        
        # Map 2: Transportation networks
        ax2 = axes[1]
        
        # Plot study area as context
        self.study_area.plot(ax=ax2, color='red', alpha=0.3, edgecolor='darkred', linewidth=1)
        
        # Plot transportation networks
        if 'roads' in self.transportation_data:
            # Plot different road types with different colors
            road_colors = {'national': 'blue', 'regional': 'green', 'local': 'orange'}
            road_widths = {'national': 3, 'regional': 2, 'local': 1}
            
            for road_type in self.transportation_data['roads']['road_type'].unique():
                roads_subset = self.transportation_data['roads'][
                    self.transportation_data['roads']['road_type'] == road_type
                ]
                roads_subset.plot(
                    ax=ax2, 
                    color=road_colors.get(road_type, 'black'),
                    linewidth=road_widths.get(road_type, 1),
                    label=f'{road_type.title()} Roads'
                )
        
        ax2.set_title('Transportation Networks')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Map 3: Detailed study area view
        ax3 = axes[2]
        
        # Plot study area with more detail
        self.study_area.plot(ax=ax3, color='red', alpha=0.7, edgecolor='darkred', linewidth=2)
        
        # Add meteorological point
        if self.meteorological_point is not None:
            self.meteorological_point.plot(ax=ax3, color='blue', markersize=150, marker='*', 
                                          edgecolor='darkblue', linewidth=2)
            
            # Add coordinates annotation
            lat = self.meteorological_point.geometry.iloc[0].y
            lon = self.meteorological_point.geometry.iloc[0].x
            ax3.annotate(f'Meteorological Station\n({lat:.4f}°N, {lon:.4f}°E)', 
                        xy=(lon, lat), xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Calculate and display study area statistics
        study_area_utm = self.study_area.to_crs('EPSG:2100')  # Greek Grid
        area_km2 = study_area_utm.area.iloc[0] / 1e6
        
        ax3.set_title(f'Study Area Detail\nArea: {area_km2:.2f} km²')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.grid(True, alpha=0.3)
        
        # Map 4: Context map (wider view)
        ax4 = axes[3]
        
        # Create a wider context around the study area
        bounds = self.study_area.bounds
        margin = 0.1  # degrees
        
        # Plot administrative boundaries for context
        if 'boundaries' in self.administrative_data:
            self.administrative_data['boundaries'].plot(
                ax=ax4, color='lightblue', edgecolor='blue', alpha=0.5, linewidth=1
            )
        
        # Plot study area as a point in the wider context
        study_center = self.study_area.centroid
        study_center.plot(ax=ax4, color='red', markersize=50, marker='o', 
                         edgecolor='darkred', linewidth=2)
        
        ax4.set_title('Regional Context')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.grid(True, alpha=0.3)
        
        # Set appropriate bounds for context map
        minx, miny, maxx, maxy = bounds.iloc[0]
        ax4.set_xlim(minx - margin, maxx + margin)
        ax4.set_ylim(miny - margin, maxy + margin)
        
        plt.tight_layout()
        plt.savefig('figures/kastoria_static_maps.png', dpi=300)
        plt.show()
        
        return fig
    
    def plot_meteorological_timeseries(self, figsize=(15, 12)):
        """
        Create comprehensive meteorological time series plots.
        """
        print("📊 Creating meteorological time series plots...")
        
        if self.meteorological_data is None:
            print("❌ No meteorological data available")
            return None
        
        # Identify meteorological variables (exclude date, lat, lon)
        met_variables = [col for col in self.meteorological_data.columns 
                        if col not in ['date', 'latitude', 'longitude']]
        
        print(f"   Plotting variables: {met_variables}")
        
        # Create subplots
        n_vars = len(met_variables)
        n_rows = (n_vars + 1) // 2  # 2 columns
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        axes = axes.flatten()
        
        # Variable information for better plotting
        var_info = {
            'T2M': {'name': 'Temperature', 'unit': '°C', 'color': 'red'},
            'PRECTOTCORR': {'name': 'Precipitation', 'unit': 'mm/day', 'color': 'blue'},
            'RH2M': {'name': 'Relative Humidity', 'unit': '%', 'color': 'green'},
            'WS2M': {'name': 'Wind Speed', 'unit': 'm/s', 'color': 'purple'}
        }
        
        for i, var in enumerate(met_variables):
            if i < len(axes):
                ax = axes[i]
                
                # Get variable info
                info = var_info.get(var, {'name': var, 'unit': 'units', 'color': 'black'})
                
                # Plot time series
                ax.plot(self.meteorological_data['date'], self.meteorological_data[var], 
                       color=info['color'], linewidth=1, alpha=0.7)
                
                # Add trend line
                x_numeric = np.arange(len(self.meteorological_data))
                z = np.polyfit(x_numeric, self.meteorological_data[var], 1)
                p = np.poly1d(z)
                ax.plot(self.meteorological_data['date'], p(x_numeric), 
                       '--', color=info['color'], linewidth=2, alpha=0.8, label='Trend')
                
                # Calculate statistics
                mean_val = self.meteorological_data[var].mean()
                std_val = self.meteorological_data[var].std()
                min_val = self.meteorological_data[var].min()
                max_val = self.meteorological_data[var].max()
                
                # Add horizontal lines for mean
                ax.axhline(mean_val, color=info['color'], linestyle=':', alpha=0.6, label=f'Mean: {mean_val:.2f}')
                
                ax.set_title(f'{info["name"]} Time Series\nMean: {mean_val:.2f} ± {std_val:.2f} {info["unit"]}')
                ax.set_xlabel('Date')
                ax.set_ylabel(f'{info["name"]} ({info["unit"]})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
                
                # Add statistics text box
                stats_text = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nStd: {std_val:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(met_variables), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Meteorological Variables Time Series - Kastoria Study Area', fontsize=16)
        plt.tight_layout()
        plt.savefig('figures/kastoria_meteorological_timeseries.png', dpi=300)
        plt.show()
        
        return fig
    
    def plot_seasonal_analysis(self, figsize=(15, 8)):
        """
        Create seasonal analysis of meteorological data.
        """
        print("🍂 Creating seasonal analysis...")
        
        if self.meteorological_data is None:
            print("❌ No meteorological data available")
            return None
        
        # Add month and season columns
        df = self.meteorological_data.copy()
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Meteorological variables
        met_vars = [col for col in df.columns 
                   if col not in ['date', 'latitude', 'longitude', 'month', 'season']]
        
        # Create seasonal box plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Variable info
        var_info = {
            'T2M': {'name': 'Temperature', 'unit': '°C'},
            'PRECTOTCORR': {'name': 'Precipitation', 'unit': 'mm/day'},
            'RH2M': {'name': 'Relative Humidity', 'unit': '%'},
            'WS2M': {'name': 'Wind Speed', 'unit': 'm/s'}
        }
        
        for i, var in enumerate(met_vars[:4]):  # Plot first 4 variables
            if i < len(axes):
                ax = axes[i]
                info = var_info.get(var, {'name': var, 'unit': 'units'})
                
                # Create box plot
                seasons_order = ['Spring', 'Summer', 'Autumn', 'Winter']
                df_plot = df[df['season'].isin(seasons_order)]
                
                sns.boxplot(data=df_plot, x='season', y=var, ax=ax, order=seasons_order)
                ax.set_title(f'{info["name"]} by Season')
                ax.set_xlabel('Season')
                ax.set_ylabel(f'{info["name"]} ({info["unit"]})')
                ax.grid(True, alpha=0.3)
                
                # Add mean values as text
                for j, season in enumerate(seasons_order):
                    season_data = df[df['season'] == season][var]
                    if len(season_data) > 0:
                        mean_val = season_data.mean()
                        ax.text(j, mean_val, f'{mean_val:.1f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Seasonal Analysis of Meteorological Variables - Kastoria', fontsize=16)
        plt.tight_layout()
        plt.savefig('figures/kastoria_seasonal_analysis.png', dpi=300)
        plt.show()
        
        return fig
    
    def create_integrated_time_series_analysis(self, figsize=(15, 12)):
        """
        Create integrated time series analysis combining meteorological and spectral data.
        """
        print("📊 Creating integrated meteorological and spectral time series analysis...")
        
        if self.spectral_indices_data is None:
            print("❌ No spectral indices data available for integration")
            return self.plot_meteorological_timeseries(figsize)
        
        # Create comprehensive integrated plot
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Temperature time series
        ax1 = axes[0]
        if 'T2M' in self.meteorological_data.columns:
            ax1.plot(self.meteorological_data['date'], self.meteorological_data['T2M'], 'r-', linewidth=1, alpha=0.7)
            temp_mean = self.meteorological_data['T2M'].mean()
            ax1.axhline(temp_mean, color='red', linestyle=':', alpha=0.6, label=f'Mean: {temp_mean:.1f}°C')
            ax1.set_title('Temperature Time Series (2018-2024)')
            ax1.set_ylabel('Temperature (°C)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Plot 2: NDVI complete time series
        ax2 = axes[1]
        if 'NDVI_mean' in self.spectral_indices_data.columns:
            ax2.plot(self.spectral_indices_data['timestep'], self.spectral_indices_data['NDVI_mean'], 
                    'g-o', linewidth=2, markersize=4, label='NDVI Mean')
            if 'NDVI_std' in self.spectral_indices_data.columns:
                ax2.fill_between(self.spectral_indices_data['timestep'], 
                               self.spectral_indices_data['NDVI_mean'] - self.spectral_indices_data['NDVI_std'],
                               self.spectral_indices_data['NDVI_mean'] + self.spectral_indices_data['NDVI_std'],
                               alpha=0.2, color='green')
            
            ndvi_mean = self.spectral_indices_data['NDVI_mean'].mean()
            ax2.set_title(f'NDVI Complete Time Series (24 timesteps)\nMean: {ndvi_mean:.3f}')
            ax2.set_ylabel('NDVI Value')
            ax2.set_xlabel('Timestep')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Plot 3: Precipitation time series
        ax3 = axes[2]
        if 'PRECTOTCORR' in self.meteorological_data.columns:
            ax3.plot(self.meteorological_data['date'], self.meteorological_data['PRECTOTCORR'], 
                    'b-', linewidth=1, alpha=0.7)
            precip_mean = self.meteorological_data['PRECTOTCORR'].mean()
            ax3.axhline(precip_mean, color='blue', linestyle=':', alpha=0.6, label=f'Mean: {precip_mean:.1f} mm/day')
            ax3.set_title('Precipitation Time Series (2018-2024)')
            ax3.set_ylabel('Precipitation (mm/day)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: NDWI complete time series
        ax4 = axes[3]
        if 'NDWI_mean' in self.spectral_indices_data.columns:
            ax4.plot(self.spectral_indices_data['timestep'], self.spectral_indices_data['NDWI_mean'], 
                    'b-o', linewidth=2, markersize=4, label='NDWI Mean')
            if 'NDWI_std' in self.spectral_indices_data.columns:
                ax4.fill_between(self.spectral_indices_data['timestep'], 
                               self.spectral_indices_data['NDWI_mean'] - self.spectral_indices_data['NDWI_std'],
                               self.spectral_indices_data['NDWI_mean'] + self.spectral_indices_data['NDWI_std'],
                               alpha=0.2, color='blue')
            
            ndwi_mean = self.spectral_indices_data['NDWI_mean'].mean()
            ax4.set_title(f'NDWI Complete Time Series (24 timesteps)\nMean: {ndwi_mean:.3f}')
            ax4.set_ylabel('NDWI Value')
            ax4.set_xlabel('Timestep')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # Plot 5: Relative Humidity
        ax5 = axes[4]
        if 'RH2M' in self.meteorological_data.columns:
            ax5.plot(self.meteorological_data['date'], self.meteorological_data['RH2M'], 
                    'g-', linewidth=1, alpha=0.7)
            humidity_mean = self.meteorological_data['RH2M'].mean()
            ax5.axhline(humidity_mean, color='green', linestyle=':', alpha=0.6, label=f'Mean: {humidity_mean:.1f}%')
            ax5.set_title('Relative Humidity Time Series (2018-2024)')
            ax5.set_ylabel('Relative Humidity (%)')
            ax5.set_xlabel('Date')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
        
        # Plot 6: BSI complete time series
        ax6 = axes[5]
        if 'BSI_mean' in self.spectral_indices_data.columns:
            ax6.plot(self.spectral_indices_data['timestep'], self.spectral_indices_data['BSI_mean'], 
                    'brown', marker='o', linewidth=2, markersize=4, label='BSI Mean')
            if 'BSI_std' in self.spectral_indices_data.columns:
                ax6.fill_between(self.spectral_indices_data['timestep'], 
                               self.spectral_indices_data['BSI_mean'] - self.spectral_indices_data['BSI_std'],
                               self.spectral_indices_data['BSI_mean'] + self.spectral_indices_data['BSI_std'],
                               alpha=0.2, color='brown')
            
            bsi_mean = self.spectral_indices_data['BSI_mean'].mean()
            ax6.set_title(f'BSI Complete Time Series (24 timesteps)\nMean: {bsi_mean:.3f}')
            ax6.set_ylabel('BSI Value')
            ax6.set_xlabel('Timestep')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        plt.suptitle('Integrated Meteorological and Spectral Indices Analysis - Kastoria Study Area', fontsize=16)
        plt.tight_layout()
        plt.savefig('figures/kastoria_integrated_timeseries_complete.png', dpi=300)
        plt.show()
        
        return fig

    def create_summary_statistics(self):
        """
        Create comprehensive summary statistics including complete spectral analysis.
        """
        print("📈 Creating comprehensive summary statistics with complete spectral analysis...")
        
        # Study area statistics
        study_area_utm = self.study_area.to_crs('EPSG:2100')
        area_km2 = study_area_utm.area.iloc[0] / 1e6
        perimeter_km = study_area_utm.length.iloc[0] / 1000
        
        # Meteorological statistics
        met_vars = [col for col in self.meteorological_data.columns 
                   if col not in ['date', 'latitude', 'longitude']]
        
        met_stats = {}
        for var in met_vars:
            met_stats[var] = {
                'mean': self.meteorological_data[var].mean(),
                'std': self.meteorological_data[var].std(),
                'min': self.meteorological_data[var].min(),
                'max': self.meteorological_data[var].max(),
                'records': len(self.meteorological_data)
            }
        
        # Spectral indices statistics (complete analysis)
        spectral_stats = {}
        if self.spectral_indices_data is not None:
            spectral_indices = ['NDVI', 'NDWI', 'BSI']
            for index_name in spectral_indices:
                mean_col = f'{index_name}_mean'
                if mean_col in self.spectral_indices_data.columns:
                    values = self.spectral_indices_data[mean_col]
                    
                    # Calculate seasonal amplitude
                    seasonal_amplitude = values.max() - values.min()
                    
                    # Calculate trend
                    x = np.arange(len(values))
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    trend_direction = "Increasing" if slope > 0 else "Decreasing"
                    
                    spectral_stats[index_name] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'seasonal_amplitude': seasonal_amplitude,
                        'trend_slope': slope,
                        'trend_r_squared': r_value**2,
                        'trend_p_value': p_value,
                        'trend_direction': trend_direction,
                        'timesteps': len(values)
                    }
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print("KASTORIA STUDY AREA - COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📍 STUDY AREA:")
        print(f"   - Area: {area_km2:.2f} km²")
        print(f"   - Perimeter: {perimeter_km:.2f} km")
        print(f"   - CRS: {self.study_area.crs}")
        
        if self.meteorological_point is not None:
            lat = self.meteorological_point.geometry.iloc[0].y
            lon = self.meteorological_point.geometry.iloc[0].x
            print(f"\n🌡️ METEOROLOGICAL STATION:")
            print(f"   - Location: {lat:.4f}°N, {lon:.4f}°E")
            print(f"   - Data period: {self.meteorological_data['date'].min().strftime('%Y-%m-%d')} to {self.meteorological_data['date'].max().strftime('%Y-%m-%d')}")
            print(f"   - Total records: {len(self.meteorological_data)}")
        
        print(f"\n📊 METEOROLOGICAL VARIABLES:")
        var_names = {
            'T2M': 'Temperature (°C)',
            'PRECTOTCORR': 'Precipitation (mm/day)',
            'RH2M': 'Relative Humidity (%)',
            'WS2M': 'Wind Speed (m/s)'
        }
        
        for var, stats in met_stats.items():
            var_name = var_names.get(var, var)
            print(f"   - {var_name}:")
            print(f"     Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"     Range: {stats['min']:.2f} to {stats['max']:.2f}")
        
        # Enhanced spectral indices summary
        if spectral_stats:
            print(f"\n🛰️ SPECTRAL INDICES (COMPLETE 24-TIMESTEP ANALYSIS):")
            for index_name, stats in spectral_stats.items():
                print(f"   - {index_name} Index:")
                print(f"     Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print(f"     Range: {stats['min']:.3f} to {stats['max']:.3f}")
                print(f"     Seasonal amplitude: {stats['seasonal_amplitude']:.3f}")
                print(f"     Trend: {stats['trend_direction']} (slope: {stats['trend_slope']:.4f})")
                print(f"     Statistical significance: R² = {stats['trend_r_squared']:.3f}, p = {stats['trend_p_value']:.3f}")
                print(f"     Temporal coverage: {stats['timesteps']} timesteps")
        
        if 'boundaries' in self.administrative_data:
            print(f"\n🗺️ ADMINISTRATIVE CONTEXT:")
            print(f"   - Administrative regions: {len(self.administrative_data['boundaries'])}")
        
        if 'roads' in self.transportation_data:
            print(f"\n🛣️ TRANSPORTATION NETWORK:")
            print(f"   - Road segments: {len(self.transportation_data['roads'])}")
            road_types = self.transportation_data['roads']['road_type'].value_counts()
            for road_type, count in road_types.items():
                print(f"     {road_type.title()}: {count} segments")
        
        print("="*80)
        
        return {
            'study_area': {'area_km2': area_km2, 'perimeter_km': perimeter_km},
            'meteorological': met_stats,
            'spectral_indices': spectral_stats
        }

# Initialize the time series analyzer
print("🚀 KASTORIA TIME SERIES ANALYSIS - STATIC MAPS & METEOROLOGICAL DATA")
print("="*70)

analyzer = KastoriaTimeSeriesAnalyzer()

# Step 1: Load all data including complete spectral analysis
print("\nSTEP 1: Loading complete dataset...")
if not analyzer.load_data():
    print("❌ Failed to load data. Please check file paths and ensure analyze.py has completed.")
    exit()

# Step 2: Fetch administrative boundaries
print("\nSTEP 2: Fetching administrative boundaries...")
analyzer.fetch_administrative_boundaries()

# Step 3: Create transportation network context
print("\nSTEP 3: Creating transportation network context...")
analyzer.fetch_transportation_networks()

# Step 4: Create comprehensive static maps
print("\nSTEP 4: Creating static maps...")
static_map_fig = analyzer.create_static_map()

# Step 5: Create integrated time series analysis (NEW)
print("\nSTEP 5: Creating integrated meteorological and spectral time series...")
integrated_fig = analyzer.create_integrated_time_series_analysis()

# Step 6: Seasonal analysis
print("\nSTEP 6: Creating seasonal analysis...")
seasonal_fig = analyzer.plot_seasonal_analysis()

# Step 7: Generate comprehensive summary statistics
print("\nSTEP 7: Generating comprehensive summary statistics...")
summary_stats = analyzer.create_summary_statistics()

# Display completion message
print(f"\n{'='*70}")
print("✅ TIME SERIES ANALYSIS COMPLETE!")
print("="*70)
print("📊 OUTPUTS GENERATED:")
print("   - Comprehensive static maps (4 views)")
print("   - Meteorological time series plots")
print("   - Seasonal analysis plots")
print("   - Summary statistics")
print("\n🗺️ STATIC MAPS INCLUDE:")
print("   - Study area with administrative boundaries")
print("   - Transportation networks context")
print("   - Detailed study area view with meteorological station")
print("   - Regional context map")
print("\n📈 TIME SERIES ANALYSIS INCLUDES:")
print("   - Temperature, precipitation, humidity, wind speed trends")
print("   - Seasonal variation analysis")
print("   - Statistical summaries and trend analysis")
print("\n📋 NEXT STEPS:")
print("   - Create interactive visualizations")
print("   - Integrate with OGC web services")
print("   - Combine with spectral indices analysis")