import warnings
from datetime import datetime, timedelta

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import seaborn as sns
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Point, Polygon

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KastoriaDataProcessor:
    """
    A class to process geospatial data for the Kastoria study area.
    Handles raster, vector, and time series data from multiple sources.
    """
    
    def __init__(self, sentinel_path="Kastoria.tif", reference_path="Kast_RefData_26Classes.tif"):
        """
        Initialize the data processor with file paths.
        
        Parameters:
        -----------
        sentinel_path : str
            Path to the Sentinel-2 time series raster file
        reference_path : str
            Path to the reference classification raster file
        """
        self.sentinel_path = sentinel_path
        self.reference_path = reference_path
        self.sentinel_data = None
        self.reference_data = None
        self.study_area_polygon = None
        self.meteorological_data = None
        
    def load_raster_data(self):
        """
        Load and explore the Sentinel-2 time series and reference data.
        """
        print("Loading Sentinel-2 time series data...")
        
        # Load Sentinel-2 data
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
            
        print(f"Sentinel-2 data loaded:")
        print(f"  - Shape: {self.sentinel_data['shape']}")
        print(f"  - Bands: {self.sentinel_data['count']}")
        print(f"  - CRS: {self.sentinel_data['crs']}")
        print(f"  - Bounds: {self.sentinel_data['bounds']}")
        
        # Load reference data
        print(f"\nLoading reference classification data...")
        with rasterio.open(self.reference_path) as src:
            self.reference_data = {
                'data': src.read(1),  # Single band
                'meta': src.meta.copy(),
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds
            }
            
        print(f"Reference data loaded:")
        print(f"  - Shape: {self.reference_data['data'].shape}")
        print(f"  - Unique classes: {np.unique(self.reference_data['data'])}")
        
        return self.sentinel_data, self.reference_data
    
    def explore_sentinel_bands(self, sample_bands=[0, 60, 120, 180, 239]):
        """
        Explore a sample of Sentinel-2 bands to understand the time series structure.
        
        Parameters:
        -----------
        sample_bands : list
            List of band indices to visualize
        """
        if self.sentinel_data is None:
            self.load_raster_data()
            
        print(f"\nExploring Sentinel-2 bands (240 total bands representing time series)...")
        
        # Create subplots for sample bands
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, band_idx in enumerate(sample_bands):
            if i < 5:  # Show 5 sample bands
                band_data = self.sentinel_data['data'][band_idx]
                axes[i].imshow(band_data, cmap='viridis')
                axes[i].set_title(f'Band {band_idx} (Time step {band_idx//10})')
                axes[i].axis('off')
        
        # Show reference classification in the last subplot
        axes[5].imshow(self.reference_data['data'], cmap='tab20')
        axes[5].set_title('Reference Classification (26 classes)')
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig('kastoria_sentinel_timeseries.png')
        plt.show()
        
        # Show basic statistics
        print(f"\nSentinel-2 Data Statistics:")
        print(f"  - Min value: {np.nanmin(self.sentinel_data['data']):.2f}")
        print(f"  - Max value: {np.nanmax(self.sentinel_data['data']):.2f}")
        print(f"  - Mean value: {np.nanmean(self.sentinel_data['data']):.2f}")
        print(f"  - Data type: {self.sentinel_data['data'].dtype}")
    
    def create_study_area_polygon(self, center_lat=40.513, center_lon=21.269, buffer_km=5):
        """
        Create a study area polygon around Kastoria city center.
        
        Parameters:
        -----------
        center_lat : float
            Latitude of the center point (Kastoria city center)
        center_lon : float
            Longitude of the center point
        buffer_km : float
            Buffer distance in kilometers
        """
        # Create center point
        center_point = Point(center_lon, center_lat)
        
        # Create a GeoDataFrame with the center point
        gdf_point = gpd.GeoDataFrame([1], geometry=[center_point], crs='EPSG:4326')
        
        # Project to a metric CRS (Greek Grid - EPSG:2100)
        gdf_point_projected = gdf_point.to_crs('EPSG:2100')
        
        # Create buffer (buffer_km * 1000 meters)
        buffer_polygon = gdf_point_projected.geometry.buffer(buffer_km * 1000)
        
        # Convert back to WGS84
        gdf_polygon = gpd.GeoDataFrame([1], geometry=buffer_polygon, crs='EPSG:2100')
        self.study_area_polygon = gdf_polygon.to_crs('EPSG:4326')
        
        print(f"\nStudy area polygon created:")
        print(f"  - Center: {center_lat:.3f}°N, {center_lon:.3f}°E")
        print(f"  - Buffer: {buffer_km} km")
        print(f"  - Area: {self.study_area_polygon.to_crs('EPSG:2100').area.iloc[0] / 1e6:.2f} km²")
        
        return self.study_area_polygon
    
    def visualize_study_area(self):
        """
        Create an interactive map showing the study area and data extent.
        """
        if self.study_area_polygon is None:
            self.create_study_area_polygon()
            
        # Get the center of the study area for the map
        bounds = self.study_area_polygon.bounds
        center_lat = (bounds.miny.iloc[0] + bounds.maxy.iloc[0]) / 2
        center_lon = (bounds.minx.iloc[0] + bounds.maxx.iloc[0]) / 2
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add study area polygon
        folium.GeoJson(
            self.study_area_polygon.__geo_interface__,
            style_function=lambda x: {
                'fillColor': 'red',
                'color': 'red',
                'weight': 2,
                'fillOpacity': 0.2
            },
            popup='Study Area'
        ).add_to(m)
        
        # Add center point
        folium.Marker(
            [center_lat, center_lon],
            popup=f'Study Area Center<br>Lat: {center_lat:.4f}<br>Lon: {center_lon:.4f}',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
    def load_existing_geojson(self, filename='kastoria_study_area.geojson'):
        """
        Load existing GeoJSON file if it exists.
        
        Parameters:
        -----------
        filename : str
            Path to the GeoJSON file
        """
        try:
            self.study_area_polygon = gpd.read_file(filename)
            area_km2 = self.study_area_polygon.to_crs('EPSG:2100').area.iloc[0] / 1e6
            print(f"✅ Existing GeoJSON loaded: {filename}")
            print(f"   - Area: {area_km2:.2f} km²")
            return self.study_area_polygon
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return None
        except Exception as e:
            print(f"❌ Error loading GeoJSON: {e}")
            return None
    
    def summarize_data_sources(self):
        """
        Provide a summary of all loaded data sources.
        """
        print("\n" + "="*60)
        print("DATA SOURCES SUMMARY")
        print("="*60)
        
        print("\n1. RASTER DATA (Source: Sentinel-2 time series)")
        if self.sentinel_data:
            print(f"   ✓ Kastoria.tif: {self.sentinel_data['count']} bands, {self.sentinel_data['shape']} pixels")
            print(f"   ✓ Spatial resolution: ~10m (estimated)")
            print(f"   ✓ CRS: {self.sentinel_data['crs']}")
        
        if self.reference_data:
            print(f"   ✓ Reference classification: 26 land cover classes")
        
        print("\n2. VECTOR DATA (Source: Accurate polygon from geojson.io)")
        if self.study_area_polygon is not None:
            area_km2 = self.study_area_polygon.to_crs('EPSG:2100').area.iloc[0] / 1e6
            print(f"   ✓ Study area polygon: {area_km2:.2f} km²")
            print(f"   ✓ CRS: {self.study_area_polygon.crs}")
            print(f"   ✓ Verification: Enhanced landmark coverage check")
            print(f"   ✓ Saved to: kastoria_study_area.geojson")
        
        print("\n3. TIME SERIES DATA (Source: NASA POWER)")
        if self.meteorological_data is not None:
            print(f"   ✓ Meteorological data: {len(self.meteorological_data)} daily records")
            variables = [col for col in self.meteorological_data.columns if col not in ['date', 'latitude', 'longitude']]
            print(f"   ✓ Variables: {', '.join(variables)}")
            print(f"   ✓ Period: {self.meteorological_data['date'].min().strftime('%Y-%m-%d')} to {self.meteorological_data['date'].max().strftime('%Y-%m-%d')}")
            print(f"   ✓ Saved to: kastoria_meteorological_data.csv")
        
        print("\n" + "="*60)
    
    def load_polygon_from_geojson(self, geojson_data, save_to_file=True, filename='kastoria_study_area.geojson'):
        """
        Load study area polygon from GeoJSON data (from geojson.io or manual creation).
        
        Parameters:
        -----------
        geojson_data : dict or str
            GeoJSON data as dictionary or file path
        save_to_file : bool
            Whether to save the GeoJSON to a file
        filename : str
            Output filename for the GeoJSON file
        """
        if isinstance(geojson_data, str):
            # If it's a file path
            self.study_area_polygon = gpd.read_file(geojson_data)
        else:
            # If it's a GeoJSON dictionary
            self.study_area_polygon = gpd.GeoDataFrame.from_features(
                geojson_data['features'], crs='EPSG:4326'
            )
            
            # Save to file if requested
            if save_to_file:
                try:
                    self.study_area_polygon.to_file(filename, driver='GeoJSON')
                    print(f"  - ✅ GeoJSON saved to: {filename}")
                except Exception as e:
                    print(f"  - ❌ Error saving GeoJSON: {e}")
        
        print(f"Polygon loaded from GeoJSON:")
        print(f"  - Bounds: {self.study_area_polygon.bounds}")
        area_km2 = self.study_area_polygon.to_crs('EPSG:2100').area.iloc[0] / 1e6
        print(f"  - Area: {area_km2:.2f} km²")
        
        return self.study_area_polygon
    
    def verify_kastoria_location(self):
        """
        Verify that the study area polygon actually covers Kastoria city.
        Uses known landmarks and coordinates to validate.
        """
        # Enhanced Kastoria landmarks with coordinates (more comprehensive)
        kastoria_landmarks = {
            'Kastoria_City_Center': (21.2685, 40.5167),
            'Kastoria_Lake_Center': (21.2750, 40.5150),
            'Byzantine_Museum': (21.2672, 40.5189),
            'Olympic_Stadium': (21.2584, 40.5201),
            'Costume_Museum': (21.2750, 40.5100),
            'Kastoria_Port': (21.2700, 40.5140),
            'Old_Town': (21.2665, 40.5195),
            'Dragon_Cave': (21.2612, 40.5089),
            'Kastoria_Airport': (21.2816, 40.4932),
            'Lake_Orestiada_North': (21.2720, 40.5200),
            'Lake_Orestiada_South': (21.2780, 40.5050),
            'Kastoria_Hospital': (21.2640, 40.5160),
            'University_Campus': (21.2590, 40.5230),
            'Central_Square': (21.2670, 40.5175)
        }
        
        if self.study_area_polygon is None:
            print("❌ No study area polygon defined!")
            return False
        
        print(f"\n🔍 VERIFYING KASTORIA LOCATION")
        print(f"{'='*50}")
        
        verification_results = {}
        landmarks_inside = 0
        important_landmarks = ['Kastoria_City_Center', 'Kastoria_Lake_Center', 'Byzantine_Museum', 'Old_Town']
        important_inside = 0
        
        for landmark, (lon, lat) in kastoria_landmarks.items():
            point = Point(lon, lat)
            is_inside = self.study_area_polygon.contains(point).iloc[0]
            verification_results[landmark] = {
                'coordinates': (lon, lat),
                'inside_polygon': is_inside
            }
            if is_inside:
                landmarks_inside += 1
                if landmark in important_landmarks:
                    important_inside += 1
            
            status = "✅ INSIDE" if is_inside else "❌ OUTSIDE"
            importance = "⭐" if landmark in important_landmarks else "  "
            print(f"{importance} {landmark:20s}: {lat:.4f}°N, {lon:.4f}°E - {status}")
        
        coverage_percentage = (landmarks_inside / len(kastoria_landmarks)) * 100
        important_coverage = (important_inside / len(important_landmarks)) * 100
        
        print(f"\n📊 COVERAGE SUMMARY:")
        print(f"   Total landmarks inside: {landmarks_inside}/{len(kastoria_landmarks)} ({coverage_percentage:.1f}%)")
        print(f"   Important landmarks: {important_inside}/{len(important_landmarks)} ({important_coverage:.1f}%)")
        
        # More nuanced validation
        if important_coverage == 100 and coverage_percentage >= 70:
            print(f"   ✅ POLYGON VALIDATION: EXCELLENT (All key landmarks covered)")
            return True
        elif important_coverage >= 75 and coverage_percentage >= 50:
            print(f"   ✅ POLYGON VALIDATION: GOOD (Most key landmarks covered)")
            return True
        elif important_coverage >= 50:
            print(f"   ⚠️  POLYGON VALIDATION: ACCEPTABLE (Some key landmarks covered)")
            return True
        else:
            print(f"   ❌ POLYGON VALIDATION: POOR (Few key landmarks covered)")
            return False
    
    def download_nasa_power_data(self, start_date='2018-01-01', end_date='2023-12-31', 
                                variables=['T2M', 'PRECTOTCORR', 'RH2M', 'WS2M'], 
                                save_to_file=True, filename='kastoria_meteorological_data.csv'):
        """
        Download meteorological data from NASA POWER API for the study area center.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        variables : list
            List of meteorological variables to download
        save_to_file : bool
            Whether to save data to CSV file
        filename : str
            Output filename for the CSV file
        """
        if self.study_area_polygon is None:
            self.create_study_area_polygon()
            
        # Get center coordinates
        bounds = self.study_area_polygon.bounds
        center_lat = (bounds.miny.iloc[0] + bounds.maxy.iloc[0]) / 2
        center_lon = (bounds.minx.iloc[0] + bounds.maxx.iloc[0]) / 2
        
        print(f"\nDownloading NASA POWER meteorological data...")
        print(f"  - Location: {center_lat:.4f}°N, {center_lon:.4f}°E")
        print(f"  - Period: {start_date} to {end_date}")
        print(f"  - Variables: {', '.join(variables)}")
        
        # Construct NASA POWER API URL
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': ','.join(variables),
            'community': 'RE',
            'longitude': center_lon,
            'latitude': center_lat,
            'start': start_date.replace('-', ''),
            'end': end_date.replace('-', ''),
            'format': 'JSON'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to pandas DataFrame
            df_list = []
            for variable in variables:
                if variable in data['properties']['parameter']:
                    var_data = data['properties']['parameter'][variable]
                    df_temp = pd.DataFrame(list(var_data.items()), 
                                         columns=['date', variable])
                    if df_list:
                        df_list[0] = df_list[0].merge(df_temp, on='date')
                    else:
                        df_list.append(df_temp)
            
            if df_list:
                self.meteorological_data = df_list[0]
                self.meteorological_data['date'] = pd.to_datetime(self.meteorological_data['date'])
                self.meteorological_data = self.meteorological_data.sort_values('date').reset_index(drop=True)
                
                # Add metadata columns
                self.meteorological_data['latitude'] = center_lat
                self.meteorological_data['longitude'] = center_lon
                
                print(f"  - Successfully downloaded {len(self.meteorological_data)} records")
                print(f"  - Date range: {self.meteorological_data['date'].min()} to {self.meteorological_data['date'].max()}")
                
                # Save to file if requested
                if save_to_file:
                    try:
                        self.meteorological_data.to_csv(filename, index=False)
                        print(f"  - ✅ Data saved to: {filename}")
                    except Exception as e:
                        print(f"  - ❌ Error saving file: {e}")
                
                return self.meteorological_data
            else:
                print("  - No data retrieved from NASA POWER API")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  - Error downloading data: {e}")
            return None
    
    def load_nasa_power_data(self, filename='kastoria_meteorological_data.csv'):
        """
        Load previously saved NASA POWER meteorological data from CSV file.
        
        Parameters:
        -----------
        filename : str
            Path to the CSV file containing meteorological data
        """
        try:
            self.meteorological_data = pd.read_csv(filename)
            self.meteorological_data['date'] = pd.to_datetime(self.meteorological_data['date'])
            
            print(f"✅ Meteorological data loaded from: {filename}")
            print(f"  - Records: {len(self.meteorological_data)}")
            print(f"  - Variables: {', '.join([col for col in self.meteorological_data.columns if col not in ['date', 'latitude', 'longitude']])}")
            print(f"  - Date range: {self.meteorological_data['date'].min()} to {self.meteorological_data['date'].max()}")
            
            return self.meteorological_data
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            print("   Use download_nasa_power_data() to download fresh data.")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def compare_polygons(self, polygon1, polygon2, names=['Generated', 'Manual']):
        """
        Compare two polygons and visualize the differences.
        
        Parameters:
        -----------
        polygon1 : GeoDataFrame
            First polygon to compare
        polygon2 : GeoDataFrame
            Second polygon to compare
        names : list
            Names for the polygons in visualization
        """
        # Calculate areas
        area1 = polygon1.to_crs('EPSG:2100').area.iloc[0] / 1e6
        area2 = polygon2.to_crs('EPSG:2100').area.iloc[0] / 1e6
        
        print(f"\n📏 POLYGON COMPARISON:")
        print(f"   {names[0]} polygon: {area1:.2f} km²")
        print(f"   {names[1]} polygon: {area2:.2f} km²")
        print(f"   Area difference: {abs(area1 - area2):.2f} km² ({abs(area1-area2)/max(area1,area2)*100:.1f}%)")
        
        # Create comparison map
        bounds1 = polygon1.bounds
        bounds2 = polygon2.bounds
        
        # Calculate map center
        all_bounds = pd.concat([bounds1, bounds2])
        center_lat = (all_bounds['miny'].min() + all_bounds['maxy'].max()) / 2
        center_lon = (all_bounds['minx'].min() + all_bounds['maxx'].max()) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add first polygon
        folium.GeoJson(
            polygon1.__geo_interface__,
            style_function=lambda x: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 3,
                'fillOpacity': 0.2
            },
            popup=f'{names[0]} Polygon ({area1:.2f} km²)'
        ).add_to(m)
        
        # Add second polygon
        folium.GeoJson(
            polygon2.__geo_interface__,
            style_function=lambda x: {
                'fillColor': 'red',
                'color': 'red',
                'weight': 3,
                'fillOpacity': 0.2
            },
            popup=f'{names[1]} Polygon ({area2:.2f} km²)'
        ).add_to(m)
        
        # Add Kastoria landmarks for reference
        kastoria_landmarks = {
            'Kastoria City Center': (21.2685, 40.5167),
            'Lake Kastoria': (21.2750, 40.5150),
            'Byzantine Museum': (21.2672, 40.5189)
        }
        
        for name, (lon, lat) in kastoria_landmarks.items():
            folium.Marker(
                [lat, lon],
                popup=name,
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
        
        return m

# Initialize the data processor
processor = KastoriaDataProcessor()

print("🗺️  KASTORIA GEOSPATIAL DATA ANALYSIS")
print("="*60)

# Step 1: Load and explore raster data
print("\nSTEP 1: Loading raster data...")
sentinel_data, reference_data = processor.load_raster_data()

# Step 2: Explore the Sentinel-2 bands structure
print("\nSTEP 2: Exploring Sentinel-2 time series structure...")
processor.explore_sentinel_bands()

# Step 3: Option to use manual polygon from geojson.io
print("\nSTEP 3: Setting up study area polygon...")

# Your improved GeoJSON from geojson.io (more accurate Kastoria boundaries)
accurate_kastoria_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [21.240863800048828, 40.55111333356784],
                        [21.202926635742188, 40.500225211369],
                        [21.28824234008789, 40.461837857613915],
                        [21.33939743041992, 40.48116484751597],
                        [21.338367462158203, 40.54954812134299],
                        [21.27450942993164, 40.56050383863174],
                        [21.240863800048828, 40.55111333356784]
                    ]
                ]
            }
        }
    ]
}

# Try to load existing GeoJSON first, or create new one
print("Checking for existing GeoJSON file...")
existing_polygon = processor.load_existing_geojson()

if existing_polygon is None:
    print("Loading accurate Kastoria polygon from geojson.io...")
    accurate_polygon = processor.load_polygon_from_geojson(accurate_kastoria_geojson, save_to_file=True)
else:
    print("Using existing polygon...")
    accurate_polygon = existing_polygon

# Also create the generated polygon for comparison
print("\nCreating generated polygon for comparison...")
generated_polygon = processor.create_study_area_polygon()

# Step 4: Verify the polygon actually covers Kastoria
print("\nSTEP 4: Verifying polygon location...")
processor.study_area_polygon = accurate_polygon  # Use accurate polygon as primary
verification_result = processor.verify_kastoria_location()

# Step 5: Compare both polygons
print("\nSTEP 5: Comparing accurate vs generated polygons...")
comparison_map = processor.compare_polygons(generated_polygon, accurate_polygon, 
                                          ['Generated (5km buffer)', 'Accurate (geojson.io)'])

# Step 6: Try to load existing NASA data, or download if not available
print("\nSTEP 6: Loading/downloading meteorological data...")
met_data = processor.load_nasa_power_data()

if met_data is None:
    print("No saved data found. Downloading from NASA POWER...")
    met_data = processor.download_nasa_power_data(save_to_file=True)

# Step 7: Summarize all data sources
processor.summarize_data_sources()

print(f"\n{'='*60}")
print("✅ INITIAL DATA LOADING COMPLETE!")
print("="*60)
print("📁 FILES CREATED:")
print("   - kastoria_meteorological_data.csv (NASA POWER data)")
print("   - kastoria_study_area.geojson (Accurate Kastoria polygon)")
print("\n🔍 VERIFICATION RESULTS:")
if verification_result:
    print("   ✅ Polygon successfully covers Kastoria landmarks")
else:
    print("   ⚠️  Consider adjusting polygon boundaries")
print("\n📋 NEXT STEPS:")
print("   - Analyze spectral indices (NDVI, BSI, NDWI)")
print("   - Perform time series analysis") 
print("   - Create interactive visualizations")
print("   - Integrate with OGC web services")

# Display the comparison map
print(f"\n🗺️  Interactive comparison map created!")
print("   - Blue polygon: Generated (5km buffer around city center)")
print("   - Red polygon: Accurate (irregular shape from geojson.io)")
print("   - Green markers: Kastoria landmarks for reference")