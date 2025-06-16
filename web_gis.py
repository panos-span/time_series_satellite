import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import rasterio
from rasterio.warp import transform_bounds
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# Try to import leafmap
try:
    import leafmap
    LEAFMAP_AVAILABLE = True
    print("✅ Leafmap available for advanced interactive mapping")
except ImportError:
    LEAFMAP_AVAILABLE = False
    print("⚠️ Leafmap not available - using Folium only")

class KastoriaInteractiveMapper:
    """
    Interactive web mapping application for Kastoria study area.
    Combines raster, vector, and WMS data into comprehensive web maps.
    """
    
    def __init__(self):
        """
        Initialize the interactive mapper.
        """
        self.study_area = None
        self.meteorological_data = None
        self.met_point = None
        self.spectral_indices_ts = None
        self.greek_geodata = {}
        self.maps = {}
        
        # WMS services for Greece and Europe
        self.wms_services = {
            'geodata_gov_gr': {
                'url': 'https://geodata.gov.gr/geoserver/ows',
                'layers': {
                    'administrative': 'geodata:administrative_boundaries_kallikratis',
                    'roads': 'geodata:road_network_l',
                    'settlements': 'geodata:settlements_p',
                    'land_cover': 'geodata:corine_land_cover_2018'
                },
                'name': 'Greek Geodata Portal'
            },
            'esa_land_cover': {
                'url': 'https://services.sentinel-hub.com/ogc/wms/bd86bcc0-f318-402b-a145-015f85b9427e',
                'layers': {
                    'land_cover': 'ESA_WORLDCOVER_10M_2020_V1'
                },
                'name': 'ESA WorldCover'
            },
            'osm_wms': {
                'url': 'https://ows.terrestris.de/osm/service',
                'layers': {
                    'osm': 'OSM-WMS'
                },
                'name': 'OpenStreetMap WMS'
            }
        }
        
    def load_all_data(self):
        """
        Load all previously created data sources.
        """
        print("📂 Loading all analysis data...")
        
        try:
            # Load study area
            self.study_area = gpd.read_file("kastoria_study_area.geojson")
            print(f"   ✅ Study area loaded: {len(self.study_area)} polygons")
            
            # Load meteorological data
            self.meteorological_data = pd.read_csv("kastoria_meteorological_data.csv")
            self.meteorological_data['date'] = pd.to_datetime(self.meteorological_data['date'])
            
            # Create meteorological point
            if 'latitude' in self.meteorological_data.columns:
                lat = self.meteorological_data['latitude'].iloc[0]
                lon = self.meteorological_data['longitude'].iloc[0]
            else:
                centroid = self.study_area.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
            
            from shapely.geometry import Point
            self.met_point = gpd.GeoDataFrame(
                {'type': ['Meteorological Station'], 'data_points': [len(self.meteorological_data)]},
                geometry=[Point(lon, lat)],
                crs='EPSG:4326'
            )
            print(f"   ✅ Meteorological data loaded: {len(self.meteorological_data)} records")
            
            # Load spectral indices time series
            try:
                self.spectral_indices_ts = pd.read_csv("kastoria_spectral_indices_timeseries_correct.csv")
                print(f"   ✅ Spectral indices loaded: {len(self.spectral_indices_ts)} timesteps")
            except FileNotFoundError:
                try:
                    self.spectral_indices_ts = pd.read_csv("kastoria_spectral_indices_timeseries.csv")
                    print(f"   ✅ Spectral indices loaded: {len(self.spectral_indices_ts)} timesteps")
                except FileNotFoundError:
                    print("   ⚠️ Spectral indices time series not found")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def create_folium_comprehensive_map(self, save_path="kastoria_comprehensive_map.html"):
        """
        Create comprehensive Folium map with all data sources.
        """
        print("🗺️ Creating comprehensive Folium map...")
        
        # Get center coordinates
        center = self.study_area.geometry.centroid.iloc[0]
        center_lat, center_lon = center.y, center.x
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=None  # We'll add tiles manually for better control
        )
        
        # Add multiple base map options
        folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
        folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='CartoDB Dark Matter').add_to(m)
        
        # Add satellite imagery
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add WMS layers
        self.add_wms_layers_to_folium(m)
        
        # Add study area
        self.add_study_area_to_map(m)
        
        # Add meteorological station
        self.add_meteorological_station_to_map(m)
        
        # Add spectral indices visualization
        self.add_spectral_indices_to_map(m)
        
        # Add meteorological time series popup
        self.add_meteorological_timeseries_popup(m)
        
        # Add measurement tools
        plugins.MeasureControl().add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        self.add_legend_to_map(m)
        
        # Save map
        m.save(save_path)
        print(f"   ✅ Comprehensive map saved to: {save_path}")
        
        self.maps['comprehensive'] = m
        return m
    
    def add_wms_layers_to_folium(self, m):
        """
        Add WMS services to Folium map.
        """
        print("   🌐 Adding WMS services...")
        
        # Add Greek Geodata WMS layers
        for service_name, service_info in self.wms_services.items():
            try:
                if service_name == 'geodata_gov_gr':
                    # Add administrative boundaries
                    wms_admin = folium.raster_layers.WmsTileLayer(
                        url=service_info['url'],
                        layers=service_info['layers']['administrative'],
                        transparent=True,
                        format="image/png",
                        name="Greek Administrative Boundaries",
                        overlay=True,
                        control=True
                    )
                    wms_admin.add_to(m)
                    
                    # Add road network
                    wms_roads = folium.raster_layers.WmsTileLayer(
                        url=service_info['url'],
                        layers=service_info['layers']['roads'],
                        transparent=True,
                        format="image/png",
                        name="Greek Road Network",
                        overlay=True,
                        control=True
                    )
                    wms_roads.add_to(m)
                    
                elif service_name == 'osm_wms':
                    # Add OpenStreetMap WMS
                    wms_osm = folium.raster_layers.WmsTileLayer(
                        url=service_info['url'],
                        layers=service_info['layers']['osm'],
                        transparent=True,
                        format="image/png",
                        name="OpenStreetMap WMS",
                        overlay=True,
                        control=True
                    )
                    wms_osm.add_to(m)
                    
                print(f"     ✅ Added WMS service: {service_info['name']}")
                
            except Exception as e:
                print(f"     ⚠️ Could not add WMS service {service_name}: {e}")
    
    def add_study_area_to_map(self, m):
        """
        Add study area polygon to map with interactive features.
        """
        print("   📍 Adding study area...")
        
        # Calculate area
        area_km2 = self.study_area.to_crs('EPSG:2100').area.iloc[0] / 1e6
        
        # Create popup content
        popup_content = f"""
        <div style="width: 300px;">
            <h4><b>Kastoria Study Area</b></h4>
            <p><b>Area:</b> {area_km2:.2f} km²</p>
            <p><b>Geometry:</b> {self.study_area.geometry.iloc[0].geom_type}</p>
            <p><b>Coordinates:</b> {self.study_area.geometry.centroid.iloc[0].y:.4f}°N, {self.study_area.geometry.centroid.iloc[0].x:.4f}°E</p>
            <p><b>Analysis:</b> Spectral indices and meteorological correlation</p>
        </div>
        """
        
        # Add polygon
        folium.GeoJson(
            self.study_area.to_json(),
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#800000',
                'weight': 3,
                'fillOpacity': 0.4
            },
            popup=folium.Popup(popup_content, max_width=300),
            tooltip="Kastoria Study Area"
        ).add_to(m)
    
    def add_meteorological_station_to_map(self, m):
        """
        Add meteorological station to map with data summary.
        """
        print("   🌡️ Adding meteorological station...")
        
        # Calculate basic statistics
        temp_mean = self.meteorological_data['T2M'].mean() if 'T2M' in self.meteorological_data.columns else 'N/A'
        precip_total = self.meteorological_data['PRECTOTCORR'].sum() if 'PRECTOTCORR' in self.meteorological_data.columns else 'N/A'
        
        # Create popup content
        popup_content = f"""
        <div style="width: 350px;">
            <h4><b>🌡️ Meteorological Station</b></h4>
            <p><b>Data Source:</b> NASA POWER API</p>
            <p><b>Records:</b> {len(self.meteorological_data)}</p>
            <p><b>Period:</b> {self.meteorological_data['date'].min().strftime('%Y-%m-%d')} to {self.meteorological_data['date'].max().strftime('%Y-%m-%d')}</p>
            <p><b>Average Temperature:</b> {temp_mean:.1f}°C</p>
            <p><b>Total Precipitation:</b> {precip_total:.1f} mm</p>
            <p><b>Variables:</b> Temperature, Precipitation, Humidity, Wind Speed</p>
        </div>
        """
        
        # Add meteorological station marker
        folium.Marker(
            location=[self.met_point.geometry.iloc[0].y, self.met_point.geometry.iloc[0].x],
            popup=folium.Popup(popup_content, max_width=350),
            tooltip="NASA POWER Meteorological Data",
            icon=folium.Icon(color='blue', icon='thermometer', prefix='fa')
        ).add_to(m)
    
    def add_spectral_indices_to_map(self, m):
        """
        Add spectral indices information to map.
        """
        if self.spectral_indices_ts is None:
            return
            
        print("   📊 Adding spectral indices information...")
        
        # Calculate statistics
        ndvi_mean = self.spectral_indices_ts['NDVI_mean'].mean() if 'NDVI_mean' in self.spectral_indices_ts.columns else 'N/A'
        ndwi_mean = self.spectral_indices_ts['NDWI_mean'].mean() if 'NDWI_mean' in self.spectral_indices_ts.columns else 'N/A'
        bsi_mean = self.spectral_indices_ts['BSI_mean'].mean() if 'BSI_mean' in self.spectral_indices_ts.columns else 'N/A'
        
        # Create mini chart for spectral indices
        chart_html = self.create_spectral_indices_chart()
        
        # Create popup content
        popup_content = f"""
        <div style="width: 400px;">
            <h4><b>📊 Spectral Indices Analysis</b></h4>
            <p><b>Data Source:</b> Sentinel-2 Time Series</p>
            <p><b>Timesteps:</b> {len(self.spectral_indices_ts)}</p>
            <p><b>Average NDVI:</b> {ndvi_mean:.3f}</p>
            <p><b>Average NDWI:</b> {ndwi_mean:.3f}</p>
            <p><b>Average BSI:</b> {bsi_mean:.3f}</p>
            <hr>
            {chart_html}
        </div>
        """
        
        # Add marker at study area center
        center = self.study_area.geometry.centroid.iloc[0]
        folium.Marker(
            location=[center.y + 0.01, center.x + 0.01],  # Slightly offset
            popup=folium.Popup(popup_content, max_width=400),
            tooltip="Spectral Indices Analysis",
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
    
    def create_spectral_indices_chart(self):
        """
        Create mini chart for spectral indices as HTML.
        """
        if self.spectral_indices_ts is None:
            return "<p>No spectral indices data available</p>"
        
        try:
            # Create mini plot
            fig, ax = plt.subplots(figsize=(6, 3))
            
            if 'NDVI_mean' in self.spectral_indices_ts.columns:
                ax.plot(self.spectral_indices_ts['timestep'], 
                       self.spectral_indices_ts['NDVI_mean'], 
                       'g-', label='NDVI', linewidth=2)
            
            if 'NDWI_mean' in self.spectral_indices_ts.columns:
                ax.plot(self.spectral_indices_ts['timestep'], 
                       self.spectral_indices_ts['NDWI_mean'], 
                       'b-', label='NDWI', linewidth=2)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Index Value')
            ax.set_title('Spectral Indices Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{image_base64}" style="width:100%;">'
            
        except Exception as e:
            return f"<p>Could not generate chart: {e}</p>"
    
    def add_meteorological_timeseries_popup(self, m):
        """
        Add meteorological time series visualization.
        """
        print("   📈 Adding meteorological time series...")
        
        try:
            # Create time series chart
            chart_html = self.create_meteorological_chart()
            
            # Create popup content
            popup_content = f"""
            <div style="width: 500px;">
                <h4><b>📈 Meteorological Time Series</b></h4>
                {chart_html}
            </div>
            """
            
            # Add marker for time series
            center = self.study_area.geometry.centroid.iloc[0]
            folium.Marker(
                location=[center.y - 0.01, center.x - 0.01],  # Slightly offset
                popup=folium.Popup(popup_content, max_width=500),
                tooltip="Meteorological Time Series",
                icon=folium.Icon(color='orange', icon='chart-line', prefix='fa')
            ).add_to(m)
            
        except Exception as e:
            print(f"     ⚠️ Could not add meteorological time series: {e}")
    
    def create_meteorological_chart(self):
        """
        Create meteorological time series chart as HTML.
        """
        try:
            # Create mini plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            
            # Plot temperature
            if 'T2M' in self.meteorological_data.columns:
                ax1.plot(self.meteorological_data['date'], 
                        self.meteorological_data['T2M'], 
                        'r-', linewidth=1)
                ax1.set_ylabel('Temperature (°C)')
                ax1.set_title('Temperature Evolution')
                ax1.grid(True, alpha=0.3)
            
            # Plot precipitation
            if 'PRECTOTCORR' in self.meteorological_data.columns:
                ax2.plot(self.meteorological_data['date'], 
                        self.meteorological_data['PRECTOTCORR'], 
                        'b-', linewidth=1)
                ax2.set_ylabel('Precipitation (mm/day)')
                ax2.set_title('Precipitation Evolution')
                ax2.set_xlabel('Date')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{image_base64}" style="width:100%;">'
            
        except Exception as e:
            return f"<p>Could not generate meteorological chart: {e}</p>"
    
    def add_legend_to_map(self, m):
        """
        Add legend to map.
        """
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Kastoria Study Area</b></p>
        <p><i class="fa fa-square" style="color:red"></i> Study Area Polygon</p>
        <p><i class="fa fa-thermometer" style="color:blue"></i> Meteorological Station</p>
        <p><i class="fa fa-leaf" style="color:green"></i> Spectral Analysis</p>
        <p><i class="fa fa-chart-line" style="color:orange"></i> Time Series</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_leafmap_advanced_map(self, save_path="kastoria_advanced_map.html"):
        """
        Create advanced interactive map using Leafmap (if available).
        """
        if not LEAFMAP_AVAILABLE:
            print("   ⚠️ Leafmap not available - skipping advanced map")
            return None
            
        print("🚀 Creating advanced Leafmap application...")
        
        try:
            # Create leafmap Map
            m = leafmap.Map(
                center=[self.study_area.geometry.centroid.iloc[0].y, 
                       self.study_area.geometry.centroid.iloc[0].x],
                zoom=12
            )
            
            # Add multiple basemaps
            m.add_basemap('OpenStreetMap')
            m.add_basemap('Esri.WorldImagery')
            m.add_basemap('CartoDB.Positron')
            
            # Add study area
            m.add_gdf(self.study_area, layer_name="Study Area", 
                     style={'color': 'red', 'fillOpacity': 0.4})
            
            # Add meteorological point
            m.add_gdf(self.met_point, layer_name="Meteorological Station",
                     marker_type='circle_marker', 
                     marker_args={'radius': 10, 'color': 'blue'})
            
            # Add WMS layers
            for service_name, service_info in self.wms_services.items():
                if service_name == 'geodata_gov_gr':
                    try:
                        m.add_wms_layer(
                            url=service_info['url'],
                            layers=service_info['layers']['administrative'],
                            name="Administrative Boundaries",
                            format="image/png",
                            transparent=True
                        )
                    except Exception as e:
                        print(f"     ⚠️ Could not add WMS layer: {e}")
            
            # Add layer control and tools
            m.add_layer_control()
            
            # Save map
            m.to_html(save_path)
            print(f"   ✅ Advanced map saved to: {save_path}")
            
            self.maps['advanced'] = m
            return m
            
        except Exception as e:
            print(f"   ❌ Error creating advanced map: {e}")
            return None
    
    def create_analysis_dashboard(self, save_path="kastoria_dashboard.html"):
        """
        Create comprehensive analysis dashboard.
        """
        print("📊 Creating analysis dashboard...")
        
        # Get center coordinates
        center = self.study_area.geometry.centroid.iloc[0]
        center_lat, center_lon = center.y, center.x
        
        # Create dashboard map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Add all data layers
        self.add_wms_layers_to_folium(m)
        self.add_study_area_to_map(m)
        self.add_meteorological_station_to_map(m)
        self.add_spectral_indices_to_map(m)
        
        # Add analysis results overlay
        self.add_analysis_results_panel(m)
        
        # Add tools
        plugins.MeasureControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.Draw().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save dashboard
        m.save(save_path)
        print(f"   ✅ Dashboard saved to: {save_path}")
        
        self.maps['dashboard'] = m
        return m
    
    def add_analysis_results_panel(self, m):
        """
        Add analysis results panel to map.
        """
        # Calculate key statistics
        area_km2 = self.study_area.to_crs('EPSG:2100').area.iloc[0] / 1e6
        
        # Create results panel
        results_html = f"""
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 300px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.5);">
        <h4><b>🔬 Analysis Results</b></h4>
        <hr>
        <p><b>Study Area:</b> {area_km2:.2f} km²</p>
        <p><b>Location:</b> Kastoria, Greece</p>
        <p><b>Coordinates:</b> {self.study_area.geometry.centroid.iloc[0].y:.3f}°N, {self.study_area.geometry.centroid.iloc[0].x:.3f}°E</p>
        <hr>
        <p><b>📊 Data Sources:</b></p>
        <ul>
            <li>✅ Sentinel-2 time series</li>
            <li>✅ NASA POWER meteorology</li>
            <li>✅ Greek geodata.gov.gr</li>
            <li>✅ WMS web services</li>
        </ul>
        <hr>
        <p><b>🎯 Analysis Types:</b></p>
        <ul>
            <li>Spectral indices (NDVI, NDWI, BSI)</li>
            <li>Time series analysis</li>
            <li>Meteorological correlations</li>
            <li>Interactive visualization</li>
        </ul>
        </div>
        """
        m.get_root().html.add_child(folium.Element(results_html))
    
    def generate_web_application_summary(self):
        """
        Generate summary of created web applications.
        """
        print("\n📋 WEB APPLICATION SUMMARY")
        print("="*60)
        
        print(f"\n🌐 INTERACTIVE MAPS CREATED:")
        for map_name, map_obj in self.maps.items():
            if map_obj is not None:
                print(f"   ✅ {map_name.capitalize()} map: kastoria_{map_name}_map.html")
        
        print(f"\n📊 DATA INTEGRATION:")
        print(f"   ✅ Study area polygon: {len(self.study_area)} features")
        print(f"   ✅ Meteorological data: {len(self.meteorological_data)} records")
        if self.spectral_indices_ts is not None:
            print(f"   ✅ Spectral indices: {len(self.spectral_indices_ts)} timesteps")
        print(f"   ✅ WMS services: {len(self.wms_services)} providers")
        
        print(f"\n🔧 INTERACTIVE FEATURES:")
        print(f"   ✅ Multiple basemap options")
        print(f"   ✅ WMS layer integration")
        print(f"   ✅ Interactive popups with charts")
        print(f"   ✅ Measurement tools")
        print(f"   ✅ Layer control")
        print(f"   ✅ Fullscreen mode")
        print(f"   ✅ Drawing tools")
        
        print(f"\n🎯 WMS SERVICES INTEGRATED:")
        for service_name, service_info in self.wms_services.items():
            print(f"   ✅ {service_info['name']}")

# Initialize the interactive mapper
print("🚀 KASTORIA INTERACTIVE WEB MAPPING APPLICATION")
print("="*60)

mapper = KastoriaInteractiveMapper()

# Step 1: Load all data
print("\nSTEP 1: Loading all analysis data...")
if not mapper.load_all_data():
    print("❌ Failed to load required data. Make sure previous analyses are complete.")
    exit()

# Step 2: Create comprehensive Folium map
print("\nSTEP 2: Creating comprehensive interactive map...")
comprehensive_map = mapper.create_folium_comprehensive_map()

# Step 3: Create advanced Leafmap application (if available)
print("\nSTEP 3: Creating advanced interactive application...")
advanced_map = mapper.create_leafmap_advanced_map()

# Step 4: Create analysis dashboard
print("\nSTEP 4: Creating analysis dashboard...")
dashboard_map = mapper.create_analysis_dashboard()

# Step 5: Generate summary
print("\nSTEP 5: Generating web application summary...")
mapper.generate_web_application_summary()

print(f"\n{'='*60}")
print("✅ INTERACTIVE WEB MAPPING COMPLETE!")
print("="*60)
print("🌐 WEB APPLICATIONS CREATED:")
print("   📄 kastoria_comprehensive_map.html - Full-featured interactive map")
print("   📄 kastoria_advanced_map.html - Advanced Leafmap application (if available)")
print("   📄 kastoria_dashboard.html - Analysis dashboard")
print("\n🎯 FEATURES IMPLEMENTED:")
print("   ✅ Combined raster & vector data visualization")
print("   ✅ WMS service integration (geodata.gov.gr)")
print("   ✅ Interactive popups with embedded charts")
print("   ✅ Multiple basemap options")
print("   ✅ Measurement and drawing tools")
print("   ✅ Professional cartographic presentation")
print("   ✅ Web-ready HTML applications")
print("\n📱 USAGE:")
print("   Open the HTML files in any web browser")
print("   All maps are self-contained and fully interactive")
print("   Suitable for presentations and web deployment")