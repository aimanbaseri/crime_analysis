import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
import requests
import io
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("Libraries imported successfully!")

# ==============================================================================
# SUMBER DATA (DATA SOURCE)
# ==============================================================================
# Nama Dataset : Crimes by District & Crime Type (Jenayah mengikut Daerah & Jenis)
# Penyedia     : Polis Diraja Malaysia (PDRM) & Jabatan Perangkaan Malaysia (DOSM)
# Platform     : Portal Data Terbuka Kebangsaan (data.gov.my)
# Lesen        : Creative Commons Attribution 4.0 International (CC BY 4.0)
# Pautan Rasmi : https://data.gov.my/data-catalogue/crime_district
# ==============================================================================

print("\n" + "="*80)
print("SUMBER DATA: Jenayah mengikut Daerah & Jenis Jenayah (data.gov.my)")
print("Penyedia Data: Polis Diraja Malaysia (PDRM) & DOSM")
print("Lesen: CC BY 4.0 (Data Terbuka Kerajaan Malaysia)")
print("="*80 + "\n")

# ==========================================
# 1. Load Malaysian Dataset (2018-2024)
# ==========================================
url_data = 'https://storage.data.gov.my/publicsafety/crime_district.parquet'
data = pd.read_parquet(url_data)

data['date'] = pd.to_datetime(data['date'])
data = data[(data['date'] >= '2018-01-01') & (data['date'] <= '2024-12-31')]

print("Total rows for 2018-2024:", len(data))
print("\nMissing values per column:")
print(data.isnull().sum())

data_clean = data.dropna().reset_index(drop=True)
print("Data after cleaning:", len(data_clean))

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
# Pivot data: States as rows, Crime types as columns
crime_pivot = data_clean.pivot_table(
    index='state', 
    columns='type', 
    values='crimes', 
    aggfunc='sum', 
    fill_value=0
).reset_index()

print("\nDeskripsi Statistik")
print(crime_pivot.describe())

# Define crime features (all columns except 'state')
crime_features = [col for col in crime_pivot.columns if col != 'state']

# Boxplot
plt.figure(figsize=(15, 6))
sns.boxplot(data=crime_pivot[crime_features])
plt.title("Distribusi Kes Jenayah per Jenis (Malaysia)")
plt.xlabel("Jenis Jenayah")
plt.ylabel("Jumlah Kes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()  # <--- Important for Python scripts

# Correlation Matrix Heatmap
corr_matrix = crime_pivot[crime_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasi Antara Jenis Jenayah (Malaysia)")
plt.tight_layout()
plt.show()  # <--- Important for Python scripts

# ==========================================
# 3. Scaling Data
# ==========================================
X = crime_pivot[crime_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("5 Baris Pertama Hasil Scaling:")
print(pd.DataFrame(X_scaled[:5], columns=crime_features))

# ==========================================
# 4. Finding Optimal Clusters
# ==========================================
inertia = []
silhouette = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method (Malaysia)")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()  # <--- Important for Python scripts

# Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette, marker="o")
plt.title("Silhouette Score (Malaysia)")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()  # <--- Important for Python scripts

print("Jumlah Cluster | Inertia | Silhouette Score")
for k, i, s in zip(K_range, inertia, silhouette):
    print(f"k={k:>2} | {i:.2f} | {s:.3f}")

# ==========================================
# 5. Final Clustering (Choose k=3)
# ==========================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

crime_pivot["Cluster"] = kmeans.labels_

print("Jumlah Anggota Setiap Cluster:")
print(crime_pivot["Cluster"].value_counts().sort_index())

print("\nContoh Data dengan Label Cluster:")
print(crime_pivot[["state", "Cluster"]].head(10))

summary = crime_pivot.groupby("Cluster")[crime_features].mean()
print("\nRata-rata Kes Jenayah per Cluster:")
print(summary)

# ==========================================
# 6. Visualization on Malaysia GeoJSON Map 
# ==========================================
print("\nGenerating Map...")

# Load the local file
geo_data = gpd.read_file("malaysia.state.geojson")

# DEBUG: Print the map's columns and names (Crucial!)
print("MAP COLUMNS:", geo_data.columns.tolist())
print("MAP STATE NAMES:", geo_data['name'].unique())

# 1. PREPARE DATASET NAMES (Standard uppercase names)
nama_ganti = {
    "Johor": "JOHOR",
    "Kedah": "KEDAH",
    "Kelantan": "KELANTAN",
    "Kuala Lumpur": "WP K LUMPUR",   # Changed to match map
    "Labuan": "WP LABUAN",           # Changed to match map
    "Melaka": "MELAKA",
    "Negeri Sembilan": "NEGERI SEMBILAN",
    "Pahang": "PAHANG",
    "Pulau Pinang": "PULAU PINANG",
    "Perak": "PERAK",
    "Perlis": "PERLIS",
    "Putrajaya": "WP PUTRAJAYA",     # Changed to match map
    "Sabah": "SABAH",
    "Sarawak": "SARAWAK",
    "Selangor": "SELANGOR",
    "Terengganu": "TERENGGANU"
}

# Apply mapping to your dataset
crime_pivot["state_code"] = crime_pivot["state"].replace(nama_ganti).str.upper().str.strip()

# 2. PREPARE MAP NAMES
geo_data["map_code"] = geo_data["name"].str.upper().str.strip()

# 3. MERGE THE DATA
geo_merged = geo_data.merge(
    crime_pivot[['state_code', 'Cluster']], 
    left_on='map_code', 
    right_on='state_code', 
    how='left'
)

# Check if any state is missing
missing = geo_merged[geo_merged['Cluster'].isna()]
if len(missing) > 0:
    print("\nWARNING: These map states did NOT match your data:")
    print(missing['map_code'].unique())
else:
    print("\nAll map states matched your data perfectly!")

# 4. Create Map
m = folium.Map(location=[4.2105, 101.9758], zoom_start=5, tiles="OpenStreetMap")

# 5. Add the Choropleth
folium.Choropleth(
    geo_data=geo_merged,
    name="choropleth",
    data=geo_merged,
    columns=['state_code', 'Cluster'],
    key_on="feature.properties.map_code",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Cluster Jenayah (Malaysia)"
).add_to(m)

# Save map to HTML
output_file = "malaysia_cluster_map.html"
m.save(output_file)
print(f"Map saved successfully! Open '{output_file}' in your web browser to view it.")

# Automatically open in browser
import webbrowser
import os
webbrowser.open('file://' + os.path.realpath(output_file))