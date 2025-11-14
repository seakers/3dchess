#!/usr/bin/env python3
"""
Script to plot wildfire points on Earth surface map.
Reads wildfire_points.csv and wildfire_events.csv and displays them on a 3D globe or 2D map.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Install with: pip install cartopy")
    print("Falling back to basic 3D sphere plot.")


def lat_lon_to_3d(lat, lon, radius=1.0):
    """
    Convert latitude and longitude to 3D Cartesian coordinates on a sphere.
    
    Parameters:
    -----------
    lat : float or array
        Latitude in degrees
    lon : float or array
        Longitude in degrees
    radius : float
        Radius of the sphere (default: 1.0 for unit sphere)
    
    Returns:
    --------
    x, y, z : arrays
        3D Cartesian coordinates
    """
    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Convert to 3D coordinates
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z


def create_sphere(radius=1.0, resolution=50):
    """
    Create a wireframe sphere for visualization.
    
    Parameters:
    -----------
    radius : float
        Radius of the sphere
    resolution : int
        Number of points along each axis
    
    Returns:
    --------
    x, y, z : arrays
        Sphere surface coordinates
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def plot_2d_earth_map(points_df, events_df, script_dir):
    """Plot points on a 2D Earth surface map using Cartopy."""
    # Extract coordinates
    points_lat = points_df['lat [deg]'].values
    points_lon = points_df['lon [deg]'].values
    
    events_lat = events_df['lat [deg]'].values
    events_lon = events_df['lon [deg]'].values
    
    # Create figure with Cartopy projection
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add Earth features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
    ax.add_feature(cfeature.LAKES, alpha=0.3, color='lightblue')
    
    # Add gridlines
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Plot all points from wildfire_points.csv
    ax.scatter(points_lon, points_lat, 
               c='blue', s=2, alpha=0.5, 
               transform=ccrs.PlateCarree(),
               label=f'Wildfire Points ({len(points_df)})',
               edgecolors='none')
    
    # Plot events from wildfire_events.csv with different color based on severity
    if 'severity' in events_df.columns:
        severity = events_df['severity'].values
        scatter = ax.scatter(events_lon, events_lat, 
                           c=severity, s=100, alpha=0.8, 
                           cmap='hot', 
                           transform=ccrs.PlateCarree(),
                           label=f'Wildfire Events ({len(events_df)})',
                           edgecolors='black', linewidth=0.5)
        # Add colorbar for severity
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                           pad=0.05, fraction=0.05, aspect=40)
        cbar.set_label('Severity', fontsize=12)
    else:
        ax.scatter(events_lon, events_lat, 
                   c='red', s=100, alpha=0.8, 
                   transform=ccrs.PlateCarree(),
                   label=f'Wildfire Events ({len(events_df)})',
                   edgecolors='black', linewidth=0.5)
    
    # Set global extent
    ax.set_global()
    
    # Set title
    ax.set_title('Wildfire Points and Events on Earth Surface Map', fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(script_dir, 'wildfire_earth_map_2d.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2D Earth map saved to {output_file}")
    
    plt.show()


def plot_3d_globe_with_cartopy(points_df, events_df, script_dir):
    """Plot points on a 3D globe with Earth surface features using Cartopy."""
    # Extract coordinates
    points_lat = points_df['lat [deg]'].values
    points_lon = points_df['lon [deg]'].values
    
    events_lat = events_df['lat [deg]'].values
    events_lon = events_df['lon [deg]'].values
    
    # Create figure with 3D orthographic projection
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
    
    # Add Earth features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.add_feature(cfeature.LAND, alpha=0.5, color='lightgray')
    ax.add_feature(cfeature.OCEAN, alpha=0.5, color='lightblue')
    
    # Add gridlines
    ax.gridlines(linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Plot all points from wildfire_points.csv
    ax.scatter(points_lon, points_lat, 
               c='blue', s=2, alpha=0.6, 
               transform=ccrs.PlateCarree(),
               label=f'Wildfire Points ({len(points_df)})',
               edgecolors='none')
    
    # Plot events from wildfire_events.csv with different color based on severity
    if 'severity' in events_df.columns:
        severity = events_df['severity'].values
        scatter = ax.scatter(events_lon, events_lat, 
                           c=severity, s=100, alpha=0.9, 
                           cmap='hot', 
                           transform=ccrs.PlateCarree(),
                           label=f'Wildfire Events ({len(events_df)})',
                           edgecolors='black', linewidth=0.5)
        # Add colorbar for severity
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                           pad=0.05, fraction=0.05, aspect=40)
        cbar.set_label('Severity', fontsize=12)
    else:
        ax.scatter(events_lon, events_lat, 
                   c='red', s=100, alpha=0.9, 
                   transform=ccrs.PlateCarree(),
                   label=f'Wildfire Events ({len(events_df)})',
                   edgecolors='black', linewidth=0.5)
    
    # Set global extent
    ax.set_global()
    
    # Set title
    ax.set_title('Wildfire Points and Events on Earth Globe (3D)', fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(script_dir, 'wildfire_earth_map_3d.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n3D Earth globe saved to {output_file}")
    
    plt.show()


def plot_3d_sphere(points_df, events_df, script_dir):
    """Plot points on a 3D sphere (fallback when Cartopy is not available)."""
    # Extract coordinates
    points_lat = points_df['lat [deg]'].values
    points_lon = points_df['lon [deg]'].values
    
    events_lat = events_df['lat [deg]'].values
    events_lon = events_df['lon [deg]'].values
    
    # Convert to 3D coordinates
    points_x, points_y, points_z = lat_lon_to_3d(points_lat, points_lon, radius=1.0)
    events_x, events_y, events_z = lat_lon_to_3d(events_lat, events_lon, radius=1.0)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create and plot sphere wireframe
    sphere_x, sphere_y, sphere_z = create_sphere(radius=1.0, resolution=30)
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray', linewidth=0.5)
    
    # Plot all points from wildfire_points.csv
    ax.scatter(points_x, points_y, points_z, 
               c='blue', s=1, alpha=0.6, label=f'Wildfire Points ({len(points_df)})')
    
    # Plot events from wildfire_events.csv with different color based on severity
    if 'severity' in events_df.columns:
        severity = events_df['severity'].values
        scatter = ax.scatter(events_x, events_y, events_z, 
                           c=severity, s=50, alpha=0.8, 
                           cmap='hot', label=f'Wildfire Events ({len(events_df)})')
        # Add colorbar for severity
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Severity', rotation=270, labelpad=15)
    else:
        ax.scatter(events_x, events_y, events_z, 
                   c='red', s=50, alpha=0.8, label=f'Wildfire Events ({len(events_df)})')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set axis limits
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Wildfire Points and Events on Earth (Sphere)')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set viewing angle (you can adjust these)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(script_dir, 'wildfire_sphere_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    plt.show()


def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    points_file = os.path.join(script_dir, 'scenarios', 'wildfire_scenario', 'data', 'World','wildfire_points_world.csv')
    events_file = os.path.join(script_dir, 'scenarios', 'wildfire_scenario', 'data', 'World','wildfire_events_world.csv')
    
    # Read CSV files
    print(f"Reading {points_file}...")
    points_df = pd.read_csv(points_file)
    print(f"Loaded {len(points_df)} points")
    
    print(f"Reading {events_file}...")
    events_df = pd.read_csv(events_file)
    print(f"Loaded {len(events_df)} events")
    
    if CARTOPY_AVAILABLE:
        # Create both 2D map and 3D globe with Earth features
        print("\nCreating 2D Earth surface map...")
        plot_2d_earth_map(points_df, events_df, script_dir)
        
        print("\nCreating 3D Earth globe...")
        plot_3d_globe_with_cartopy(points_df, events_df, script_dir)
    else:
        # Fallback to basic 3D sphere
        print("\nCreating 3D sphere (Cartopy not available)...")
        plot_3d_sphere(points_df, events_df, script_dir)


if __name__ == '__main__':
    main()

