"""
Bisecting K-Means Clustering Algorithm for Image Data Analysis

This script implements a specialized bisecting K-means clustering algorithm for image data that:
1. Loads image data from CSV format files
2. Applies variance threshold feature selection to remove low-variance features
3. Uses t-SNE (t-Distributed Stochastic Neighbor Embedding) for dimensionality reduction
4. Normalizes the data for better clustering performance
5. Iteratively bisects clusters using a modified approach that considers furthest points
6. Assigns cluster labels to data points and saves results

The algorithm is optimized for high-dimensional image data and uses t-SNE to preserve
local structure while reducing dimensionality for more effective clustering.
"""

import random
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


def load_and_preprocess_image_data(filename, variance_threshold=0, tsne_method='exact'):
    """
    Load image data from CSV file and preprocess with feature selection and t-SNE.
    
    Args:
        filename (str): Path to the CSV data file
        variance_threshold (float): Threshold for variance-based feature selection
        tsne_method (str): Method for t-SNE computation ('exact' or 'barnes_hut')
        
    Returns:
        list: Preprocessed data points as list of lists after t-SNE transformation
    """
    print("Loading image data...")
    
    tokenizer = TweetTokenizer()
    data_points = []
    
    with open(filename, "r") as file:
        for line in file:
            # Replace commas with spaces for proper tokenization
            line = line.replace(",", " ")
            point = [float(number) for number in tokenizer.tokenize(line)]
            data_points.append(point)
    
    print(f"Loaded {len(data_points)} data points with {len(data_points[0])} features")
    print("Starting feature selection and dimensionality reduction...")
    
    # Apply variance threshold feature selection
    selector = VarianceThreshold(variance_threshold)
    data_points = selector.fit_transform(data_points)
    print(f"After variance threshold: {data_points.shape[1]} features retained")
    
    # Apply t-SNE dimensionality reduction
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(method=tsne_method, random_state=42)
    data_points = tsne.fit_transform(data_points)
    print(f"After t-SNE: {data_points.shape[1]} dimensions")
    
    # Normalize the data
    data_points = normalize(data_points)
    
    return data_points.tolist()


def calculate_sse(cluster_center, cluster_points):
    """
    Calculate Sum of Squared Errors for a cluster.
    
    Args:
        cluster_center (list): Center point of the cluster
        cluster_points (list): List of points in the cluster
        
    Returns:
        float: Average SSE for the cluster
    """
    if not cluster_points:
        return 0
    
    total_error = sum(mean_squared_error(cluster_center, point) for point in cluster_points)
    return total_error / len(cluster_points)


def calculate_cluster_center(cluster_points):
    """
    Calculate the centroid of a cluster.
    
    Args:
        cluster_points (list): List of points in the cluster
        
    Returns:
        list: Centroid coordinates
    """
    if not cluster_points:
        return []
    
    num_features = len(cluster_points[0])
    center = [0.0] * num_features
    
    for point in cluster_points:
        for i in range(num_features):
            center[i] += point[i]
    
    return [coord / len(cluster_points) for coord in center]


def find_highest_sse_cluster(clusters, cluster_centers):
    """
    Find the cluster with the highest Sum of Squared Errors.
    
    Args:
        clusters (list): List of clusters (each cluster is a list of points)
        cluster_centers (list): List of cluster centers
        
    Returns:
        tuple: (highest_sse_value, cluster_index, cluster_points)
    """
    highest_sse = 0
    highest_index = 0
    
    for i, cluster in enumerate(clusters):
        if i < len(cluster_centers) and cluster:
            sse = calculate_sse(cluster_centers[i], cluster)
            if sse > highest_sse:
                highest_sse = sse
                highest_index = i
    
    return highest_sse, highest_index, clusters[highest_index]


def find_furthest_points(cluster_points, current_center):
    """
    Find the two points furthest from the current cluster center.
    
    Args:
        cluster_points (list): Points in the cluster
        current_center (list): Current cluster center
        
    Returns:
        tuple: (furthest_point1, furthest_point2)
    """
    if len(cluster_points) < 2:
        return cluster_points[0] if cluster_points else [], []
    
    # Find the furthest point from current center
    max_distance = 0
    furthest_point1 = cluster_points[0]
    
    for point in cluster_points:
        distance = mean_squared_error(current_center, point)
        if distance > max_distance:
            max_distance = distance
            furthest_point1 = point
    
    # Find the second furthest point (different from the first)
    max_distance = 0
    furthest_point2 = cluster_points[0]
    
    for point in cluster_points:
        if point != furthest_point1:
            distance = mean_squared_error(current_center, point)
            if distance > max_distance:
                max_distance = distance
                furthest_point2 = point
    
    return furthest_point1, furthest_point2


def bisect_cluster_with_furthest_points(cluster_points, current_center, max_iterations=40):
    """
    Bisect a cluster using a strategy that starts with furthest points.
    
    Args:
        cluster_points (list): Points in the cluster to bisect
        current_center (list): Current center of the cluster
        max_iterations (int): Maximum number of iterations for convergence
        
    Returns:
        tuple: (cluster1, cluster2, center1, center2, final_sse)
    """
    if len(cluster_points) < 2:
        return cluster_points, [], calculate_cluster_center(cluster_points), [], float('inf')
    
    best_sse = float('inf')
    best_clusters = ([], [])
    best_centers = ([], [])
    
    num_features = len(cluster_points[0])
    first_run = True
    
    for iteration in range(max_iterations):
        if first_run:
            # Use furthest points strategy for initial centers
            center1, center2 = find_furthest_points(cluster_points, current_center)
            first_run = False
        else:
            # Use random centers for subsequent iterations
            center1 = cluster_points[random.randint(0, len(cluster_points) - 1)][:]
            center2_idx = random.randint(0, len(cluster_points) - 1)
            while cluster_points[center2_idx] == center1:
                center2_idx = random.randint(0, len(cluster_points) - 1)
            center2 = cluster_points[center2_idx][:]
        
        # Assign points to clusters based on nearest center
        cluster1 = []
        cluster2 = []
        
        for point in cluster_points:
            dist1 = mean_squared_error(center1, point)
            dist2 = mean_squared_error(center2, point)
            
            if dist1 < dist2:
                cluster1.append(point)
            else:
                cluster2.append(point)
        
        # Skip if one cluster is empty
        if not cluster1 or not cluster2:
            continue
        
        # Calculate new centers
        new_center1 = calculate_cluster_center(cluster1)
        new_center2 = calculate_cluster_center(cluster2)
        
        # Calculate SSE
        sse1 = calculate_sse(new_center1, cluster1)
        sse2 = calculate_sse(new_center2, cluster2)
        total_sse = (sse1 + sse2) / 2
        
        # Update best solution if improved
        if total_sse < best_sse:
            best_sse = total_sse
            best_clusters = (cluster1[:], cluster2[:])
            best_centers = (new_center1[:], new_center2[:])
        
        print(f"Iteration {iteration + 1}: SSE = {total_sse:.6f}")
    
    return best_clusters[0], best_clusters[1], best_centers[0], best_centers[1], best_sse


def reassign_points(clusters, cluster_centers):
    """
    Reassign all points to their nearest cluster centers.
    
    Args:
        clusters (list): Current clusters
        cluster_centers (list): Current cluster centers
        
    Returns:
        list: Updated clusters after reassignment
    """
    all_points = []
    for cluster in clusters:
        all_points.extend(cluster)
    
    # Clear existing clusters
    new_clusters = [[] for _ in range(len(cluster_centers))]
    
    # Reassign each point to nearest center
    for point in all_points:
        min_distance = float('inf')
        best_cluster = 0
        
        for i, center in enumerate(cluster_centers):
            if center:  # Only consider non-empty centers
                distance = mean_squared_error(center, point)
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = i
        
        new_clusters[best_cluster].append(point)
    
    return new_clusters


def bisecting_kmeans_for_images(data_points, k=10):
    """
    Perform bisecting k-means clustering optimized for image data.
    
    Args:
        data_points (list): List of data points to cluster
        k (int): Number of desired clusters
        
    Returns:
        tuple: (final_clusters, cluster_centers)
    """
    print("Starting bisecting k-means clustering for image data...")
    print(f"Target number of clusters: {k}")
    
    # Initialize with all points in one cluster
    clusters = [data_points[:]]
    cluster_centers = [calculate_cluster_center(data_points)]
    
    while len(clusters) < k:
        print(f"\nCurrent number of clusters: {len(clusters)}")
        
        # Find cluster with highest SSE
        highest_sse, highest_index, cluster_to_split = find_highest_sse_cluster(clusters, cluster_centers)
        print(f"Selected cluster {highest_index + 1} for bisection (SSE: {highest_sse:.6f})")
        
        if not cluster_to_split:
            print("Warning: Selected cluster is empty")
            break
        
        # Bisect the selected cluster using furthest points strategy
        current_center = cluster_centers[highest_index]
        cluster1, cluster2, center1, center2, final_sse = bisect_cluster_with_furthest_points(
            cluster_to_split, current_center
        )
        
        if not cluster1 or not cluster2:
            print("Warning: Could not successfully bisect cluster")
            break
        
        print(f"Bisection complete (Final SSE: {final_sse:.6f})")
        
        # Replace the original cluster with two new clusters
        clusters.pop(highest_index)
        cluster_centers.pop(highest_index)
        
        clusters.extend([cluster1, cluster2])
        cluster_centers.extend([center1, center2])
        
        # Reassign all points to nearest centers
        print("Reassigning points to nearest centers...")
        print("Pre-adjustment cluster sizes:", [len(cluster) for cluster in clusters])
        
        clusters = reassign_points(clusters, cluster_centers)
        
        # Update cluster centers after reassignment
        for i in range(len(clusters)):
            if clusters[i]:  # Only update if cluster is not empty
                cluster_centers[i] = calculate_cluster_center(clusters[i])
        
        # Print updated cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters]
        print("Post-adjustment cluster sizes:", cluster_sizes)
    
    print(f"\nClustering complete! Final number of clusters: {len(clusters)}")
    return clusters, cluster_centers


def save_results(original_data, final_clusters, filename="results-image.dat"):
    """
    Save cluster assignments to a file.
    
    Args:
        original_data (list): Original data points
        final_clusters (list): Final clusters after bisecting k-means
        filename (str): Output filename
    """
    print(f"Saving results to {filename}...")
    
    with open(filename, "w") as result_file:
        for data_point in original_data:
            cluster_id = 0
            for i, cluster in enumerate(final_clusters):
                if data_point in cluster:
                    cluster_id = i + 1
                    break
            result_file.write(f"{cluster_id}\n")
    
    print("Results saved successfully!")


def main():
    """Main function to orchestrate the image clustering process."""
    print("Bisecting K-Means Clustering Algorithm - Image Version")
    print("=" * 60)
    
    # Get input parameters
    filename = input("Enter the test file name: ")
    k = 10  # Default value for image clustering
    
    try:
        # Load and preprocess image data
        data_points = load_and_preprocess_image_data(filename)
        original_data = data_points[:]  # Keep copy for result mapping
        
        # Perform bisecting k-means
        final_clusters, cluster_centers = bisecting_kmeans_for_images(data_points, k)
        
        # Display final results
        print(f"\nFinal Results:")
        print(f"Number of clusters: {len(final_clusters)}")
        for i, cluster in enumerate(final_clusters):
            print(f"Cluster {i + 1}: {len(cluster)} points")
        
        # Save results
        save_results(original_data, final_clusters)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()