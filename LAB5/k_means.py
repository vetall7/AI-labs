import numpy as np

def initialize_centroids_forgy(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False), :]


def calculate_distance(data, centroids):
    num_data = data.shape[0]
    num_clusters = centroids.shape[0]

    distances = np.zeros((num_data, num_clusters))

    for i in range(num_data):
        for j in range(num_clusters):
            dist_sq = 0
            for k in range(data.shape[1]):
                dist_sq += (data[i, k] - centroids[j, k]) ** 2
            distances[i, j] = np.sqrt(dist_sq)

    return distances

def initialize_centroids_kmeans_pp(data, k):
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0], 1), :]
    for i in range(1, k):
        distances = calculate_distance(data, centroids[:i, :])
        min_distances = np.min(distances, axis=1)
        max_distance_idx = np.argmax(min_distances)
        centroids[i] = data[max_distance_idx]
    return centroids


def assign_to_cluster(data, centroids):
    distances = calculate_distance(data, centroids)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments


def update_centroids(data, assignments):
    unique_clusters = np.unique(assignments)
    centroids = np.zeros((len(unique_clusters), data.shape[1]))
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = data[assignments == cluster_id]
        centroid = np.mean(cluster_data, axis=0)
        centroids[i] = centroid

    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

