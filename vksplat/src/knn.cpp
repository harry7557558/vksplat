#include "knn.h"

#include <stdexcept>


// Build KD-tree recursively
std::unique_ptr<NearestNeighbors3D::KDNode> NearestNeighbors3D::build_tree(
    std::vector<std::pair<glm::vec3, size_t>>& points, 
    int depth
) {
    if (points.empty()) return nullptr;
    
    int axis = depth % 3;
    
    // Sort points by the current axis
    std::sort(points.begin(), points.end(), 
                [axis](const auto& a, const auto& b) {
                    return a.first[axis] < b.first[axis];
                });
    
    size_t median = points.size() / 2;
    auto node = std::make_unique<KDNode>(points[median].first, points[median].second, axis);
    
    // Split points
    std::vector<std::pair<glm::vec3, size_t>> left_points(points.begin(), points.begin() + median);
    std::vector<std::pair<glm::vec3, size_t>> right_points(points.begin() + median + 1, points.end());
    
    node->left = build_tree(left_points, depth + 1);
    node->right = build_tree(right_points, depth + 1);
    
    return node;
}


// Search KD-tree for k nearest neighbors
void NearestNeighbors3D::search_tree(
    const KDNode* node,
    const glm::vec3& query_point,
    int k,
    std::priority_queue<NeighborResult, std::vector<NeighborResult>, 
            std::less<NeighborResult>>& neighbors
) const {
    if (!node) return;
    
    float dist = glm::distance(node->point, query_point);
    
    if ((int)neighbors.size() < k) {
        neighbors.push({dist, node->index, node->point});
    } else if (dist < neighbors.top().distance) {
        neighbors.pop();
        neighbors.push({dist, node->index, node->point});
    }
    
    int axis = node->axis;
    float diff = query_point[axis] - node->point[axis];
    
    // Choose which subtree to search first
    const KDNode* first = (diff < 0) ? node->left.get() : node->right.get();
    const KDNode* second = (diff < 0) ? node->right.get() : node->left.get();
    
    search_tree(first, query_point, k, neighbors);
    
    // Check if we need to search the other subtree
    if ((int)neighbors.size() < k || std::abs(diff) < neighbors.top().distance) {
        search_tree(second, query_point, k, neighbors);
    }
}


// Search within radius
void NearestNeighbors3D::radius_search_tree(
    const KDNode* node,
    const glm::vec3& query_point,
    float radius,
    std::vector<NeighborResult>& neighbors
) const {
    if (!node) return;
    
    float dist = glm::distance(node->point, query_point);
    
    if (dist <= radius) {
        neighbors.push_back({dist, node->index, node->point});
    }
    
    int axis = node->axis;
    float diff = query_point[axis] - node->point[axis];
    
    // Search left subtree
    if (diff - radius <= 0) {
        radius_search_tree(node->left.get(), query_point, radius, neighbors);
    }
    
    // Search right subtree
    if (diff + radius >= 0) {
        radius_search_tree(node->right.get(), query_point, radius, neighbors);
    }
}


// Fit the model with training data (similar to sklearn.fit())
void NearestNeighbors3D::fit(const std::vector<glm::vec3>& X) {
    data_points = X;
    
    // Create points with original indices
    std::vector<std::pair<glm::vec3, size_t>> indexed_points;
    indexed_points.reserve(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        indexed_points.emplace_back(X[i], i);
    }
    
    root = build_tree(indexed_points);
    is_fitted = true;
}


// Find k nearest neighbors (similar to sklearn.kneighbors())
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<size_t>>> 
NearestNeighbors3D::kneighbors(
    const std::vector<glm::vec3>& X,
    int n_neighbors,
    bool return_distance
) const {
    if (!is_fitted) {
        throw std::runtime_error("Model must be fitted before calling kneighbors()");
    }
    
    std::vector<std::vector<float>> distances;
    std::vector<std::vector<size_t>> indices;
    
    if (return_distance) distances.reserve(X.size());
    indices.reserve(X.size());
    
    for (const auto& query_point : X) {
        std::priority_queue<NeighborResult, std::vector<NeighborResult>, 
                            std::less<NeighborResult>> neighbors;
        
        search_tree(root.get(), query_point, n_neighbors, neighbors);
        
        std::vector<float> query_distances;
        std::vector<size_t> query_indices;
        
        // Extract results (reverse order due to priority queue)
        std::vector<NeighborResult> results;
        while (!neighbors.empty()) {
            results.push_back(neighbors.top());
            neighbors.pop();
        }
        std::reverse(results.begin(), results.end());
        
        query_distances.reserve(results.size());
        query_indices.reserve(results.size());
        
        for (const auto& result : results) {
            if (return_distance) query_distances.push_back(result.distance);
            query_indices.push_back(result.index);
        }
        
        if (return_distance) distances.push_back(std::move(query_distances));
        indices.push_back(std::move(query_indices));
    }
    
    return std::make_pair(distances, indices);
}


// Find neighbors within a given radius (similar to sklearn.radius_neighbors())
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<size_t>>>
NearestNeighbors3D::radius_neighbors(
    const std::vector<glm::vec3>& X,
    float radius,
    bool return_distance
) const {
    if (!is_fitted) {
        throw std::runtime_error("Model must be fitted before calling radius_neighbors()");
    }
    
    std::vector<std::vector<float>> distances;
    std::vector<std::vector<size_t>> indices;
    
    if (return_distance) distances.reserve(X.size());
    indices.reserve(X.size());
    
    for (const auto& query_point : X) {
        std::vector<NeighborResult> neighbors;
        radius_search_tree(root.get(), query_point, radius, neighbors);
        
        // Sort by distance
        std::sort(neighbors.begin(), neighbors.end(), 
                    [](const NeighborResult& a, const NeighborResult& b) {
                        return a.distance < b.distance;
                    });
        
        std::vector<float> query_distances;
        std::vector<size_t> query_indices;
        
        query_distances.reserve(neighbors.size());
        query_indices.reserve(neighbors.size());
        
        for (const auto& neighbor : neighbors) {
            if (return_distance) query_distances.push_back(neighbor.distance);
            query_indices.push_back(neighbor.index);
        }
        
        if (return_distance) distances.push_back(std::move(query_distances));
        indices.push_back(std::move(query_indices));
    }
    
    return std::make_pair(distances, indices);
}


// Single point k-nearest neighbors query
std::pair<std::vector<float>, std::vector<size_t>>
NearestNeighbors3D::kneighbors_single(const glm::vec3& query_point, int n_neighbors, bool return_distance) const {
    auto result = kneighbors({query_point}, n_neighbors, return_distance);
    return std::make_pair(
        return_distance ? result.first[0] : std::vector<float>(),
        result.second[0]
    );
}


// Single point radius neighbors query
std::pair<std::vector<float>, std::vector<size_t>>
NearestNeighbors3D::radius_neighbors_single(const glm::vec3& query_point, float radius, bool return_distance) const {
    auto result = radius_neighbors({query_point}, radius, return_distance);
    return std::make_pair(
        return_distance ? result.first[0] : std::vector<float>(),
        result.second[0]
    );
}


// Get the fitted data
const std::vector<glm::vec3>& NearestNeighbors3D::get_data() const {
    if (!is_fitted) {
        throw std::runtime_error("Model must be fitted before accessing data");
    }
    return data_points;
}
