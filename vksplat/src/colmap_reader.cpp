#include "colmap_reader.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <sstream>


float median(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0f;
    } else {
        return values[n/2];
    }
}

glm::vec3 median_vec3(const std::vector<glm::vec3>& vectors) {
    if (vectors.empty()) return glm::vec3(0.0f);
    
    std::vector<float> x_vals, y_vals, z_vals;
    for (const auto& v : vectors) {
        x_vals.push_back(v.x);
        y_vals.push_back(v.y);
        z_vals.push_back(v.z);
    }
    
    return glm::vec3(median(x_vals), median(y_vals), median(z_vals));
}

void eigh_jacobi(const glm::dmat3& matrix, glm::dmat3& eigenvectors, glm::dvec3& eigenvalues, int max_iterations = 12) {
    glm::dmat3 A = matrix;
    eigenvectors = glm::dmat3(1.0); // Identity matrix
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find the largest off-diagonal element
        int p = 0, q = 1;
        double max_off_diag = abs(A[0][1]);
        
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (abs(A[i][j]) > max_off_diag) {
                    max_off_diag = abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check for convergence
        if (max_off_diag < 1e-8) break;
        
        // Compute rotation angle
        double theta = 0.5 * atan2(2.0 * A[p][q], A[q][q] - A[p][p]);
        double c = cos(theta);
        double s = sin(theta);
        
        // Apply Givens rotation
        glm::dmat3 G = glm::dmat3(1.0);
        G[p][p] = c; G[p][q] = -s;
        G[q][p] = s; G[q][q] = c;
        
        A = glm::transpose(G) * A * G;
        eigenvectors = eigenvectors * G;
    }
    
    // Extract eigenvalues from diagonal
    eigenvalues = glm::dvec3(A[0][0], A[1][1], A[2][2]);
}


template<typename T>
T readBinary(std::ifstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

std::vector<double> readDoubleArray(std::ifstream& file, size_t count) {
    std::vector<double> values(count);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(double));
    return values;
}

std::string readNullTerminatedString(std::ifstream& file) {
    std::string result;
    char c;
    while (file.read(&c, 1) && c != '\0') {
        result += c;
    }
    return result;
}


const std::map<int, ColmapReader::CameraModel> ColmapReader::CAMERA_MODEL_IDS = {
    {0, CameraModel(0, "SIMPLE_PINHOLE", 3)},
    {1, CameraModel(1, "PINHOLE", 4)},
    {2, CameraModel(2, "SIMPLE_RADIAL", 4)},
    {3, CameraModel(3, "RADIAL", 5)},
    {4, CameraModel(4, "OPENCV", 8)},
    {5, CameraModel(5, "OPENCV_FISHEYE", 8)},
    {6, CameraModel(6, "FULL_OPENCV", 12)},
    {7, CameraModel(7, "FOV", 5)},
    {8, CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4)},
    {9, CameraModel(9, "RADIAL_FISHEYE", 5)},
    {10, CameraModel(10, "THIN_PRISM_FISHEYE", 12)},
    {11, CameraModel(11, "RAD_TAN_THIN_PRISM_FISHEYE", 16)}
};

const std::map<std::string, ColmapReader::CameraModel> ColmapReader::CAMERA_MODEL_NAMES = {
    {"SIMPLE_PINHOLE", CameraModel(0, "SIMPLE_PINHOLE", 3)},
    {"PINHOLE", CameraModel(1, "PINHOLE", 4)},
    {"SIMPLE_RADIAL", CameraModel(2, "SIMPLE_RADIAL", 4)},
    {"RADIAL", CameraModel(3, "RADIAL", 5)},
    {"OPENCV", CameraModel(4, "OPENCV", 8)},
    {"OPENCV_FISHEYE", CameraModel(5, "OPENCV_FISHEYE", 8)},
    {"FULL_OPENCV", CameraModel(6, "FULL_OPENCV", 12)},
    {"FOV", CameraModel(7, "FOV", 5)},
    {"SIMPLE_RADIAL_FISHEYE", CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4)},
    {"RADIAL_FISHEYE", CameraModel(9, "RADIAL_FISHEYE", 5)},
    {"THIN_PRISM_FISHEYE", CameraModel(10, "THIN_PRISM_FISHEYE", 12)},
    {"RAD_TAN_THIN_PRISM_FISHEYE", CameraModel(11, "THIN_PRISM_FISHEYE", 16)}
};


std::map<int, ColmapReader::Camera> ColmapReader::readCamerasBinary(const std::string& path) {
    std::map<int, Camera> cameras;
    std::ifstream file(path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    uint64_t num_cameras = readBinary<uint64_t>(file);
    
    for (uint64_t i = 0; i < num_cameras; ++i) {
        int camera_id = readBinary<int>(file);
        int model_id = readBinary<int>(file);
        uint64_t width = readBinary<uint64_t>(file);
        uint64_t height = readBinary<uint64_t>(file);
        
        auto model_it = CAMERA_MODEL_IDS.find(model_id);
        if (model_it == CAMERA_MODEL_IDS.end()) {
            throw std::runtime_error("Unknown camera model ID: " + std::to_string(model_id));
        }
        
        const CameraModel& model = model_it->second;
        std::vector<double> params = readDoubleArray(file, model.num_params);
        
        cameras[camera_id] = Camera(camera_id, model.model_name, width, height, params);
    }
    
    return cameras;
}


std::map<int, ColmapReader::Image> ColmapReader::readImagesBinary(const std::string& path) {
    std::map<int, Image> images;
    std::ifstream file(path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    uint64_t num_images = readBinary<uint64_t>(file);
    
    for (uint64_t i = 0; i < num_images; ++i) {
        int image_id = readBinary<int>(file);
        
        // Read quaternion (w, x, y, z)
        float qw = (float)readBinary<double>(file);
        float qx = (float)readBinary<double>(file);
        float qy = (float)readBinary<double>(file);
        float qz = (float)readBinary<double>(file);
        glm::quat qvec(qw, qx, qy, qz);
        
        // Read translation vector
        float tx = (float)readBinary<double>(file);
        float ty = (float)readBinary<double>(file);
        float tz = (float)readBinary<double>(file);
        glm::vec3 tvec(tx, ty, tz);
        
        int camera_id = readBinary<int>(file);
        std::string image_name = readNullTerminatedString(file);
        
        uint64_t num_points2D = readBinary<uint64_t>(file);
        
        std::vector<glm::vec2> xys;
        std::vector<uint64_t> point3D_ids;
        xys.reserve(num_points2D);
        point3D_ids.reserve(num_points2D);
        
        for (uint64_t j = 0; j < num_points2D; ++j) {
            double x = readBinary<double>(file);
            double y = readBinary<double>(file);
            uint64_t point_id = readBinary<uint64_t>(file);
            
            xys.emplace_back((float)x, (float)y);
            point3D_ids.push_back(point_id);
        }
        
        images[image_id] = Image(image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids);
    }
    
    return images;
}


std::map<uint64_t, ColmapReader::Point3D> ColmapReader::readPoints3DBinary(const std::string& path) {
    std::map<uint64_t, Point3D> points3D;
    std::ifstream file(path, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    uint64_t num_points = readBinary<uint64_t>(file);
    
    for (uint64_t i = 0; i < num_points; ++i) {
        uint64_t point3D_id = readBinary<uint64_t>(file);
        
        // Read XYZ coordinates
        double x = readBinary<double>(file);
        double y = readBinary<double>(file);
        double z = readBinary<double>(file);
        glm::vec3 xyz(x, y, z);
        
        // Read RGB color
        uint8_t r = readBinary<uint8_t>(file);
        uint8_t g = readBinary<uint8_t>(file);
        uint8_t b = readBinary<uint8_t>(file);
        glm::u8vec3 rgb(r, g, b);
        
        double error = readBinary<double>(file);
        
        uint64_t track_length = readBinary<uint64_t>(file);
        
        std::vector<int> image_ids;
        std::vector<int> point2D_idxs;
        image_ids.reserve(track_length);
        point2D_idxs.reserve(track_length);
        
        for (uint64_t j = 0; j < track_length; ++j) {
            int img_id = readBinary<int>(file);
            int pt2d_idx = readBinary<int>(file);
            
            image_ids.push_back(img_id);
            point2D_idxs.push_back(pt2d_idx);
        }
        
        points3D[point3D_id] = Point3D(point3D_id, xyz, rgb, error, image_ids, point2D_idxs);
    }
    
    return points3D;
}


std::map<int, ColmapReader::Camera> ColmapReader::readCamerasText(const std::string& path) {
    std::map<int, Camera> cameras;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream ss(line);
        int camera_id, width, height;
        std::string model;
        
        ss >> camera_id >> model >> width >> height;
        
        std::vector<double> params;
        double p;
        while (ss >> p) {
            params.push_back(p);
        }
        
        cameras[camera_id] = Camera(camera_id, model, width, height, params);
    }
    
    return cameras;
}

std::map<int, ColmapReader::Image> ColmapReader::readImagesText(const std::string& path) {
    std::map<int, Image> images;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        if (line.empty() || line[0] == '#') continue;
        
        // First line: image metadata
        std::istringstream ss(line);
        int image_id, camera_id;
        double qw, qx, qy, qz, tx, ty, tz;
        std::string image_name;
        
        ss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> image_name;
        
        glm::quat qvec((float)qw, (float)qx, (float)qy, (float)qz);
        glm::vec3 tvec((float)tx, (float)ty, (float)tz);
        
        // Second line: 2D points
        std::string points_line;
        if (!std::getline(file, points_line)) break;
        
        std::istringstream pss(points_line);
        std::vector<glm::vec2> xys;
        std::vector<uint64_t> point3D_ids;
        
        double x, y;
        int64_t point_id;
        while (pss >> x >> y >> point_id) {
            xys.emplace_back((float)x, (float)y);
            point3D_ids.push_back((uint64_t)point_id);
        }
        
        images[image_id] = Image(image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids);
    }
    
    return images;
}

std::map<uint64_t, ColmapReader::Point3D> ColmapReader::readPoints3DText(const std::string& path) {
    std::map<uint64_t, Point3D> points3D;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream ss(line);
        uint64_t point3D_id;
        double x, y, z, error;
        int r, g, b;
        
        ss >> point3D_id >> x >> y >> z >> r >> g >> b >> error;
        
        glm::vec3 xyz((float)x, (float)y, (float)z);
        glm::u8vec3 rgb((uint8_t)r, (uint8_t)g, (uint8_t)b);
        
        std::vector<int> image_ids;
        std::vector<int> point2D_idxs;
        
        int img_id, pt2d_idx;
        while (ss >> img_id >> pt2d_idx) {
            image_ids.push_back(img_id);
            point2D_idxs.push_back(pt2d_idx);
        }
        
        points3D[point3D_id] = Point3D(point3D_id, xyz, rgb, error, image_ids, point2D_idxs);
    }
    
    return points3D;
}


std::vector<glm::mat4> transform_cameras(const glm::mat4& matrix, const std::vector<glm::mat4>& camtoworlds) {
    std::vector<glm::mat4> result;
    result.reserve(camtoworlds.size());
    
    for (const auto& cam : camtoworlds) {
        glm::mat4 transformed = matrix * cam;
        
        // Normalize the rotation part (first 3 columns should have unit length)
        glm::vec3 scale_x = glm::vec3(transformed[0][0], transformed[1][0], transformed[2][0]);
        float scaling = glm::length(scale_x);
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                transformed[i][j] /= scaling;
            }
        }
        
        result.push_back(transformed);
    }
    
    return result;
}

std::vector<glm::vec3> transform_points(const glm::mat4& matrix, const std::vector<glm::vec3>& points) {
    std::vector<glm::vec3> result;
    result.reserve(points.size());
    
    glm::mat3 rotation = glm::mat3(matrix);
    glm::vec3 translation = glm::vec3(matrix[3]);
    
    for (const auto& point : points) {
        result.push_back(rotation * point + translation);
    }
    
    return result;
}

glm::mat4 similarity_from_cameras(const std::vector<glm::mat4>& c2w, bool strict_scaling = false, const std::string& center_method = "focus") {
    if (c2w.empty()) return glm::mat4(1.0f);
    
    std::vector<glm::vec3> poss;
    std::vector<glm::vec3> ups;
    std::vector<glm::vec3> fwds;
    
    // Extract camera positions and orientations
    for (const auto& cam : c2w) {
        poss.push_back(glm::vec3(cam[3]));
        
        glm::mat3 rotation = glm::mat3(cam);
        glm::vec3 up = rotation * glm::vec3(0.0f, -1.0f, 0.0f);
        glm::vec3 forward = rotation * glm::vec3(0.0f, 0.0f, 1.0f);
        
        ups.push_back(up);
        fwds.push_back(forward);
    }
    
    // (1) Estimate world up direction
    glm::vec3 world_up(0.0f);
    for (const auto& up : ups) {
        world_up += up;
    }
    world_up /= static_cast<float>(ups.size());
    world_up = glm::normalize(world_up);
    
    // Align world up with camera space up (0, -1, 0)
    glm::vec3 up_camspace(0.0f, -1.0f, 0.0f);
    float c = glm::dot(up_camspace, world_up);
    glm::vec3 cross_product = glm::cross(world_up, up_camspace);
    
    glm::mat3 R_align;
    if (c > -1.0f) {
        glm::mat3 skew(0.0f);
        skew[1][0] = -cross_product.z; skew[2][0] = cross_product.y;
        skew[0][1] = cross_product.z;  skew[2][1] = -cross_product.x;
        skew[0][2] = -cross_product.y; skew[1][2] = cross_product.x;
        
        R_align = glm::mat3(1.0f) + skew + (skew * skew) * (1.0f / (1.0f + c));
    } else {
        // 180-degree rotation about x-axis
        R_align = glm::mat3(-1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f);
    }
    
    // Apply rotation to positions and forwards
    for (auto& pos : poss) {
        pos = R_align * pos;
    }
    for (auto& fwd : fwds) {
        fwd = R_align * fwd;
    }
    
    // (2) Recenter the scene
    glm::vec3 translate;
    if (center_method == "focus") {
        // find the closest point to the origin for each camera's center ray
        std::vector<glm::vec3> nearest_points;
        for (size_t i = 0; i < poss.size(); ++i) {
            glm::vec3 nearest = poss[i] + dot(-poss[i], fwds[i]) * fwds[i];
            nearest_points.push_back(nearest);
        }
        translate = -median_vec3(nearest_points);
    } else if (center_method == "poses") {
        translate = -median_vec3(poss);
    } else {
        translate = glm::vec3(0.0f);
    }
    
    // (3) Rescale the scene
    std::vector<float> distances;
    for (const auto& pos : poss) {
        distances.push_back(length(pos + translate));
    }
    
    float scale;
    if (strict_scaling) {
        scale = 1.0f / *std::max_element(distances.begin(), distances.end());
    } else {
        scale = 1.0f / median(distances);
    }
    
    // Create transformation matrix
    glm::mat4 transform(1.0f);
    transform = glm::scale(transform, glm::vec3(scale));
    transform = glm::mat4(glm::mat3(R_align)) * transform;
    transform[3] = glm::vec4(translate * scale, 1.0f);
    
    return transform;
}

glm::mat4 align_principal_axes(const std::vector<glm::vec3>& point_cloud) {
    if (point_cloud.empty()) return glm::mat4(1.0f);
    
    // Compute centroid
    glm::vec3 centroid = median_vec3(point_cloud);
    
    // Compute covariance matrix
    glm::dmat3 covariance(0.0f);
    glm::vec3 mean(0.0f);
    for (const auto& point : point_cloud) {
        glm::dvec3 translated_point = glm::dvec3(point - centroid);
        mean += translated_point;
        covariance += glm::outerProduct(translated_point, translated_point);
    }
    covariance /= double(point_cloud.size() - 1);
    mean /= double(point_cloud.size());
    covariance -= glm::outerProduct(mean, mean);
    
    // Compute eigenvectors and eigenvalues
    glm::dmat3 eigenvectors_d;
    glm::dvec3 eigenvalues_d;
    eigh_jacobi(covariance, eigenvectors_d, eigenvalues_d);
    glm::mat3 eigenvectors(eigenvectors_d);
    glm::vec3 eigenvalues(eigenvalues_d);
    
    // Sort eigenvectors by eigenvalues (descending order)
    std::vector<std::pair<float, int>> eigen_pairs;
    for (int i = 0; i < 3; ++i)
        eigen_pairs.push_back({eigenvalues[i], i});
    std::sort(eigen_pairs.rbegin(), eigen_pairs.rend());
    
    glm::mat3 sorted_eigenvectors;
    for (int i = 0; i < 3; ++i) {
        int idx = eigen_pairs[i].second;
        sorted_eigenvectors[i] = eigenvectors[idx];
    }
    eigenvectors = sorted_eigenvectors;
    
    // Check orientation - ensure right-handed coordinate system
    if (glm::determinant(eigenvectors) < 0)
        eigenvectors[0] = -eigenvectors[0];
    
    // Create SE(3) transformation matrix
    glm::mat3 rotation_matrix = glm::transpose(eigenvectors);
    glm::vec3 new_translation = -rotation_matrix * centroid;
    
    glm::mat4 transform(1.0f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            transform[i][j] = rotation_matrix[i][j];
    transform[3] = glm::vec4(new_translation, 1.0f);
    
    return transform;
}


glm::mat4 ColmapReader::normalize_world_space(
    std::vector<glm::mat4> &camtoworlds, 
    std::vector<glm::vec3> &points, 
    bool normalize
) {
    glm::mat4 transform = glm::mat4(1.0f);
    if (!normalize)
        return transform;

    // Step 1: Similarity transform from cameras
    glm::mat4 T1 = similarity_from_cameras(camtoworlds);
    camtoworlds = transform_cameras(T1, camtoworlds);
    points = transform_points(T1, points);

    // Step 2: Align principal axes
    glm::mat4 T2 = align_principal_axes(points);
    camtoworlds = transform_cameras(T2, camtoworlds);
    points = transform_points(T2, points);
    
    transform = T2 * T1;
    
    // Step 3: Fix for upside down
    std::vector<float> z_coords;
    for (const auto& point : points) {
        z_coords.push_back(point.z);
    }
    
    float z_median = median(z_coords);
    float z_mean = std::accumulate(z_coords.begin(), z_coords.end(), 0.0f) / z_coords.size();
    
    if (z_median > z_mean) {
        // Rotate 180 degrees around x-axis (flip z)
        glm::mat4 T3(1.0f);
        T3[1][1] = -1.0f; // flip y
        T3[2][2] = -1.0f; // flip z
        
        camtoworlds = transform_cameras(T3, camtoworlds);
        points = transform_points(T3, points);
        transform = T3 * transform;
    }

    return transform;
}


#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
