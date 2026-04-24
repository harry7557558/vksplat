#include <iosfwd>
#include <vector>
#include <map>
#include <string>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM

#include "stb_image.h"


class ColmapReader {
public:

    struct CameraModel {
        int model_id;
        std::string model_name;
        int num_params;
        
        CameraModel(int id, const std::string& name, int params) 
            : model_id(id), model_name(name), num_params(params) {}
    };

    struct Camera {
        int id;
        std::string model;
        uint64_t width;
        uint64_t height;
        std::vector<double> params;
        
        Camera() = default;
        Camera(int id_, const std::string& model_, uint64_t w, uint64_t h, const std::vector<double>& p)
            : id(id_), model(model_), width(w), height(h), params(p) {}
    };

    struct Image {
        int id;
        glm::quat qvec;
        glm::vec3 tvec;
        int camera_id;
        std::string name;
        std::vector<glm::vec2> xys;
        std::vector<uint64_t> point3D_ids;
        
        Image() = default;
        Image(int id_, const glm::quat& q, const glm::vec3& t, int cam_id, const std::string& n,
            const std::vector<glm::vec2>& xy, const std::vector<uint64_t>& pt_ids)
            : id(id_), qvec(q), tvec(t), camera_id(cam_id), name(n), xys(xy), point3D_ids(pt_ids) {}
    };

    struct Point3D {
        uint64_t id;
        glm::vec3 xyz;
        glm::u8vec3 rgb;
        double error;
        std::vector<int> image_ids;
        std::vector<int> point2D_idxs;
        
        Point3D() = default;
        Point3D(uint64_t id_, const glm::vec3& pos, const glm::u8vec3& color, double err,
                const std::vector<int>& img_ids, const std::vector<int>& pt2d_idxs)
            : id(id_), xyz(pos), rgb(color), error(err), image_ids(img_ids), point2D_idxs(pt2d_idxs) {}
    };

    std::map<int, Camera> readCamerasBinary(const std::string& path);
    std::map<int, Image> readImagesBinary(const std::string& path);
    std::map<uint64_t, Point3D> readPoints3DBinary(const std::string& path);

    std::map<int, Camera> readCamerasText(const std::string& path);
    std::map<int, Image> readImagesText(const std::string& path);
    std::map<uint64_t, Point3D> readPoints3DText(const std::string& path);

    static glm::mat4 normalize_world_space(
        std::vector<glm::mat4> &camtoworlds, 
        std::vector<glm::vec3> &points, 
        bool normalize = true
    );

private:
    static const std::map<int, CameraModel> CAMERA_MODEL_IDS;
    static const std::map<std::string, CameraModel> CAMERA_MODEL_NAMES;

};

