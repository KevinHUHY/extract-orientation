#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <exception>
#include <fstream>

using namespace std;
using namespace cv;

const double PI = 3.14159265358979323846;

Vec3b hsv2rgb (float h, float s, float v);

class Pixel {
public:
    Vec2i position;
    int cluster;
    float angle;
    float magnitude;
    float weight;
    Pixel () {};
};

typedef vector<vector<Pixel>> PixelMap;
typedef vector<vector<int>> IndexCluster;

bool comparePixelByAngle (const Pixel& p1, const Pixel& p2)
{
    return p1.angle < p2.angle;
}

class GaussianFilter {
public:
    const float sigma;
    const float miu;
    GaussianFilter (float sigma_, float miu_);
    float operator() (float x);
};

GaussianFilter::GaussianFilter (float sigma_, float miu_)
    : sigma(sigma_)
    , miu(miu_) {}

float GaussianFilter::operator() (float x)
{
    return exp(-0.5 * pow((x-miu)/sigma, 2)) / (sigma * sqrt(2*PI));
}

void bilateral_filter (const Vec2i& centerPosition, vector<Pixel>& neighbors, const Mat& color_image)
{
    static GaussianFilter spatialFilter(2.0f, 0.0f);
    static GaussianFilter colorFilter(10.0f, 0.0f);

    const Vec3b& center_color = color_image.at<Vec3b>(centerPosition[0], centerPosition[1]);

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const Pixel& neighbor = neighbors[i];
        const Vec3b& neighbor_color = color_image.at<Vec3b>(neighbor.position[0], neighbor.position[1]);

        float spatial_distance = norm(centerPosition - neighbor.position);
        float color_distance = norm(center_color - neighbor_color);

        neighbors[i].weight = spatialFilter(spatial_distance) * colorFilter(color_distance);
    }
}

float interpolate_magnitude (const vector<Pixel>& neighbors)
{
    float sumWeightedMagnitudes = 0.0f;
    float sumWeights = 0.0f;

    for (size_t i = 0; i < neighbors.size(); ++i) {
        sumWeights += neighbors[i].weight;
        sumWeightedMagnitudes += neighbors[i].weight * neighbors[i].magnitude;
    }
    return sumWeightedMagnitudes / sumWeights;
}

// not const here, input will be sorted according to its angle
float interpolate_angle (vector<Pixel>& neighbors)
{
    sort(neighbors.begin(), neighbors.end(), comparePixelByAngle);
    float minDiff = neighbors.back().angle - neighbors[0].angle;
    int minIndex = 0;
    for (int i = 1; i < neighbors.size(); ++i) {
        float diff = neighbors[i-1].angle + PI - neighbors[i].angle;
        if (diff < minDiff) {
            minDiff = diff;
            minIndex = i;
        }
    }

    float avgAngle = 0.0f;
    float sumMagWeights = 0.0f;
    for (int i = 0; i < neighbors.size(); ++i) {
        float magWeight = neighbors[i].weight * neighbors[i].magnitude;
        if (i < minIndex) {
            avgAngle += (neighbors[i].angle + PI) * magWeight;
        } else {
            avgAngle += neighbors[i].angle * magWeight;
        }
        sumMagWeights += magWeight;
    }
    avgAngle /= sumMagWeights;
    if (avgAngle >= PI/2) {
        avgAngle -= PI;
    }
    return avgAngle;
}

void printPixels (const vector<Pixel>& qualified_neighbors)
{
    for (int i = 0; i < qualified_neighbors.size(); ++i) {
        cout << "r: " << qualified_neighbors[i].position[0]
             << ", c: " << qualified_neighbors[i].position[1]
             << ", angle: " << qualified_neighbors[i].angle
             << ", magnitudes: " << qualified_neighbors[i].magnitude << endl;
    }
}

// k is kernelSize;
void updateCell (int r, int c, int k, PixelMap& pixel_map, const PixelMap& old_pixel_map, const Mat& color_image, bool ignore_mag)
{
    assert(pixel_map.size() != 0);
    const int rows = pixel_map.size();
    const int cols = pixel_map[0].size();
    const int leftMost = max(0, c-k/2);
    const int rightMost = min(cols-1, c+k/2);
    const int upMost = max(0, r-k/2);
    const int downMost = min(rows-1, r+k/2);

    vector<Pixel> qualified_neighbors;

    for (int rr = upMost; rr <= downMost; ++rr) {
        for (int cc = leftMost; cc <= rightMost; ++cc) {
            if (pixel_map[rr][cc].cluster == pixel_map[r][c].cluster
                && (ignore_mag || old_pixel_map[rr][cc].magnitude >= old_pixel_map[r][c].magnitude)) {
                qualified_neighbors.push_back(old_pixel_map[rr][cc]);
            }
        }
    }

    assert(qualified_neighbors.size() >= 1);
    if (qualified_neighbors.size() == 1) {
        return;
    }

    // set weight for neighbors
    bilateral_filter(Vec2i(r,c), qualified_neighbors, color_image);
    pixel_map[r][c].magnitude = interpolate_magnitude(qualified_neighbors);

    // next statement will sort the qualified_neighbors according to angles
    pixel_map[r][c].angle = interpolate_angle(qualified_neighbors);
}

void iterate (int k, PixelMap& pixel_map, const Mat& color_image, bool ignore_mag)
{
    assert(pixel_map.size() != 0);
    const int rows = pixel_map.size();
    const int cols = pixel_map[0].size();

    PixelMap old_pixel_map = pixel_map;

    // #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            updateCell(r, c, k, pixel_map, old_pixel_map, color_image, ignore_mag);
        }
    }
}

void calc_gradients (const string& image_name, Mat& angles, Mat& magnitudes)
{
    const int ddepth = CV_32F;

    Mat src = imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
    Mat gradX, gradY;
    // Sobel(src, gradX, ddepth, 1, 0, 5);
    // Sobel(src, gradY, ddepth, 0, 1, 5);
    Scharr(src, gradX, ddepth, 1, 0);
    Scharr(src, gradY, ddepth, 0, 1);

    divide(gradX, -gradY, angles);
    // #pragma omp parallel for
    for (int r = 0; r < angles.rows; ++r) {
        for (int c = 0; c < angles.cols; ++c) {
            angles.at<float>(r,c) = atan(angles.at<float>(r,c));
        }
    }

    magnitude(gradX, gradY, magnitudes);
}

void save_angle_graph (const string& image_name, const PixelMap& pixel_map)
{
    assert(pixel_map.size() != 0);
    const int rows = pixel_map.size();
    const int cols = pixel_map[0].size();

    Mat angle_image = Mat(rows, cols, CV_8UC3);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            try {
                angle_image.at<Vec3b>(r,c) = hsv2rgb(pixel_map[r][c].angle, 1.0, 1.0);
            } catch (double a) {
                cout << r << ", " << c << "; Invalid angle: " << a << endl;
                assert(false);
            }
        }
    }

    cout << "saving " << image_name << endl;
    cvtColor(angle_image, angle_image, CV_RGB2BGR);
    imwrite(image_name, angle_image);
}

// void saveAngleGreyGraph (const string& image_name, const Mat& angles)
// {
//     const int rows = angles.rows;
//     const int cols = angles.cols;
//     Mat angle_image = Mat(rows, cols, CV_8U);
//
//     for (int r = 0; r < rows; ++r) {
//         for (int c = 0; c < cols; ++c) {
//             float ratio = (angles.at<float>(r,c) + PI/2) / PI;
//             if (ratio > 1.0f) {
//                 cout << "saveAngleGreyGraph: " << r << ", " << c << ", ratio: " << ratio << endl;
//                 ratio = 1.0f;
//             } else if (ratio < 0.0f) {
//                 cout << "saveAngleGreyGraph: " << r << ", " << c << ", ratio: " << ratio << endl;
//                 ratio = 0.0f;
//             }
//             // unsigned char greyVal = ratio * 255;
//             angle_image.at<unsigned char>(r,c) = (unsigned char) (ratio * 255);
//         }
//     }
//     imwrite(image_name, angle_image);
// }

void save_angle_to_file (const string& file_name, const PixelMap& pixel_map)
{
    assert(pixel_map.size() != 0);
    const int rows = pixel_map.size();
    const int cols = pixel_map[0].size();

    ofstream out_file(file_name);
    out_file << rows << " " << cols << endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out_file << pixel_map[r][c].angle << " ";
        }
        out_file << endl;
    }
    out_file.close();
}

void save_magnitute_to_file (const string& file_name, const PixelMap& pixel_map)
{
    assert(pixel_map.size() != 0);
    const int rows = pixel_map.size();
    const int cols = pixel_map[0].size();

    ofstream out_file(file_name);
    out_file << rows << " " << cols << endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out_file << pixel_map[r][c].magnitude << " ";
        }
        out_file << endl;
    }
    out_file.close();
}

// void saveMagnitudeGraph (const string& image_name, const Mat& magnitudes)
// {
//     const int rows = magnitudes.rows;
//     const int cols = magnitudes.cols;
//
//     Mat imageOfMagnitudes = Mat(rows, cols, CV_8U);
//     double max_magnitude;
//     minMaxLoc(magnitudes, 0, &max_magnitude);
//
//     // #pragma omp parallel for
//     for (int r = 0; r < rows; ++r) {
//         for (int c = 0; c < cols; ++c) {
//             imageOfMagnitudes.at<unsigned char>(r,c) = (unsigned char)(magnitudes.at<float>(r,c) / max_magnitude * 255);
//         }
//     }
//     imwrite(image_name, imageOfMagnitudes);
// }

template <class T>
vector<vector<T>> load_matrix_from_file (const string& file_name)
{
    ifstream in_file(file_name);
    int rows, cols;
    in_file >> rows >> cols;
    assert(rows > 0 && cols > 0);
    vector<vector<T>> matrix(rows, vector<T>(cols));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in_file >> matrix[r][c];
        }
    }
    in_file.close();
    return matrix;
}

IndexCluster load_index_cluster (const string& file_name)
{
    return load_matrix_from_file<int>(file_name);
}

PixelMap construct_pixel_map(const string& image_name, const string& cluster_file_name)
{
    Mat angles, magnitudes;
    calc_gradients(image_name, angles, magnitudes);

    IndexCluster index_cluster = load_index_cluster(cluster_file_name);

    const int rows = angles.rows;
    const int cols = angles.cols;
    assert(rows == magnitudes.rows && cols == magnitudes.cols);
    assert(rows == index_cluster.size() && cols == index_cluster[0].size());

    PixelMap pixel_map(rows, vector<Pixel>(cols, Pixel()));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            pixel_map[r][c].position = Vec2i(r,c);
            pixel_map[r][c].cluster = index_cluster[r][c];
            pixel_map[r][c].angle = angles.at<float>(r,c);
            pixel_map[r][c].magnitude = magnitudes.at<float>(r,c);
        }
    }
    return pixel_map;
}

int main(const int argc, const char* argv[])
{
    if (argc != 5) {
        cout << "usage: image_name, cluster_file_name, num_of_iter, save_step_size" << endl;
        return 0;
    }

    const string image_name = argv[1];
    const string cluster_file_name = argv[2];
    const int iteration_times = atoi(argv[3]);
    const int save_step = atoi(argv[4]);

    Mat color_image = imread(image_name, CV_LOAD_IMAGE_COLOR);
    PixelMap pixel_map = construct_pixel_map(image_name, cluster_file_name);

    save_magnitute_to_file(image_name+"_original_mag.txt", pixel_map);

    for (int i = 0; i < iteration_times; ++i) {
        cout << "iter " << i+1 << endl;
        if (i < iteration_times / 2) {
            iterate(7, pixel_map, color_image, false);
        } else {
            iterate(7, pixel_map, color_image, true);
        }

        if ((i+1) % save_step == 0) {
            string out_name = "result_pic/" + image_name + "_" + to_string(i+1) + "_iter";
            save_angle_to_file(out_name + ".txt", pixel_map);
            save_angle_graph(out_name + ".jpg", pixel_map);
        }
    }
    return 0;
}

// h in rad
Vec3b hsv2rgb (float h, float s, float v)
{
    float ah = (h + PI/2) / PI * 360;
    int hi = int(ah / 60);
    float f = ah / 60 - hi;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    if (hi == 0) {
        return Vec3b(v*255, t*255, p*255);
    } else if (hi == 1) {
        return Vec3b(q*255, v*255, p*255);
    } else if (hi == 2) {
        return Vec3b(p*255, v*255, t*255);
    } else if (hi == 3) {
        return Vec3b(p*255, q*255, v*255);
    } else if (hi == 4) {
        return Vec3b(t*255, p*255, v*255);
    } else if (hi == 5 || ah == 360) {
        return Vec3b(v*255, p*255, q*255);
    } else {
        return Vec3b(0, 0, 0);
        std::cerr << "error angle:" << hi << std::endl;
        // throw(h/(PI*2)*360);
    }
}
