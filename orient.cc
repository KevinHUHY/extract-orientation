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
    float angle;
    float magnitude;
    float weight;
    Pixel (const Vec2i& position_, float angle_, float magnitude_);
};

Pixel::Pixel (const Vec2i& position_, float angle_, float magnitude_)
    : position(position_)
    , angle(angle_)
    , magnitude(magnitude_)
    , weight(0.0f) {}

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

void bilateralFilter (const Vec2i& centerPosition, vector<Pixel>& neighbors, const Mat& coloredImage)
{
    static GaussianFilter spatialFilter(2.0f, 0.0f);
    static GaussianFilter colorFilter(10.0f, 0.0f);

    const Vec3b& centerColor = coloredImage.at<Vec3b>(centerPosition[0], centerPosition[1]);

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const Pixel& neighbor = neighbors[i];
        const Vec3b& neighborColor = coloredImage.at<Vec3b>(neighbor.position[0], neighbor.position[1]);

        float spatialDistance = norm(centerPosition - neighbor.position);
        float colorDistance = norm(centerColor - neighborColor);

        neighbors[i].weight = spatialFilter(spatialDistance) * colorFilter(colorDistance);
    }
}

float interpolateMagnitude (const vector<Pixel>& neighbors)
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
float interpolateAngle (vector<Pixel>& neighbors)
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

void printPixels (const vector<Pixel>& qualifiedNeighbors)
{
    for (int i = 0; i < qualifiedNeighbors.size(); ++i) {
        cout << "r: " << qualifiedNeighbors[i].position[0]
             << ", c: " << qualifiedNeighbors[i].position[1]
             << ", angle: " << qualifiedNeighbors[i].angle
             << ", magnitudes: " << qualifiedNeighbors[i].magnitude << endl;
    }
}

// k is kernelSize;
void updateCell (int r, int c, int k, Mat& nextAngles, Mat& nextMagnitudes,
                 const Mat& angles, const Mat& magnitudes, const Mat& coloredImage)
{
    const int rows = angles.rows;
    const int cols = angles.cols;
    const int leftMost = max(0, c-k/2);
    const int rightMost = min(cols-1, c+k/2);
    const int upMost = max(0, r-k/2);
    const int downMost = min(rows-1, r+k/2);

    vector<Pixel> qualifiedNeighbors;

    for (int rr = upMost; rr <= downMost; ++rr) {
        for (int cc = leftMost; cc <= rightMost; ++cc) {
            if (magnitudes.at<float>(rr,cc) >= magnitudes.at<float>(r,c)) {
                qualifiedNeighbors.push_back(Pixel(Vec2i(rr,cc), angles.at<float>(rr,cc), magnitudes.at<float>(rr,cc)));
            }
        }
    }

    assert(qualifiedNeighbors.size() >= 1);
    if (qualifiedNeighbors.size() == 1) {
        return;
    }
    bilateralFilter(Vec2i(r,c), qualifiedNeighbors, coloredImage);
    nextMagnitudes.at<float>(r,c) = interpolateMagnitude(qualifiedNeighbors);
    // next statement will sort the qualifiedNeighbors according to angles
    nextAngles.at<float>(r,c) = interpolateAngle(qualifiedNeighbors);
}

void iterate (int k, Mat& angles, Mat& magnitudes, Mat& nextAngles, Mat& nextMagnitudes, const Mat& coloredImage)
{
    const int rows = angles.rows;
    const int cols = angles.cols;
    // #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            updateCell(r, c, k, nextAngles, nextMagnitudes, angles, magnitudes, coloredImage);
        }
    }

    swap(angles, nextAngles);
    swap(magnitudes, nextMagnitudes);
}

void calcGradients (const string& imageName, Mat& angles, Mat& magnitudes)
{
    const int ddepth = CV_32F;

    Mat src = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
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

void saveAngleGraph (const string& imageName, const Mat& angles,
                     const Mat& magnitudes, float threshold=0.0f)
{
    const int rows = angles.rows;
    const int cols = angles.cols;

    Mat imageOfAngles = Mat(rows, cols, CV_8UC3);
    double maxMagnitude;
    minMaxLoc(magnitudes, 0, &maxMagnitude);
    float t = maxMagnitude * threshold;

    // #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (magnitudes.at<float>(r,c) > t) {
                try {
                    imageOfAngles.at<Vec3b>(r,c) = hsv2rgb(angles.at<float>(r,c), 1.0, 1.0);
                } catch (double a) {
                    cout << r << ", " << c << "; Invalid angle: " << a << endl;
                    assert(false);
                }
            } else {
                imageOfAngles.at<Vec3b>(r,c) = Vec3b(0,0,0);
            }
        }
    }

    cvtColor(imageOfAngles, imageOfAngles, CV_RGB2BGR);
    imwrite(imageName, imageOfAngles);
}

void saveAngleGreyGraph (const string& imageName, const Mat& angles)
{
    const int rows = angles.rows;
    const int cols = angles.cols;
    Mat imageOfAngles = Mat(rows, cols, CV_8U);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float ratio = (angles.at<float>(r,c) + PI/2) / PI;
            if (ratio > 1.0f) {
                cout << "saveAngleGreyGraph: " << r << ", " << c << ", ratio: " << ratio << endl;
                ratio = 1.0f;
            } else if (ratio < 0.0f) {
                cout << "saveAngleGreyGraph: " << r << ", " << c << ", ratio: " << ratio << endl;
                ratio = 0.0f;
            }
            // unsigned char greyVal = ratio * 255;
            imageOfAngles.at<unsigned char>(r,c) = (unsigned char) (ratio * 255);
        }
    }
    imwrite(imageName, imageOfAngles);
}

void saveAngleToFile (const string& fileName, const Mat& angles)
{
    const int rows = angles.rows;
    const int cols = angles.cols;

    ofstream out_file(fileName);
    out_file << rows << " " << cols << endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out_file << angles.at<float>(r,c) << " ";
        }
        out_file << endl;
    }
    out_file.close();
}

void saveMagnitudeGraph (const string& imageName, const Mat& magnitudes)
{
    const int rows = magnitudes.rows;
    const int cols = magnitudes.cols;

    Mat imageOfMagnitudes = Mat(rows, cols, CV_8U);
    double maxMagnitude;
    minMaxLoc(magnitudes, 0, &maxMagnitude);

    // #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            imageOfMagnitudes.at<unsigned char>(r,c) = (unsigned char)(magnitudes.at<float>(r,c) / maxMagnitude * 255);
        }
    }
    imwrite(imageName, imageOfMagnitudes);
}

int main(const int argc, const char* argv[])
{

    const string imageName = "Starry_Night.jpg";
    const int iterationTimes = 20;
    Mat coloredImage = imread(imageName, CV_LOAD_IMAGE_COLOR);
    Mat angles, magnitudes;
    calcGradients(imageName, angles, magnitudes);
    // saveAngleGraph("cppNoFiltered.jpg", angles, magnitudes, 0.0f);
    // saveAngleGraph("cppFiltered01.jpg", angles, magnitudes, 0.1f);
    // saveAngleGraph("cppFiltered015.jpg", angles, magnitudes, 0.15f);

    Mat nextAngles = angles.clone();
    Mat nextMagnitudes = magnitudes.clone();
    // saveAngleGraph("cppIterate0.jpg", angles, magnitudes, 0.0f);
    // saveMagnitudeGraph("cppIterate0_Mag.jpg", magnitudes);

    for (int i = 0; i < iterationTimes; ++i) {
        cout << i << "th iteration." << endl;
        iterate(5, angles, magnitudes, nextAngles, nextMagnitudes, coloredImage);
        if ((i+1) % 5 == 0) {
            string prefix = "newit" + to_string(i+1);
            saveAngleToFile(prefix+"_angles.txt", angles);
            saveAngleGraph(prefix+"_angles.jpg", angles, magnitudes, 0.0f);
            // saveAngleGreyGraph(prefix+"GreyAngle.jpg", angles);
            // saveMagnitudeGraph(prefix+"Mag.jpg", magnitudes);
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
        throw(h/(PI*2)*360);
    }
}
