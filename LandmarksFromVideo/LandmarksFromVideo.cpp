// LandmarksFromVideo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/cpp17/optional.hpp"

#include "Eigen/Core"

#include <boost/filesystem.hpp>
#include <iostream>

using namespace eos;
using namespace std;
using namespace cv;
using namespace cv::face;
using namespace boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;

vector<pair <int, int>> get_directories(const string& s, string& id)
{
    vector<pair <int, int>> frame_pos;
    for (auto& p : recursive_directory_iterator(s))
        if (is_directory(p.path()))
        {
            if (p.path().string().find(id) != string::npos)
            {
                std::ifstream start_frame_file(p.path().string()+"\\startframe.txt"), n_frame_file(p.path().string() + "\\nframe.txt");
                pair<int, int> start_n;
                string sint;
                getline(start_frame_file, sint);
                start_n.first = stoi(sint);
                getline(n_frame_file, sint);
                start_n.second = stoi(sint);
                frame_pos.push_back(start_n);
            }
        }
    return frame_pos;
}

int main()
{
    /*EOS Initialization*/
    string modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputbasename;
    modelfile = "C:\\Program Files\\eos\\eos\\share\\sfm_shape_3448.bin";
    mappingsfile = "C:\\Program Files\\eos\\eos\\share\\ibug_to_sfm.txt";
    blendshapesfile = "C:\\Program Files\\eos\\eos\\share\\expression_blendshapes_3448.bin";
    contourfile = "C:\\Program Files\\eos\\eos\\share\\sfm_model_contours.json";
    edgetopologyfile = "C:\\Program Files\\eos\\eos\\share\\sfm_3448_edge_topology.json";
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile);
    }
    catch (const std::runtime_error& e)
    {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    core::LandmarkMapper landmark_mapper;
    try
    {
        landmark_mapper = core::LandmarkMapper(mappingsfile);
    }
    catch (const std::exception& e)
    {
        cout << "Error loading the landmark mappings: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    // The expression blendshapes:
    const vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);
    morphablemodel::MorphableModel morphable_model_with_expressions(
        morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(), cpp17::nullopt,
        morphable_model.get_texture_coordinates());
    // These two are used to fit the front-facing contour to the ibug contour landmarks:
    const fitting::ModelContour model_contour =
        contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
    const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
    // The edge topology is used to speed up computation of the occluding face contour fitting:
    const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);


    /*Face and landmark detector*/
    string id = "_Lm6l2vaF5o";
    string resolution = "480p";
    string extension = ".mp4";
    vector<pair<int, int>> frame_pos= get_directories("C:\\Projects\\Python projects\\DeepFake\\obama_data", id);
    path opencv_dir = path(getenv("OPENCV_DIR"));
    path faceDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "haarcascades" / "haarcascade_frontalface_alt2.xml";
    path landmarkDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "FaceLandmarksModels" / "lbfmodel.yaml";
    CascadeClassifier faceDetector(faceDetectorPath.string());
    Ptr<FacemarkLBF> landmarkDetector = FacemarkLBF::create();
    landmarkDetector->loadModel(landmarkDetectorPath.string());
    string video_dir= "E:\\obama_dataset\\video";
    VideoCapture video = VideoCapture(video_dir+"\\"+id+"\\"+id+"_"+resolution+extension);
    path dataFilePath = path("E:\\obama_dataset\\facial_landmarks\\"+id);
    create_directory(dataFilePath);
    int part = 1;

    
    for (vector<pair<int, int>>::const_iterator iter = frame_pos.begin(); iter != frame_pos.end(); ++iter)
    {
        //path subdir = dataFilePath / to_string(part);
        part++;
        //create_directory(subdir);
        video.set(CV_CAP_PROP_POS_FRAMES, iter->first);
        int frame_ctr = iter->first;
        int endFrames = iter->first+iter->second;
        while (frame_ctr < endFrames)
        {
            Mat frame, grayFrame;
            bool status = video.read(frame);
            if (status)
            {
                vector<Rect> faces;
                cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
                faceDetector.detectMultiScale(grayFrame, faces,1.1,5);
                if (faces.size()>1)
                {
                    cout << "More faces detected" << endl;
                    faces.resize(1);
                }
                vector<vector<Point2f>> landmarks;
                bool success = landmarkDetector->fit(grayFrame, faces, landmarks);
                if (success)
                {
                    
                    /*string fileName = "frame" + std::to_string(frame_ctr) + ".pts";
                    string frameName = "frame" + std::to_string(frame_ctr) + ".png";
                    std::ofstream dataFile;
                    dataFile.open((subdir / fileName).string(), std::ofstream::out | std::ofstream::app);
                    dataFile << "version: 1" << '\n';
                    dataFile << "n_points:  68" << '\n';
                    dataFile << "{" << '\n';
                    for (int i = 0; i < landmarks.size(); i++)
                    {
                        for (int j = 0; j < landmarks[i].size(); j++)
                        {
                            dataFile << landmarks[i][j].x << " " << landmarks[i][j].y << '\n';
                        }
                    }
                    dataFile << "}" << '\n';
                    dataFile.close();
                    imwrite((subdir / frameName).string(), grayFrame);*/
                    LandmarkCollection<Eigen::Vector2f> landmarksM;
                    landmarksM.reserve(68);
                    for (int j = 0; j < 68; j++)
                    {
                        Landmark<Eigen::Vector2f> landmark;
                        landmark.name = to_string(j + 1);
                        landmark.coordinates[0] = landmarks[0][j].x - 1.0f;
                        landmark.coordinates[1] = landmarks[0][j].y - 1.0f;
                        landmarksM.emplace_back(landmark);
                    }
                    core::Mesh mesh;
                    fitting::RenderingParameters rendering_params;
                    std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
                        morphable_model_with_expressions, landmarksM, landmark_mapper, grayFrame.cols, grayFrame.rows, edge_topology,
                        ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f);
                }
                cout << frame_ctr << "/" << endFrames-1 << endl;
                frame_ctr++;
            }
        }
    }

    destroyAllWindows();
}