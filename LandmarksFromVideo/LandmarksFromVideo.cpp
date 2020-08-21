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

vector<pair <int, int>> get_video_info(path s, string& yt_id)
{
    vector<pair <int, int>> frame_pos;
    for (auto& p : recursive_directory_iterator(s))
        if (is_directory(p.path()))
        {
            if (p.path().string().find(yt_id) != string::npos)
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

inline string get_yt_id(path subdir)
{
    return subdir.stem().string();
}

path get_video(path subdir, string res, string yt_id)
{
    
    path video;
    for (auto& p : directory_iterator(subdir))
    {
        if (is_regular_file(p.path()))
        {
            if (extension(p.path()) == ".mp4")
            {
                bool q = p.path().string().find(res) != string::npos;
                if (q)
                {
                    video = p.path();
                }
            }
        }
    }
    return video;
}

vector<path> get_subdir(path dir)
{
    vector<path> subdirs;
    for (auto& p : directory_iterator(dir))
        if (is_directory(p.path()))
        {
            subdirs.push_back(p.path());
        }
    return subdirs;
}

vector<Rect> best_face(vector<Rect> faces, Rect prev_face)
{
    vector<Rect> best_fit;
    double min_d = 1E10;
    int min_ind;
    for (int i = 0; i < faces.size(); i++)
    {
        double d = sqrt(pow((faces[i].x - prev_face.x), 2) + pow((faces[i].y - prev_face.y), 2));
        if (d < min_d)
        {
            min_d = d;
            min_ind = i;
        }
    }
    best_fit.push_back(faces[min_ind]);
    return best_fit;
}

void log_mesh(std::ofstream& file, core::Mesh& m, int frame)
{
    vector<int> vertex_ind = { 398, 315, 413, 329, 825, 736, 812, 841, 693, 411, 264, 431, 416, 423, 828, 817, 442, 404 };
    file << to_string(frame) << ",";
    for (int i = 0; i < vertex_ind.size(); i++)
    {
        if (i != vertex_ind.size() - 1)
        {
            file << to_string(m.vertices[i].x()) << ',' << to_string(m.vertices[i].y()) << ',' << to_string(m.vertices[i].z()) << ',';
        }
        else
        {
            file << to_string(m.vertices[i].x()) << ',' << to_string(m.vertices[i].y()) << ',' << to_string(m.vertices[i].z());
        }
    }
    file << endl;
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
    path opencv_dir = path(getenv("OPENCV_DIR"));
    path faceDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "haarcascades" / "haarcascade_frontalface_alt2.xml";
    path landmarkDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "FaceLandmarksModels" / "lbfmodel.yaml";
    path video_data_root = "E:\\obama_dataset\\video";
    path video_info_root = "E:\\obama_dataset\\obama_data_SO";
    path mouth_shapes_root = "E:\\obama_dataset\\mouth_shapes\\480p\\haar";
    string resolution = "480p";
    string extension = ".mp4";
    vector<path> video_dirs = get_subdir(video_data_root);
    CascadeClassifier faceDetector(faceDetectorPath.string());
    Ptr<FacemarkLBF> landmarkDetector = FacemarkLBF::create();
    landmarkDetector->loadModel(landmarkDetectorPath.string());
    
    for (int i = 0; i < video_dirs.size(); i++)
    {
        string yt_id = get_yt_id(video_dirs[i]);
        cout << "Procesing: " << yt_id <<" with ind: "<<i<< endl;
        path video_path = get_video(video_dirs[i], resolution, yt_id);
        vector<pair<int, int>> frame_pos = get_video_info(video_info_root, yt_id);
        path save_folder_root = mouth_shapes_root / yt_id;
        if (is_directory(save_folder_root))
        {
            cout << "ALREADY PROCESSED!!!" << '\n';
        }
        else
        {
            create_directories(save_folder_root);
            VideoCapture video = VideoCapture(video_path.string());
            int part = 1;
            for (vector<pair<int, int>>::const_iterator iter = frame_pos.begin(); iter != frame_pos.end(); ++iter)
            {
                path subdir = save_folder_root / to_string(part);
                part++;
                create_directory(subdir);
                video.set(CV_CAP_PROP_POS_FRAMES, iter->first);
                int frame_ctr = iter->first;
                int endFrames = iter->first + iter->second;
                Rect prev_face;
                bool process = true;
                std::ofstream data_log;
                path data_log_path = subdir / "mouth_shapes.csv";
                data_log.open(data_log_path.string(), std::ios::out | std::ios::app);
                while ((frame_ctr <= endFrames) && process)
                {
                    Mat frame, grayFrame;
                    bool status = video.read(frame);
                    if (status)
                    {
                        vector<Rect> faces, all_faces;
                        bool first_frame = frame_ctr == iter->first;
                        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
                        faceDetector.detectMultiScale(grayFrame, all_faces, 1.2, 6);
                        switch (all_faces.size())
                        {
                        case 0:
                        {
                            if (first_frame)
                            {
                                process = false;
                                std::ofstream out;
                                path out_path = subdir / "report.txt";
                                out.open(out_path.string(), std::ios::out | std::ios::app);
                                out << "This part of the video is skipped!\n";
                                out.close();
                            }
                            else
                            {
                                faces.push_back(prev_face);
                            }
                        }
                        break;
                        case 1:
                        {
                            faces = all_faces;
                        }
                        break;
                        default:
                        {
                            if (first_frame)
                            {
                                faces = best_face(all_faces, prev_face);
                            }
                            else
                            {
                                faces.push_back(all_faces[0]);
                            }
                        }
                        break;
                        }
                        if (process)
                        {
                            prev_face = faces[0];
                            vector<vector<Point2f>> landmarks;
                            bool success = landmarkDetector->fit(grayFrame, faces, landmarks);
                            if (success)
                            {
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

                                tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model_with_expressions, landmarksM, landmark_mapper, grayFrame.cols, grayFrame.rows, edge_topology, ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f);
                                log_mesh(data_log, mesh, frame_ctr);
                            }
                            cout << '\r' << std::setw(5)<< frame_ctr << "/" << std::setw(5)<< endFrames << std::flush;
                            frame_ctr++;
                        }
                        
                    }
                }
            }
            destroyAllWindows();
        }
        cout<< "\nFinished: " << yt_id << endl;
    }
}