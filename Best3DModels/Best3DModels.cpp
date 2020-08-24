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
#include <boost/tokenizer.hpp>
#include <iostream>
#include <string>

using namespace eos;
using namespace std;
using namespace cv;
using namespace cv::face;
using namespace boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;

typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
vector<int> vertex_ind = { 398, 315, 413, 329, 825, 736, 812, 841, 693, 411, 264, 431, 416, 423, 828, 817, 442, 404 };

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

void add_shape(vector<pair<vector<Point3f>,int>>& mouth_shapes, core::Mesh& m,int frame)
{
    
    vector<Point3f> new_mouth_shape;
    for (int i = 0; i < vertex_ind.size(); i++)
    {
        Point3f new_point;
        new_point.x = m.vertices[vertex_ind[i]].x();
        new_point.y = m.vertices[vertex_ind[i]].y();
        new_point.z = m.vertices[vertex_ind[i]].z();
        new_mouth_shape.push_back(new_point);
    }
    mouth_shapes.push_back(pair<vector<Point3f>, int>(new_mouth_shape, frame));
}

inline double calculate_error(vector<Point3f> shape1, vector<Point3f> shape2)
{
    double err = 0;
    for (int i = 0; i < shape1.size(); i++)
    {
        err += sqrt(pow(shape1[i].x - shape2[i].x, 2) + pow(shape1[i].y - shape2[i].y, 2) + pow(shape1[i].z - shape2[i].z, 2));
    }
    return err;
}

int find_best_shape(vector<pair<vector<Point3f>, int>>& mouth_shapes, vector<Point3f> RNN_mouth_shape)
{
    double min_err = INFINITY;
    int best_frame;
    for (int i = 0; i < mouth_shapes.size(); i++)
    {
        double err = calculate_error(mouth_shapes[i].first, RNN_mouth_shape);
        if (err<min_err)
        {
            best_frame = mouth_shapes[i].second;
            min_err = err;
        }
    }
    return best_frame;
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
    vector<pair<vector<Point3f>, int>> mouth_shapes;

    /*Face and landmark detector*/
    path opencv_dir = path(getenv("OPENCV_DIR"));
    path faceDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "haarcascades" / "haarcascade_frontalface_alt2.xml";
    path landmarkDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "FaceLandmarksModels" / "lbfmodel.yaml";
    CascadeClassifier faceDetector(faceDetectorPath.string());
    Ptr<FacemarkLBF> landmarkDetector = FacemarkLBF::create();
    landmarkDetector->loadModel(landmarkDetectorPath.string());

    path target_video = "C:\\obama_dataset\\video\\_csblV1PJ4o\\Weekly Address_ Climate Change Can No Longer Be Ignored.mp4";
    path face_models_path = "C:\\Users\\vseselj\\Desktop\\out";
    VideoCapture video = VideoCapture(target_video.string());
    int start_frame = 300;
    int end_frame = video.get(CAP_PROP_FRAME_COUNT)-400;
    video.set(CV_CAP_PROP_POS_FRAMES, start_frame);
    for (int i = start_frame; i < end_frame; i++)
    {
        Mat frame, grayFrame, grayFrame1;
        bool status = video.read(frame);
        if (status)
        {
            vector<Rect> all_faces, faces;
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
            cvtColor(grayFrame, grayFrame1, COLOR_GRAY2BGR);
            faceDetector.detectMultiScale(grayFrame, all_faces, 1.2, 6);
            vector<vector<Point2f>> landmarks;
            bool success = landmarkDetector->fit(grayFrame, all_faces, landmarks);
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
                // Create models
                core::Mesh mesh;
                fitting::RenderingParameters rendering_params;
                tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model_with_expressions, landmarksM, landmark_mapper, grayFrame1.cols, grayFrame1.rows, edge_topology, ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f);
                const Eigen::Matrix<float, 3, 4> affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, grayFrame1.cols, grayFrame1.rows);
                const core::Image4u isomap = render::extract_texture(mesh, affine_from_ortho, core::from_mat(grayFrame1), true);
                string outputbasename = face_models_path.string() +"\\"+ to_string(i);
                path outputfile = outputbasename + ".png";
                // Save the mesh as textured obj:
                outputfile.replace_extension(".obj");
                core::write_textured_obj(mesh, outputfile.string());
                // And save the isomap:
                outputfile.replace_extension(".isomap.png");
                imwrite(outputfile.string(), core::to_mat(isomap));
                cout << '\r' << std::setw(5) << i << "/" << std::setw(5) << end_frame << std::flush;
                add_shape(mouth_shapes, mesh, i);
            }
        }
    }
    destroyAllWindows();
    path RNN_mouth_shapes_path = "";
    std::ifstream RNN_mouth_shapes(RNN_mouth_shapes_path.string());
    string line;
    vector<string> vec;
    
    path best_shape_folder = "";
    while (std::getline(RNN_mouth_shapes, line))
    {
        vector<Point3f> RNN_mouth_shape;
        Tokenizer toc(line);
        vec.assign(toc.begin(), toc.end());
        for (int i = 1; i < vec.size(); i += 3)
        {
            Point3f point;
            point.x = stof(vec[i]);
            point.y = stof(vec[i+1]);
            point.z = stof(vec[i+2]);
            RNN_mouth_shape.push_back(point);
        }
        int best_frame = find_best_shape(mouth_shapes, RNN_mouth_shape);
        string inputbasename = face_models_path.string() + "\\" + to_string(best_frame);
        string outputbasename = best_shape_folder.string() + "\\" + to_string(best_frame);
        copy_file(inputbasename + ".png", outputbasename + ".png");
        copy_file(inputbasename + ".obj", outputbasename + ".obj");
        copy_file(inputbasename + ".isomap.png", outputbasename + ".isomap.png");
    }
}