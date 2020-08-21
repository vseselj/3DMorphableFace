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
    CascadeClassifier faceDetector(faceDetectorPath.string());
    Ptr<FacemarkLBF> landmarkDetector = FacemarkLBF::create();
    landmarkDetector->loadModel(landmarkDetectorPath.string());

    path target_video = "";
}