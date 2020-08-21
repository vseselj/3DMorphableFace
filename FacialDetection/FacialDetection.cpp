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

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace boost::filesystem;
using namespace eos;
using eos::core::Landmark;
using eos::core::LandmarkCollection;

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
vector<pair <int, int>> get_video_info(path s, string& yt_id)
{
	vector<pair <int, int>> frame_pos;
	for (auto& p : recursive_directory_iterator(s))
		if (is_directory(p.path()))
		{
			if (p.path().string().find(yt_id) != string::npos)
			{
				std::ifstream start_frame_file(p.path().string() + "\\startframe.txt"), n_frame_file(p.path().string() + "\\nframe.txt");
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
	path opencv_dir = path(getenv("OPENCV_DIR"));
	path faceDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "haarcascades" / "haarcascade_frontalface_alt2.xml";
	path landmarkDetectorPath = (opencv_dir.parent_path()).parent_path() / "etc" / "FaceLandmarksModels" / "lbfmodel.yaml";
	path video_data_root = "E:\\obama_dataset\\video";
	path video_info_root = "E:\\obama_dataset\\obama_data_SO";
	string resolution = "480p";
	string extension = ".mp4";
	vector<path> video_dirs = get_subdir(video_data_root);
	CascadeClassifier faceDetector(faceDetectorPath.string());
	int i = 50;
	string yt_id = get_yt_id(video_dirs[i]);
	cout << "Procesing: " << yt_id << " with ind: " << i << endl;
	path video_path = get_video(video_dirs[i], resolution, yt_id);
	vector<pair<int, int>> frame_pos = get_video_info(video_info_root, yt_id);
	VideoCapture video = VideoCapture(video_path.string());
	vector<pair<int, int>>::const_iterator iter = frame_pos.begin();
	int frame_ctr = iter->first;
	int endFrames = iter->first + 3;
	Ptr<FacemarkLBF> landmarkDetector = FacemarkLBF::create();
	landmarkDetector->loadModel(landmarkDetectorPath.string());
	while (frame_ctr <= endFrames)
	{
		video.set(CV_CAP_PROP_POS_FRAMES, frame_ctr);
		Mat frame, grayFrame;
		bool status = video.read(frame);
		vector<Rect> faces, all_faces;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Mat3b grayBGR,grayBGR1;
		cvtColor(grayFrame, grayBGR, COLOR_GRAY2BGR);
		cvtColor(grayFrame, grayBGR1, COLOR_GRAY2BGR);
		faceDetector.detectMultiScale(grayFrame, all_faces, 1.2, 6);
		//cv::rectangle(grayBGR, all_faces[0], Scalar(0, 0, 255), 2);
		path save = "C:\\Users\\vseselj\\Documents\\MSC_report\\res\\" + to_string(frame_ctr) + ".jpg";
		//imwrite(save.string(), grayBGR);
		vector<vector<Point2f>> landmarks;
		bool success = landmarkDetector->fit(grayFrame, all_faces, landmarks);
		//face::drawFacemarks(grayBGR1, landmarks[0], Scalar(0, 0, 255));
		save = "C:\\Users\\vseselj\\Documents\\MSC_report\\res\\" + to_string(frame_ctr) + "lm.jpg";
		//imwrite(save.string(), grayBGR1);
		Mat outimg = grayFrame.clone();
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
		tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model_with_expressions, landmarksM, landmark_mapper, grayBGR.cols, grayBGR.rows, edge_topology, ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f);
		// The 3D head pose can be recovered as follows:
		float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
		// and similarly for pitch and roll.

		// Extract the texture from the image using given mesh and camera parameters:
		const Eigen::Matrix<float, 3, 4> affine_from_ortho =
			fitting::get_3x4_affine_camera_matrix(rendering_params, grayBGR.cols, grayBGR.rows);
		const core::Image4u isomap =
			render::extract_texture(mesh, affine_from_ortho, core::from_mat(grayBGR), true);
		string outputbasename = "C:\\Users\\vseselj\\Desktop\\out\\"+to_string(frame_ctr);
		// Draw the fitted mesh as wireframe, and save the image:
		render::draw_wireframe(outimg, mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
			fitting::get_opencv_viewport(grayBGR.cols, grayBGR.rows));
		path outputfile = outputbasename + ".png";
		cv::imwrite(outputfile.string(), outimg);
		// Save the mesh as textured obj:
		outputfile.replace_extension(".obj");
		core::write_textured_obj(mesh, outputfile.string());
		// And save the isomap:
		outputfile.replace_extension(".isomap.png");
		imwrite(outputfile.string(), core::to_mat(isomap));
		frame_ctr++;
	}
	destroyAllWindows();
}