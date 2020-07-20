#include <iostream>
#include <boost/filesystem.hpp>

using namespace std;
namespace bfs = boost::filesystem;

vector<bfs::path> get_subdir(bfs::path dir)
{
    vector<bfs::path> subdirs;
    for (auto& p : bfs::directory_iterator(dir))
        if (bfs::is_directory(p.path()))
        {
            subdirs.push_back(p.path());
        }
    return subdirs;
}

bfs::path get_hres_video(bfs::path subdir, string lower_res, bool& exist)
{
    exist = true;
    bfs::path video;
    for (auto& p : bfs::directory_iterator(subdir))
    {
        if (bfs::is_regular_file(p.path()))
        {
            if (bfs::extension(p.path()) == ".mp4")
            {
                bool q = p.path().string().find(lower_res) == string::npos;
                exist = exist && q; 
                if (q)
                {
                   video = p.path();
                }
            }
        }
    }
    return video;
}

string command(bfs::path video, string target_res)
{
    bfs::path parent_folder = video.parent_path();
    bfs::path youtubeID = parent_folder.stem();
    string out_video_name = youtubeID.string() + target_res + ".mp4";
    bfs::path out_video = parent_folder / out_video_name;
    bfs::path out_log = parent_folder / "out_log.txt";
    bfs::path error_log = parent_folder / "error_log.txt";
    string input = '"' + video.string() + '"';
    string output = '"' + out_video.string() + '"';
    string cmd = "ffmpeg -i " + input +" -s hd480 -c:v libx264 -crf 23 -c:a aac -strict -2 " + output;
    return cmd;
}

int main()
{
    string target_resolution = "_480p";
	bfs::path video_data_root("E:\\obama_dataset\\video");
    vector <bfs::path> video_data_dirs = get_subdir(video_data_root);
    int result = 0;
    for (int i = 0; i < video_data_dirs.size(); i++)
    {
        bool exist = true;
        bfs::path video_path = get_hres_video(video_data_dirs[i], target_resolution, exist);
        if (exist)
        {
            string cmd = command(video_path, target_resolution);
            int result_tmp = std::system(cmd.c_str());
            result += result_tmp;
        }
        else
        {
            cout << "ALREADY CREATED!!" << endl;
        }
    }
    cout << "Errors:: " << result << endl;
}
