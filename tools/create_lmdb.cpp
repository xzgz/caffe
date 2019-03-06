// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace std;
vector<vector<pair<string, int> > > getFilesList(string dirpath);


#define LINUX_PLATFORM
//#define WIN32_PLATFORM

int main(int argc, char** argv) {
    const string root_folder = "/media/xzgz/ubudata/Ubuntu/Code/python-study/chess-project/train";
//    const string root_folder = "/media/xzgz/ubudata/Ubuntu/Code/python-study/chess-project/test";
    const string db_name = "/media/xzgz/ubudata/Ubuntu/Code/python-study/chess-project/chess_train_lmdb";
//    const string db_name = "/media/xzgz/ubudata/Ubuntu/Code/python-study/chess-project/chess_test_lmdb";
    const bool is_color = true;
    // Whether check that all the datum have the same size.
    const bool check_size = false;
    // Randomly shuffle the order of images and their labels.
    const bool shuffle_data = true;
    const string encode_type = "";
    const string backend = "lmdb";
    // If any of resize_width, resize_height equal to 0, don't resize.
//    int resize_width = 28;
//    int resize_height = 28;
    int resize_width = 0;
    int resize_height = 0;

    vector<vector<pair<string, int> > > filename_label = getFilesList(root_folder);
    vector<pair<string, int> > fpath_label = filename_label[0];
    vector<pair<string, int> > class_name_label = filename_label[1];
    ofstream name_file((root_folder + "/class_name_label.txt").c_str());
    for (int i = 0; i < class_name_label.size(); i++)
    {
        std::pair<std::string, int> fm_lb = class_name_label.at(i);
        cout << fm_lb.first << "  " << fm_lb.second << endl;
        name_file << fm_lb.first << " " << fm_lb.second << endl;
    }
    name_file.close();
//    for (int i = 0; i < fpath_label.size(); i++)
//    {
//        std::pair<std::string, int> fm_lb = fpath_label.at(i);
//        cout << fm_lb.first << "  " << fm_lb.second << endl;
//    }

    if (shuffle_data) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        shuffle(fpath_label.begin(), fpath_label.end());
    }
    LOG(INFO) << "A total of " << fpath_label.size() << " images.";

    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(db_name, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    Datum datum;
    int count = 0;

    for (int i = 0; i < fpath_label.size(); i++) {
        bool status;
        std::string enc = encode_type;
        status = ReadImageToDatum(fpath_label[i].first, fpath_label[i].second,
                resize_height, resize_width, is_color, enc, &datum);
        if (status == false) continue;
        // sequential
        string key_str = caffe::format_int(i, 8) + "_" + fpath_label[i].first;

        // Put in db
        string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (++count % 1000 == 0) {
            // Commit db
            txn->Commit();
            txn.reset(db->NewTransaction());
            LOG(INFO) << "Processed " << count << " files.";
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }

    return 0;
}


#ifdef LINUX_PLATFORM
#include <memory.h>
#include <dirent.h>
vector<vector<pair<string, int> > > getFilesList(string dirpath) {
    vector<vector<pair<string, int> > > filename_label;
    static vector<pair<string, int> > fpath_label;
    static vector<pair<string, int> > class_name_label;
    static int label = -1;
    DIR *dir = opendir(dirpath.c_str());

    if (dir == NULL)
    {
        cout << "Open dir error!" << endl;
        return filename_label;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            label += 1;
            class_name_label.push_back(make_pair(entry->d_name, label));
            string dirNew = dirpath + "/" + entry->d_name;
            getFilesList(dirNew);
//            vector<vector<pair<string, int> > > temp_filename_label = getFilesList(dirNew);
//            fpath_label.insert(filename_label[0].end(), temp_filename_label[0].begin(), temp_filename_label[0].end());
        }
        else {
            string name = entry->d_name;
            string filename = dirpath + "/" + name;
            fpath_label.push_back(make_pair(filename, label));
        }
    }
    closedir(dir);
    filename_label.push_back(fpath_label);
    filename_label.push_back(class_name_label);
    return filename_label;
}
#endif

#ifdef WIN32_PLATFORM
#include <io.h>
vector<vector<pair<string, int> > > getFilesList(string dir)
{
    vector<vector<pair<string, int> > > filename_label;
    static vector<pair<string, int> > fpath_label;
    static vector<pair<string, int> > class_name_label;
    static int label = -1;
    string dir2 = dir + "\\*.*";

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dir2.c_str(), &findData);
	if (handle == -1) {
		cout << "Open dir error!" << endl;
		return filename_label;
	}
	while (_findnext(handle, &findData) == 0)
	{
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;
            label += 1;
            class_name_label.push_back(make_pair(findData.name, label));
            string dirNew = dir + "\\" + findData.name;
            getFilesList(dirNew);
//            std::vector<std::pair<std::string, int> > temp_filename_label = getFilesList(dirNew);
//            filename_label.insert(filename_label.end(), temp_filename_label.begin(), temp_filename_label.end());
		}
		else
		{
			string filename = dir + "\\" + findData.name;
            fpath_label.push_back(std::make_pair(filename, label));
		}
	}
	_findclose(handle);
	filename_label.push_back(fpath_label);
    filename_label.push_back(class_name_label);
	return filename_label;
}
#endif
