/*
 * convert_thumos.cpp
 *
 *  Created on: Mar 21, 2016
 *      Author: zehuany
 */

/*
 * convert_ucf.cpp
 *
 *  Created on: Mar 20, 2016
 *      Author: zehuany
 */


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
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include <ctime>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "leveldb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_string(mode, "rgb",
    "Optional: What type of data should be established ('rgb','flow')");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::vector<std::pair<int,int> > lines_duration;
  std::vector<std::pair<int,int> >lines_annotations;
  std::string filename;
  int label;
  int length,offset, bindex, eindex;
  while (infile >> filename >> length >>offset>> label>>bindex>>eindex) {
    lines.push_back(std::make_pair(filename, label));
    lines_duration.push_back(std::make_pair(length,offset));
    lines_annotations.push_back(std::make_pair(bindex,eindex));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    rng_t* rng=caffe_rng();
    shuffle(lines.begin(), lines.end(),rng);
    shuffle(lines_duration.begin(), lines_duration.end(),rng);
    shuffle(lines_annotations.begin(),lines_annotations.end(),rng);

  }
  LOG(INFO) << "A total of " << lines.size() << " videos.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;
  char tmp[30];
  std::clock_t    start;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    string fn,fn1;
    for (int fid=0; fid<lines_duration[line_id].first;fid++){
	start = std::clock();
    	if (FLAGS_mode=="rgb"){
			sprintf(tmp,"img_%05d.jpg",fid+lines_duration[line_id].second);
			fn  =lines[line_id].first + "/" + tmp;
                      sprintf(tmp,"img_%08d.jpg",fid+lines_duration[line_id].second);
                        fn1  =lines[line_id].first + "/" + tmp;
			if (encoded && !enc.size()) {
			  // Guess the encoding type from the file name
			  size_t p = fn.rfind('.');
			  if ( p == fn.npos )
				LOG(WARNING) << "Failed to guess the encoding of '" << tmp << "'";
			  enc = fn.substr(p);
			  std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
			}
			status = ReadImageToDatum(root_folder+fn,
				0, resize_height, resize_width, is_color,
				enc, &datum);
    	}
    	else {
    		sprintf(tmp,"flow_%05d.jpg",fid+lines_duration[line_id].second);
			fn  =lines[line_id].first + "/" + tmp;
                sprintf(tmp,"flow_%08d.jpg",fid+lines_duration[line_id].second);
                        fn1  =lines[line_id].first + "/" + tmp;
			status = ReadFlowToDatum(root_folder+lines[line_id].first + "/",
							0, resize_height, resize_width, is_color,enc, &datum,0,fid+lines_duration[line_id].second);
    	}
		if (status == false)
			continue;
		if (check_size) {
		  if (!data_size_initialized) {
			data_size = datum.channels() * datum.height() * datum.width();
			data_size_initialized = true;
		  } else {
			const std::string& data = datum.data();
			CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
				<< data.size();
		  }
		}
		if (fid>=lines_annotations[line_id].first-1 && fid< lines_annotations[line_id].second )
			datum.set_label(lines[line_id].second);
		// sequential
		//int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", fid+1,fn.c_str());

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		//txn->Put(string(key_cstr, length), out);
               txn->Put(fn1, out);
	       LOG(ERROR) <<(std::clock() - start) / (double)CLOCKS_PER_SEC<<" s";

		if (++count % 1000 == 0) {
		  // Commit db
		  txn->Commit();
		  txn.reset(db->NewTransaction());
		  LOG(ERROR) << "Processed " << count << " files.";
		}
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}





