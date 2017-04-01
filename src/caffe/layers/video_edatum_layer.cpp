/*
 * video_datum_layer.cpp
 *
 *  Created on: Mar 22, 2016
 *      Author: zehuany
 */
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "caffe/video_edatum_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif


namespace caffe{
void ReadVideoDatum(shared_ptr<db::Cursor> cursor,vector<Datum>& vfdata){
	int new_id;
	char key_cstr[30];
	int index;
	string fn;
	Datum datum;
	for (int n=0;n<1620;n++){
		new_id=n/2+1;
		sprintf(key_cstr,"_%04d",new_id);
		//fn=fn+key_cstr;i
		//LOG(INFO)<<key_cstr;
		fn=cursor->key();
		index=fn.find_first_of('_');
        //       LOG(INFO)<<fn<<"  "<<new_id<<" "<<n;
		fn=fn.substr(index+1,index+4);
                //LOG(INFO)<<fn;
		CHECK(std::atoi(fn.c_str())==new_id);
		datum.ParseFromString(cursor->value());
		CHECK(DecodeDatumNative(&datum));
		vfdata.push_back(datum);
		cursor->Next();
	}

}
template <typename Dtype>
VideoEDatumLayer<Dtype>:: ~VideoEDatumLayer<Dtype>(){
	this->JoinPrefetchThread();
}
template <typename Dtype>
vector<int>  VideoEDatumLayer<Dtype>::_get_inds(const string& videoname){
	vector<int> inds;
	int start,end;
    int video_size = this->layer_param_.video_datum_param().video_size();
    map<string,int>::iterator iter;
	int index=videoname.find_first_of('_');
    //LOG(INFO)<<fn<<"  "<<new_id;
	string fn=videoname.substr(0,index);
	iter=starts[lines_id_].find(fn);
	CHECK(iter!=starts[lines_id_].end())<<fn<<" not found";
	start=iter->second;
	iter=ends[lines_id_].find(fn);
	CHECK(iter!=ends[lines_id_].end())<<fn<<" not found";
	end=iter->second;
	CHECK(start<end)<<"end should be bigger than start";
    float pratio=float(end-start+1)/(video_size*pos);
    float nratio=float(800.0+start-end-1)/(video_size*(1-pos));
    int new_id, offset=1;
    for (int i=0;i<video_size*(1-pos);i++){
	    caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
	    new_id=offset+(*frame_rng)()%std::max(1,int(nratio));
	    offset=floor((i+1)*nratio);
	    inds.push_back(new_id<start?new_id:new_id+end-start+1);
	}
	offset = start;
	for (int i=0;i<video_size*pos;i++){
	    caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
	    new_id=offset+(*frame_rng)()%std::max(1,int(pratio));
	    inds.push_back(new_id);
	    offset=start+floor((i+1)*pratio);
	}
	CHECK(offset-1<=end);
	std::sort(inds.begin(),inds.end());
	return inds;


}
template <typename Dtype>
void VideoEDatumLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const string& source = this->layer_param_.video_datum_param().source();
	const int new_length = this->layer_param_.video_datum_param().new_length();
	const string dprefix = this->layer_param_.video_datum_param().prefix();
	pos = this->layer_param_.video_datum_param().pos();
    cur_input_mode_ = SEQUENCE;

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	std:: ifstream mfile;
	int start,end;
	string filename;
	while (infile >> filename){
		lines_.push_back(filename);
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;
	char mname[20];
    int num=lines_.size();
    db_.resize(num);
    cursor_.resize(num);
    shuffle_key_pool_.resize(num);
    starts.resize(num);
    ends.resize(num);
    for (int i=0;i<num;i++){
		db_[i].reset(db::GetDB(this->layer_param_.video_datum_param().backend()));
		// To make reading concurrently for leveldb
		string command1=string("rm ")+lines_[i]+ "/LOCK";
		string command2=string("touch ")+lines_[i]+ "/LOCK";
		system(command1.c_str());
		system(command2.c_str());
		db_[i]->Open(lines_[i], db::READ);
		cursor_[i].reset(db_[i]->NewCursor());
		// reading meta
		sprintf(mname,"data/thumos/%s-%d.txt",dprefix.c_str(),i==0?i:i+1);
		//LOG(INFO)<<mname;
		mfile.open(mname);
		CHECK (mfile.is_open())<<"Cannot open meta file";
		while(mfile>> filename >> start>>end){
			starts[i].insert(pair<string,int>(filename,start));
			ends[i].insert(pair<string,int>(filename,end));
		}
		mfile.close();
    }

	Datum datum;
	datum.ParseFromString(cursor_[0]->value());
	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	LOG(INFO)<<top_shape[0]<<top_shape[1]<<top_shape[2]<<top_shape[3];
	top_shape[1]=top_shape[1]*new_length;
	this->transformed_data_.Reshape(top_shape);
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

    int video_size = this->layer_param_.video_datum_param().video_size();
    int crop_size = this->layer_param_.transform_param().crop_size();
    if (crop_size>0){
		top[0]->Reshape(video_size, top_shape[1], crop_size, crop_size);
		this->prefetch_data_.Reshape(video_size, top_shape[1], crop_size, crop_size);
    }
    else {
		top[0]->Reshape(video_size, top_shape[1], datum.height(), datum.width());
		this->prefetch_data_.Reshape(video_size, top_shape[1], datum.height(), datum.width());
    }

	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(video_size, 1, 1, 1);
	this->prefetch_label_.Reshape(video_size, 1, 1, 1);

    for (int i=0;i<num;i++){
     	cursor_[i]->SeekToFirst();
    }
}

template <typename Dtype>
void VideoEDatumLayer<Dtype>::InternalThreadEntry(){

	Timer timer;
	Datum datum;
	const int kMaxKeyLength = 50;
	char key_cstr[kMaxKeyLength];
	string fn;
	int index;
	CHECK(this->prefetch_data_.count());
	CHECK(this->transformed_data_.count());
	VideoDatumParameter video_data_param = this->layer_param_.video_datum_param();
	const int new_length = video_data_param.new_length();
	const int lines_size = lines_.size();
	const int video_size = video_data_param.video_size();
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
    LOG(INFO)<<"begin to process"<<lines_[lines_id_]<<"....."<<cursor_[lines_id_]->key();
    Timer otimer;
    otimer.Start();
	if (video_data_param.modality() == VideoDatumParameter_Modality_RGB){
        int new_id=0;
        if (cur_input_mode_ == SEQUENCE) {
	      shuffle_key_pool_[lines_id_].push_back(cursor_[lines_id_]->key());

	    }
	    else if (cur_input_mode_ == SHUFFLE){
	      cursor_[lines_id_]->Lookup(*shuffle_cursor_);
	    }
	    //LOG(INFO)<<cursor_[lines_id_]->key();
	    // get inds
	    vector<int> inds = _get_inds(cursor_[lines_id_]->key());

		for (int item_id = 0; item_id < video_size; item_id++){
			new_id = inds[item_id];
			sprintf(key_cstr,"_%04d",new_id);
			fn=cursor_[lines_id_]->key();
			index=fn.find_first_of('_');
			fn=fn.substr(index+1,index+4);
			while (std::atoi(fn.c_str())!=new_id){
				cursor_[lines_id_]->Next();
				fn=cursor_[lines_id_]->key();
				index=fn.find_first_of('_');
				fn=fn.substr(index+1,index+4);
	        }
			CHECK(std::atoi(fn.c_str())==new_id);
			datum.ParseFromString(cursor_[lines_id_]->value());
			int offset1 = this->prefetch_data_.offset(item_id);
			int offset2 = this->prefetch_label_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset1);
			this->data_transformer_->Transform(datum, &(this->transformed_data_));
			top_label[offset2] = datum.label();
		}

		LOG(INFO)<<"reading video takes "<<otimer.MilliSeconds()<<".ms";
		//next iteration
		while (new_id++<=810){
			cursor_[lines_id_]->Next();
		}
	    if (cur_input_mode_ == SEQUENCE) {
			if (!cursor_[lines_id_]->valid()){
				cursor_[lines_id_]->SeekToFirst();
		        if (this->layer_param_.video_datum_param().shuffle() == true){
		        	shuffle(shuffle_key_pool_[lines_id_].begin(), shuffle_key_pool_[lines_id_].end());
		        }
		        lines_id_++;
				if (lines_id_ >= lines_size) {
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					cur_input_mode_ = this->layer_param_.video_datum_param().shuffle()?SHUFFLE:cur_input_mode_;
					shuffle_cursor_ = shuffle_key_pool_[lines_id_].begin();
					LOG(INFO)<<"Entering shuffling mode after first epoch";
				}
	      	}
	    } else if (cur_input_mode_ == SHUFFLE){
		    if (!cursor_[lines_id_]->valid())
				cursor_[lines_id_]->SeekToFirst();
	      	shuffle_cursor_++;
	      	if (shuffle_cursor_ == shuffle_key_pool_[lines_id_].end()){
		        LOG(INFO)<<"Start to next base and shuffle again";
		        shuffle(shuffle_key_pool_[lines_id_].begin(), shuffle_key_pool_[lines_id_].end());
		        lines_id_++;
				if (lines_id_ >= lines_size) {
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
				}
				shuffle_cursor_ = shuffle_key_pool_[lines_id_].begin();
	      	}
	    }
    }
    else{
        int new_id,start_id;
        string* data_string;
        Datum fdatum;
        if (cur_input_mode_ == SEQUENCE) {
	      shuffle_key_pool_[lines_id_].push_back(cursor_[lines_id_]->key());

	    }else if (cur_input_mode_ == SHUFFLE){
	    	LOG(INFO)<<cursor_[lines_id_]->key();
	      	cursor_[lines_id_]->Lookup(*shuffle_cursor_);
	    }
	    Timer timer;
	    timer.Start();
	    //vector<Datum> vfdata;
	    //vfdata.clear();
	    vector<int> inds = _get_inds(cursor_[lines_id_]->key());
	    //ReadVideoDatum(cursor_[lines_id_],vfdata);
	    //LOG(INFO)<<"reading video takes "<<timer.MilliSeconds()<<".ms";
	    //CHECK(vfdata.size()==1620);

        /***** read as a whole
		for (int item_id = 0; item_id < video_size; item_id++){
			start_id=(inds[item_id]-1)*2;
        	for (int n=0;n<new_length;n++){
                //new_id=start_id+n/2;
                new_id=start_id+n;
				datum=vfdata[std::min(1619,new_id)];
                if (n==0){
				   int offset2 = this->prefetch_label_.offset(item_id);
				   top_label[offset2] = datum.label();
		           fdatum=datum;
				   fdatum.set_channels(datum.channels()*new_length);
	               data_string=fdatum.mutable_data();
                }
                else {
                    data_string->append(datum.data());
                }
            }
			int offset1 = this->prefetch_data_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset1);
			this->data_transformer_->Transform(fdatum, &(this->transformed_data_));
		}*****/
		//**** reading separately
		for (int item_id = 0; item_id < video_size; item_id++){
			start_id=inds[item_id];
			for (int n=0;n<new_length;n++){
				new_id=n/2+start_id;
				sprintf(key_cstr,"_%04d",new_id);
				fn=cursor_[lines_id_]->key();
				index=fn.find_first_of('_');
				fn=fn.substr(index+1,index+4);
				while (std::atoi(fn.c_str())!=new_id){
					cursor_[lines_id_]->Next();
					fn=cursor_[lines_id_]->key();
					index=fn.find_first_of('_');
					fn=fn.substr(index+1,index+4);
	        	}
				CHECK(std::atoi(fn.c_str())==new_id);
				datum.ParseFromString(cursor_[lines_id_]->value());
				CHECK(DecodeDatumNative(&datum));
				if (n==0){
				   int offset2 = this->prefetch_label_.offset(item_id);
				   top_label[offset2] = datum.label();
		           fdatum=datum;
				   fdatum.set_channels(datum.channels()*new_length);
	               data_string=fdatum.mutable_data();
                }
                else {
                    data_string->append(datum.data());
                }
                cursor_[lines_id_]->Next();
			}
			if (cursor_[lines_id_]->valid())
				for (int n=0;n<new_length;n++)
					cursor_[lines_id_]->Last();
			int offset1 = this->prefetch_data_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset1);
			this->data_transformer_->Transform(fdatum, &(this->transformed_data_));
		}
		while (start_id++<=810){
			cursor_[lines_id_]->Next();
			cursor_[lines_id_]->Next();
		}


		LOG(INFO)<<"reading video takes "<<otimer.MilliSeconds()<<".ms";
	    if (cur_input_mode_ == SEQUENCE) {
			if (!cursor_[lines_id_]->valid()){
				cursor_[lines_id_]->SeekToFirst();
		        if (this->layer_param_.video_datum_param().shuffle() == true){
		          shuffle(shuffle_key_pool_[lines_id_].begin(), shuffle_key_pool_[lines_id_].end());
		        }
		        lines_id_++;
				if (lines_id_ >= lines_size) {
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					cur_input_mode_ = this->layer_param_.video_datum_param().shuffle()?SHUFFLE:cur_input_mode_;
					shuffle_cursor_ = shuffle_key_pool_[lines_id_].begin();
					LOG(INFO)<<"Entering shuffling mode after first epoch";

				}
		      }
	    } else if (cur_input_mode_ == SHUFFLE){
		    if (!cursor_[lines_id_]->valid())
				cursor_[lines_id_]->SeekToFirst();
	    	shuffle_cursor_++;
	    	if (shuffle_cursor_ == shuffle_key_pool_[lines_id_].end()){
		        LOG(INFO)<<"Start to next base and shuffle again";
		        shuffle(shuffle_key_pool_[lines_id_].begin(), shuffle_key_pool_[lines_id_].end());
		        lines_id_++;
				if (lines_id_ >= lines_size) {
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
				}
				shuffle_cursor_ = shuffle_key_pool_[lines_id_].begin();
	      }
	    }
	}
}

INSTANTIATE_CLASS(VideoEDatumLayer);
REGISTER_LAYER_CLASS(VideoEDatum);
}
