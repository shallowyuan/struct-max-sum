#ifndef  CAFFE_EDATUM_H_ 
#define  CAFFE_EDATUM_H_ 

#include "caffe/data_layers.hpp"
#include <map>

namespace caffe {
template <typename Dtype>
class VideoEDatumLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit VideoEDatumLayer(const LayerParameter& param)
	: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~VideoEDatumLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "VideoEDatum"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 2; }
	inline int getclassid()  {return class_;}
    void setclassid(int classid);
       
protected:
	virtual void InternalThreadEntry();
	vector<shared_ptr<db::DB> >db_;
	vector<shared_ptr<db::Cursor> >cursor_;
	vector<map<string,int> > starts;
	vector<map<string,int> > ends;

	vector<string> lines_;
	int lines_id_;
	enum InputMode{
			SEQUENCE, SHUFFLE
	};

	InputMode cur_input_mode_;
	vector<vector<string> > shuffle_key_pool_;
	vector<string>::iterator shuffle_cursor_;
	shared_ptr<Caffe::RNG> frame_prefetch_rng_;

	// for seperate class training 
	int class_;
	float pos;
private:
	vector<int> _get_inds(const string& vname);
};
}
#endif
