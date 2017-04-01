
#ifndef CAFFE_TEST_LAYER_HPP_
#define CAFFE_TEST_LAYER_HPP_

#include <vector>

#include "caffe/loss_layers.hpp"


namespace caffe {
template <typename Dtype>
class OverlapLossLayer : public LossLayer<Dtype> {
 public:
  explicit OverlapLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),start(0),end(-1),slabel(-1) {}
  virtual ~OverlapLossLayer(){
    r_sum.clear();
    max_sum.clear();
    m_begin.clear();
    m_end.clear();
  }
  void  Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) ;
  virtual inline const char* type() const { return "OverlapLoss"; }
  //inline const vector<Dtype>& get_max_sum() const { return max_sum; }
  inline const vector<vector<int> >&   get_max_begin() const {return m_begin; }
  inline const vector<vector<int> >&   get_max_end  () const {return m_end; }
  virtual void Inference (const Dtype* array, const Dtype * label,  int num, int dim, int k);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

  //vector<Dtype> r_sum, max_sum;
  vector<vector<int> > m_begin, m_end;
  //constant margin
  vector<vector<Dtype> > max_sum;
  //vector<vector<std::pair<int,int> > > m_ind;
 private:
  int r_begin;
  bool ctype;
  int start,end,slabel;
  Blob<Dtype> loss_t;
  vector<Dtype> r_sum;

};
}
#endif
