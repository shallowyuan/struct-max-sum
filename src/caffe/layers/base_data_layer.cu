#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
vector<Dtype> BasePrefetchingDataLayer<Dtype>::lastdata;

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_data_);
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
      
        
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_label_);
    // Copy the labels.
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }/*
  const Dtype* bottom_data=top[1]->cpu_data();
        if (this->lastdata.size()==0){
          for (int i=0;i<this->prefetch_label_.count();i++)
            lastdata.push_back(bottom_data[i]);
                      LOG(INFO)<<"first pass-------------------------------"<<lastdata.size();

          }
        else{
          for (int i=0;i<this->prefetch_label_.count();i++)
            CHECK(this->lastdata[i]==bottom_data[i]);
          LOG(INFO)<<"checking pass "<<lastdata.size();
        }*/
#ifdef USE_MPI
  //advance (all_rank - (my_rank+1)) mini-batches to be ready for next run
  BaseDataLayer<Dtype>::OffsetCursor(top[0]->num() * (Caffe::MPI_all_rank() - 1));
#endif
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
