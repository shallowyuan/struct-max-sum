
#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

//#include "caffe/common_layers.hpp"
#include "caffe/data_reshape_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

template <typename TypeParam>
class DataReshapeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  DataReshapeLayerTest()
      : blob_bottom_(new Blob<Dtype>(20, 21, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);


  }
  virtual ~DataReshapeLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DataReshapeLayerTest, TestDtypesAndDevices);

TYPED_TEST(DataReshapeLayerTest, TestBackwar) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  DataReshapeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
}
