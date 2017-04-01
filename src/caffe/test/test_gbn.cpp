#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

//#include "caffe/common_layers.hpp"
#include "caffe/gbn_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

template <typename TypeParam>
class CBNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  CBNLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 7, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    std::ofstream ofile;
    ofile.open("test.txt",ios::out|ios::app);
    ofile<<std::endl;
    ofile<<"data"<<std::endl;
    for (int i=0;i<3;i++){
      for (int j=0;j<18;j++)
        ofile<<blob_bottom_->cpu_data()[j*3+i]<<" ";
       ofile<<std::endl;
    }

  }
  virtual ~CBNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CBNLayerTest, TestDtypesAndDevices);

TYPED_TEST(CBNLayerTest, TestForward_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);

  CBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}


TYPED_TEST(CBNLayerTest, TestBackward_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  CBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CBNLayerTest, TestForward_LayerNormalize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_layer_normalize(true);

  CBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < num; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < channels; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(j, i, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * channels;
    var /= height * width * channels;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(CBNLayerTest, TestBackward_LayerNormalize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_layer_normalize(true);

  CBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
TYPED_TEST(CBNLayerTest, TestForwardGlobal_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_global_normalize(true);

  CBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();
  Dtype sum = 0, var = 0;

  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
  }
    sum /= height * width * num * channels;
    var /= height * width * num * channels;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
}


TYPED_TEST(CBNLayerTest, TestBackwardGlobal_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_global_normalize(true);

  CBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
TYPED_TEST(CBNLayerTest, TestForward_ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_sme(true);



  CBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();
  for (int j = 0; j < channels/3; j=j+3) {
     Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data1 = this->blob_top_->data_at(i, j, k, l);
          Dtype data2 = this->blob_top_->data_at(i, j+1, k, l);
          Dtype data3 = this->blob_top_->data_at(i, j+2, k, l);

          sum += data1+data2+data3;
          var += data1 * data1+ data2 * data2 + data3* data3;
        }
      }
    }
    sum /= height * width * num * 3;
    var /= height * width * num * 3;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(CBNLayerTest, TestBackward_ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_sme(true);


  CBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CBNLayerTest, TestForwardGlobal__ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_global_normalize(true);
  gbn_param->set_sme(true);

  CBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();
      Dtype sum = 0, var = 0;

  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
  }
    sum /= height * width * num * channels;
    var /= height * width * num * channels;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
}

TYPED_TEST(CBNLayerTest, TestBackwardGlobal_ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_sme(true);
  gbn_param->set_global_normalize(true);


  CBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
/*
#ifdef USE_CUDNN
template <typename TypeParam>
class CuDNNCBNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  CuDNNCBNLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNCBNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNCBNLayerTest, TestDtypesAndDevices);

TYPED_TEST(CuDNNCBNLayerTest, TestForward_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);

  CuDNNCBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(CuDNNCBNLayerTest, TestBackward_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  CuDNNCBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNCBNLayerTest, TestForwardGlobal_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_global_normalize(true);

  CuDNNCBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();
      Dtype sum = 0, var = 0;

  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
  }
    sum /= height * width * num * channels;
    var /= height * width * num * channels;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
}


TYPED_TEST(CuDNNCBNLayerTest, TestBackwardGlobal_OneChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_global_normalize(true);

  CuDNNCBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
TYPED_TEST(CuDNNCBNLayerTest, TestForward_ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_sme(true);



  CuDNNCBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int j = 0; j < channels; ++j) {
    Dtype sum = 0, var = 0;
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
    sum /= height * width * num;
    var /= height * width * num;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(CuDNNCBNLayerTest, TestBackwardGlobal_ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_sme(true);
  gbn_param->set_global_normalize(true);


  CuDNNCBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNCBNLayerTest, TestForwardGlobal__ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_global_normalize(true);
  gbn_param->set_sme(true);

  CuDNNCBNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();
    Dtype sum = 0, var = 0;
  for (int j = 0; j < channels; ++j) {
    for (int i = 0; i < num; ++i) {
      for ( int k = 0; k < height; ++k ) {
        for ( int l = 0; l < width; ++l ) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
    }
  }
    sum /= height * width * num * channels;
    var /= height * width * num * channels;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
}

TYPED_TEST(CuDNNCBNLayerTest, TestBackwardGlobal_ThreeChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CuDNNGBNParameter* gbn_param = layer_param.mutable_gbn_param();
  gbn_param->set_eps(0.);
  gbn_param->set_sme(true);
  gbn_param->set_global_normalize(true);


  CuDNNCBNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-4);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#endif
*/

}

