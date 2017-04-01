/*
 * test_structsvm_loss.cpp
 *
 *  Created on: Feb 29, 2016
 *      Author: zehuany
 */
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/StructSoft_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class StructSoftLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  StructSoftLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(30, 4, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(30, 4, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(160);
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    begin=caffe_rng_rand()%30;
    end=begin+caffe_rng_rand() %(30-begin)+1;
    label=caffe_rng_rand()%2+1;
    caffe_set(0,Dtype(0),blob_bottom_label_->mutable_cpu_data());
    for (int i = begin; i <= end; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = label;
    }
    std::ofstream ofile;
  	ofile.open("test.txt",ios::out|ios::app);
    ofile<<std::endl;
    ofile<<"label"<<std::endl;
    for(int i=0;i<10;i++)
    	ofile<<float(blob_bottom_label_->mutable_cpu_data()[i])<<" ";
    ofile<<"data"<<std::endl;
    for (int i=0;i<7;i++){
    	for (int j=0;j<10;j++)
    		ofile<<blob_bottom_data_->cpu_data()[j*7+i]<<" ";
    	 ofile<<std::endl;
    }

    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~StructSoftLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int begin,end,label;
};

TYPED_TEST_CASE(StructSoftLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(StructSoftLossLayerTest, TestGradientC1_LOC) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  StructSVMLossParameter* structsvm_loss_param=layer_param.mutable_structsvm_loss_param();
  structsvm_loss_param->set_bias(0.5);

  StructSoftLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1700);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(StructSoftLossLayerTest, TestGradientC1_CLS) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  StructSVMLossParameter* structsvm_loss_param=layer_param.mutable_structsvm_loss_param();
  structsvm_loss_param->set_only_locate(false);
  structsvm_loss_param->set_bias(0.3);

  StructSoftLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1700);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe




