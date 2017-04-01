#ifndef CAFFE_UTIL_HDF5_H_
#define CAFFE_UTIL_HDF5_H_

#include <string>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "H5Cpp.h"      // C++ API header file
using namespace H5;

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void hdf5_load_d_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_d_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_d_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob,
    bool write_diff = false);

int hdf5_load_int(hid_t loc_id, const string& dataset_name);
void hdf5_save_int(hid_t loc_id, const string& dataset_name, int i);
string hdf5_load_string(hid_t loc_id, const string& dataset_name);
string hdf5_load_string_CPlus(const string& filename, const string& dataset_name);
void hdf5_save_string(hid_t loc_id, const string& dataset_name,
                      const string& s);

int hdf5_get_num_links(hid_t loc_id);
string hdf5_get_name_by_idx(hid_t loc_id, int idx);

}  // namespace caffe

#endif   // CAFFE_UTIL_HDF5_H_