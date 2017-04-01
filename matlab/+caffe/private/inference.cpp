/*

 * inference.cpp
 *
 *  Created on: Apr 6, 2016
 *      Author: zehuany
 */
#include <vector>
#include <cfloat>
#include "mex.h"
using namespace std;

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }

static mxArray* singlevec_to_mx_vec(const vector<vector<float> > & single_vec) {
  mxArray* mx_vec = mxCreateDoubleMatrix(single_vec.size(), single_vec[0].size(),mxREAL);
  double* vec_mem_ptr = mxGetPr(mx_vec);
  int count=0;
  for (int i = 0; i < single_vec[0].size(); i++)
	  for (int j = 0; j < single_vec.size(); j++){
	  vec_mem_ptr[count++] = (double)single_vec[j][i];
  }
  return mx_vec;
}
static mxArray* structvec_to_mx_vec(const vector<vector<std::pair<int, int> > > & single_vec) {

  const int handle_field_num = 2;
  const char* handle_fields[handle_field_num] = { "start", "end" };
  mxArray* mx_vec= mxCreateStructMatrix(single_vec.size(), single_vec[0].size(), handle_field_num, handle_fields);
  int count=0;
  for (int i = 0; i < single_vec[0].size(); i++)
	  for (int j = 0; j < single_vec.size(); j++){
		  mxSetField(mx_vec, count, "start", mxCreateDoubleScalar(single_vec[j][i].first));
		  mxSetField(mx_vec, count++, "end", mxCreateDoubleScalar(single_vec[j][i].second));
  }
  return mx_vec;
}
template <typename Dtype>
void Inference (const Dtype* array,  int num, int dim, int k, vector<vector<Dtype> >& max_sum, vector<vector<std::pair<int, int> > >& m_ind){
    //K_largest problem for temporal localization
    //remember to set max_sum to negative infinity
    Dtype s_first,s_middle,s_end;

    vector<Dtype> r_sum(num+1, Dtype(-DBL_MAX));
    vector<Dtype> r_min(k,Dtype(DBL_MAX));
    vector<Dtype> cand_k(k,0);
    vector<int>   min_end(k,0);
    bool ctype=dim>30;

    //  one-channel distribution

    // three-channels distribution
    vector<Dtype>r_max(k,Dtype(-DBL_MAX));
    vector<int> max_start(k,0);

    // one channel distribution
    if (!ctype) {
        for (int i = 1; i < dim; ++i){
            // re-initialization
            for (int m=0;m<k;m++){
                r_min[m]=Dtype(DBL_MAX);
                min_end[m]=0;
            }
            r_min[0]=0;
            r_sum[0]=0;
            for (int j=1; j <= num; ++j){
                r_sum[j]=r_sum[j-1]+array[i+(j-1)*dim];
                int p=0;
                for (int l=0;l<k;l++){

                    cand_k[l]=r_sum[j]-r_min[l];
                    // one by one merge
                    while ((p<k) && (cand_k[l]<=max_sum[i-1][p])) p++;
                    if (p==k)
                        break;
                    for (int q=k-1;q>p;q--){
                        max_sum[i-1][q]=max_sum[i-1][q-1];
                        m_ind[i-1][q]=m_ind[i-1][q-1];
                    }
                    max_sum[i-1][p]=cand_k[l];
                    m_ind[i-1][p].first=min_end[l]+1;
                    m_ind[i-1][p].second=j;
                }
                // insert
                for  (int l=0;l<k;l++){
                    if (r_sum[j]<r_min[l]){
                        for (int n=k-1;n>l;--n){
                            r_min[n]=r_min[n-1];
                            min_end[n]=min_end[n-1];
                        }
                        r_min[l]=r_sum[j];
                        min_end[l]=j;
                        break;
                    }
                    else continue;
                }
            }
        }
    }
    // three-channels distribution
    else {
        for (int i = 1; i < dim; i=i+3) {
        	int temp=(i-1)/3;
            for (int m=0;m<k;m++)
                r_max[m]=Dtype(-DBL_MAX);
            for (int j=0; j < num; ++j){
                //insert it to max_sum

                if (j==0){
                    //insert it to r_max then continue
                    r_max[0]=array[i];
                    max_start[0]=0;
                }
                else{
                    // insert it to max_sum
                    int p=0;

                    s_first=array[j*dim+i];
                    s_middle=array[j*dim+i+1];
                    s_end=array[j*dim+i+2];
                    for (int l=0;l<k;l++){
                        // one by one merge
                        if (r_max[l]==Dtype(-DBL_MAX))
                            break;
                        cand_k[l]=r_max[l]+s_end;
                        while ((p<k) && (cand_k[l]<=max_sum[temp][p])) p++;
                        if (p==k)
                            break;
                        for (int q=k-1;q>p;q--){
                            max_sum[temp][q]=max_sum[temp][q-1];
                            m_ind[temp][q]=m_ind[temp][q-1];
                        }
                        max_sum[temp][p]=cand_k[l];
                        m_ind[temp][p].first=max_start[l];
                        m_ind[temp][p].second=j;
                    }
                    //max_sum[i].resize(k);
                    // insert it to r_max
                    for (int l=0;l<k;l++)
                        if (r_max[l]!=Dtype(-DBL_MAX))
                            r_max[l]=r_max[l]+s_middle;
                        else
                            break;
                    //cand_k[l]=
                    //insert s_first
                    for  (int l=0;l<k;l++){
                        if (s_first>r_max[l]){
                            for (int n=k-1;n>l;--n){
                                r_max[n]=r_max[n-1];
                                max_start[n]=max_start[n-1];
                            }
                            r_max[l]=s_first;
                            max_start[l]=j;
                            break;
                        }
                        else continue;
                    }
                }

            }
        }
    }
    max_start.clear();
    r_max.clear();
    min_end.clear();
    r_min.clear();
    cand_k.clear();
}

void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs == 4 && nlhs==2, "Usage: [max_sum, max_ind]=inference(array,num,dim,k");
  // Handle input command
  vector<vector<float> > max_sum;
  vector<vector<std::pair<int, int> > > m_ind;
  int num=mxGetScalar(prhs[1]);
  int dim=mxGetScalar(prhs[2]);
  int k  =mxGetScalar(prhs[3]);
  float* array=(float*)mxGetData(prhs[0]);
  int temp;
  bool ctype=dim>30;
	if (!ctype) {
	    temp=dim-1;
	}
	else{
		temp=(dim-1)/3;
	}
   max_sum.resize(temp);
   m_ind.resize(temp);
   for (int i=0;i<temp;i++){
	 max_sum[i].assign(k,-DBL_MAX);
	 m_ind[i].resize(k);
   }
   Inference<float>(array,num,dim,k,max_sum,m_ind);
   plhs[0]=singlevec_to_mx_vec(max_sum);
   plhs[1]=structvec_to_mx_vec(m_ind);
   for (int i=0;i<temp;i++){
	 max_sum[i].clear();
	 m_ind[i].clear();
   }
   max_sum.clear();
   m_ind.clear();
}


