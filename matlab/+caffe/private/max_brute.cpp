/*
 * max_bruteforce.cpp
 *
 *  Created on: Apr 6, 2016
 *      Author: zehuany
 */
#include <vector>
#include<cfloat>
#include <algorithm>
#include "mex.h"

using namespace std;

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

class MAX{
public:
    double max_sum;
    std::pair<int,int> m_ind;
    bool operator<(const MAX &rhs) const { return max_sum < rhs.max_sum; }
};
class MaxGenerator{
public:
    int K;
    vector<MAX> k_max;
    MaxGenerator(int k):K(k){
        k_max.resize(k);
    }
    ~MaxGenerator(){
    	k_max.clear();
    }
    void generate3_(const float* array, const int * label, int num, int dim, int sdim){ //background + 3dimension
        vector<MAX> allsum;
        MAX max;
        double integ[num][num];
        for (int i=0;i<num;i++)
            for (int j=0;j<num;j++)
                integ[i][j]=0;
        for (int i=0;i<num;i++){
            integ[i][i]=array[i*dim+(sdim-1)*3+2];
            for (int j=i+1;j<num;j++)
                integ[i][j]=integ[i][j-1]+array[j*dim+(sdim-1)*3+2];
        }

        for (int i=0;i<num-1;i++){
            for (int j=i+1;j<num;j++){
                max.m_ind.first=i;
                max.m_ind.second=j;
                max.max_sum= array[i*dim+(sdim-1)*3+1]+array[j*dim+(sdim-1)*3+3]+integ[i+1][j-1];
                allsum.push_back(max);
            }
        }
        std::sort(allsum.rbegin(),allsum.rend());
        for (int i=0;i<K;i++)
            k_max[i]=allsum[i];
    }
    void generate3_l(const float* array, const int * label, int num, int dim,int sdim){ //background + 3dimension
        vector<MAX> allsum;
        MAX max;
        double integ[num][num];
        int    integl[num][num];
        int begin=0, end=-1;
        bool coin=false;
  // compute integl
        for (int i=0; i<num;i++)
            if ((!coin)&&(label[i]==sdim)){
                begin=i;
                coin=true;
            }
            else if (coin && (label[i]==0)){
                end=i-1;
                coin=false;
            }
        if (coin)
        	end=num-1;
        for (int i=0;i < num; i++)
            for (int j = i; j < num; j++)
                if ((i>end) || (j<begin))
                    integl[i][j]=end-begin+j-i+2;
                else
                    integl[i][j]=std::abs(begin-i)+std::abs(end-j);
    // compute integ
        for (int i=0;i<num;i++)
            for (int j=0;j<num;j++)
                integ[i][j]=0;
        for (int i=0;i<num-1;i++){
            integ[i][i]=array[i*dim+(sdim-1)*3+2];
            for (int j=i+1;j<num;j++)
                integ[i][j]=integ[i][j-1]+array[j*dim+(sdim-1)*3+2];
        }

        for (int i=0;i<num-1;i++){
            for (int j=i+1;j<num;j++){
                max.m_ind.first=i;
                max.m_ind.second=j;
                max.max_sum= array[i*dim+(sdim-1)*3+1]+array[j*dim+(sdim-1)*3+3]+integ[i+1][j-1]+integl[i][j];
                allsum.push_back(max);
            }
        }
        std::sort(allsum.rbegin(),allsum.rend());
        for (int i=0;i<K;i++)
            k_max[i]=allsum[i];
    }
    void generate1_l(const float* array, const int * label, int num, int dim, int sdim){ //background + 1dimension
        vector<MAX> allsum;
        MAX max;
        double integ[num][num];
        int    integl[num][num];
        int begin=0, end=-1;

        bool coin=false;
        // compute integl
        for (int i=0; i<num;i++)
            if ((!coin)&&(label[i]==sdim)){
                begin=i;
                coin=true;
            }
            else if (coin && (label[i]==0)){
                end=i-1;
                coin=false;
            }
        if (coin)
        	end=num-1;
        for (int i=0;i < num; i++)
            for (int j = i; j < num; j++)
                if ((i>end) || (j<begin)){
                    integl[i][j]=end-begin+j-i+2;
                    //cout<<i<<" "<<j<<" "<<j<<integl[i][j]<<endl;
                }
                else
                    integl[i][j]=std::abs(begin-i)+std::abs(end-j);
        // compute integ
        for (int i=0;i<num;i++)
            for (int j=0;j<num;j++)
                integ[i][j]=0;
        for (int i=0;i<num;i++){
            integ[i][i]=array[i*dim+sdim];
            for (int j=i+1;j<num;j++)
                integ[i][j]=integ[i][j-1]+array[j*dim+sdim];
        }

        for (int i=0;i<num;i++){
            for (int j=i;j<num;j++){
                max.m_ind.first=i;
                max.m_ind.second=j;
                max.max_sum= integ[i][j]+integl[i][j];
                //cout<< integ[i][j]<<" "<<integl[i][j]<<endl;
                allsum.push_back(max);
            }
        }
        std::sort(allsum.rbegin(),allsum.rend());
        for (int i=0;i<K;i++)
            k_max[i]=allsum[i];
    }
    void generate1_(const float* array, const int * label, int num, int dim, int sdim){ //background + 1dimension
            vector<MAX> allsum;
            MAX max;
            double integ[num][num];
            // compute integ
            for (int i=0;i<num;i++)
                for (int j=0;j<num;j++)
                    integ[i][j]=0;
            for (int i=0;i<num;i++){
                integ[i][i]=array[i*dim+sdim];
                for (int j=i+1;j<num;j++)
                    integ[i][j]=integ[i][j-1]+array[j*dim+sdim];
            }

            for (int i=0;i<num;i++){
                for (int j=i;j<num;j++){
                    max.m_ind.first=i;
                    max.m_ind.second=j;
                    max.max_sum= integ[i][j];
                    //cout<< integ[i][j]<<" "<<integl[i][j]<<endl;
                    allsum.push_back(max);
                }
            }
            std::sort(allsum.rbegin(),allsum.rend());
            for (int i=0;i<K;i++)
                k_max[i]=allsum[i];
        }

};


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

void mexFunction(MEX_ARGS) {
  //mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs == 4 && nlhs==2, "Usage: [max_sum, max_ind]=inference(array,num,dim,k");
  // Handle input command
  vector<vector<float> > max_sum;
  vector<vector<std::pair<int, int> > > m_ind;
  int num=mxGetScalar(prhs[1]);
  int dim=mxGetScalar(prhs[2]);
  int k  =mxGetScalar(prhs[3]);
  //mxCHECK(mxIsDouble(prhs[0]),"array should be double matrix");
  //mxCHECK(mxGetN(prhs[0])==num,"array should be double matrix");
  //mxCHECK(mxGetM(prhs[0])==dim,"array should be double matrix");
  float* array=(float*)mxGetData(prhs[0]);
  mexPrintf("num: %d, dim: %d, k: %d \n",mxGetM(prhs[0]),mxGetN(prhs[0]),k);
  int count=0;
  /*
  for (int i=0; i< num; i++)
	  for (int j=0; j< dim; j++)
		  mexPrintf("%f ",array[count++]);
		  */
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
   MaxGenerator allk(k);


       //allk.generate_f(&array[0][0],num,dim);
   for (int i=0;i<temp;i++){
	   if (!ctype)
      allk.generate1_(array,0,num,dim,i+1);
	   else
		   allk.generate3_(array,0,num,dim,i+1);
  	//mexPrintf("%d ",allk.k_max.size());
      for (int j=0;j<k;j++){
    	  max_sum[i].push_back(float(allk.k_max[j].max_sum));
    	  m_ind[i].push_back(allk.k_max[j].m_ind);
    	  //mexPrintf("%f ",allk.k_max[j].max_sum);

      }
   }
   plhs[0]=singlevec_to_mx_vec(max_sum);
   plhs[1]=structvec_to_mx_vec(m_ind);
   for (int i=0;i<temp;i++){
	 max_sum[i].clear();
	 m_ind[i].clear();
   }
   max_sum.clear();
   m_ind.clear();
}



