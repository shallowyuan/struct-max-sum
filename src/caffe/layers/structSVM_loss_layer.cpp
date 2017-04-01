/*
 * structSVM_loss_layer.cpp
 *
 *  Created on: Feb 23, 2016
 *      Author: zehuany
 */
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void StructSVMLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      CHECK_EQ(bottom[0]->num(), bottom[1]->num())
          << "The data and label should have the same number.";
      vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
      top[0]->Reshape(loss_shape);
      int num=bottom[0]->num();
      int dim=bottom[0]->count()/num;
      loss_t.Reshape(bottom[0]->shape());
      r_sum.assign(num+1, Dtype(-DBL_MAX));
      hyper=this->layer_param_.structsvm_loss_param().chyper();
      normalize=this->layer_param_.structsvm_loss_param().normalize();
      lamda2=this->layer_param_.structsvm_loss_param().mweight();
      // resize 2 for hard loss. Hard loss has been deprecated but maintain orginal code here.
      switch (this->layer_param_.structsvm_loss_param().losstype()){
          case StructSVMLossParameter_LossType_SETDIFF:
              m_begin.resize(dim-1);
              m_end.resize(dim-1);
              max_sum.resize(dim-1);
              for (int i=0;i<dim-1;i++){
                   max_sum[i].assign(2,Dtype(-DBL_MAX));
                   m_end[i].resize(2);
                   m_begin[i].resize(2);
              }
              break;
          case StructSVMLossParameter_LossType_OVERLAP:
             // max_sum.assign(dim-1,Dtype(-DBL_MAX));
              m_begin.resize(dim-1);
              m_end.resize(dim-1);
                  clsloss.resize(dim-1);
                  cal_gradient.resize(dim-1);
                  max_sum.resize(dim-1);
              for (int i=0;i<dim-1;i++){
                   max_sum[i].assign(2,Dtype(-DBL_MAX));
                   m_end[i].resize(2);
                   m_begin[i].resize(2);
              }
              ctype=false;
              break;
          case StructSVMLossParameter_LossType_OVERLAP3CHANNEL:
             // max_sum.assign((dim-1)/3,Dtype(-DBL_MAX));
                max_sum.resize((dim-1)/3);
              m_begin.resize((dim-1)/3);
              m_end.resize((dim-1)/3);
              clsloss.resize((dim-1)/3);
              cal_gradient.resize((dim-1)/3);
              for (int i=0;i<(dim-1)/3;i++){
                   max_sum[i].assign(2,Dtype(-DBL_MAX));
                   m_end[i].resize(2);
                   m_begin[i].resize(2);
              }
              ctype=true;
              break;
          default:
          LOG(FATAL) << "Unknown Losstype";
      }
}


template <typename Dtype>
void StructSVMLossLayer<Dtype>::Inference_augmented(const Dtype* array, const Dtype* label, int num, int dim){
    Dtype s_first,s_middle,s_end;
    Dtype s_s;
    bool convert;
    Dtype t_length;
    Dtype lamda=1;
    switch (this->layer_param_.structsvm_loss_param().losstype()){
    case StructSVMLossParameter_LossType_SETDIFF:
        //........................|y_i|-|y_i\cap\y_i^*|.....................................
        for (int i = 1; i < dim; ++i){
            s_middle =array[i]+ (label[0]==i?lamda:Dtype(0));
            max_sum[i-1][0]=s_middle;
            r_sum[0]=s_middle;
            r_begin=0;
            m_begin[i-1][0]=0;
            m_end[i-1][0]=0;
            for (int j=1; j< num; ++j){
                s_middle=array[i+j*dim]+ (label[j]==i?lamda:Dtype(0));
                if (r_sum[j-1]>0){
                    r_sum[j]=r_sum[j-1]+s_middle;
                }
                else{
                    r_sum[j]=s_middle;
                    r_begin=j;
                }
                if (r_sum[j]>max_sum[i-1][0]){
                    max_sum[i-1][0]=r_sum[j];
                    m_begin[i-1][0]=r_begin;
                    m_end[i-1][0]  =j;
                }
            }
        }
        break;
    case StructSVMLossParameter_LossType_OVERLAP:
        for (int i=0;i<dim-1;i++){
            max_sum[i].assign(2,Dtype(-DBL_MAX));
        }
        //..........................|y_i\cup y_i^*|-|y_i\cap y_i^*|
        for (int i = 1; i < dim; ++i){
            s_middle=array[i]+ (label[0]!=i && slabel==i?lamda:Dtype(0));
            max_sum[i-1][0]=s_middle;
            r_sum[0]=s_middle;
            r_begin=0;
            m_begin[i-1][0]=0;
            m_end[i-1][0]=0;
            s_s=0;
            convert=false;
            for (int j=1; j< num; ++j){
                s_middle=array[i+j*dim]+ (label[j]!=i && slabel==i?lamda:Dtype(0));
                if (r_sum[j-1]>s_s){
                    r_sum[j]=r_sum[j-1]+s_middle;
                }
                else{
                    r_sum[j]=s_middle+s_s;
                    r_begin=j;
                }
                //t_length = label[j]!=i?0:std::min(end-start+1,std::max(0,end-j));
                t_length = slabel!=i?0:std::min(end-start+1,std::max(0,end-j));
                t_length=t_length*lamda;
                if (r_sum[j]+t_length>max_sum[i-1][0]){
                    max_sum[i-1][0]=r_sum[j]+t_length;
                    m_begin[i-1][0]=r_begin;
                    m_end[i-1][0]  =j;
                }
                if (label[j]==i)
                    s_s+=lamda;
                else if (convert){
                    s_s=0;
                    convert=false;
                }
            }
        }
        break;
    case StructSVMLossParameter_LossType_OVERLAP3CHANNEL:
        //..........................s_first+s_middle*+s_end-----|y_i\cup y_i^*|-|y_i\cap y_i^*|
        for (int i=0;i<(dim-1)/3;i++){
            max_sum[i].assign(2,Dtype(-DBL_MAX));
        }
        for (int i = 1; i < dim; i=i+3){
            int temp=(i-1)/3;
            s_first=array[i]+ (label[0]!=temp+1 && slabel==temp+1?lamda:Dtype(0));
            s_end  =array[i+dim+2]+ (label[0]!=temp+1 && slabel== temp+1?lamda:Dtype(0));
            max_sum[temp][0]=s_first+s_end;
            r_sum[0]=s_first;
            r_begin=0;
            m_begin[temp][0]=0;
            m_end[temp][0]=1;
            s_s=0;
            convert=false;
            for (int j=1; j< num; ++j){
                s_first=array[i+j*dim]+ (label[j]!=temp+1 && slabel== temp+1?lamda:Dtype(0));
                s_end  =array[i+j*dim+2]+ (label[j]!=temp+1 && slabel== temp+1?lamda:Dtype(0));
                s_middle =array[i+j*dim+1]*lamda2+ (label[j]!=temp+1 && slabel== temp+1?lamda:Dtype(0));
                t_length = slabel!=temp+1?0:std::min(end-start+1,std::max(0,end-j));
                t_length=t_length*lamda;
                if (r_sum[j-1]+s_middle>s_s+s_first){
                    r_sum[j]=r_sum[j-1]+s_middle;
                }
                else{
                    r_sum[j]=s_first+s_s;
                    r_begin=j;
                }
                if (r_sum[j-1]+s_end+t_length>max_sum[temp][0]){
                    max_sum[temp][0]=r_sum[j-1]+s_end+t_length;
                    m_begin[temp][0]=r_begin;
                    m_end[temp][0]  =j;
                }
                if (label[j]==temp+1)
                    s_s+=lamda;
                else if (convert){
                    s_s=0;
                    convert=false;
                }
            }
        }
        break;
            default:
                LOG(FATAL) << "Unknown Losstype";
    }
}
template <typename Dtype>
void StructSVMLossLayer<Dtype>::Inference (const Dtype* array, const Dtype * label,  int num, int dim, int k){
    //K_largest problem for temporal localization
    Dtype s_first,s_middle,s_end;
    vector<Dtype> r_min(k,Dtype(DBL_MAX));
    vector<Dtype> cand_k(k,0);
    vector<int>   min_end(k,0);
    vector<Dtype>r_max(k,Dtype(-DBL_MAX));
    vector<int> max_start(k,0);

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
                        m_begin[i-1][q]=m_begin[i-1][q-1];
                        m_end[i-1][q]=m_end[i-1][q-1];

                    }
                    max_sum[i-1][p]=cand_k[l];
                    m_begin[i-1][p]=min_end[l];
                    m_end[i-1][p]=j-1;
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
                            m_begin[temp][q]=m_begin[temp][q-1];
                            m_end[temp][q]=m_end[temp][q-1];

                        }
                        max_sum[temp][p]=cand_k[l];
                        m_begin[temp][p]=max_start[l];
                        m_end[temp][p]=j;
                    }
                    // insert it to r_max
                    for (int l=0;l<k;l++)
                        if (r_max[l]!=Dtype(-DBL_MAX))
                            r_max[l]=r_max[l]+s_middle;
                        else
                            break;
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

template <typename Dtype>
void StructSVMLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype lamda=1;

    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    int sdim;
    Dtype sloss=0,sg = 0;
    static Dtype iouloss=0;
    static int iter_=0;
    static vector<Dtype> losses;
    static vector<Dtype> IoUs;

    // ----------------------calculate the end and begin ------------------------------
    start=num;
    end=-1;
    slabel=0;

    bool mask=false;
    for (int i=0; i < num; i++){
        if (label[i]!=0){
            if (!mask){
                mask=true;
                start=i;
            }
            slabel=label[i];
        }
        else if (mask){
            mask=false;
            end=i-1;
        }
    }
    if (mask)
        end=num-1;

    //----------------------find maximum box for each label-----------------------------
    Inference_augmented(bottom_data, label, num, dim);

    caffe_set(count, Dtype(0), bottom_diff);

    // estimated y, sdim should start with 0
    CHECK(slabel>0) << "background videos are not allowed";
    sdim=slabel-1;

    Dtype plength,wlength,ratio;
    switch(this->layer_param_.structsvm_loss_param().losstype()){

        case StructSVMLossParameter_LossType_SETDIFF:
            // not implemented
            break;
        // one channel structsvm
        case StructSVMLossParameter_LossType_OVERLAP:
            plength=m_end[sdim][0]-m_begin[sdim][0]+1;
            wlength=end-start+1;
            CHECK(wlength==8); // should be commentted afterwards
            ratio=wlength/plength;

            for (int i=0; i < num; i++){
                  if (label[i]!=0)
                     sg = sg + bottom_data[i*dim + static_cast<int>(label[i])];
            }

            sloss=(max_sum[sdim][0]-sg)/wlength;
            if (normalize){
                sloss=0;
                for (int i=0;i<num;i++)
                    if ((i>=m_begin[sdim][0])&&(i<=m_end[sdim][0])&&(i>=start)&&(i<=end))
                        sloss +=0;
                    else if (i>=m_begin[sdim][0]&&i<=m_end[sdim][0])
                        sloss +=bottom_data[i*dim + slabel]/plength;
                    else if (i>=start&&i<=end)
                        sloss -=bottom_data[i*dim + static_cast<int>(label[i])]/wlength;

            }

            //propogaete sloss(change after normalize)
            for (int i=0; i < num; i++){
                if (label[i]!=0)
                      bottom_diff[i*dim + static_cast<int>(label[i])]=Dtype(1);
            }
            for (int i=m_begin[sdim][0];i<=m_end[sdim][0];i++)
                bottom_diff[i*dim + sdim+1] = normalize?ratio:Dtype(1);

            // add classfing loss
            if (!this->layer_param_.structsvm_loss_param().only_locate()){
                for (int i=0;i<dim-1;i++){
                    cal_gradient[i] = false;
                    clsloss[i] = 0;
                    plength=m_end[i][0]-m_begin[i][0]+1;
                    if (i!=sdim){
                        clsloss[i] = std::max(lamda*wlength+max_sum[i][0]-sg,Dtype(0))/wlength;
                        cal_gradient[i]=clsloss[i]>0?true:false;
                        if (normalize && cal_gradient[i]){
                            ratio=wlength/plength;
                            clsloss[i] = ((lamda*wlength+max_sum[i][0])*ratio-sg)/wlength;
                        }
                    }
                }
            }
            break;
        case StructSVMLossParameter_LossType_OVERLAP3CHANNEL:
            CHECK(start<end)<<"end should be bigger than start";
            plength=std::max(1,m_end[sdim][0]-m_begin[sdim][0]-1);
            wlength=std::max(1,end-start-1);
            ratio =wlength/plength;
            CHECK(wlength==6)<<start<<" "<<end;

            for (int i=0; i < num; i++)
                if (label[i]!=0)
                   sg = sg + bottom_data[i*dim + static_cast<int>((label[i]-1)*3+2)];
            sg=sg-bottom_data[start*dim + static_cast<int>((label[start]-1)*3+2)]+
              bottom_data[start*dim + static_cast<int>((label[start]-1)*3+1)]+
              bottom_data[end*dim + static_cast<int>((label[end]-1)*3+3)]-
              bottom_data[end*dim + static_cast<int>((label[end]-1)*3+2)];

            // calculate sloss
            sloss=std::max (Dtype(0),max_sum[sdim][0]-sg)/(end-start+1);
            if (normalize){
                sloss=0;
                for (int i=0;i<num;i++)
                    if ((i>m_begin[sdim][0])&&(i<m_end[sdim][0])&&(i>start)&&(i<end))
                        sloss +=0;
                    else if (i>m_begin[sdim][0]&&i<m_end[sdim][0])
                        sloss +=bottom_data[i*dim + sdim*3+2]/plength;
                    else if (i>start&&i<end)
                        sloss -=bottom_data[i*dim + sdim*3+2]/wlength;
                sloss=lamda2*sloss+bottom_data[m_begin[sdim][0]*dim+sdim*3+1]+bottom_data[m_end[sdim][0]*dim+sdim*3+3]-
                    bottom_data[start*dim+sdim*3+1]-bottom_data[end*dim+sdim*3+3];
                sloss=sloss/Dtype(3);
            }


            //add cls closs
            if (!this->layer_param_.structsvm_loss_param().only_locate()){
                for (int i=0;i<(dim-1)/3;i++){
                    cal_gradient[i] = false;
                    clsloss[i] = 0;
                    plength=std::max(1,m_end[i][0]-m_begin[i][0]-1);
                    Dtype pside=bottom_data[m_begin[i][0]*dim+i*3+1]+bottom_data[m_end[i][0]*dim+i*3+3];
                    Dtype gside= bottom_data[start*dim+sdim*3+1]+bottom_data[end*dim+sdim*3+3];
                    if (i!=sdim){
                        clsloss[i] = std::max(lamda*(end-start+1)+max_sum[i][0]-sg,Dtype(0))/(end-start+1);
                        cal_gradient[i]=clsloss[i]>0?true:false;
                        if (normalize && cal_gradient[i]){
                            clsloss[i] = lamda*3+(max_sum[i][0]-pside)*lamda2/plength-(sg-gside)*lamda2/wlength+pside-gside;
                            clsloss[i] = clsloss[i]/Dtype(3);
                        }
                    }
                }

            }

            for (int i=start+1; i < end; i++)
                if (label[i]!=0)
                    bottom_diff[i*dim + static_cast<int>((label[i]-1)*3+2)]=Dtype(lamda2); // replement afer normalize delete L2 norm


            for (int i = m_begin[sdim][0]+1; i < m_end[sdim][0]; i++)
                  bottom_diff[i*dim+sdim*3+2]=normalize?ratio*lamda2:Dtype(lamda2);

            bottom_diff[start*dim+sdim*3+1]=Dtype(1);
            bottom_diff[end*dim+sdim*3+3]=Dtype(1);

            bottom_diff[m_begin[sdim][0]*dim+sdim*3+1]=Dtype(1);
            bottom_diff[m_end[sdim][0]*dim+sdim*3+3]=Dtype(1);
            dim=(dim-1)/3+1;
            break;
        default:
            LOG(FATAL) << "Unknown Losstype";
    }

    Dtype* loss = top[0]->mutable_cpu_data();

    loss[0] = sloss;

    // add cls loss
    if (!this->layer_param_.structsvm_loss_param().only_locate()){
        for (int i=0;i< dim-1;i++)
            loss[0] += clsloss[i]*hyper/(dim-2);
    }

    /** calulate overlap **/
    int bi=std::max(0,std::min(end,m_end[sdim][0])-std::max(start,m_begin[sdim][0])+1);
    int bu=end-start+m_end[sdim][0]-m_begin[sdim][0]-bi+2;

    Dtype iou=Dtype(bi)/Dtype(bu);

    // smoothed loss  output
    LOG(INFO)<<slabel<<"---iou:"<<iou<<"----loss:"<<loss[0]<<"----gt:"<<sg<<"---pre:"<<max_sum[sdim][0];
    if (IoUs.size() < 100) {
        IoUs.push_back(iou);
        int size = IoUs.size();
        iouloss = (iouloss * (size - 1) + iou) / Dtype(size);
    } else {
        int idx = iter_% 100;
        iouloss += (iou - IoUs[idx]) / 100.0;
        IoUs[idx]=iou;
    }
    if (++iter_%100==0)
      this->ofile<<iouloss<<std::endl;
}

template <typename Dtype>
void StructSVMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      Dtype* loss_temp=loss_t.mutable_cpu_data();
      Dtype wlength=0,plength;
      const Dtype* label = bottom[1]->cpu_data();
      int num = bottom[0]->num();
      int count = bottom[0]->count();
      int dim = count / num;
      int sdim = slabel-1;

      caffe_set(count, Dtype(0), loss_temp);

      switch(this->layer_param_.structsvm_loss_param().losstype()){

          case StructSVMLossParameter_LossType_SETDIFF:
              // nor implement
              break;
          // one channel structsvm
          case StructSVMLossParameter_LossType_OVERLAP:
              wlength=end-start+1;
              for (int i = 0; i < num; ++i) {
                 if ((i>=m_begin[sdim][0])&&(i<=m_end[sdim][0])&&(i>=start)&&(i<=end))
                     bottom_diff[i * dim + sdim +1] = 0;
                 else
                     bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
              }

              if (!this->layer_param_.structsvm_loss_param().only_locate()){
                  for (int i=0;i < dim-1; ++i){
                      if (cal_gradient[i]){
                          plength=m_end[i][0]-m_begin[i][0]+1;
                          for (int j=0;j < num; j++){
                              if ((j>=m_begin[i][0])&&(j<=m_end[i][0]))
                                  loss_temp[j * dim +  i+1] += normalize?wlength/plength:Dtype(1);
                              if ((j>=start)&&(j<=end))
                                  loss_temp[j*dim + static_cast<int>(label[j])] += Dtype(-1);
                          }
                      }
                  }
              }
              break;
          case StructSVMLossParameter_LossType_OVERLAP3CHANNEL:
              wlength=std::max(1,end-start-1);
              for (int i = 0; i < num; ++i) {
                  if ((i>m_begin[sdim][0])&&(i<m_end[sdim][0])&&(i>start)&&(i<end))
                      bottom_diff[i * dim + sdim*3 +2] = 0;
                  else if (label[i]!=0){
                      if (i==start)
                          bottom_diff[i * dim + static_cast<int>((label[i]-1)*3 + 1)] *= -1;
                      else if (i==end)
                          bottom_diff[i * dim + static_cast<int>((label[i]-1)*3 + 3)] *= -1;
                      else
                          bottom_diff[i * dim + static_cast<int>((label[i]-1)*3 + 2)] *= -1;
                  }//caffe mul
              }
              if (end == m_end[sdim][0])
                  bottom_diff[end * dim + sdim * 3 + 3]=0;
              if (start == m_begin[sdim][0])
                  bottom_diff[start * dim + sdim * 3 + 1]=0;

              if (!this->layer_param_.structsvm_loss_param().only_locate()){
                  for (int i=0;i < (dim-1)/3; ++i){
                      if (cal_gradient[i]){
                          plength=std::max(1,m_end[i][0]-m_begin[i][0]-1);
                          for (int j=0;j < num; j++){
                              if ((j>m_begin[i][0])&&(j<m_end[i][0]))
                                  loss_temp[j * dim +  i * 3 + 2] += normalize?wlength*lamda2/plength:Dtype(lamda2);
                              if ((j>start)&&(j<end))
                                  loss_temp[j*dim + sdim * 3 + 2] += Dtype(-lamda2);

                          }

                          loss_temp[start*dim+sdim*3+1] += normalize?Dtype(-wlength):Dtype(-1);
                          loss_temp[end*dim+sdim*3+3] += normalize?Dtype(-wlength):Dtype(-1);
                          loss_temp[m_begin[i][0]*dim+i*3+1] += normalize?Dtype(wlength):Dtype(1);
                          loss_temp[m_end[i][0]*dim+i*3+3] += normalize?Dtype(wlength):Dtype(1);
                      }
                  }
              }
              wlength=normalize?wlength*3:(end-start+1);
              dim=(dim-1)/3+1;
              break;
          default:
              LOG(FATAL) << "Unknown LossType";
      }

      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(count, loss_weight/wlength, bottom_diff);
      if (!this->layer_param_.structsvm_loss_param().only_locate()){
          caffe_scal(count,loss_weight*hyper/(wlength*(dim-2)),loss_temp);
          caffe_add(count,bottom_diff,loss_temp,bottom_diff);
      }
    }
}

INSTANTIATE_CLASS(StructSVMLossLayer);
REGISTER_LAYER_CLASS(StructSVMLoss);

}  // namespace caffe




