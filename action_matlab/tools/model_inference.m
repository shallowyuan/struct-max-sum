function model_inference(model_file, model_def_file, type)

%% this is inference only for entire test set without any filter

addpath(genpath('../../matlab/'));
addpath(genpath('../'));
rdbname='../../data/thumos/testbase';
%fdbname='/scratch/jiadeng_fluxg/zehuany/t_flow_thumos_test';
outdir='testmat';
modeldir = '../../data/models/';
mdefdir= '../../models/action_detection/';
cname='test_configure.txt';


database1 = leveldb.DB(rdbname);
rcursor=database1.cursor('RDONLY', true);
assert(rcursor.first());
sizes_vid = [240,320];
load('groundtruth.mat');
rate=6;
display('Begin perform inference..');
gpu_id = 0;

model_file = [modeldir  model_file]
model_def_file = [mdefdir  model_def_file]

[~,vnames,~, durations]=textread(cname,'%d %s %f %d',-1);
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net1 = caffe.Net(model_def_file, model_file, 'test');

count=1;
for i=1:length(vnames)
    vid_name = vnames{i};
    if ~exist([outdir vid_name num2str(type) '.mat'],'file')
    fprintf('RGB Inference for %s------------\n',vid_name);
    switch type
        case 0
            output=SpatialPrediction_l(['thumos14_toptical/' vid_name], net1, sizes_vid(1), sizes_vid(2), rcursor, durations(i), rate);
        case 1
            output=TemporalPrediction_l(['thumos14_toptical/' vid_name], net1, sizes_vid(1), sizes_vid(2), rcursor, durations(i), rate);
        otherwise
            warning('No file are generated.')
        save([outdir vid_name num2str(type) '.mat'],'output');
    end
end
return output
