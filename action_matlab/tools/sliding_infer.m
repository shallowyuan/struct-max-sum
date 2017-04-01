
addpath(genpath('../'));
cname='test_configure.txt';
K=100;
rate=6;
type=0;
display('Begin perform inference..');


load('test_set_meta.mat');
[~, vnames,~,~]=textread(cname,'%d %s %f %d',-1);


is=ismember({test_videos.video_name},vnames);
frates=[test_videos(is).frame_rate_FPS];
frates=frates/rate;
load('groundtruth.mat');
load('prior.mat','prior');

rname='result.txt';
fid1=fopen(rname,'w');
count=1;
snames=vnames;
for i=1:length(vnames)
    vid_name = vnames{i};
    fprintf('Inference for video %d %s------------\n',i,vid_name);
    load(['output/' vid_name num2str(type) '.mat'],'output');
    sum_r=[];
    ind_r=[];
    tic
    for m=1:10:size(output,2)-100
        [rsum, rind]=inference(output(:,m:m+100),100,size(output,1),K);
        sum_r=[sum_r rsum];
        ind_r=[ind_r rind];
    end
    toc

    count=1;
    result1=[];
    for j=1:size(sum_r,2)
        for c=1:size(sum_r,1)
            interval=ind_r(c,j).end/frates(i)-ind_r(c,j).start/frates(i);
            pscore=sum_r(c,j);
            if interval<0.3
                continue;
            end
            result1(count).vname=vnames{i};
            result1(count).start=single(ind_r(c,j).start/frates(i));
            result1(count).end=single(ind_r(c,j).end/frates(i));
            result1(count).label=th14classids(c);
            % Apply the prior of durations for each actions
            result1(count).score=pscore/(ind_r(c,j).end-ind_r(c,j).start+1);
            result1(count).score =result1(count).score*normpdf(interval,prior(c).mean,prior(c).sdev);
            count=count+1;
        end
    end
    if count==1
        continue;
    end
    result1=nms_detection(result1,0.6);
    result1=result1([result1.score]>0);
    for c=1:length(result1)
       fprintf(fid1,'%s %.1f %.1f %d %f\n',result1(c).vname,result1(c).start,result1(c).end, result1(c).label,result1(c).score);
    end
end
fprintf('begin to write...\n');
fclose(fid1);
