function pwin=nms_detection(allwindows,overlap)
pwin=[];
labels=[allwindows.label];
ulabels=unique(labels);
for j=1:length(ulabels)
windows=allwindows(labels==ulabels(j));
starts=[windows.start];
ends=[windows.end];
du=ends-starts;
scores=[windows.score];
[~,inds]=sort(scores);
count=1;
picks=[];
while ~isempty(inds)
    i=inds(end);
    picks(count)=i;
    count=count+1;
    tt1=max(starts(inds(1:end-1)),starts(i));
    tt2=min(ends(inds(1:end-1)),ends(i));
    bi=max(0,tt2-tt1);
    iou=bi./(du(inds(1:end-1))+ends(i)-starts(i)-bi);
    inds=inds(iou<overlap);
end
pwin=[pwin windows(picks)];
end
end