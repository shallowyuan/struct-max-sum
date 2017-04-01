function output=TemporalPrediction_l(vid_name, net, NUM_HEIGHT, NUM_WIDTH, cursor, num_images, rate)

%% inference for rgb frames input,the input should be cursor of leveldb or lmdb
%% return output with dimensionxframenumber

L = 10;
d  = load('flow_mean.mat');
FLOW_MEAN = d.image_mean;
FLOW_MEAN = imresize(FLOW_MEAN,[224,224]);

batch_size = 8;
num_batches = ceil(num_images/(batch_size*rate));
video = zeros(NUM_HEIGHT,NUM_WIDTH,L*2,batch_size);
flow_flip = zeros(NUM_HEIGHT,NUM_WIDTH,L*2,batch_size);
output=[];
images = zeros(224, 224, L*2, batch_size*10, 'single');
assert(cursor.find(sprintf('%s_%08d.jpg',[vid_name,'/flow'],1)));
for bb = 1 : num_batches
    start= 1 + batch_size*(bb-1);
    range = start: min(num_images/rate,batch_size*bb);
    for i = start: min(num_images/rate,batch_size*bb)
        count=1;
        for ii=(i-1)*rate+1:1:min(num_images,(i-1)*rate+10)
            if ii~=(i-1)*rate+1
                cursor.next();
            end
            keystr=sprintf('%s_%08d.jpg',[vid_name,'/flow'],ii);
            assert(strcmp(keystr,cursor.key));
            [flow,~]=net.fromDatum(cursor.value);
            video(:,:,count:count+1,i-start+1) = imresize(flow,[NUM_HEIGHT,NUM_WIDTH],'bilinear');
            flow_flip(:,:,count,i-start+1) = 255- video(:,end:-1:1,count,i-start+1);
            flow_flip(:,:,count+1,i-start+1) = video(:,end:-1:1,count+1,i-start+1);
            count=count+2;
        end
        for ii=count:2:19
            video(:,:,ii:ii+1,i-start+1)=video(:,:,count-2:count-1,i-start+1);
            flow_flip(:,:,ii:ii+1,i-start+1)=flow_flip(:,:,count-2:count-1,i-start+1);
        end
        while count>3
            cursor.previous();
            count=count-2;
        end
        %video(:,:,2,i-start+1) = imresize(flow_y,[NUM_HEIGHT,NUM_WIDTH],'bilinear');
        ty=rate;
        while ty>0 && cursor.next()
            ty=ty-1;
        end
    end
    flow = video(:,:,:,1:length(range));

    flow_1 = flow(1:224,1:224,:,:);
    flow_2 = flow(1:224,end-223:end,:,:);
    flow_3 = flow(ceil(NUM_HEIGHT/2)-112:ceil(NUM_HEIGHT/2)+111,ceil(NUM_WIDTH/2)-112:ceil(NUM_WIDTH/2)+111,:,:);
    flow_4 = flow(end-223:end,1:224,:,:);
    flow_5 = flow(end-223:end,end-223:end,:,:);
    flow_f_1 = flow_flip(1:224,1:224,:,:);
    flow_f_2 = flow_flip(1:224,end-223:end,:,:);
    flow_f_3 = flow_flip(ceil(NUM_HEIGHT/2)-112:ceil(NUM_HEIGHT/2)+111,ceil(NUM_WIDTH/2)-112:ceil(NUM_WIDTH/2)+111,:,:);
    flow_f_4 = flow_flip(end-223:end,1:224,:,:);
    flow_f_5 = flow_flip(end-223:end,end-223:end,:,:);
    tmp=cat(4,flow_1,flow_2,flow_3,flow_4,flow_5,flow_f_1,flow_f_2,flow_f_3,flow_f_4,flow_f_5);

    for ii = 1 : size(tmp,4)
        img = single(tmp(:,:,:,ii));
        images(:,:,:,ii) = permute(img -FLOW_MEAN,[2,1,3]);
    end
    
    tput=net.forward({images});
    tput = squeeze(tput{1});
    if isempty(output)
        output = zeros(size(tput,1), floor(num_images/rate), 'single');
    end
    tput = squeeze(max(reshape(tput,[size(tput,1),size(range,1),10]),[],3))
    
    output(:,range) = tput(:,mod(range-1,batch_size)+1);
end
%output=output';
end

