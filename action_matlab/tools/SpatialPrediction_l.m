function  output=SpatialPrediction_l(vid_name, net, NUM_HEIGHT, NUM_WIDTH, cursor, duration,rate)

%% inference for rgb frames input,the input should be cursor of leveldb or lmdb
%% return output with dimensionxframenumber



d = load('VGG_mean.mat');
IMAGE_MEAN = d.image_mean;
IMAGE_MEAN = imresize(IMAGE_MEAN,[224,224]);


batch_size = 16;
video = zeros(224,224, 3, batch_size*10,'single');
num_images = duration;
num_batches = ceil(num_images/(batch_size*rate));
disp(duration);
output=[];
assert(cursor.find(sprintf('%s_%08d.jpg',[vid_name,'/img'],1)));
fprintf('%s\n',cursor.key);
for bb = 1 : num_batches
    start = 1 + (bb-1)*batch_size;
    tic
    for i = start : min((duration-1)/rate+1,start+batch_size-1)
        keystr=sprintf('%s_%08d.jpg',[vid_name,'/img'],(i-1)*rate+1);
        assert(strcmp(keystr,cursor.key));
        [tmp,~]=net.fromDatum(cursor.value);

        tmp=imresize(tmp, [NUM_HEIGHT, NUM_WIDTH], 'bilinear');

    	rgb_flip(:,:,:) = tmp(:,end:-1:1,:);

    	rgb_1 = tmp(1:224,1:224,:,:);
    	rgb_2 = tmp(1:224,end-223:end,:,:);
    	rgb_3 = tmp(8:8+223,50:50+223,:,:);
    	rgb_4 = tmp(end-223:end,1:224,:,:);
    	rgb_5 = tmp(end-223:end,end-223:end,:,:);
    	rgb_f_1 = rgb_flip(1:224,1:224,:,:);
    	rgb_f_2 = rgb_flip(1:224,end-223:end,:,:);
    	rgb_f_3 = rgb_flip(8:8+223,50:50+223,:,:);
    	rgb_f_4 = rgb_flip(end-223:end,1:224,:,:);
    	rgb_f_5 = rgb_flip(end-223:end,end-223:end,:,:);

    	rgb = cat(4,rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5);
        video(:,:,:,(i-start)*10+1:(i-start)*10+10) = rgb;
        ty=rate;
        while ty>0 && cursor.next()
            ty=ty-1;
        end
    end
    toc
    range=start : min(duration/rate,start+batch_size-1);
    
    video = video(:,:,[3,2,1],:);
    video = bsxfun(@minus,video,IMAGE_MEAN);
    images = permute(video,[2,1,3,4]);
    tic
    tput=net.forward({images});
    toc
    tput = squeeze(tput{1});
    %%---------------------------------------------
    tput=reshape(tput,size(tput,1),10,[]);
    tput=squeeze(max(tput,[],2));
    %%--------------------------------------------
    if isempty(output)
        output = zeros(size(tput,1), floor(num_images/rate), 'single');
    end
    
    output(:,range) = tput(:,mod(range-1,batch_size)+1);
end
end
