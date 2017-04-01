model = './models/bvlc_reference_caffenet/deploy.prototxt';
weights = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
net1  = caffe.Net(model,weights,'train');
layernames=net1.get_layernames();
