    function [image, label] = fromDatum(datum)
	%FROMDATUM Convert datum to image and label.
	%
	% [image, label] = caffe_pb.fromDatum(datum);
	%
	% See also caffe_pb.toDatum caffe_pb.toEncodedDatum
     	[image, label] = caffe_('fromDatum', datum);
<<<<<<< HEAD
	end
=======
	end
>>>>>>> cb1ff1662eb34c2af8a7c4ffd06ef8c4693e1c94
