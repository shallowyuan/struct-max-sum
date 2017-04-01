data=randn(1000,61);
[sum,ind]=max_brute(single(data'),1000,61,6);

[sum_,ind_]=inference(single(data'),1000,61,6);