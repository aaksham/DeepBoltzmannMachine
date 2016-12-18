function [J]=test_wb(X,W1,W2,c,b,a)
m=size(X,1);
J=0;
H2=size(W2,2);
sampling_times=2;
for t=1:m
    v=X(t,:);
    h2=binornd(1,0.5,1,H2);
    %for sampling=1:sampling_times
        p_h1=sigmoid(v*W1+h2*W2'+b);
        h1=binornd(1,p_h1);
        p_h2=sigmoid(h1*W2+a);
        h2=binornd(1,p_h2);
    %end
    p_v=sigmoid(h1*W1'+c);
    v_sampled=binornd(1,p_v);
    cross_entropy=v*log(p_v)'+(1-v)*log(1-p_v)';
    J=J-cross_entropy;
end
J=J/m;   