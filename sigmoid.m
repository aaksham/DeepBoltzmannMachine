function [s] = sigmoid(a)
s = zeros(size(a));
for i=1:size(a,1),
 for j=1:size(a,2),
  s(i,j)=1/(1+exp(-1*a(i,j)));
 end
 end