function [weights]=initialize(input_size,output_size)

b=sqrt(6)/sqrt(input_size+output_size);
rng(1e3);
weights=unifrnd(-1*b,b,input_size,output_size);

end