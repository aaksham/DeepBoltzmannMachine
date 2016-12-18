clear
trainset=load('digitstrain.txt');

valset=load('digitsvalid.txt');
Xval=double(valset(:,1:size(valset,2)-1)>=0.5);

K=100;%batch size
N=K;
max_epochs=500;
early_stopper_limit=5;
learning_rate=0.02;

input_layer_size=784;
D=input_layer_size;
H1=100;
H2=100;
W1=initialize_wts(D,H1);
W2=initialize_wts(H1,H2);
%adding biases
c=zeros(1,D);
b=zeros(1,H1);
a=zeros(1,H2);

v_bar=binornd(1,0.5,K,D);
h1_bar=binornd(1,0.5,K,H1);
h2_bar=binornd(1,0.5,K,H2);

W1final=zeros(size(W1));
W2final=zeros(size(W2));
cfinal=zeros(size(c));
bfinal=zeros(size(b));
afinal=zeros(size(a));

track_train_error=zeros(max_epochs,1);
track_val_error=zeros(max_epochs,1);
prev_J_val=100000;

num_samples=size(trainset,1);
num_batches=num_samples/N;

detail_string=sprintf('wb_epochs_%d_lr_%.3f_esl_%d_hl_%d',max_epochs,learning_rate,early_stopper_limit,H1);

early_stopper=0;
convergence_flag=0;
tic;
for epoch=1:max_epochs
    shuffled_trainset=trainset(randperm(num_samples),:);
    Xtrain=shuffled_trainset(:,1:size(shuffled_trainset,2)-1);
    X=double(Xtrain>=0.5);
    si=1;
    for batch=1:num_batches
        xb=X(si:si+N-1,:);
        si=si+N;
        mu1=unifrnd(0,1,N,H1);
        mu2=unifrnd(0,1,N,H2);
        %E_data
        for t=1:N
            x=xb(t,:);
            mu1_j=mu1(t,:);
            mu2_m=mu2(t,:);
            for i=1:10
                mu1_j=sigmoid(x*W1+mu2_m*W2'+b);
                mu2_m=sigmoid(mu1_j*W2+a);
            end
            mu1(t,:)=mu1_j;
            mu2(t,:)=mu2_m;
        end
        
        %E_model
        
        
        vsampled=zeros(K,D);
        h1sampled=zeros(K,H1);
        h2sampled=zeros(K,H2);
        for t=1:K
            v=v_bar(t,:);
            h1=h1_bar(t,:);
            h2=h2_bar(t,:);
            %for st=1:2
                p_h1=sigmoid(v*W1+h2*W2'+b);
                h1_sampled=binornd(1,p_h1);
                p_h2=sigmoid(h1_sampled*W2+a);
                h2_sampled=binornd(1,p_h2);
                h2=h2_sampled;
            %end
            p_v=sigmoid(h1_sampled*W1'+c);
            v_sampled=binornd(1,p_v);
            
            vsampled(t,:)=v_sampled;
            h1sampled(t,:)=h1_sampled;
            h2sampled(t,:)=h2_sampled;
        end
        v_bar=vsampled;
        h1_bar=h1sampled;
        h2_bar=h2sampled;

        W1=W1+learning_rate*(1/N)*(xb'*mu1-vsampled'*h1sampled);
        W2=W2+learning_rate*(1/N)*(mu1'*mu2-h1sampled'*h2sampled);
        a=a+learning_rate*(1/N)*sum(mu2-h2sampled);
        b=b+learning_rate*(1/N)*sum(mu1-h1sampled);
        c=c+learning_rate*(1/N)*sum(xb-vsampled);
    end
    J=test_wb(X,W1,W2,c,b,a)/784;
    J_val=test_wb(Xval,W1,W2,c,b,a)/784;
    track_train_error(epoch)=J;
    track_val_error(epoch)=J_val;
    if (mod(epoch,50)==0)
        epoch
    end
    if prev_J_val<J_val
       if early_stopper==0
           W1final=W1;
           W2final=W2;
           cfinal=c;
           bfinal=b;
           afinal=a;
           thetas=[cfinal' W1final];
           thetas=[thetas;[0 afinal];[0 bfinal]];
           W2padded=[zeros(size(W2,1),1) W2final];
           thetas=[thetas;W2padded];
           fname=strcat('theta_',detail_string,'.mat');
           save(fname,'thetas');
       end
       if convergence_flag==0
            convergence_flag=1;   
       end
       early_stopper=early_stopper+1;
       if early_stopper>=early_stopper_limit
           break
       end
    else
       if convergence_flag==1
          convergence_flag=0;
          early_stopper=0;
       end
    end
    prev_J_val=J_val;
end
toc
elapsedTime=toc/60 
output=[track_train_error track_val_error];
fname=strcat('dbm_',detail_string,'.csv');  
csvwrite(fname,output);