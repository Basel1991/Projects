function [CSF_Prob,WM_Prob,GM_Prob,SegmentedImage]=EM_Probability(image_MRI,image_GT)

    %% Set the number of clusters and Stopping Threshold
    clusterNumbers=3;
    StoppingThreshold=0.05;
    image_MRI = rescale_img(image_MRI,255);
    
    % assuming good initialization by using the image_GT
    max_iter = 75;
    
    Image3D=reshape(image_MRI,size(image_MRI,1)*size(image_MRI,2)*size(image_MRI,3),1);
    GT3D=reshape(image_GT,size(image_GT,1)*size(image_GT,2)*size(image_GT,3),1);

    IndexCFS=find(GT3D==1);
    IndexWM=find(GT3D==2);
    IndexGM=find(GT3D==3);

    Index3D=cat(1,IndexCFS,IndexWM,IndexGM);

    Data_CSF=Image3D(IndexCFS);
    Data_WM=Image3D(IndexWM);
    Data_GM=Image3D(IndexGM);

    DataCell={Data_CSF,Data_WM,Data_GM};

    Data3D=cat(1,Data_CSF,Data_WM,Data_GM);

    for i=1:1:clusterNumbers
       mean_GMM(i)=(1/length(DataCell{i})).*(sum(DataCell{i}));
       proportion_GMM(i)=(length(DataCell{i}))/length(Data3D(:,1));
       ClusterData=DataCell{i};
       meanData=mean(ClusterData); 
       Standard_Dev(i)=sqrt((sum((ClusterData-meanData).^2))./length(ClusterData(:,1)));
    end

    %% Expectation Maximization Algorithm
    Iterations=1;
    disp('---------------------Processing--------------------')

    while(Iterations < max_iter)
    %     Expectation which is evaluating the responsibilities using the current parameter values
        GM=Gaussian_Mixture(Data3D,mean_GMM,Standard_Dev,proportion_GMM,clusterNumbers);
        sum_all_Cluster = sum(GM,2)+eps;
        loglikelihood_Current=sum(log(sum_all_Cluster));
        latentVariable=GM./sum_all_Cluster; %Posterior Probability

        % Maximization which is re-estimate the parameters using the current responsibilities
        for cluster=1:1:clusterNumbers
            proportion_GMM(cluster)=sum(latentVariable(:,cluster));
            mean_GMM(cluster)=(sum(latentVariable(:,cluster).*Data3D))./(sum(latentVariable(:,cluster)));     
            temp_1 = bsxfun (@ minus, Data3D,mean_GMM(cluster));
            temp=latentVariable(:,cluster).*temp_1;
            Standard_Dev(i) = (1/proportion_GMM(cluster))* (temp_1' * temp) ;
            Standard_Dev(i)=sqrt(Standard_Dev(i));
        end

        % Stopping Criterion fixation
        GM=Gaussian_Mixture(Data3D,mean_GMM,Standard_Dev,proportion_GMM,clusterNumbers);
        sum_all_Cluster = sum(GM,2);
        loglikelihood_Updated=sum(log(sum_all_Cluster));

        difference_loglikelihood=loglikelihood_Updated-loglikelihood_Current;

        disp(['Error--> ','Iteration = ',num2str(Iterations),' --> ',num2str(difference_loglikelihood)]); % Display difference_loglikelihood for each Iteration

        if(abs(difference_loglikelihood) < StoppingThreshold) 
            break; 
        end

        Iterations=Iterations+1;
    end
    disp('-------------------Process DONE!!!--------------------')
%% Getting Prob. for each Tissue types.
[~,PixelClassification]=max(latentVariable,[],2);

Segmentation=zeros(size(Image3D,1),1);
for i=1:1:size(Index3D,1)
    Segmentation(Index3D(i))=PixelClassification(i);
end

SegmentedImage=reshape(Segmentation,size(image_MRI));

CSF=zeros(size(Image3D,1),1);
WM=zeros(size(Image3D,1),1);
GM=zeros(size(Image3D,1),1);

for i=1:1:size(Index3D,1)
    CSF(Index3D(i))=latentVariable(i,1);
    WM(Index3D(i))=latentVariable(i,2);
    GM(Index3D(i))=latentVariable(i,3);
end
CSF_Prob=reshape(CSF,size(image_MRI));
WM_Prob=reshape(WM,size(image_MRI));
GM_Prob=reshape(GM,size(image_MRI));
end


%% Multi-variate Gaussian Mixture PDF Function
function GMM=Gaussian_Mixture(Data_Vector_3D,mean_GMM,Standard_Dev,proportion_GMM,HowManyCluster)
for i=1:1:HowManyCluster
    GMM(:,i) = proportion_GMM(i).*normpdf(Data_Vector_3D,mean_GMM(i),Standard_Dev(i));
end
end