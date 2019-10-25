% Generating spontaneous activity waves on different geometries

clear
clc;
close all

% Initialize Retinal nodes and their properties

flag = 0;

numRetina = 1;
totNeurons_Retina = 2000;
retinaParams_old = {};

for i = 1:numRetina
    
    retinaParams_old(i).numNeurons = totNeurons_Retina;
    re = rand(totNeurons_Retina,1);
    
    retinaParams_old(i).x = 28*rand(totNeurons_Retina,2);
    
    centroid_RGC = mean(retinaParams_old(i).x);
    dist_center_to_all = bsxfun(@minus, retinaParams_old(i).x, centroid_RGC);
    
    dist_center_to_all = pdist2(retinaParams_old(i).x, centroid_RGC);
    gaussian_val = 6*exp(-(dist_center_to_all)/10);
    
    retinaParams_old(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams_old(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams_old(i).c = [-65+15*re.^2];
    %retinaParams(i).d = [8-6*re.^2];
    retinaParams_old(i).d = bsxfun(@minus, 8, gaussian_val);
    
    retinaParams_old(i).D = squareform(pdist(retinaParams_old(i).x));
    D = retinaParams_old(i).D;
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>10).*exp(-D/10); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end

%% Load arbitrary geometry
iter = 4; % Change the file that loads different boundary geometries (iter -> 1 to 5)
fbdryName = sprintf('Boundary_sheet%d.mat',iter);
load(fbdryName)

% Find all coordinates within boundary
[in, on] = inpolygon(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),bdry_geometrySheet(:,1),bdry_geometrySheet(:,2));
ind_withinGeometry = union(find(in==1),find(on==1));

numRetina = 1;
totNeurons_Retina = length(ind_withinGeometry);
retinaParams = {};

for i = 1:numRetina
    
    retinaParams(i).numNeurons = totNeurons_Retina;
    retinaParams(i).x = retinaParams_old(i).x(ind_withinGeometry,:);
    
    centroid_RGC = mean(retinaParams(i).x);
    dist_center_to_all = bsxfun(@minus, retinaParams(i).x, centroid_RGC);
    
    dist_center_to_all = pdist2(retinaParams(i).x, centroid_RGC);
    gaussian_val = 6*exp(-(dist_center_to_all)/10);
    
    retinaParams(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams(i).c = [-65+15*re.^2];
    retinaParams(i).d = bsxfun(@minus, 8, gaussian_val);
    
    retinaParams(i).D = squareform(pdist(retinaParams(i).x));
    D = retinaParams(i).D;
    retinaParams(i).Dk = 5*(D<2)- 2*(D>10).*exp(-D/10); 
    retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));
    
    retinaParams(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams(i).u = retinaParams(i).b.*retinaParams(i).v;
    retinaParams(i).firings = [];
    
end

%% 
a = [5];
for outerRadius = a

% Altering wave-size!
retinaParams(i).Dk = 5*(D<2)- 2*(D>outerRadius).*exp(-D/10); 
retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));

    
totTime = 2000; %Total time of simulation

pairWise_allRGC = sum(pdist(retinaParams(1).x));
heatMap_wave = zeros(totNeurons_Retina,1); % # of times each neuron spikes

numActiveNodes_wave = [];
clusterDisp_wave = [];
radius_wave = [];

for t = 1:totTime % simulation of totTime (ms)
    
    spikeMat = zeros(totNeurons_Retina,1);
    
    for i = 1:numRetina
        
        retinaParams(i).array_act = []; %Active nodes
        retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; % Noisy input
        
        retinaParams(i).fired = find(retinaParams(i).v >= 30);
        fired = retinaParams(i).fired; 
        
        retinaParams(i).array_act = retinaParams(i).x(fired,:);

        spikeMat(fired + (i-1)*totNeurons_Retina,1) = 1;
        
        %Find number of nodes that fired AND cluster contiguity
        pairWise_firingNode = sum(pdist(retinaParams(i).array_act));
        contig_firing = -log(pairWise_firingNode/pairWise_allRGC);
 
        if length(fired)>30
             heatMap_wave(fired) = heatMap_wave(fired)+1;
             flag = 1;
        
        end
        
        if flag == 1
        
        figure(2);
        hold on
        scatter(retinaParams(i).x(:,2),retinaParams(i).x(:,1),'k','filled')
        scatter(bdry_geometrySheet(:,1),bdry_geometrySheet(:,2),[],[0,0,0]+0.75, 'filled')
        if size(retinaParams(i).array_act,2) ~=0
            scatter(retinaParams(i).array_act(:,2),retinaParams(i).array_act(:,1),[],'r','filled')
            axis off
            pause(0.3)
        end
        axis([0 inf 0 inf])
        %plotName = sprintf('E:/GURU_DATA/RetWave_img3/retWave_t=%d_bdry_%d_outerRad_%d.png',t, iter, outerRadius);
        %saveas(gca, plotName)        
        end
       
        retinaParams(i).firings = [retinaParams(i).firings; t+0*fired, fired];
        retinaParams(i).v(fired) = retinaParams(i).c(fired);
        retinaParams(i).u(fired) = retinaParams(i).u(fired) + retinaParams(i).d(fired);
        retinaParams(i).I = retinaParams(i).I + sum(retinaParams(i).Dk(:,fired),2);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).u = retinaParams(i).u + retinaParams(i).a.*(retinaParams(i).b.*retinaParams(i).v - retinaParams(i).u); 
        
        numActiveNodes_wave(t) = length(fired);
        clusterDisp_wave = [clusterDisp_wave, contig_firing'];
        
        centroid_wave = mean(retinaParams(i).array_act,1);
        
        dist = pdist2(retinaParams(i).array_act, centroid_wave);
        radius_wave(t,:) = [centroid_wave, prctile(dist, 80)];
        
    end
    
end
end


% PLOT HEATMAP WITH TIME (CUMULATIVE SPIKES OVER ENTIRE SIMULATION)

figure;
hold on
scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],heatMap_wave(:,end),'filled')
colorbar
scatter(bdry_geometrySheet(:,1),bdry_geometrySheet(:,2),[],[0 0 0] +0.74,'filled')
axis([0 inf 0 inf])
title('probability of neuron firing - hotspots of wave')
plotName = sprintf('retWave_t=%d_bdry_%d_outerRad_%d.png', t, iter, outerRadius);        
saveas(gca, plotName)
set(gca,'Visible','Off')

