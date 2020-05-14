%% Spatiotemporal wave generator in layer-I (square shaped layer)

% clear;
% clc;
% close all
% 
% % Initialize Retinal nodes and their properties
% 
numRetina = 1;
totNeurons_Retina = 1500;
squareLen = 20;
retinaParams_old = {};    % This is a DATA STRUCTURE that holds many parameters

% %{
for i = 1:numRetina
    
    retinaParams_old(i).numNeurons = totNeurons_Retina;
    re = rand(totNeurons_Retina,1);
    
    retinaParams_old(i).x = squareLen*rand(totNeurons_Retina,2);
    
    centroid_RGC = mean(retinaParams_old(i).x);
    
    retinaParams_old(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams_old(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams_old(i).c = [-65+15*re.^2];
    % retinaParams_old(i).d = [8-6*re.^2];
    retinaParams_old(i).d = bsxfun(@minus, 8, gaussian_val);
    
    retinaParams_old(i).D = squareform(pdist(retinaParams_old(i).x));
    D = retinaParams_old(i).D;
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>4).*exp(-D/10); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end
%}

fnum = 1; % What is this for? Seems like dummy variable
retinaParams = retinaParams_old;

%% 
a = [4];
for outerRadius = a

% Controlling wave-size by altering sensor-node connectivity in layer-I!
retinaParams(i).Dk = 5*(D<2)- 2*(D>outerRadius).*exp(-D/10); 
retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));

totTime = 3000; %Total time of simulation

pairWise_allRGC = sum(pdist(retinaParams(i).x));
heatMap_wave = zeros(totNeurons_Retina,1); % # of times each neuron spikes

numActiveNodes_wave = [];
clusterDisp_wave = [];
radius_wave = [];

for t = 1:totTime % simulation of totTime (ms)
    if mod(t,30) == 0 , disp(num2str(t)) , end    
    spikeMat = zeros(totNeurons_Retina,1);
    
    for i = 1:numRetina
        
        retinaParams(i).array_act = []; %Active nodes
        retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; % Noisy input
        
        retinaParams(i).fired = find(retinaParams(i).v >= 30); % Nodes only fire when activity above 30
        fired = retinaParams(i).fired; % INDEX of nodes that have fired
        
        retinaParams(i).array_act = retinaParams(i).x(fired,:); 

        spikeMat(fired + (i-1)*totNeurons_Retina,1) = 1; % not used here?
        
        %Find number of nodes that fired AND cluster contiguity
        pairWise_firingNode = sum(pdist(retinaParams(i).array_act));
        contig_firing = -log(pairWise_firingNode/pairWise_allRGC);
        
        if length(fired)>30 % Why this criteria?
             heatMap_wave(fired) = heatMap_wave(fired)+1;
        end
        
        %Plot traveling wave
        figure(2); scatter(retinaParams(i).x(:,2),retinaParams(i).x(:,1),'k','filled'); hold on
  
        if size(retinaParams(i).array_act,2) ~=0
            scatter(retinaParams(i).array_act(:,2),retinaParams(i).array_act(:,1),[],'r','filled'); axis off; pause(0.3)
        end
        
        fnum = fnum + 1; % What is this for? Seems like dummy variable
        
        retinaParams(i).firings = [retinaParams(i).firings; t+0*fired, fired]; % keeps track nodes that fired at distinct time t
        retinaParams(i).v(fired) = retinaParams(i).c(fired);
        retinaParams(i).u(fired) = retinaParams(i).u(fired) + retinaParams(i).d(fired);
        retinaParams(i).I = retinaParams(i).I + sum(retinaParams(i).Dk(:,fired),2);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        %Why split this explicit sum up in two? 
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).u = retinaParams(i).u + retinaParams(i).a.*(retinaParams(i).b.*retinaParams(i).v - retinaParams(i).u); 
        
        % Keeping track of certain parameters: 
        % (1): Number of active nodes; (2): active nodes contiguity; (3): wave radius 
        
        numActiveNodes_wave(t) = length(fired);
        clusterDisp_wave = [clusterDisp_wave, contig_firing'];
        
        centroid_wave = mean(retinaParams(i).array_act,1);
        
        dist = pdist2(retinaParams(i).array_act, centroid_wave);
        radius_wave(t,:) = [centroid_wave, prctile(dist, 80)];
        
    end
    
end

end

%% PLOTS

% RASTER PLOT!
figure;
for i = 1:numRetina
    plot(retinaParams(i).firings(:,1),retinaParams(i).firings(:,2)+(i-1)*1000,'.')
    hold on
end
xlabel('Time')
ylabel('Neurons')
title('Raster plot')

% Temporal correlation matrix
F = zeros(totNeurons_Retina,max(retinaParams(i).firings(:,1)));
for ff = 1:length(retinaParams(i).firings(:,1))
    F(retinaParams(i).firings(ff,2),retinaParams(i).firings(ff,1)) = 1;
end
figure, spy(F), axis square
Ct = F*F';
figure, h = pcolor(Ct); axis square, set(h,'EdgeColor','none'),set(gca,'Ydir','reverse'),colormap jet;

% HEATMAP WITH TIME (CUMULATIVE SPIKES OVER ENTIRE SIMULATION)
figure;
% heatMap_wave(heatMap_wave == 0) = 0.1;
scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],heatMap_wave(:,end),'filled')
colorbar
title('probability of neuron firing - hotspots of wave')

