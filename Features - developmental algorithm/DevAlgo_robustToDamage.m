% Developmental algorithm robust to defective sensor-nodes in the input layer

clear;
clc;
close all

% Initialize Retinal nodes and their properties

numRetina = 1;
totNeurons_Retina = 1500;
squareLength = 28;
retinaParams_old = {};

for i = 1:numRetina
    
    retinaParams_old(i).numNeurons = totNeurons_Retina;
    re = rand(totNeurons_Retina,1);
    
    retinaParams_old(i).x = squareLength*rand(totNeurons_Retina,2);
    
    centroid_RGC = mean(retinaParams_old(i).x);
    dist_center_to_all = bsxfun(@minus, retinaParams_old(i).x, centroid_RGC);
    
    dist_center_to_all = pdist2(retinaParams_old(i).x, centroid_RGC);
    gaussian_val = 6*exp(-(dist_center_to_all)/10);
    
    retinaParams_old(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams_old(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams_old(i).c = [-65+15*re.^2];
    retinaParams_old(i).d = bsxfun(@minus, 8, gaussian_val);
    
    retinaParams_old(i).D = squareform(pdist(retinaParams_old(i).x));
    D = retinaParams_old(i).D;
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>10).*exp(-D/10); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end

retinaParams = retinaParams_old;

LGN_num = [400];

for outerRadius = 4%:2:24

for numLGN= LGN_num

%% Parameters of the LGN

eta = 0.1;
decay_rt = 0.01;
maxInnervations = totNeurons_Retina;

connectedNeurons = [];
initConnections  = [];

mu_wts = 2.5;
sigma_wts = 0.14;

% Choose random nodes on the arbitGeometry Retina -- and layer up!
layer_LGN = randi([1, totNeurons_Retina],numLGN,1);
LGN_pos2d = retinaParams(1).x(layer_LGN,:);

LGN_pos3d = [LGN_pos2d, ones(size(LGN_pos2d,1),1)];

synapticMatrix_retinaLGN = zeros(totNeurons_Retina, numLGN); 
% Normalizing synaptic matrix
for i = 1:numLGN
    synapticMatrix_retinaLGN(:,i) = normrnd(mu_wts, sigma_wts, [totNeurons_Retina,1]);
    synapticMatrix_retinaLGN(:,i) = synapticMatrix_retinaLGN(:,i)/mean(synapticMatrix_retinaLGN(:,i))*mu_wts;
end

[synapticMatrix_retinaLGN, retinaParams, sortIdx] = LGN_learn_rF(numLGN, outerRadius, retinaParams, LGN_pos3d, eta, decay_rt, mu_wts,...
    synapticMatrix_retinaLGN, numRetina);
   
end

end

%% Draw holes in layer-I -- represent chunks of dead pixels [Damaging some nodes in the input-layer!!]

figure; 
scatter(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),'filled')

h1 = imfreehand();
h2 = imfreehand();
h3 = imfreehand();
h4 = imfreehand();
h5 = imfreehand();


sketch_boundary1 = h1.getPosition;
sketch_boundary2 = h2.getPosition;
sketch_boundary3 = h3.getPosition;
sketch_boundary4 = h4.getPosition;
sketch_boundary5 = h5.getPosition;

figure; 
scatter(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),'filled')
hold on; 

% Find all coordinates within boundary
[in1, on1] = inpolygon(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),sketch_boundary1(:,1),sketch_boundary1(:,2));
[in2, on2] = inpolygon(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),sketch_boundary2(:,1),sketch_boundary2(:,2));
[in3, on3] = inpolygon(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),sketch_boundary3(:,1),sketch_boundary3(:,2));
[in4, on4] = inpolygon(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),sketch_boundary4(:,1),sketch_boundary4(:,2));
[in5, on5] = inpolygon(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),sketch_boundary5(:,1),sketch_boundary5(:,2));

ind_withinGeometry = intersect(intersect(intersect(intersect(find(in1==0),find(in2==0)),find(in3==0)),find(in4==0)),find(in5==0));

figure; 
scatter(retinaParams_old(1).x(:,2),retinaParams_old(1).x(:,1),'k','filled')
hold on
scatter(retinaParams_old(1).x(ind_withinGeometry,2),retinaParams_old(1).x(ind_withinGeometry,1),'r','filled')

pixelKill = setdiff(1:totNeurons_Retina, ind_withinGeometry); % These pixels are malfunctioning.

%  Learn new receptive fields!

synapticMatrix_retinaLGN_adapted = LGN_learn_rF_aDam(numLGN, outerRadius, retinaParams, LGN_pos3d, eta, decay_rt, mu_wts,...
    synapticMatrix_retinaLGN, numRetina, pixelKill, sortIdx);



