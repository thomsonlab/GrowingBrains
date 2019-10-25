% Growing a neural network from a single unit on an arbitrary scaffold

clear; clc
close all

iter_bdry = 1; %iter_bdry -> 1 to 5 (can pick any arbitrary scaffold for growing a multi-layered NN)
fname_bdry = sprintf('Boundary_sheet%d.mat',iter_bdry);

load(fname_bdry)
%area_sheet = polyarea(bdry_geometrySheet(:,1),bdry_geometrySheet(:,2));

% Seed the sheet and let cells divide/migrate and form parallel layers (layering)
numSeeds = 1;

while(1)
    [pos_seed] = 5+20*rand(numSeeds,2);
    [in,on] = inpolygon(pos_seed(:,1),pos_seed(:,2),bdry_geometrySheet(:,1),bdry_geometrySheet(:,2));
    if sum(in) == numSeeds
        break;
    end
end

% Seeded nodes multiply and migrate within the sheet (Terminate when
% critical density ~2.5-3.5 nodes/unit area)

numCells = numSeeds;

MIGRATE_DIST = 2;
DIV_DIST = 1;
T_SPAN = 25;
LOCAL_RAD = 1;
LOCAL_DENSE = 11;
HAYFLICK_LIM = 40;
LAYER_DIST = 1;
PROB_LAYER = 0.4;

allNodes_pos = pos_seed;
D = squareform(pdist(allNodes_pos));

% Defining v,u parameters for all nodes in the system (LAYER-I parameters)
retinaParams = {};

retinaParams.numNeurons = size(allNodes_pos,1);
re = rand(retinaParams.numNeurons,1);
retinaParams.x = [allNodes_pos(:,2),allNodes_pos(:,1)];
retinaParams.a = 0.02;
retinaParams.b = 0.2;
retinaParams.c = -65+15*re^2;
retinaParams.d = [8-6*re.^2];

retinaParams.D = squareform(pdist(retinaParams.x));
D = retinaParams.D;
retinaParams.Dk = 5*(D<2)- 2*(D>4).*exp(-D/10); 
retinaParams.Dk = retinaParams.Dk - diag(diag(retinaParams.Dk));
    
retinaParams.v = -65*ones(retinaParams.numNeurons,1); % Initial values of v
retinaParams.u = retinaParams.b.*retinaParams.v;
retinaParams.firings = [];
retinaParams.hfLimit = repmat(HAYFLICK_LIM,retinaParams.numNeurons,1);
retinaParams.time = repmat(T_SPAN,retinaParams.numNeurons,1);
retinaParams.layered = zeros(retinaParams.numNeurons,1);

spikeMat = zeros(retinaParams.numNeurons,1); % Input vector for the layer above (whenever the need arises)


% LAYER-2 parameters (parameters of the LGN)
eta = 0.1;
decay_rt = 0.01;
maxInnervations = retinaParams.numNeurons;

numLGN = 0;
LGN_pos2d = [];
LGN_pos3d = [];
LGN_threshold = [];
LGNactivity = [];
LGN_synapticChanges = [];
startLayer_t = NaN;
stopLayer_t = NaN;
mu_wts = 2.5;
sigma_wts = 0.14;
yAct_allLGN = [];
synapticMatrix_retinaLGN = zeros(retinaParams.numNeurons, numLGN);
maxInd_LGN = [];
s2Matrix = [];

figure(1);
hold on
scatter(bdry_geometrySheet(:,1),bdry_geometrySheet(:,2),'k','filled')
scatter(allNodes_pos(:,1),allNodes_pos(:,2),'filled')
hold off

cell_num = 1;
ctr_img = 1;
%layered = zeros(retinaParams.numNeurons,1);

t_LGN = []; %Time of existence for each LGN

rfSizes = [150:50:750]';
time_rf = zeros(size(rfSizes,1),1);
numRfcompInit = 0;
retPos = [];
t = 0;
temp_vec = [];
totNeurons = [];
choice = 1;
lgn_pos = [];


%%
while(1)
    
    %for cell_num = 1:size(allNodes_pos,1)
        
        % At time t-\Delta (update the properties of the node, ie v_n and u_n)
        retinaParams.array_act = []; %Active nodes
        retinaParams.I = [3*randn(retinaParams.numNeurons,1)]; % Noisy input
        retinaParams.fired = find(retinaParams.v >= 30);
        fired = retinaParams.fired; 
        retinaParams.array_act = retinaParams.x(fired,:);

        spikeMat = zeros(retinaParams.numNeurons,1);
        spikeMat(fired,1) = 1; %input vector for layer above

        retinaParams.firings = [retinaParams.firings; t+0*fired, fired];
        retinaParams.v(fired) = retinaParams.c(fired);
        retinaParams.u(fired) = retinaParams.u(fired) + retinaParams.d(fired);
        if isempty(retinaParams.Dk)
            % Leave retinaParams.I as is.
        else
            retinaParams.I = retinaParams.I + sum(retinaParams.Dk(:,fired),2);
        end

        retinaParams.v = retinaParams.v + 0.5*(0.04*retinaParams.v.^2 + 5*retinaParams.v + 140 - retinaParams.u + retinaParams.I);
        retinaParams.v = retinaParams.v + 0.5*(0.04*retinaParams.v.^2 + 5*retinaParams.v + 140 - retinaParams.u + retinaParams.I);
        retinaParams.u = retinaParams.u + retinaParams.a.*(retinaParams.b.*retinaParams.v - retinaParams.u); 
    

    if cell_num == 1
    
        t = t+1
        
        delete(gca)
        figure(2)
        scatter3(bdry_geometrySheet(:,1),bdry_geometrySheet(:,2),zeros(size(bdry_geometrySheet,1),1),'m','filled')
        hold on
        scatter3(retinaParams.x(:,2),retinaParams.x(:,1),zeros(retinaParams.numNeurons,1),'k','filled')
        hold on
        if ~isempty(retinaParams.array_act)
            scatter3(retinaParams.array_act(:,2),retinaParams.array_act(:,1),zeros(size(retinaParams.array_act,1),1),'r','filled')
        end
        hold on
        if ~isempty(LGN_pos3d)
                
            s2Matrix= synapticMatrix_retinaLGN;
            s2Matrix(s2Matrix<0.1) = NaN;
            s2Matrix = ~isnan(s2Matrix);
            
            retPos = find(s2Matrix(:,1) == 1);
            
            lgn_pos = LGN_pos3d(1,:);
            vec_plot = [];
            for iter = 1:size(retPos,1)
               vec_plot = [vec_plot; [retinaParams.x(retPos(iter),:),0];lgn_pos];
            end
            hold on
            plot3(vec_plot(:,2),vec_plot(:,1),vec_plot(:,3),'k-')
            hold on
            scatter3(LGN_pos3d(:,2),LGN_pos3d(:,1),LGN_pos3d(:,3),'b','filled')
            
            if yAct_allLGN(maxInd_LGN)>0
                scatter3(LGN_pos3d(maxInd_LGN,2),LGN_pos3d(maxInd_LGN,1),LGN_pos3d(maxInd_LGN,3),'r','filled')
            end
            
            axis([0 inf 0 inf 0 0.8])
        end
        set(gca,'Visible','off')
        fname = strcat('growingLayer_img/',sprintf('dev_%d_bdrySheet_%d.mat',ctr_img,iter_bdry));
        x_3d = [retinaParams.x, zeros(retinaParams.numNeurons,1)];
        LGN_2_ret = [x_3d(retPos,:)];
        storeFireNodes = [retinaParams.array_act,zeros(size(retinaParams.array_act,1),1)];
        %save(fname, 'x_3d','LGN_pos3d','LGN_2_ret','storeFireNodes','maxInd_LGN','lgn_pos','s2Matrix')
        
        fname = strcat('growingLayer_img/',sprintf('dev_%d_bdrySheet_%d.png',ctr_img,iter_bdry));
        %saveas(gca, fname);
        
        pause(0.2)

    % slower time-scale process of layering - slower than firing activity - similar to node division ! 
    if ~isempty(LGN_pos3d)
        LGN_pos3d(:,3) = LGN_pos3d(:,3) + 0.005;
        LGN_pos3d(LGN_pos3d(:,3)>0.8,3) = 0.8;
        t_LGN = t_LGN + 1;
    end
       
    end
    
    %if and(mod(ctr_img, 1000) == 0,choice~=0)
     if mod(ctr_img,1000) == 0
        
        %fname = strcat('growingLayer_matFiles/',sprintf('dev_%d_bdrySheet_%d.mat',ctr_img, iter_bdry));
        %save(fname, 'retinaParams','bdry_geometrySheet','t','t_LGN','LGN_pos3d','synapticMatrix_retinaLGN','yAct_allLGN','maxInd_LGN')
        
        delete(gca)
        figure(2)
        scatter3(bdry_geometrySheet(:,1),bdry_geometrySheet(:,2),zeros(size(bdry_geometrySheet,1),1),'m','filled')
        hold on
        scatter3(retinaParams.x(:,2),retinaParams.x(:,1),zeros(retinaParams.numNeurons,1),'k','filled')
        hold on
        if ~isempty(retinaParams.array_act)
            scatter3(retinaParams.array_act(:,2),retinaParams.array_act(:,1),zeros(size(retinaParams.array_act,1),1),'r','filled')
        end
        hold on
        if ~isempty(LGN_pos3d)
                
            s2Matrix= synapticMatrix_retinaLGN;
            s2Matrix(s2Matrix<0.1) = NaN;
            s2Matrix = ~isnan(s2Matrix);
            
            retPos = find(s2Matrix(:,1) == 1);
            
            lgn_pos = LGN_pos3d(1,:);
            vec_plot = [];
            for iter = 1:size(retPos,1)
               vec_plot = [vec_plot; [retinaParams.x(retPos(iter),:),0];lgn_pos];
            end
            hold on
            plot3(vec_plot(:,2),vec_plot(:,1),vec_plot(:,3),'k-')
            hold on
            scatter3(LGN_pos3d(:,2),LGN_pos3d(:,1),LGN_pos3d(:,3),'b','filled')
            
            if yAct_allLGN(maxInd_LGN)>0
                scatter3(LGN_pos3d(maxInd_LGN,2),LGN_pos3d(maxInd_LGN,1),LGN_pos3d(maxInd_LGN,3),'r','filled')
            end
            axis([0 inf 0 inf 0 0.8])
            
        end
        set(gca,'Visible','off')
        fname = strcat('growingLayer_img/',sprintf('dev_%d_bdrySheet_%d.mat',ctr_img,iter_bdry));
        x_3d = [retinaParams.x, zeros(retinaParams.numNeurons,1)];
        LGN_2_ret = [x_3d(retPos,:)];
        storeFireNodes = [retinaParams.array_act,zeros(size(retinaParams.array_act,1),1)];
        %save(fname, 'x_3d','LGN_pos3d','LGN_2_ret','storeFireNodes','maxInd_LGN')
        %save(fname, 'x_3d','LGN_pos3d','LGN_2_ret','storeFireNodes','maxInd_LGN','lgn_pos','s2Matrix')
        
        %fname = strcat('growingLayer_img/',sprintf('dev_%d_bdrySheet_%d.png',ctr_img,iter_bdry));
        %saveas(gca, fname);
        pause(0.2)
        
    
    end
    
        
%     Learning by the LGN    
    if numLGN>0
        
        % Hebbian learning for LGN nodes

        y1_allLGN = [];
        thresh_LGN = [];
        y1_allLGN = spikeMat'*synapticMatrix_retinaLGN;
        y1_allLGN(y1_allLGN<0) = 0;
        thresh_LGN = LGN_threshold';
        
        %[size(LGNactivity),numLGN]
        LGNactivity(end+1,:) = y1_allLGN;
        

        yAct_allLGN = bsxfun(@minus, y1_allLGN, thresh_LGN);
        yAct_allLGN(yAct_allLGN<0) = 0;
        [maxAct, maxInd_LGN] = max(yAct_allLGN);
        
        % Check if max node is greater than threshold
    
        if yAct_allLGN(maxInd_LGN) > 0

        % Modify weights ONLY for maxInd_LGN
        x_input = spikeMat;
        wt_input = synapticMatrix_retinaLGN(:,maxInd_LGN);

        wt_input = wt_input + 0.5*(eta*(yAct_allLGN(maxInd_LGN))*x_input);
        wt_input = wt_input + 0.5*(eta*(yAct_allLGN(maxInd_LGN))*x_input);

        synapticMatrix_retinaLGN(:,maxInd_LGN) = wt_input;

        % Keep track of number of synapses added/pruned!
        LGN_synapticChanges(maxInd_LGN) = LGN_synapticChanges(maxInd_LGN) + 1;

        % Modifying threshold! If threshold is much larger than activity :reduce ELSE increase
        LGN_threshold(maxInd_LGN) = LGN_threshold(maxInd_LGN) + 0.005*yAct_allLGN(maxInd_LGN);

        % Normalize weights to a constant strength
        synapticMatrix_retinaLGN(:,maxInd_LGN) = synapticMatrix_retinaLGN(:,maxInd_LGN)/mean(synapticMatrix_retinaLGN(:,maxInd_LGN))*mu_wts;

        end
        
    end
    
    if and(numLGN>0, ~isempty(find(mod(t_LGN,1000)==0)))
    
        t_LGN(1)
        s2Matrix= synapticMatrix_retinaLGN;
        s2Matrix(s2Matrix<0.1) = NaN;
        s2Matrix = ~isnan(s2Matrix);
        
        rgc_connected = [];
        
        for ind_rf = 1:length(rfSizes)
            rf = rfSizes(ind_rf);
            temp1 = find(sum(s2Matrix)<rf);
            
            if and(length(temp1)>0.9*numLGN, time_rf(ind_rf) ==0)
                time_rf(ind_rf) = t;
            end
        end
        
        numRfcomp = length(time_rf)-length(find(time_rf ==0));
        
        if numRfcomp>=1
            
        if numRfcomp>numRfcompInit
            
            u = find(time_rf ~=0);
            minRFsize = u(1);
            
            s2Matrix= synapticMatrix_retinaLGN;
            s2Matrix(s2Matrix<0.1) = NaN;
            s2Matrix = ~isnan(s2Matrix);        
            
            temp1 = find(sum(s2Matrix)<rfSizes(minRFsize));
 
            for j = temp1%1:numLGN
                rgc_connected = [rgc_connected, find(s2Matrix(:,j)==1)'];
            end
            percent_node = length(unique(rgc_connected))/retinaParams.numNeurons;
            
            % Save all variables in workspace (.mat)
            %save(strcat('LGN_rf_arbitGeo/',sprintf('LGN_%d_%d_r=%d.mat',numLGN, totNeurons_Retina, outerRadius)));
        end
        numRfcompInit = numRfcomp;
        end
        
        %if and(numRfcomp == length(rfSizes), t_LGN(1)>1e5)
        if and(numRfcomp >= 11, t_LGN(1)>1e5)
        
            break
        end
        
        tempVar = mod(t_LGN,1000);
    
        % AFter every 1 sec (1000 ms) -- update the threshold to prevent random shifting % LGN_params.threshold
        for i = 1:numLGN
            
            if and(tempVar(i) == 0, LGN_synapticChanges(i) < 200)
                LGN_threshold(i) = max(LGNactivity(:,i))*1/5;
            end
        end
        LGNactivity = max(LGNactivity,[],1);
        
    end

    flag = 0;
    v = rand(1);
    
    if v<1
        choice = 1;
    end

        
    if isempty(D)
        choice = 1;
    end
    
        if choice == 1 % Divide
        
            % Check if it's crowded or not!
            if isempty(D)
                
                numTries = 0;
                
                while(1)
                
                % pick a random point on a disk of radius (0.5 units)
                theta1 = 2*pi*rand(1);
                
                r = DIV_DIST+DIV_DIST*rand;
                newPts = [allNodes_pos(cell_num,:); allNodes_pos(cell_num,1) + r*cos(theta1), allNodes_pos(cell_num,2)+r*sin(theta1)];
                
                temp_nodes = allNodes_pos;
                temp_nodes = [temp_nodes(1:cell_num,:);newPts;temp_nodes(cell_num+1:end,:)];
                temp_nodes(cell_num,:) = [];
                
                % Check if both points lie within the polygon
                [in] = inpolygon(newPts(:,1),newPts(:,2),bdry_geometrySheet(:,1),bdry_geometrySheet(:,2));
                
                if sum(in) == 2
                    allNodes_pos = temp_nodes;
                    flag = 1;
                    break;
                end
                
                if numTries == 5
                    break;
                end
                
                numTries = numTries + 1;
                
                end
                
                % update retinaParams
                
                if flag == 1
                
                    retinaParams.x = [allNodes_pos(:,2),allNodes_pos(:,1)];
                    retinaParams.numNeurons = size(allNodes_pos,1);
                    
                    %layered = zeros(retinaParams.numNeurons,1);
                    
                    re = rand(2,1);
                    newNodes_a = 0.02*ones(2,1);
                    newNodes_b = 0.2*ones(2,1);
                    newNodes_c = -65+15*re.^2;
                    newNodes_d = 8-6*re.^2;
                    newNodes_v = -65*ones(2,1);
                    newNodes_u = newNodes_b.*newNodes_v;
                    %newWts_retLGN = normrnd(mu_wts, sigma_wts, [2,numLGN]);
                    newHf = repmat(retinaParams.hfLimit(cell_num)-1,2,1);
                    newTime = repmat(T_SPAN,2,1);
                    newLayeredVar = zeros(2,1);
                    
                    retinaParams.a = [retinaParams.a(1:cell_num);newNodes_a;retinaParams.a(cell_num+1:end)];
                    retinaParams.b = [retinaParams.b(1:cell_num);newNodes_b;retinaParams.b(cell_num+1:end)];
                    retinaParams.c = [retinaParams.c(1:cell_num);newNodes_c;retinaParams.c(cell_num+1:end)];
                    retinaParams.d = [retinaParams.d(1:cell_num);newNodes_d;retinaParams.d(cell_num+1:end)];
                    retinaParams.v = [retinaParams.v(1:cell_num);newNodes_v;retinaParams.v(cell_num+1:end)];
                    retinaParams.u = [retinaParams.u(1:cell_num);newNodes_u;retinaParams.u(cell_num+1:end)];
                    retinaParams.hfLimit = [retinaParams.hfLimit(1:cell_num);newHf;retinaParams.hfLimit(cell_num+1:end)];
                    retinaParams.time = [retinaParams.time(1:cell_num);newTime;retinaParams.time(cell_num+1:end)];
                    retinaParams.layered = [retinaParams.layered(1:cell_num);newLayeredVar;retinaParams.layered(cell_num+1:end)];
                    
                    retinaParams.a(cell_num) = [];
                    retinaParams.b(cell_num) = [];
                    retinaParams.c(cell_num) = [];
                    retinaParams.d(cell_num) = [];
                    retinaParams.v(cell_num) = [];
                    retinaParams.u(cell_num) = [];
                    %synapticMatrix_retinaLGN(cell_num,:) = [];
                    retinaParams.hfLimit(cell_num) = [];
                    retinaParams.time(cell_num) = [];
                    retinaParams.layered(cell_num) = [];
                    retinaParams.D = squareform(pdist(retinaParams.x));
                    D = retinaParams.D;
                    retinaParams.Dk = 5*(D<2)- 2*(D>4).*exp(-D/10); 
                    retinaParams.Dk = retinaParams.Dk - diag(diag(retinaParams.Dk));
                    
                    % Normalization
                    %synapticMatrix_retinaLGN = synapticMatrix_retinaLGN./mean(synapticMatrix_retinaLGN,1)*mu_wts;
                    
                    retinaParams.firings = [];
                    
                    if cell_num + 2 > retinaParams.numNeurons
                        cell_num = 1;
                    else
                        cell_num = cell_num + 2;
                    end
                    
                end
            
            else               
            if and(and(length(find(D(cell_num,:)<LOCAL_RAD))<= 3, retinaParams.hfLimit(cell_num)>0), ...
                retinaParams.time(cell_num)>0)
            
                numTries = 0;
                
                while(1)
                
                % pick a random point on a disk of radius (0.5 units)
                theta1 = 2*pi*rand(1);
                
                r = DIV_DIST+DIV_DIST*rand;
                
                newPts = [allNodes_pos(cell_num,:); allNodes_pos(cell_num,1) + r*cos(theta1), allNodes_pos(cell_num,2)+r*sin(theta1)];
                
                % Check if both points lie within the polygon
                [in] = inpolygon(newPts(:,1),newPts(:,2),bdry_geometrySheet(:,1),bdry_geometrySheet(:,2));
                
                temp_nodes = allNodes_pos;
                        
                dist_new = pdist2(newPts(2,:), temp_nodes);
                
                %if and(sum(in) == 2, max(sum(dist_new<0.5))<=LOCAL_DENSE)
                if and(sum(in) == 2, sum(dist_new<LOCAL_RAD)<=LOCAL_DENSE)
                   %allNodes_pos = temp_nodes;
                    allNodes_pos = [allNodes_pos(1:cell_num,:);newPts;allNodes_pos(cell_num+1:end,:)];
                    allNodes_pos(cell_num,:) = [];
                    flag = 1;
                    break;
                end
                
                if numTries == 50
                    break
                end
                
                numTries = numTries + 1;
                
                end
                
                % update retinaParams
                
                if flag == 1 % There was cell division
                
                    retinaParams.x = [allNodes_pos(:,2),allNodes_pos(:,1)];
                    retinaParams.numNeurons = size(allNodes_pos,1);
                    %layered = zeros(retinaParams.numNeurons,1);
                    
                    re = rand(2,1);
                    newNodes_a = 0.02*ones(2,1);
                    newNodes_b = 0.2*ones(2,1);
                    newNodes_c = -65+15*re.^2;
                    newNodes_d = 8-6*re.^2;
                    newNodes_v = -65*ones(2,1);
                    newNodes_u = newNodes_b.*newNodes_v;
                    newHf = repmat(retinaParams.hfLimit(cell_num)-1,2,1);
                    newTime = repmat(T_SPAN,2,1);
                    newLayeredVar = zeros(2,1);
                    
                    retinaParams.a = [retinaParams.a(1:cell_num);newNodes_a;retinaParams.a(cell_num+1:end)];
                    retinaParams.b = [retinaParams.b(1:cell_num);newNodes_b;retinaParams.b(cell_num+1:end)];
                    retinaParams.c = [retinaParams.c(1:cell_num);newNodes_c;retinaParams.c(cell_num+1:end)];
                    retinaParams.d = [retinaParams.d(1:cell_num);newNodes_d;retinaParams.d(cell_num+1:end)];
                    retinaParams.v = [retinaParams.v(1:cell_num);newNodes_v;retinaParams.v(cell_num+1:end)];
                    retinaParams.u = [retinaParams.u(1:cell_num);newNodes_u;retinaParams.u(cell_num+1:end)];
                    retinaParams.hfLimit = [retinaParams.hfLimit(1:cell_num);newHf;retinaParams.hfLimit(cell_num+1:end)];
                    retinaParams.time = [retinaParams.time(1:cell_num);newTime;retinaParams.time(cell_num+1:end)];
                    retinaParams.layered = [retinaParams.layered(1:cell_num);newLayeredVar;retinaParams.layered(cell_num+1:end)];
                   
                    retinaParams.a(cell_num) = [];
                    retinaParams.b(cell_num) = [];
                    retinaParams.c(cell_num) = [];
                    retinaParams.d(cell_num) = [];
                    retinaParams.v(cell_num) = [];
                    retinaParams.u(cell_num) = [];
                    %synapticMatrix_retinaLGN(cell_num,:) = [];
                    %layered(cell_num) = [];
                    retinaParams.hfLimit(cell_num) = [];
                    retinaParams.time(cell_num) = [];
                    retinaParams.layered(cell_num) = [];
                    
                    retinaParams.D = squareform(pdist(retinaParams.x));
                    D = retinaParams.D;
                    retinaParams.Dk = 5*(D<2)- 2*(D>4).*exp(-D/10); 
                    retinaParams.Dk = retinaParams.Dk - diag(diag(retinaParams.Dk));
                    
                    retinaParams.firings = [];
                    
                    if cell_num + 2 > retinaParams.numNeurons
                        cell_num = 1;
                    else
                        cell_num = cell_num + 2;
                    end
                    
                    if numLGN>0
                        synapticMatrix_retinaLGN = [synapticMatrix_retinaLGN; zeros(1,size(synapticMatrix_retinaLGN,2))];
                        % Normalization of weights..
                        synapticMatrix_retinaLGN = synapticMatrix_retinaLGN./mean(synapticMatrix_retinaLGN,1)*mu_wts;
                    end
                    
                end
                
                end
            end
            %retinaParams.layered = zeros(retinaParams.numNeurons,1);
            
        end
          

        if flag == 0 % Did not divide
            
            if retinaParams.time(cell_num)>0
                retinaParams.time(cell_num) = retinaParams.time(cell_num)-1;
            end
            
            if cell_num + 1 > retinaParams.numNeurons
                cell_num = 1;
            else
                cell_num = cell_num + 1;
            end
        end

        % Check if critical time reached zero & if that particular cell hasn't layered.
        
        if and(retinaParams.layered(cell_num) == 0, retinaParams.time(cell_num) == 0)
        
            % Check if any unit within LAYER_DIST hasn't layered
            temp_vec2 = find(D(cell_num,:)<LAYER_DIST);
            temp_vec2(temp_vec2==cell_num) = [];
            
            length(find(retinaParams.layered == 1))
            if length(find(retinaParams.layered(temp_vec2)==1))==0
                
                u_layer = rand(1);
                if u_layer<PROB_LAYER
                
                    disp('Start layering')
                    numLGN
                    retinaParams.layered(cell_num) = 1;
                    numLGN = numLGN + 1;
                    LGNactivity = [max(LGNactivity,[],1),0];
                    LGN_pos2d = [LGN_pos2d; retinaParams.x(cell_num,:)];
                    LGN_pos3d = [LGN_pos3d; [retinaParams.x(cell_num,:),0]];

                    LGN_threshold = [LGN_threshold; normrnd(70,2,1,1)];
                    LGN_synapticChanges = [LGN_synapticChanges, 0];
                    t_LGN = [t_LGN, 0];

                    %newWts_retLGN = normrnd(mu_wts, sigma_wts, [retinaParams.numNeurons, 1]);
                    newWts_retLGN = zeros(retinaParams.numNeurons,1);
                    newWts_retLGN(cell_num) = normrnd(mu_wts,sigma_wts,1);
                    diffSize = retinaParams.numNeurons - size(synapticMatrix_retinaLGN,1); % Difference in # of neurons in layer-1
                    
                    if isempty(synapticMatrix_retinaLGN)                        
                        synapticMatrix_retinaLGN = newWts_retLGN;
                    else
                        synapticMatrix_retinaLGN = [synapticMatrix_retinaLGN; zeros(diffSize, size(synapticMatrix_retinaLGN,2))];                    
                        synapticMatrix_retinaLGN = [synapticMatrix_retinaLGN, newWts_retLGN];
                    end
                    % Normalization
                    synapticMatrix_retinaLGN = synapticMatrix_retinaLGN./mean(synapticMatrix_retinaLGN,1)*mu_wts;
                    disp('added unit')
                end
            end
        end
  
    ctr_img = ctr_img + 1;
    

end



