%% Implementing Developmental algorithm
% 1. Spatiotemporal wave generator in layer-I
% 2. Learning rule implemented in layer-II
% ==> Results in a pooling architecture

% clear
% clc
% close all

for numSim = 1:1

% clearvars -except numSim
% clc
% close all

% Initialize Retinal nodes and their properties

numRetina = 1;
totNeurons_Retina = 1600;
squareLength = 20;
retinaParams_old = {};

for i = 1:numRetina
    
    retinaParams_old(i).numNeurons = totNeurons_Retina;
    re = rand(totNeurons_Retina,1);
    
    retinaParams_old(i).x = squareLength*rand(totNeurons_Retina,2);
    
    centroid_RGC = mean(retinaParams_old(i).x);
    dist_center_to_all = bsxfun(@minus, retinaParams_old(i).x, centroid_RGC);
    
    dist_center_to_all = pdist2(retinaParams_old(i).x, centroid_RGC);
    gaussian_val = 6*exp(-(dist_center_to_all)/10);
    
    %% Parameters of the dynamical system
    retinaParams_old(i).a = [0.02*ones(totNeurons_Retina,1)];
    retinaParams_old(i).b = [0.2*ones(totNeurons_Retina,1)];
    retinaParams_old(i).c = [-65+15*re.^2];  %random noise on activity field
    %retinaParams(i).d = [8-6*re.^2];
    retinaParams_old(i).d = bsxfun(@minus, 8, gaussian_val);
    
    retinaParams_old(i).D = squareform(pdist(retinaParams_old(i).x));
    D = retinaParams_old(i).D;
    retinaParams_old(i).Dk = 5*(D<2)- 2*(D>5).*exp(-D/10); 
    retinaParams_old(i).Dk = retinaParams_old(i).Dk - diag(diag(retinaParams_old(i).Dk));
    
    retinaParams_old(i).v = -65*ones(totNeurons_Retina,1); % Initial values of v
    retinaParams_old(i).u = retinaParams_old(i).b.*retinaParams_old(i).v;
    retinaParams_old(i).firings = [];
    
end

retinaParams = retinaParams_old;

% figure;
% scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),'k','filled')
% pause(0.2)

LGN_num = [200];

for outerRadius = 6%:2:24

for numLGN= LGN_num
      
% Change wave-size!
retinaParams(i).Dk = 5*(D<2)- 2*(D>outerRadius).*exp(-D/10); 
retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));
% 3D equivalent of positions
x_3d = [ retinaParams(i).x, zeros( size(retinaParams(i).x,1),1 ) ];

%% Parameters of the LGN
eta = 0.1;          % Eta_learn learning rate for modifying weights
decay_rt = 0.01;
maxInnervations = totNeurons_Retina;
%maxInnervations = 1499;

LGN_params = {};
connectedNeurons = [];
initConnections  = [];

% THIS is the WEIGHT MATRIX W that connects layer 1 & 2
synapticMatrix_retinaLGN = zeros(totNeurons_Retina, numLGN);

% Choose random nodes on the arbitGeometry Retina -- and layer up!
layer_LGN = randi([1, totNeurons_Retina],numLGN,1);
LGN_pos2d = retinaParams(1).x(layer_LGN,:);
LGN_pos3d = [LGN_pos2d, ones(size(LGN_pos2d,1),1)];

di = pdist2(x_3d, LGN_pos3d);

% Normalizing synapticMatrix
mu_wts = 2.5;
sigma_wts = 0.14;
for i = 1:numLGN
    synapticMatrix_retinaLGN(:,i) = normrnd(mu_wts, sigma_wts, [totNeurons_Retina,1]);
    synapticMatrix_retinaLGN(:,i) = synapticMatrix_retinaLGN(:,i)/mean(synapticMatrix_retinaLGN(:,i))*mu_wts;
end

LGN_synapticChanges = zeros(numLGN,1);
LGN_threshold = normrnd(70,2,numLGN,1); % Normal distribution around 70 +- 2 (std. dev)

LGNactivity = [];
initSynapticMatrix_retinaLGN = synapticMatrix_retinaLGN;

heatMap_wave = zeros(totNeurons_Retina,1); % # of times each neuron spikes

rfSizes = [150:50:750]';

%% Spontaneous synchronous bursts of Retina(s)
t = 0;

tic
rgc_connected = [];
time_rf = zeros(size(rfSizes,1),1);
numRfcompInit = 0;

while(1)
    t = t+1;
    
    if (t>1e6)
        break
    end
   
    if mod(t,1000) == 0
        
        % Binary Matrix 
        s2Matrix= synapticMatrix_retinaLGN;
        s2Matrix(s2Matrix<0.1) = NaN;
        s2Matrix = ~isnan(s2Matrix);
        
        for ind_rf = 1:length(rfSizes) % [150:50:750]
            rf = rfSizes(ind_rf);
            temp1 = find(sum(s2Matrix)<rf);
            
            if and(length(temp1)>0.9*numLGN, time_rf(ind_rf) ==0)
                time_rf(ind_rf) = t;
            end
        end
        
        numRfcomp = length(time_rf)-length(find(time_rf ==0)) % print-out
        
        if numRfcomp>=1            
            if numRfcomp>numRfcompInit

                u = find(time_rf ~=0);
                minRFsize = u(1);

                s2Matrix= synapticMatrix_retinaLGN;
                s2Matrix(s2Matrix<0.1) = NaN;
                s2Matrix = ~isnan(s2Matrix);        

                temp1 = find(sum(s2Matrix)<rfSizes(minRFsize));

                figure;
                ctr = 1; jj = 0;
                for j = jj+1:jj+20%numLGN        
                        subplot(4,5,j)
                        clear l

                        l = find(synapticMatrix_retinaLGN(:,temp1(j))>0.1);
                        hold on
                        scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],'k','filled')
                        scatter(retinaParams(1).x(l,2),retinaParams(1).x(l,1),[],'r','filled')
                        hold on
                        title(sprintf('LGN %d',temp1(j)))

                        ctr = ctr + 1;
                end 
                %saveas(gca,strcat('LGN_rf_arbitGeo/',sprintf('2DLGN_%d_%d_r=%d.fig',numLGN, totNeurons_Retina, outerRadius)));

                for j = temp1%1:numLGN
                    rgc_connected = [rgc_connected, find(~isnan(s2Matrix(:,j)))'];
                end
                percent_node = length(unique(rgc_connected))/totNeurons_Retina;

                % Save all variables in workspace (.mat)
                %save(strcat('LGN_rf_arbitGeo/',sprintf('LGN_%d_%d_r=%d.mat',numLGN, totNeurons_Retina, outerRadius)));
            end
            numRfcompInit = numRfcomp;
        end
        
        if numRfcomp == length(rfSizes)
            break
        end        
    
%         % After every 1 sec (1000 ms) -- update the threshold to prevent random shifting
%         for i = 1:numLGN            
%             if LGN_synapticChanges(i) < 200
%                 LGN_threshold(i) = max(LGNactivity(:,i))*1/5; %
%             end        
%         end
%         LGNactivity = max(LGNactivity); 
        disp(t)
        
    end
    
    spikeMat = zeros(numRetina*totNeurons_Retina,1);
    
    i=1; %for i = 1:numRetina
    retinaParams(i).array_act = [];
    retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; 

    retinaParams(i).fired = find(retinaParams(i).v >= 30);
    fired = retinaParams(i).fired;

    retinaParams(i).array_act = retinaParams(i).x(fired,:);

    retinaParams(i).firings = [retinaParams(i).firings; t+0*fired, fired];
    retinaParams(i).v(fired) = retinaParams(i).c(fired);
    retinaParams(i).u(fired) = retinaParams(i).u(fired) + retinaParams(i).d(fired);
    retinaParams(i).I = retinaParams(i).I + sum(retinaParams(i).Dk(:,fired),2);
    retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
    retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
    retinaParams(i).u = retinaParams(i).u + retinaParams(i).a.*(retinaParams(i).b.*retinaParams(i).v - retinaParams(i).u); 

    spikeMat(fired+(i-1)*totNeurons_Retina,1) = ones(length(fired),1);

    if (length(fired)>30 && mod(t,50) == 0)
%         if (mod(t,100) == 0)
        heatMap_wave(fired) = heatMap_wave(fired)+1;            
        figure(2); 
        hold on
        scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),'k','filled')
        if size(retinaParams(1).array_act,1) ~=0
            scatter(retinaParams(1).array_act(:,2),retinaParams(1).array_act(:,1),'r','filled')
        end   
        gifname2='wave.gif';
        %{    
              frame = getframe(gcf); 
              im = frame2im(frame); 
              [imind,cm] = rgb2ind(im,256);

              % Write to the GIF File 
              if t==1
                  imwrite(imind,cm,gifname2,'gif','DelayTime',0.1,'Loopcount',Inf); 
              else 
                  imwrite(imind,cm,gifname2,'gif','DelayTime',0.1,'WriteMode','append'); 
              end 
        %}

        figure(3), h = pcolor(synapticMatrix_retinaLGN); title(['Weight Matrix t=' num2str(t)]), set(h,'EdgeColor','none'),set(gca,'Ydir','reverse'),colormap cool;
        gifname3='matrix.gif';
        %{    
              frame = getframe(gcf); 
              im = frame2im(frame); 
              [imind,cm] = rgb2ind(im,256);

              % Write to the GIF File 
              if t==1
                  imwrite(imind,cm,gifname3,'gif','DelayTime',0.1,'Loopcount',Inf); 
              else 
                  imwrite(imind,cm,gifname3,'gif','DelayTime',0.1,'WriteMode','append'); 
              end 
        %}
    end        
    %end
  
    %% Hebbian learning for LGN nodes      
    %y1_allLGN = []; thresh_LGN = [];
    y1_allLGN = spikeMat'*synapticMatrix_retinaLGN; % H(v_i(t)-30)*W 
    y1_allLGN(y1_allLGN<0) = 0; % ReLU activation function
    thresh_LGN = LGN_threshold'; % load updated threshold
    
    LGNactivity(end+1,:) = y1_allLGN; % Snapshot matrix of inputs to all LGN units over time

    yAct_allLGN = bsxfun(@minus, y1_allLGN, thresh_LGN); % amount by how much input signal lies over threshold
    yAct_allLGN(yAct_allLGN<0) = 0; % ReLU activation function
    [~, maxInd_LGN] = max(yAct_allLGN); % find Index of >> WINNER <<
    [~, maxkInd_LGN] = maxk(yAct_allLGN,10); % ALTERNATIVELY: find Index of k best performers
    figure(4), plot(1:numLGN,yAct_allLGN,1:numLGN,thresh_LGN), title(['Spikes t=' num2str(t)]), ylim([0,4000]),%legend('y1_allLGN','thresh_LGN')
    gifname4='spikes.gif'; 
    %{    
          frame = getframe(gcf); 
          im = frame2im(frame); 
          [imind,cm] = rgb2ind(im,256);

          % Write to the GIF File 
          if t==1
              imwrite(imind,cm,gifname4,'gif','DelayTime',0.1,'Loopcount',Inf); 
          else 
              imwrite(imind,cm,gifname4,'gif','DelayTime',0.1,'WriteMode','append'); 
          end 
    %} 
    
    % Check if max node is greater than threshold   
    if yAct_allLGN(maxInd_LGN) > 0    
        % Modify weights ONLY for maxInd_LGN -- the WINNER processing unit
% %{
        x_input = spikeMat; % array of either 0 or 1
        wt_input = synapticMatrix_retinaLGN(:,maxInd_LGN); % get all weights leading to winner unit
        
        tic
        wt_input = wt_input + eta*yAct_allLGN(maxInd_LGN)*x_input; % weight update for the winner
        synapticMatrix_retinaLGN(:,maxInd_LGN) = wt_input; % update W
        toc

        % Update/Keep track of number of times a particular winner unit was updated!
        LGN_synapticChanges(maxInd_LGN) = LGN_synapticChanges(maxInd_LGN) + 1;

        % Modifying threshold! If threshold is much larger than activity: reduce ELSE increase
        LGN_threshold(maxInd_LGN) = LGN_threshold(maxInd_LGN) + 0.005*yAct_allLGN(maxInd_LGN);
        
        % Normalize weights to a constant strength
        synapticMatrix_retinaLGN(:,maxInd_LGN) = synapticMatrix_retinaLGN(:,maxInd_LGN)/mean(synapticMatrix_retinaLGN(:,maxInd_LGN))*mu_wts;
%}        
        % Alternatively, also modify the weights around fixed radius of the winner
%{
        for rr = maxInd_LGN-5 : maxInd_LGN+5
            if (maxInd_LGN+rr >= 1 && maxInd_LGN+rr <= LGN_num)
                x_input = spikeMat; % array of either 0 or 1
                wt_input = synapticMatrix_retinaLGN(:,maxInd_LGN+rr); % get all weights leading to winner unit
                wt_input = wt_input + eta*(yAct_allLGN(maxInd_LGN+rr))*x_input; % weight update for the winner
                synapticMatrix_retinaLGN(:,maxInd_LGN+rr) = wt_input; % update W
                % Update/Keep track of number of times a particular winner unit was updated!
                LGN_synapticChanges(maxInd_LGN+rr) = LGN_synapticChanges(maxInd_LGN+rr) + 1;
                % Modifying threshold! If threshold is much larger than activity: reduce ELSE increase
                LGN_threshold(maxInd_LGN+rr) = LGN_threshold(maxInd_LGN+rr) + 0.005*yAct_allLGN(maxInd_LGN+rr);
                % Normalize weights to a constant strength
                synapticMatrix_retinaLGN(:,maxInd_LGN+rr) = synapticMatrix_retinaLGN(:,maxInd_LGN+rr)/mean(synapticMatrix_retinaLGN(:,maxInd_LGN+rr))*mu_wts;
            end
        end; rr=0;
%}
        % Alternatively, modify for all k-best units (winner and k-1 runner-ups)
%{
        for rr = maxkInd_LGN
            x_input = spikeMat; % array of either 0 or 1
            wt_input = synapticMatrix_retinaLGN(:,rr); % get all weights leading to winner unit
            wt_input = wt_input + (yAct_allLGN(rr)/yAct_allLGN(maxInd_LGN))^8 ... % take ratio of runner-up to winner into account
                                  *eta*(yAct_allLGN(rr))*x_input; % weight update for the winner
            synapticMatrix_retinaLGN(:,rr) = wt_input; % update W
            % Update/Keep track of number of times a particular winner unit was updated!
            LGN_synapticChanges(rr) = LGN_synapticChanges(rr) + 1;
            % Modifying threshold! If threshold is much larger than activity: reduce ELSE increase
            LGN_threshold(rr) = LGN_threshold(rr) + 0.005*yAct_allLGN(rr);
            % Normalize weights to a constant strength
            synapticMatrix_retinaLGN(:,rr) = synapticMatrix_retinaLGN(:,rr)/mean(synapticMatrix_retinaLGN(:,rr))*mu_wts;
        end; rr=0;
%}
    end    
end

end

end

matFileName = sprintf('RGC-LGNnetwork_MNIST_%d.mat',numSim);
save(matFileName, 'retinaParams', 's2Matrix', 'synapticMatrix_retinaLGN')

end