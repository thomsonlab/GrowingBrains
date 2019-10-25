function [synapticMatrix_retinaLGN, retinaParams, sortIdx, percent_node] = LGN_learn_receptiveFields(numLGN, outerRadius, retinaParams, LGN_pos3d, eta, decay_rt, mu_wts, synapticMatrix_retinaLGN, numRetina)
   
i = 1;
D = retinaParams(i).D;
totNeurons_Retina = retinaParams(i).numNeurons;

% Change wave-size!
retinaParams(i).Dk = 5*(D<2)- 2*(D>outerRadius).*exp(-D/10); 
retinaParams(i).Dk = retinaParams(i).Dk - diag(diag(retinaParams(i).Dk));
x_3d = [retinaParams(i).x, zeros(size(retinaParams(i).x,1),1)];
di = pdist2(x_3d, LGN_pos3d);

LGN_synapticChanges = zeros(numLGN,1);
LGN_threshold = normrnd(70,2,numLGN,1);
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
flag = 0;
while(1)
    t = t+1;
    
    if (t>5e4)
        break
    end
    
   
    if mod(t, 500) == 0
        %if flag == 0
        %    all_lgnIdx = find(LGN_synapticChanges>200);
            %[lgnSynCh, lgn_idx] = max(LGN_synapticChanges);
        %    if length(all_lgnIdx) > 0
        %        lgn_idx = datasample(all_lgnIdx, 1);
        %    else
        %        lgn_idx = datasample(1:numLGN,1);
        %    end
        %   lgn_idx
        %    if t>1000    
        %        flag = 1;
        %    end
        %end
        
        %[lgnSynCh, lgn_idx] = max(LGN_synapticChanges);
        if flag == 0
        [sortVal, sortIdx] = sort(LGN_synapticChanges, 'descend'); 
        if t>1000
        flag = 1;
        end
        end
        
        % Plot the 2 layers while self-organizing
        %figure(1);
        scatter3(retinaParams.x(:,2),retinaParams.x(:,1),zeros(retinaParams.numNeurons,1),'k','filled')
        hold on
        if ~isempty(retinaParams.array_act)
            scatter3(retinaParams.array_act(:,2),retinaParams.array_act(:,1),zeros(size(retinaParams.array_act,1),1),'r','filled')
        end

        s2Matrix= synapticMatrix_retinaLGN;
        s2Matrix(s2Matrix<0.1) = NaN;
        s2Matrix = ~isnan(s2Matrix);

        for tempVar = 1:3
            
            lgn_idx = sortIdx(tempVar);
        
            retPos = find(s2Matrix(:,lgn_idx) == 1);

            lgn_pos = LGN_pos3d(lgn_idx,:);
            vec_plot = [];
            for iter = 1:size(retPos,1)
               vec_plot = [vec_plot; [retinaParams.x(retPos(iter),:),0];lgn_pos];
            end
            hold on
            plot3(vec_plot(:,2),vec_plot(:,1),vec_plot(:,3),'color',[0 0 0]+0.7);
            hold on
            scatter3(LGN_pos3d(:,2),LGN_pos3d(:,1),LGN_pos3d(:,3),'b','filled')
        
        end
        set(gca, 'Visible','off')
        saveas(gca,strcat('E:\GURU_DATA\Project 1\faultTolerance2\',sprintf('undamaged_nodes_%d.png',t)));
        clf
        
    end
    
    
    if mod(t,1000) == 0
    
        s2Matrix= synapticMatrix_retinaLGN;
        s2Matrix(s2Matrix<0.1) = NaN;
        s2Matrix = ~isnan(s2Matrix);
        
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

            %figure; 
            %scatter3(x_3d(:,2),x_3d(:,1),x_3d(:,3),'k','filled')
            %hold on

            %for i = 1:length(temp1)

                %l = find(s2Matrix(:,temp1(i))==1);
                %c = rand(1,3);
                %scatter3(LGN_pos3d(temp1(i),2),LGN_pos3d(temp1(i),1),LGN_pos3d(temp1(i),3),[],c,'filled')
                %hold on
                %scatter3(x_3d(l,2),x_3d(l,1),x_3d(l,3),[],repmat(c,length(l),1),'filled')
            %end
            
            %axis([0 inf 0 inf 0 inf])
            
            
           
% 
%             hold off
            %saveas(gca,strcat('LGN_rf_1500_trial2/',sprintf('3DLGN_%d_%d_r=%d.fig',numLGN, totNeurons_Retina, outerRadius)));
        
            %figure;
            %ctr = 1;
            %for j = 1:20%numLGN        
             %       subplot(4,5,j)
             %       clear l

             %      l = find(synapticMatrix_retinaLGN(:,temp1(j))>0.1);
             %       hold on
             %       scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],'k','filled')
             %       scatter(retinaParams(1).x(l,2),retinaParams(1).x(l,1),[],'r','filled')
             %       hold on
                    %scatter(post_synapticPos(temp1(j),2),post_synapticPos(temp1(j),1),[],'b','filled')
                    %scatter(retinaParams(1).x(LGN_params(j).connectLGN,2),retinaParams(1).x(LGN_params(j).connectLGN,1),'b','filled')
             %       title(sprintf('LGN %d',temp1(j)))

              %      ctr = ctr + 1;
            %end 
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
        
    
        % AFter every 1 sec (1000 ms) -- update the threshold to prevent random shifting % LGN_params.threshold
        for i = 1:numLGN
            
            if LGN_synapticChanges(i) < 200
                LGN_threshold(i) = max(LGNactivity(:,i))*1/5;
            end
            
            
        end
        LGNactivity = max(LGNactivity);
        disp(t)
        
    end
    
    spikeMat = zeros(numRetina*totNeurons_Retina,1);
    
    for i = 1:numRetina
       
        retinaParams(i).array_act = [];
        retinaParams(i).I = [3*randn(totNeurons_Retina,1)]; 
        
        retinaParams(i).fired = find(retinaParams(i).v >= 30);
        fired = retinaParams(i).fired;
        
        retinaParams(i).array_act = retinaParams(i).x(fired,:);
        
        %retinaParams(i).firings = [retinaParams(i).firings; t+0*fired, fired];
        retinaParams(i).v(fired) = retinaParams(i).c(fired);
        retinaParams(i).u(fired) = retinaParams(i).u(fired) + retinaParams(i).d(fired);
        retinaParams(i).I = retinaParams(i).I + sum(retinaParams(i).Dk(:,fired),2);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).v = retinaParams(i).v + 0.5*(0.04*retinaParams(i).v.^2 + 5*retinaParams(i).v + 140 - retinaParams(i).u + retinaParams(i).I);
        retinaParams(i).u = retinaParams(i).u + retinaParams(i).a.*(retinaParams(i).b.*retinaParams(i).v - retinaParams(i).u); 

        spikeMat(fired+(i-1)*totNeurons_Retina,1) = ones(length(fired),1);
        
        if length(fired)>30
            heatMap_wave(fired) = heatMap_wave(fired)+1;
            
%             figure(2);
%             hold on
%             scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),'k','filled')
%             if size(retinaParams(1).array_act,1) ~=0
%                 scatter(retinaParams(1).array_act(:,2),retinaParams(1).array_act(:,1),'r','filled')
%             end
            
        end
        
    end
  
%     if (length(fired)>30)
%     
%     % Plot retina spontaneous synchronous bursts
%     figure(1);
%     subplot(2,2,1);
%     hold on
%     scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),'k','filled')
%     if size(retinaParams(1).array_act,1) ~=0
%         scatter(retinaParams(1).array_act(:,2),retinaParams(1).array_act(:,1),'r','filled')
%     end
%     
%     %figure(2);% Plot LGN connectivity
%     for j = 1:numLGN        
%         subplot(2,2,j+1)
%         hold on
%         scatter(retinaParams(1).x(:,2),retinaParams(1).x(:,1),[],synapticMatrix_retinaLGN(:,j),'filled')
%         
%         %scatter(retinaParams(1).x(LGN_params(j).connectLGN,2),retinaParams(1).x(LGN_params(j).connectLGN,1),'b','filled')
%         title(sprintf('LGN %d; %d; t=%d',j,LGN_params(j).synapticChanges,t))
%         colorbar
%     end 
%     
%     pause(0.2)
%     
%     end

    % Hebbian learning for LGN nodes
      
    y1_allLGN = [];
    thresh_LGN = [];
    y1_allLGN = spikeMat'*synapticMatrix_retinaLGN;
    y1_allLGN(y1_allLGN<0) = 0;
    thresh_LGN = LGN_threshold';
    
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

