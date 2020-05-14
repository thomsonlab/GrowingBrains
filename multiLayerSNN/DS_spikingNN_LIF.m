%% Dynamical Systems Form of Spiking NN Developmental algorithm %%
clc
clear
close all

%% System Constants and Parameters
nR = 1600;      % # Neurons in Retina
nL = 300;       % # Neurons in LGN
nV = 400;       % # Neurons in V1n length of Retina

%% Retina Structure Parameters
Ret = {};       % Retina Data Structure
% centroid_RGC = mean(Ret.nx); dist_center_to_all = pdist2(Ret.nx, centroid_RGC); gaussian_val = 6*exp(-(dist_center_to_all)/10);

Ret.tf = zeros(nR,1);
Ret.th = ones(nR,1);        %variable retina thresh
Ret.b = 0*ones(nR,1);
Ret.v_reset = 0 + 0.1*randn(nR,1).^2;     %Noise on activity field

sqR = 20;
Ret.nx = sqR*rand(nR,2);
Ret.ri = 2; Ret.ro = 5;                 %Inner & outer rad of Ret-wave
Ret.D = squareform(pdist(Ret.nx));      %Distance matrix
Ret.S = 16*(Ret.D < Ret.ri)- 3*(Ret.D > Ret.ro).*exp(-Ret.D / 10); 
Ret.S = Ret.S - diag(diag(Ret.S));      %Adjacency matrix between Neurons

Ret.v = 0*ones(nR,1); %I.C. of v
Ret.eta = [];
Ret.hist = [];
Ret.H = sparse(zeros(nR,1)); % equivalent to "spikeMat"

%% LGN Structure Parameters
LGN = {};           % LGN Data Structure
LGN.eta_learn = 0.1;
LGN.synapticChanges = zeros(nL,1);
LGN.thresh = normrnd(70,2,nL,1); % Normal distribution around 70 +- 2 (std. dev)
LGN.activity = [];

sqL = 6; 
LGN.nx = sqL*rand(nL,2);
LGN.ri = 1.2; LGN.ro = 1.3;
LGN.D = squareform(pdist(LGN.nx));      
LGN.S_d = 0.04*(LGN.D < LGN.ri)- 0.01*(LGN.D > LGN.ro).*exp(-LGN.D / 10); 
LGN.S = 60*(LGN.D < LGN.ri)- 15*(LGN.D > LGN.ro).*exp(-LGN.D / 10); 
LGN.S = LGN.S - diag(diag(LGN.S));      %Adjacency matrix between Neurons in LGN

LGN.v = 0*ones(nR,1);               %Voltage
LGN.eta = [];                       %Noise input
LGN.H = sparse(zeros(nL,1));        %"spikeMat"
LGN.th = ones(nL,1);                %variable retina thresh
LGN.b = 0*ones(nL,1);               %constant bias input
LGN.v_reset = 0 + 0.1*randn(nL,1).^2;  %Noisy voltage reset

%% V1 Structure Parameters
V1 = {};           % LGN Data Structure
V1.eta_learn = 0.15;
V1.synapticChanges = zeros(nV,1);
V1.thresh = normrnd(30,2,nV,1); % Normal distribution around 70 +- 2 (std. dev)
V1.activity = [];

%% Synaptic Weight Matrix I (Ret-LGN1)
mu_W1 = 2.5;
sigma_W1 = 0.14;
W1 = normrnd(mu_W1, sigma_W1, [nR,nL]);
W1 = W1./mean(W1)*mu_W1;    % Weight Matrix between Retina-LGN1 normrnd initialized

%% Synaptic Weight Matrix II (LGN1-V1)
mu_W2 = 2.5;
sigma_W2 = 0.14;
W2 = normrnd(mu_W2, sigma_W2, [nL,nV]);
W2 = W2./mean(W2)*mu_W2;    % Weight Matrix between Retina-LGN1 normrnd initialized

% heatMap_wave = zeros(nR,1); % # of times each neuron spikes
% rfSizes = [150:50:750]';    % counter for size of receptive field

%% Time loop & Initialization
Tend = 3e2; dt = 0.1; t_intrvls = 0:Tend;
fnoise = 3*randn(nR,length(t_intrvls)); % Pre-generate noise
fnoiseI = interp1(t_intrvls',fnoise',[0:dt:Tend]','linear')';
% fnoise = randn(nR,1).*ones(nR,length(t_intrvls)); % Pre-generate noise

Ret.v = 0*ones(nR,1); Ret.u = zeros(nR,1); %I.C. of v, u
Xu = zeros(nR,t_intrvls(end)); Xu(:,1) = Ret.u; %Snapshot matrix u
Xv = zeros(nR,t_intrvls(end)); Xv(:,1) = Ret.v; %Snapshot matrix v
Xth = zeros(nR,t_intrvls(end)); Xth(:,1) = Ret.th;

LGN.v = 0*ones(nL,1); LGN.u = zeros(nL,1); %I.C. of v, u
Lu = zeros(nL,t_intrvls(end)); Lu(:,1) = LGN.u; %Snapshot matrix u
Lv = zeros(nL,t_intrvls(end)); Lv(:,1) = LGN.v; %Snapshot matrix v
Lth = zeros(nL,t_intrvls(end)); Lth(:,1) = LGN.th;
Ly = zeros(nL,t_intrvls(end)); 

tau_v = 2; tau_u = 0.3;
tau_th = 60; th_plus = 9; v_th = 1;
firedMat = cell(1,length(t_intrvls)); fireLMat = cell(1,length(t_intrvls));
tic
for tt = 0:dt:t_intrvls(end-1)
%{
    if mod(tt,1000)==0
        % After every 1 sec (1000 ms) -- update the thresh to prevent random shifting % LGN_params.thresh
        for i = 1:nL            
            if LGN.synapticChanges(i) < 200
                LGN.thresh(i) = max(LGN.activity(:,i))*1/5; %
            end        
        end
        LGN.activity = max(LGN.activity); 
        disp(tt)
    end
%}
%     Ret.eta = fnoiseI(:,round(tt/dt+1)); 
    Ret.eta = fnoise(:,round(tt)+1);
    
    % Solve Wave Dynamical System in Retina
    Ret.u = (Ret.S*Ret.H);%.*exp(-(tt-Ret.tf)/tau_u)/tau_u ;
    Ret.v = RK4(@(v)(-1/tau_v *v + Ret.u + Ret.b + Ret.eta),  dt,Ret.v);
    Ret.th = RK4(@(th)(1/tau_th*(v_th-th).*(1-Ret.H) + th_plus*Ret.H),  dt,Ret.th);

%     Ret.v = Ret.v + dt*(- 1/tau_v * Ret.v + Ret.u + Ret.b + Ret.eta);
%     Ret.th = Ret.th + dt*(1/tau_th*(v_th-Ret.th).*(1-Ret.H) + th_plus*Ret.H);

    % Discontinuous Update rule
    fired = find(Ret.v >= Ret.th);
    Ret.v(fired) = Ret.v_reset(fired);
%     Ret.tf(fired) = tt;
    Ret.H = sparse(zeros(nR,1));
    Ret.H(fired,1) = ones(length(fired),1);
    
    % Inputs to LGN
    LGN.x = W1' * Ret.H; % H(v_i(t)-30)*W 
%     LGN.activity(end+1,:) = LGN.x; % Snapshot matrix of inputs to all LGN units over time

    % LGN Activation function
    LGN.y = max(LGN.x - LGN.thresh, 0); % amount by how much input signal lies over thresh % ReLU activation function
%     [win, maxInd] = max(LGN.y);     % find Index of >> WINNER << 
    [wink, maxInd] = maxk(LGN.y,1); % ALTERNATIVELY: find Index of k best performers
    
    % LGN Competition rule
%     LGN.y(LGN.y<win) = 0;         % only allow >> WINNER << to participate
    LGN.y(LGN.y<wink(end)) = 0;     % ALTERNATIVELY: allow k best performers to participate
    LGN.y(maxInd) = (LGN.y(maxInd)/(wink(1)+eps)).^4 .*LGN.y(maxInd);
    
    % "Output" of LGN
    LGN.y = sparse(LGN.y); % Sparsity declaration for memory/speed
    
    % Solve W1 Matrix dynamical system & Update thresh
    W1 = W1 + dt*(LGN.eta_learn* Ret.H*LGN.y'); 
    LGN.thresh = LGN.thresh + dt*(0.005*LGN.y);
    W1(:,maxInd) = W1(:,maxInd)./mean(W1(:,maxInd))*mu_W1;
    
    % Solve Wave Dynamical System in LGN
    LGN.u = LGN.S*LGN.H + LGN.S_d*LGN.y;    
    LGN.v = RK4(@(v)(-1/tau_v *v + LGN.u ),  dt,LGN.v);
    LGN.th = RK4(@(th)(1/tau_th*(v_th-th).*(1-LGN.H) + th_plus*LGN.H),  dt,LGN.th);
    
    % Discontinuous Update rule in LGN
    fireL = find(LGN.v >= LGN.th);
    LGN.v(fireL) = LGN.v_reset(fireL);
    LGN.H = sparse(zeros(nL,1));
    LGN.H(fireL,1) = ones(length(fireL),1);
    
    % Inputs to V1
    V1.x = W2' * LGN.H;
    
    % V1 Activation function
    V1.y = max(V1.x - V1.thresh, 0); % amount by how much input signal lies over thresh % ReLU activation function
    [wink, maxInd] = maxk(V1.y,40); % ALTERNATIVELY: find Index of k best performers
    
    % V1 Competition rule
    V1.y(V1.y<wink(end)) = 0;     % ALTERNATIVELY: allow k best performers to participate
    V1.y(maxInd) = (V1.y(maxInd)/(wink(1)+eps)).^4 .*V1.y(maxInd);
    
    % Assemble system
    V1.y = sparse(V1.y); % Sparsity declaration for memory/speed   
    
    % Solve W2 Matrix dynamical system & Update thresh
    W2 = W2 + dt*(V1.eta_learn* LGN.H*V1.y'); 
    V1.thresh = V1.thresh + dt*(0.005*V1.y);
    W2(:,maxInd) = W2(:,maxInd)./mean(W2(:,maxInd))*mu_W2;

    if mod(tt,1) == 0
        disp(tt)
        Xv(:,tt+1) = Ret.v; Xu(:,tt+1) = Ret.u; Xth(:,tt+1) = Ret.th;
        Lv(:,tt+1) = LGN.v; Lu(:,tt+1) = LGN.u; Lth(:,tt+1) = LGN.th; Ly(:,tt+1) = LGN.y;        
        firedMat{tt+1} = fired; fireLMat{tt+1} = fireL;
        %heatMap_wave(fired) = heatMap_wave(fired)+1;
        figure(3), 
        subplot(1,2,1), plot(1:nL,LGN.y,1:nL,LGN.thresh), title(['LGN Spikes t=' num2str(tt)]), ylim([0,4000])
        subplot(1,2,2), plot(1:nV,V1.y,1:nV,V1.thresh), title(['V1 Spikes t=' num2str(tt)]), ylim([0,2000])
%         figure(7), plot(1:nL,LGN.S_d*LGN.y, 1:nL,LGN.S*LGN.H)
        if mod(tt,100) == 0
            figure(4), 
            subplot(1,2,1), h1 = pcolor(W1); title(['W1 Matrix t=' num2str(tt)]), set(h1,'EdgeColor','none'),set(gca,'Ydir','reverse'),colormap cool;
            subplot(1,2,2), h2 = pcolor(W2); title(['W2 Matrix t=' num2str(tt)]), set(h2,'EdgeColor','none'),set(gca,'Ydir','reverse'),colormap jet;
        end
    end
end
toc
disp('Spatio-temp wave computed') 
figure(1),subplot(1,3,1),plot(t_intrvls(1:end-1),Xv(1:40:end,:),'-o');                          title('Ret.v')
        subplot(1,3,2),plot(t_intrvls(1:end-1),Xu(1:40:end,:),'-o',t_intrvls(1:end),fnoise,'+');title('Ret.u')
        subplot(1,3,3),plot(t_intrvls(1:end-1),Xth(1:40:end,:),'-o');                           title('Ret.th')
figure(2),subplot(1,3,1),plot(t_intrvls(1:end-1),Lv(1:20:end,:),'-o');                          title('LGN.v')
        subplot(1,3,2),plot(t_intrvls(1:end-1),Lu(1:20:end,:),'-o');                            title('LGN.u')
        subplot(1,3,3),plot(t_intrvls(1:end-1),Lth(1:20:end,:),'-o');                           title('LGN.th')

%% Visualization of Wave
for ii = 1:length(t_intrvls)-1
    figure(5); title(['Ret-Wave t = ' num2str(ii)])
    hold on
    scatter(Ret.nx(:,2),Ret.nx(:,1),'k','filled')
    scatter(Ret.nx(firedMat{ii},2),Ret.nx(firedMat{ii},1),'r','filled')
    
    figure(6), subplot(1,2,1), scatter(LGN.nx(:,2),LGN.nx(:,1),[],Ly(:,ii),'filled'), axis square, title('y_{LGN}'), ...
               subplot(1,2,2), hold on, axis square, scatter(LGN.nx(:,2),LGN.nx(:,1),'k','filled'),...
                                                     scatter(LGN.nx(fireLMat{ii},2),LGN.nx(fireLMat{ii},1),'r','filled'),title(['LGN-Wave t = ' num2str(ii)])
    %{    
          gifname2='wave.gif';
          frame = getframe(gcf); 
          im = frame2im(frame); 
          [imind,cm] = rgb2ind(im,256);

          % Write to the GIF File 
          if ii==1
              imwrite(imind,cm,gifname2,'gif','DelayTime',0.1,'Loopcount',Inf); 
          else 
              imwrite(imind,cm,gifname2,'gif','DelayTime',0.1,'WriteMode','append'); 
          end 
    %}
end