%% Dynamical Systems Form of Spiking NN Developmental algorithm %%clear
clc
% clear

%% System Constants and Parameters %%
nR = 1600;      % # Neurons in Retina
nL = 200;       % # Neurons in LGN
sqrL = 20;      % Domain length of Retina
ri = 2; ro = 5; % Inner & outer rad of wave

%% Retina Structure Parameters %%
Ret = {};       % Retina Data Structure
Ret.x = sqrL*rand(nR,2);
centroid_RGC = mean(Ret.x); dist_center_to_all = pdist2(Ret.x, centroid_RGC); gaussian_val = 6*exp(-(dist_center_to_all)/10);

Ret.a = 0.02*ones(nR,1);
Ret.b = 0.2*ones(nR,1);
Ret.c = -65 + 15*rand(nR,1).^2;     %Noise on activity field
Ret.d = 8 - gaussian_val;

Ret.D = squareform(pdist(Ret.x));   %Distance matrix
Ret.S = 40*(Ret.D < ri)- 15*(Ret.D > ro).*exp(-Ret.D / 10); 
Ret.S = Ret.S - diag(diag(Ret.S));  %Adjacency matrix between Neurons

Ret.v = -65*ones(nR,1); %I.C. of v
Ret.u = Ret.b .* Ret.v; %I.C. of u
Ret.eta = [];
Ret.hist = [];

%% LGN Structure Parameters %%
LGN = {};           % LGN Data Structure
LGN.eta_learn = 0.1;
LGN.synapticChanges = zeros(nL,1);
LGN.threshold = normrnd(70,2,nL,1); % Normal distribution around 70 +- 2 (std. dev)
LGN.activity = [];

% Synaptic Weight Matrix
mu_W1 = 2.5;
sigma_W1 = 0.14;
W1 = normrnd(mu_W1, sigma_W1, [nR,nL]);
W1 = W1./mean(W1)*mu_W1;    % Weight Matrix between Retina-LGN1 normrnd initialized

heatMap_wave = zeros(nR,1); % # of times each neuron spikes
rfSizes = [150:50:750]';    % counter for size of receptive field

Tend = 8e2; dt = 0.1; t_intrvls = 0:Tend;
Ret.v = -65*ones(nR,1); Ret.u = Ret.b .* Ret.v; %I.C. of v, u
Xu = zeros(nR,t_intrvls(end)); Xu(:,1) = Ret.u; %Snapshot matrix u
Xv = zeros(nR,t_intrvls(end)); Xv(:,1) = Ret.v; %Snapshot matrix v
% fnoise = 3*randn(nR,length(t_intrvls)); % Pre-generate noise
fnoiseI = interp1(t_intrvls',fnoise',[0:dt:Tend]','linear')';
% fnoise = randn(nR,1).*ones(nR,length(t_intrvls)); % Pre-generate noise

fired = [];
firedMat = cell(1,length(t_intrvls));
tic
for tt = 0:dt:t_intrvls(end-1)
%{
    if mod(tt,1000)==0
        % After every 1 sec (1000 ms) -- update the threshold to prevent random shifting % LGN_params.threshold
        for i = 1:nL            
            if LGN.synapticChanges(i) < 200
                LGN.threshold(i) = max(LGN.activity(:,i))*1/5; %
            end        
        end
        LGN.activity = max(LGN.activity); 
        disp(tt)
    end
%}
%     Ret.eta = fnoiseI(:,round(tt/dt+1)); 
    Ret.eta = fnoise(:,round(tt)+1);
%{
    IC1 = [Ret.v; Ret.u]; timespan = [tt:0.1:tt+1]; dt=1; 
    options = odeset('RelTol',1e-3,'AbsTol',1e-6); %,'Stats','on','OutputFcn',@odeplot); warning off;
    tic, [~,uv] = ode45(@(t,y) ode_RetWave(t,y,Ret,fnoise(:,round(tt/dt)+1),nR,dt,fired), timespan, IC1, options); toc
    Ret.v = uv(end,1:nR)'; Ret.u = uv(end,nR+1:2*nR)';
%}    
    % Discontinious Update rule
    fired = find(Ret.v >= 30);
    Ret.v(fired) = Ret.c(fired);
    Ret.u(fired) = Ret.u(fired) + Ret.d(fired);
    Ret.H = sparse(zeros(nR,1)); % equivalent to "spikeMat"
    Ret.H(fired,1) = ones(length(fired),1);   
%{
    % Solve Wave Dynamical System
    Ret.v = Ret.v + dt*(0.04* Ret.v.^2 + 5* Ret.v + 140 - Ret.u ... 
               + sum(Ret.S(:,fired),2) + Ret.eta);
    Ret.u = Ret.u + dt*(Ret.a.* (Ret.b.*Ret.v - Ret.u));
%}
    Ret.v = RK4(@(v)(0.04* v.^2 + 5* v + 140 - Ret.u ... 
                + Ret.S*Ret.H + Ret.eta),   dt,Ret.v);
    Ret.u = RK4(@(u)(Ret.a.* (Ret.b.*Ret.v - u)),   dt,Ret.u);    
%{
    v_dot =@(v)(0.04* v.^2 + 5* v + 140 - Ret.u ... 
                + Ret.S*Ret.H + Ret.eta); %insert function to be solved
        k1 = dt*v_dot(Ret.v);
        k2 = dt*v_dot(Ret.v+.5*k1);
        k3 = dt*v_dot(Ret.v+.5*k2);
        k4 = dt*v_dot(Ret.v+k3);
    Ret.v = Ret.v+(k1+2*k2+2*k3+k4)/6;
    
    u_dot =@(u)(Ret.a.* (Ret.b.*Ret.v - u)); %insert function to be solved
        k1 = dt*u_dot(Ret.u);
        k2 = dt*u_dot(Ret.u+.5*k1);
        k3 = dt*u_dot(Ret.u+.5*k2);
        k4 = dt*u_dot(Ret.u+k3);
    Ret.u = Ret.u+(k1+2*k2+2*k3+k4)/6;
%}    
    %Inputs to LGN
    LGN.x = W1' * Ret.H; % H(v_i(t)-30)*W 
%     LGN.activity(end+1,:) = LGN.x; % Snapshot matrix of inputs to all LGN units over time

    %LGN Activation function
    LGN.y = sparse(max(LGN.x - LGN.threshold, 0)); % amount by how much input signal lies over thresh % ReLU activation function
%     [win, maxInd] = max(LGN.y);     % find Index of >> WINNER << 
    [wink, maxInd] = maxk(LGN.y,10); % ALTERNATIVELY: find Index of k best performers
    
    %LGN Competition rule
%     LGN.y(LGN.y<win) = 0;         % only allow >> WINNER << to participate
    LGN.y(LGN.y<wink(end)) = 0;     % ALTERNATIVELY: allow k best performers to participate
    LGN.y(maxInd) = (LGN.y(maxInd)/(wink(1)+eps)).^4 .*LGN.y(maxInd);
    
    %Assemble system
%     LGN.y = sparse(LGN.y); % Sparsity declaration for memory/speed
%{    
    IC2 = [W1(:)]; timespan = [tt:0.1:tt+1]; dt=1;
    tic, [~,w] = ode45(@(t,y) ode_HebLGN(t,y,Ret,LGN,nR,nL), timespan, IC2, options); toc
    W1 = reshape(w(end,1:nR*nL)',[nR,nL]); tmp2 = w(end,(nR*nL+1):(nR+1)*nL)';
%}    
    
    % Solve Matrix dynamical system & Update threshold
    W1 = W1 + dt*(LGN.eta_learn* Ret.H*LGN.y'); 
    LGN.threshold = LGN.threshold + dt*(0.005*LGN.y);
    W1(:,maxInd) = W1(:,maxInd)./mean(W1(:,maxInd))*mu_W1;
    if mod(tt,1) == 0
        disp(tt); 
        Xv(:,tt+1) = Ret.v;
        Xu(:,tt+1) = Ret.u;
        firedMat{tt+1} = fired;
        heatMap_wave(fired) = heatMap_wave(fired)+1;
        figure(3), plot(1:nL,LGN.y,1:nL,LGN.threshold), title(['Spikes t=' num2str(tt)]), ylim([0,4000])
        if mod(tt,50) == 0
            figure(4), h = pcolor(W1); title(['Weight Matrix t=' num2str(tt)]), set(h,'EdgeColor','none'),set(gca,'Ydir','reverse'),colormap cool;
        end
    end
end
toc
disp('Spatio-temp wave computed') 
figure(1),plot(t_intrvls(1:end-1),Xv(1:8:end,:),'-o');%ylim([-80,-20]), 
figure(5),plot(t_intrvls(1:end-1),Xu(1:16:end,:),'-o',t_intrvls(1:end),fnoise,'+');

for ii = 1:length(firedMat)
    figure(2); title(['Wave t = ' num2str(ii)])
    hold on
    scatter(Ret.x(:,2),Ret.x(:,1),'k','filled')
    scatter(Ret.x(firedMat{ii},2),Ret.x(firedMat{ii},1),'r','filled')
end
