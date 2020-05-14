function dydt = ode_HebLGN(t,y,Ret,LGN,nR,nL)
%     %Inputs
%     W1 = reshape(y(1:nR*nL,1),[nR,nL]);
%     thresh = y((nR*nL+1):(nR+1)*nL ,1); beta = 0.005; %threshold increase time constant
% %     thresh = LGN.threshold;
%     LGN.x = W1' * Ret.H; % H(v_i(t)-30)*W 
% %     LGN.activity(end+1,:) = LGN.x; % Snapshot matrix of inputs to all LGN units over time
% 
%     %Activation function
%     LGN.y = max(LGN.x - thresh, 0); % amount by how much input signal lies over thresh % ReLU activation function
%     [win, maxInd] = max(LGN.y);     % find Index of >> WINNER << 
% %     [wink, maxkInd] = maxk(LGN.y,10); % ALTERNATIVELY: find Index of k best performers
%     
%     %Competition rule
%     LGN.y(LGN.y<win) = 0;         % only allow >> WINNER << to participate
% %     LGN.y(LGN.y<wink(end)) = 0;     % ALTERNATIVELY: allow k best performers to participate    
% 
%     %Assemble system
%     LGN.y = sparse(LGN.y); % Sparsity declaration for memory/speed
    f_Heb = LGN.eta_learn* Ret.H*LGN.y';
%     f_Heb = LGN.eta_learn* kron(LGN.y,Ret.H);

    % Modifying thresh! Increase thresh with increasing spike activity
%     dydt = [f_Heb(:); beta*LGN.y];
    dydt = f_Heb(:);

%     % Update/Keep track of number of times a particular winner unit was updated!
%     LGN.synapticChanges(maxInd) = LGN.synapticChanges(maxInd) + 1;

%     % Normalize weights to a constant strength
%     W1(:,maxInd) = W1(:,maxInd)/mean(W1(:,maxInd))*mu_wts;
end