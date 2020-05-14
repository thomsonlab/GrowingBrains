function dydt = ode_RetWave(t,y,Ret,fnoise,nR,dt,fired)
%     fired = find(y(1:nR,1) >= 30);
%     y(fired,1) = Ret.c(fired);
%     y(fired+nR+1,1) = Ret.u(fired) + Ret.d(fired);
    Ret.v = y(1:nR,1); Ret.u = y(nR+1:2*nR,1);
    Ret.eta = fnoise;
%     tt = round(t/dt)+1; Ret.eta = fnoise(:,tt);
    
    dydt(1:nR,1) = 0.04* Ret.v.^2 + 5* Ret.v + 140 - Ret.u ... 
               + sum(Ret.S(:,fired),2) + Ret.eta;
%     dydt(fired,1) = 1000*(Ret.c(fired) - Ret.v(fired))/dt;
    dydt(nR+1:2*nR,1) = Ret.a.* (Ret.b.*Ret.v - Ret.u);
%     dydt(fired+nR+1,1) = Ret.d(fired);
end