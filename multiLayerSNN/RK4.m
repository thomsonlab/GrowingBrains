function dydt = RK4(y_dot,dt,IC)
    k1 = dt*y_dot(IC);
    k2 = dt*y_dot(IC+.5*k1);
    k3 = dt*y_dot(IC+.5*k2);
    k4 = dt*y_dot(IC+k3);
    dydt = IC+(k1+2*k2+2*k3+k4)/6;
end

% h = 0.05;  % time step size
% t = 0:h:10;  % time interval of t
% y = zeros(1,length(t));
% y(1) = 1;   % IC value for y
% n = length(t)-1;
% y_dot =@(t,y)(- y +1 - dirac(t-1)); %insert function to be solved
% for i = 1:n
%     k1 = h*y_dot(t(i),y(i));
%     k2 = h*y_dot(t(i)+.5*h,y(i)+.5*k1);
%     k3 = h*y_dot(t(i)+.5*h,y(i)+.5*k2);
%     k4 = h*y_dot(t(i)+h,y(i)+k3);
%     y(i+1) = y(i)+(k1+2*k2+2*k3+k4)/6;
% end
% [t,y_check] = ode45(y_dot,t,y(1));
% figure; 
% plot(t,y,t,y_check,'-.'); title('ode45 Check')