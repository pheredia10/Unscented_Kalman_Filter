% function y=f_function(x,th,t)
% 
% y=[th(1)*x(1,t-1)+x(2,t-1)^2+x(3,t-1)^2;(th(2)-x(1,t-1))*x(2,t-1);th(3)*x(3,t-1)^2];
% end

% Working Function
function y=f_function_input(x,th,t,u)

% y=5.*[th(1,t-1)*cos(x(1,t-1))*x(3,t-1)^2;th(2,t-1)*sin(x(2,t-1))^2;th(3,t-1)*cos(x(3,t-1))^2];


y=5*[th(1,t-1)*cos(x(1,t-1));th(2,t-1)*cos(x(2,t-1))^2;th(3,t-1)*sin(x(3,t-1))^2]+[sin(u);cos(u);u];
% y=[th(1,t-1)*cos(x(1,t-1));th(2,t-1)*cos(x(2,t-1))^2;th(3,t-1)*cos(x(3,t-1))^2];

 
 %y=[-th(1,t-1)*x(1,t-1);-th(2,t-1)*x(2,t-1);-th(3,t-1)*x(3,t-1)];
end