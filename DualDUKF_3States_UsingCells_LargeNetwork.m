clear all
% close all
clc
%Initialize
L=1;
m=10;
% q=3; %output dimension
n=3;
p=3;% number of parameters
T=800;
kappa=1;
x=[1;1;1];
x_hat(:,1,1)=[1.5;1.5;0];
x_hat(:,1,2)=[1;1;-3];
x_hat(:,1,3)=[-1.5;-1.5;0];
x_hat(:,1,4)=[-1.5;0;-1];
x_hat(:,1,5)=[1;1;1];
x_hat(:,1,6)=[1.1;1;-1];
x_hat(:,1,7)=[1;-1;1];
x_hat(:,1,8)=[1.8;1;.1];
for i=9:m
    x_hat(:,1,i)=[2;1;.1];
end

%th=[0.3;1.5;1];
th=[0.3;.5;.1];
th_hat(:,1,1)=[0.2;.1;.1];%[.2;2;.5];
th_hat(:,1,2)=[.1;.1;.5];%[.1;2;1];
th_hat(:,1,3)=[0;.8;.1];%[.5;1;1.5];
th_hat(:,1,4)=[0.2;2;.15];%[.7;.7;0];
th_hat(:,1,5)=[0.2;.1;.1];%[.2;2;.5];
th_hat(:,1,6)=[.2;.1;.5];%[.1;2;1];
th_hat(:,1,7)=[0;.8;.1];%[.5;1;1.5];
th_hat(:,1,8)=[0.2;.1;.5];%[.7;.7;0];
for i=9:m
    th_hat(:,1,i)=[0.2;2;.5];
end
for i=1:m
P(:,:,1,i)=eye(n);
end
for i=1:m
P_th(:,:,1,i)=eye(n);
end

Q=1e-7*eye(n);%(.01^2)*eye(n); %.01^2
delQ=0;%1e-8*eye(n);
delQ_th=0;%1e-8*eye(p);%eye(p);
Q_th=1e-7*eye(p);%(.02^2)*eye(p);%.02^2

% C(:,:,1)=[0,0,0;0,1,1;0,0,1]; %[1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1]; %Measurment matrix
% C(:,:,2)=[1,1,0;0,1,0;0,0,0];
% C(:,:,3)=[1,0,0;0,0,0;1,0,1];
% C(:,:,4)=[1,1,0;0,0,0;0,0,1];

% C1=[0,0,0;0,1,1;0,0,1]; %[1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1]; %Measurment matrix
% C2=[1,1,0;0,1,0;0,0,0];
% C3=[1,0,0;0,0,0;1,0,1];
% C4=[1,1,0;0,0,0;0,0,1];

% C1=[1,0,0];
% C2=[0,1,0];
% C3=[0,1,0];
% C4=[1,0,0];
% C5=[0,1,0];
% C6=[0,0,1];
% C7=[1,0,0];
% C8=[0,1,0];

% c1=eig(C1*C1');
% C3*C3';
% c3=eig(C3*C3');

C1=[1,0,0;-1,0,1];
C2=[0,1,0;0,1,1];
C3=[0,1,0;0,0,1];
C4= [1,1,0;1,0,0];
C5=[0,1,0;1,1,0];
C6=[0,0,1;1,0,1];
C7=[1,0,0;0,1,0];
C8=[0,1,0;0,0,1];

C={C1;C2;C3;C4;C5;C6;C7;C8};
C_test=[];
for i=9:m
    C{i}=[0,1,1;1,0,1];
   
    %C{i}=[0,0,1];
end

% for i=1:m
%      C_test=[C_test;C{i}];
% end
% E=C;

% C_test=[];
% 
% for i=1:m
%     
%  C{i}=rand(2,n);
% % C{i}=randi([1,2],1,n);
% C_test=[C_test;C{i}];
% 
% end
% 
% Ccond=eig(C_test*C_test')

% %  E=C;
% E_test=[];
% for i=1:m
% E{i}=rand(2,p);
% % E{i}=randi([1,2],2,p);
% E_test=[E_test;E{i}];
% 
% end
% Econd=eig(blkdiag(C_test,E_test)*blkdiag(C_test,E_test)')
%%
% C_test=[[1,0,0;0,1,0];[0,1,0];[0,0,1];[-1,0,1]];
%C_test=[C1;C2;C3;C4;C5;C6;C7;C8];
% rank(C_test)
R={};

for i=1:m
    q(i)=size(C{i},1); 
    if m<9
    R(:,:,i)={1e-2*eye(q(i))};%{((0.01*i)^2)*eye(q(i))};
    else
    R(:,:,i)={1e-2*eye(q(i))};%{((0.01*8)^2)*eye(q(i))};
    end
end
delR=0;%1e-1;
%  R_th(:,:,1:m)={(0.01^2)*eye(p)};
%   R_th=R;


% R(:,:,1)={(0.01^2)*eye(q(1))};
% R(:,:,2)={(0.02^2)*eye(q(2))};
% R(:,:,3)={(0.03^2)*eye(q(3))};
% R(:,:,4)={(0.04^2)*eye(q(4))};
%Adjacency Matrix
% adj=[1,1,1,0;
%      1,1,1,0;
%      1,1,1,1;
%      0,0,1,1];

% First Strongly Connected Graph
% adj=[1,0,0,0,0,0,1,0;
%      1,1,0,0,0,0,0,0;
%      0,1,1,0,0,0,0,0;
%      0,0,1,1,0,0,0,0;
%      0,0,0,1,1,0,0,1;
%      0,0,0,0,1,1,0,0;
%      0,0,0,0,0,1,1,0;
%      1,0,0,0,0,0,0,1];
 
%  adj=[1,0,0,0,1,0,1,0;
%      1,1,0,0,1,0,0,0;
%      0,1,1,0,1,0,0,0;
%      0,0,1,1,0,0,1,0;
%      0,1,0,1,1,0,0,0;
%      0,0,0,0,1,1,0,1;
%      0,0,0,1,0,1,1,0;
%      1,0,1,0,1,0,0,1];

adj0=ones(m,m);
adj=tril(adj0)-tril(adj0,-2)+ triu(adj0)-triu(adj0,2)-eye(m,m);
% adj(m,m-1)=0;
adj(m,1)=1;
adj(1,m)=1;
adj(1,m-1)=1;
adj(m-1,1)=1;
adj(5,m)=1;
adj(m,5)=1;
%
%Metropolis Weights
 for i=1:m
     sumw(i)=0;
     degi=length(find(adj(:,i)));
     for j=1:m
     degj=length(find(adj(j,:)));
     if adj(i,j)~=0 && i~=j
         weights(i,j)=1/(max(degi,degj));
         sumw(i)=sumw(i)+weights(i,j);
     else
         weights(i,j)=0;
     end
     
     end
     weights(i,i)=1-sumw(i);
 end
 adj=weights.*adj;

% % Uniform Weights(Column Stochastic)
% for j=1:m
% degi=length(find(adj(:,j)));
% for i=1:m
%     adj(i,j)=adj(i,j)/degi;
% end
% end

% B=[0;1;1];
% u=0;
iterations=1;
 for ind=1:iterations
 z_hat={};
  z={};
  K={};
  Pzz={};
  Pxz={};
  Fprod_eig_th=[];
  Hprod_eig_th={};
  errors=zeros(m*n,1);
error_norm_sq=zeros(T-1,1);
for t=2:T
    u=sin(t-1);
    noise_w=0*sqrt(Q)*randn(n,1);
    noise_th=0*sqrt(Q_th)*randn(p,1);
    th(:,t)=th(:,t-1)+noise_th;
    x(:,t)=f_function_input(x,th,t,u)+noise_w;
    
 
    
    for i=1:m
        noise_v=sqrt(R{:,:,i})*randn(q(i),1);
       
%         noise_v_th=sqrt(R_th{:,:,i})*randn(q(i),1);
        
        z(:,t,i)={C{i}*x(:,t)+noise_v};
        %z_th(:,t,i)={E{i}*th(:,t)+noise_v_th};
        
        sigma(:,t-1,i,1)=x_hat(:,t-1,i);
    
        square_root(:,:)=(chol((n+kappa)*P(:,:,t-1,i))); % Or use Cholesky decompositions
        for s=2:n+1
            sigma(:,t-1,i,s)=x_hat(:,t-1,i)+square_root(:,s-1);
        end
        for s=n+2:2*n+1
            sigma(:,t-1,i,s)=x_hat(:,t-1,i)-square_root(:,s-n-1);
        end
        for s=1:2*n+1
        
        func_sigma=(f_function_input(sigma(:,:,i,s),th_hat(:,:,i),t,u));% [th_hat(1,t-1,i)*cos(sigma(1,t-1,i,s));th_hat(2,t-1,i)*sin(sigma(2,t-1,i,s))^2;th_hat(1,t-1,i)*cos(sigma(3,t-1,i,s))^2];
        sigma_predict(:,t-1,i,s)=func_sigma;
        
        end
        W(1)=(kappa/(n+kappa));
        for s=2:2*n+1
            W(s)=1/(2*(n+kappa));
        end
        
        sumx=0;
        sumP=0;
        for s=1:2*n+1
            sumx=sumx+W(s)*sigma_predict(:,t-1,i,s);     
        end
        x_hat_predict(:,t-1,i)=sumx;
        for s=1:2*n+1
            sumP=sumP+W(s)*(sigma_predict(:,t-1,i,s)-sumx)*...
                (sigma_predict(:,t-1,i,s)-sumx)';
            
            gamma(:,t,i,s)={C{i}*sigma_predict(:,t-1,i,s)};
        end
        sumP=sumP+Q+delQ;
        P_predict(:,:,t-1,i)=sumP;
        
        
        %%%%% Extra: Not part of aglorithm%%%%%%%%%%%%%%%%%
        sumPx_xpred=0;
        for s=1:2*n+1
            sumPx_xpred=sumPx_xpred+W(s)*(sigma(:,t-1,i,s)-x_hat(:,t-1,i))*...
                (sigma_predict(:,t-1,i,s)-sumx)';
        end
        sumPx_xpred=sumPx_xpred+ Q+delQ;
        F(:,:,t-1,i)=sumPx_xpred*P(:,:,t-1,i)^-1;
      
        Fprod_eig(:,t-1,i)=eig(F(:,:,t-1,i)*F(:,:,t-1,i)');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        sumz=0;
       for s=1:2*n+1
           sumz=sumz+W(s)*gamma{:,t,i,s};
       end
       
       z_hat(:,t,i)={sumz};
       sumPzz=0;
       sumPxz=0;
       sumtest=0;
       for s=1:2*n+1
           sumPzz=sumPzz+W(s)*((gamma{:,t,i,s}-z_hat{:,t,i})...
               *(gamma{:,t,i,s}-z_hat{:,t,i})');
   
           sumPxz=sumPxz+ W(s)*(sigma_predict(:,t-1,i,s)...
               -x_hat_predict(:,t-1,i))*(gamma{:,t,i,s}-z_hat{:,t,i})';
       end
       
%        sumPzz=sumtest;
       sumPzz=sumPzz+R{:,:,i}+delR;
       Pzz(:,:,t,i)={sumPzz};
       Pxz(:,:,t,i)={sumPxz};
       K(:,:,t,i)={sumPxz*(sumPzz)^(-1)};
        
    end
    
    
    
  
    for i1=1:m
        K_i=K{:,:,t,i1};
        P_i_zz=Pzz{:,:,t,i1};
        
        x_hat(:,t,i1)=x_hat_predict(:,t-1,i1)+ K_i*(z{:,t,i1}-z_hat{:,t,i1});
        
        P_i=P_predict(:,:,t-1,i1)-K_i*(P_i_zz)*K_i';
        x_hatL(:,i1,1)=x_hat(:,t,i1);
        P_L(:,:,i1,1)=P_i;
    end
    
    for l=1:L
        for i=1:m
            
            neighbors_i=find(adj(i,:));

            sumx=zeros(n,1);
            sumP=0;
            for j=neighbors_i
               sumx=sumx+adj(i,j)*x_hatL(:,j,l);
               sumP=sumP+adj(i,j)*P_L(:,:,j,l);
            end
           
            x_hatL(:,i,l+1)=sumx;
            P_L(:,:,i,l+1)=sumP;
            
            
        end
    end
    for i=1:m
       
        x_hat(:,t,i)=x_hatL(:,i,L+1);
        P(:,:,t,i)=P_L(:,:,i,L+1);
    end
    
    
%     [th_hat,P_th,Fprod_eig_th,Hprod_eig_th]=distributedUKF_Parameters_withCells(th_hat,z,P_th,R,Q_th,C,adj,L,m,q,p,kappa,t,x_hat,delQ_th,Fprod_eig_th,Hprod_eig_th,u,delR);
        [th_hat,P_th,Hprod_eig_th]=distributedUKF_Parameters_CellsandInputs(th_hat,z,P_th,R,Q_th,C,adj,L,m,p,kappa,t,x_hat,delQ_th,u,@f_function_input);
        

    augP=[];
    for i=1:m
        errors((i*n-n)+1:i*n)=x(:,t)-x_hat(:,t,i);
        augP=blkdiag(augP,P(:,:,t,i));
    end
%     lyap_func(t)=errors'*(augP)^-1*errors;
    sumLyap=0;
    for i=1:m
    sumLyap=sumLyap+.5*norm(x(:,t)-x_hat(:,t,i))^2;
    end
    lyap_func(t)=sumLyap;
    
    sumLyap_th=0;
    for i=1:m
    sumLyap_th=sumLyap_th+.5*norm(th(:,t)-th_hat(:,t,i))^2;
    end
    lyap_func_th(t)=sumLyap_th;
    
    
    error_norm_sq(t-1)=norm(errors)^2;
    xi(ind,t-1)=error_norm_sq(t-1);
end

 end
 for t=1:T-1
 xi_mean(t)=mean(xi(:,t));
 end
%%
% for t=2:100
% noise_w=Q*randn(2,1);
%     x(:,t)=[0.3*x(1,t-1)+x(2,t-1)^2;(1.5-x(1,t-1))*x(2,t-1)]+noise_w;
% end
 %legend({'$x$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$'},'Interpreter','latex')

%close all
figure
plot(1:T,lyap_func)
title('$V_{x}(t)$','Interpreter','latex')
figure
plot(1:T,lyap_func_th)
title('$V_{\theta}(t)$','Interpreter','latex')
%%

figure
plot(1:T,x)
ylabel('States')
legend('State 1','State 2','State 3')
figure
hold
%%

% plot(1:T,x(1,:))
% plot(1:T,x_hat(1,:,1))
% plot(1:T,x_hat(1,:,5))
% plot(1:T,x_hat(1,:,10))
% ylabel('State')
% legend('Actual','Agent 1','Agent 5',' Agent 10')
% title('State 1')
% 
% figure
% hold
% plot(1:T,x(2,:))
% plot(1:T,x_hat(2,:,1))
% plot(1:T,x_hat(2,:,5))
% plot(1:T,x_hat(2,:,10))
% ylabel('State')
% title('State 2')
% legend('Actual','Agent 1','Agent 5',' Agent 10')
% figure
% hold
% plot(1:T,x(3,:))
% plot(1:T,x_hat(3,:,1))
% plot(1:T,x_hat(3,:,5))
% plot(1:T,x_hat(3,:,10))
% ylabel('State')
% title('State 3')
% legend('Actual','Agent 1','Agent 5',' Agent 10')
% 
% figure
% hold
% plot(1:T,th(1,:))
% plot(1:T,th_hat(1,:,1))
% plot(1:T,th_hat(1,:,5))
% plot(1:T,th_hat(1,:,10))
% ylabel('Parameter')
% title('Parameter 1')
% legend('Actual','Agent 1','Agent 5',' Agent 10')
% 
% figure
% hold
% plot(1:T,th(2,:))
% plot(1:T,th_hat(2,:,1))
% plot(1:T,th_hat(2,:,5))
% plot(1:T,th_hat(2,:,10))
% ylabel('Parameter')
% title('Parameter 2')
% legend('Actual','Agent 1','Agent 5',' Agent 10')
% 
% figure
% hold
% plot(1:T,th(3,:))
% plot(1:T,th_hat(3,:,1))
% plot(1:T,th_hat(3,:,5))
% plot(1:T,th_hat(3,:,10))
% ylabel('Parameter')
% title('Parameter 3')
% legend('Actual','Agent 1','Agent 5',' Agent 10')
% figure
% 
% 
% plot(1:T-1,xi_mean)
% ylabel('mean$\{||e||^2\}$','Interpreter','latex')
% xlabel('Time Step')
% title('Mean squared state estimation error')

%%
figure
plot(1:T,x-x_hat(:,:,2))
xlabel('Time Step')
ylabel('Error')

% figure
% hold
% plot(1:T,x(1,:))
% plot(1:T,x_hat(1,:,1))
% plot(1:T,x_hat(1,:,10))
% plot(1:T,x_hat(1,:,20))
% ylabel('State')
% legend('Actual','Agent 1','Agent 10',' Agent 20')
% title('State 1')
% 
% figure
% hold
% plot(1:T,x(2,:))
% plot(1:T,x_hat(2,:,1))
% plot(1:T,x_hat(2,:,10))
% plot(1:T,x_hat(2,:,20))
% ylabel('State')
% title('State 2')
% legend('Actual','Agent 1','Agent 10',' Agent 20')
% figure
% hold
% plot(1:T,x(3,:))
% plot(1:T,x_hat(3,:,1))
% plot(1:T,x_hat(3,:,10))
% plot(1:T,x_hat(3,:,20))
% ylabel('State')
% title('State 3')
% legend('Actual','Agent 1','Agent 10',' Agent 20')
% 
% figure
% hold
% plot(1:T,th(1,:))
% plot(1:T,th_hat(1,:,1))
% plot(1:T,th_hat(1,:,10))
% plot(1:T,th_hat(1,:,20))
% ylabel('Parameter')
% title('Parameter 1')
% legend('Actual','Agent 1','Agent 10',' Agent 20')
% 
% figure
% hold
% plot(1:T,th(2,:))
% plot(1:T,th_hat(2,:,1))
% plot(1:T,th_hat(2,:,10))
% plot(1:T,th_hat(2,:,20))
% ylabel('Parameter')
% title('Parameter 2')
% legend('Actual','Agent 1','Agent 10',' Agent 20')
% 
% figure
% hold
% plot(1:T,th(3,:))
% plot(1:T,th_hat(3,:,1))
% plot(1:T,th_hat(3,:,10))
% plot(1:T,th_hat(3,:,20))
% ylabel('Parameter')
% title('Parameter 3')
% legend('Actual','Agent 1','Agent 10',' Agent 20')
% figure


% plot(1:T,(x-x_hat(:,:,1))./x)
% ylabel('Error')
% title('Agent 1')
% figure
% plot(1:T,(x-x_hat(:,:,2))./x)
% ylabel('Error')
% title('Agent 2')
% figure
% plot(1:T,x_hat(:,:,3))
% ylabel('Error')
% title('Agent 3')
% figure
% plot(1:T,x-x_hat(:,:,4))
% ylabel('Error')
% title('Agent 4')
% figure
% plot(1:T,x-x_hat(:,:,5))
% ylabel('Error')
% title('Agent 5')
% figure
% plot(1:T,x-x_hat(:,:,6))
% ylabel('Error')
% title('Agent 6')
% figure
% plot(1:T,(x-x_hat(:,:,7)))
% ylabel('Error')
% title('Agent 7')
% figure
% plot(1:T,(x(:,:)-x_hat(:,:,8))./x)
% ylabel('Error')
% title('Agent 8')

% figure
% plot(1:t-1,Fprod_eig(:,:,1))
% figure
% plot(1:t-1,Fprod_eig_th(:,:,1))
% figure
% plot(1:t,Hprod_eig_th(:,:,1))
% figure

% plot(1:T,x(1,:),1:T,x_hat(1,:,1),1:T,x_hat(1,:,2),1:T,x_hat(1,:,3),1:T,x_hat(1,:,4))
% xlabel('Time Step')
% ylabel('State 1')
% legend({'$x$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$'},'Interpreter','latex')
% figure
% plot(1:T,x(2,:),1:T,x_hat(2,:,1),1:T,x_hat(2,:,2),1:T,x_hat(2,:,3),1:T,x_hat(2,:,4))
% xlabel('Time Step')
% ylabel('State 2')
% legend({'$x$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$'},'Interpreter','latex')
% figure
% plot(1:T,x(3,:),1:T,x_hat(3,:,1),1:T,x_hat(3,:,2),1:T,x_hat(3,:,3),1:T,x_hat(3,:,4))
% xlabel('Time Step')
% ylabel('State 3')
% legend({'$x$','$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$'},'Interpreter','latex')
% figure
% plot(1:T,th(1,:),1:T,th_hat(1,:,1),1:T,th_hat(1,:,2),1:T,th_hat(1,:,3))
% xlabel('Time Step')
% ylabel('Parameter 1')
% legend({'$\theta$','$\hat{\theta}_1$','$\hat{\theta}_2$','$\hat{\theta}_3$'},'Interpreter','latex')
% figure
% plot(1:T,x(1,:)-x_hat(1,:,1),1:T,x(1,:)-x_hat(1,:,2),1:T,x(1,:)-x_hat(1,:,3),1:T,x(1,:)-x_hat(1,:,4))
% xlabel('Time Step')
% ylabel('State 1 Estimation Error')
% legend({'$\hat{x}_1$','$\hat{x}_2$','$\hat{x}_3$','$\hat{x}_4$'},'Interpreter','latex')


%     sumx=0;
%     sumP=0;
%     for s=1:2*n+1
%         sumx=sumx+W(s)*sigma_predict(:,t,i,s);
%     end
%     x_hat_predict(:,t,i)=sumx;
%     for s=1:2*n+1
%         sumP=sumP+W(s)*(sigma_predict(:,t,i,s)-sumx)*...
%             (sigma_predict(:,t,i,s)-sumx)'+ Q;
%     end
%     P_predict(:,t,i)=sumP;