function [x_hat,P,Fprod_eig,Hprod_eig]=distributedUKF_Parameters_withCells(x_hat,z,P,R,Q,C,adj,L,m,q,n,kappa,t,state,delQ,Fprod_eig,Hprod_eig,u,delR)


    for i=1:m
        
        sigma(:,t-1,i,1)=x_hat(:,t-1,i);
     
        square_root(:,:)=(chol((n+kappa)*P(:,:,t-1,i))); % Or use Cholesky decompositions
        for s=2:n+1
        sigma(:,t-1,i,s)=x_hat(:,t-1,i)+square_root(:,s-1);
        end
        for s=n+2:2*n+1
        sigma(:,t-1,i,s)=x_hat(:,t-1,i)-square_root(:,s-n-1);
        end
        for s=1:2*n+1
        
        func_sigma=sigma(:,t-1,i,s);
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
                (sigma_predict(:,t-1,i,s)-sumx)'+ Q+delQ;
            state_pred=f_function_input(state(:,:,i),sigma_predict(:,:,i,s),u,t);%[sigma_predict(1,t-1,i,s)*state(1,t-1)+state(2,t-1)^2+state(3,t-1)^2;(sigma_predict(2,t-1,i,s)-state(1,t-1))*state(2,t-1);sigma_predict(3,t-1,i,s)*state(3,t-1)^2];
          
            gamma(:,t,i,s)={C{i}*state_pred};
        end
        P_predict(:,:,t-1,i)=sumP;
%         display(P_predict(:,:,t-1,i),'P_predict')
        
        sumPx_xpred=0;
        for s=1:2*n+1
            sumPx_xpred=sumPx_xpred+W(s)*(sigma(:,t-1,i,s)-x_hat(:,t-1,i))*...
                (sigma_predict(:,t-1,i,s)-sumx)'+ Q+delQ;
        end
        F(:,:,t-1,i)=sumPx_xpred*P(:,:,t-1,i)^-1;
      
        Fprod_eig(:,t-1,i)=eig(F(:,:,t-1,i)*F(:,:,t-1,i)');
        
        sumz=0;
       for s=1:2*n+1
           sumz=sumz+W(s)*gamma{:,t,i,s};
       end
       z_hat(:,t,i)={sumz};
       sumPzz=0;
       sumPxz=0;
       for s=1:2*n+1
           sumPzz=sumPzz+W(s)*(gamma{:,t,i,s}-z_hat{:,t,i})...
               *(gamma{:,t,i,s}-z_hat{:,t,i})'+ R{:,:,i}+delR;
           sumPxz=sumPxz+ W(s)*(sigma_predict(:,t-1,i,s)...
               -x_hat_predict(:,t-1,i))*(gamma{:,t,i,s}-z_hat{:,t,i})';
       end
       Pzz(:,:,t,i)={sumPzz};
       Pxz(:,:,t,i)={sumPxz};
       
   
       H(:,:,t,i)={Pxz{:,:,t,i}'*P_predict(:,:,t-1,i)^(-1)};
       Hprod_eig(:,t,i)={eig(H{:,:,t,i}*H{:,:,t,i}')};
       
    
       K(:,:,t,i)={sumPxz*(sumPzz)^(-1)};
        
    end
    
    
    
  
    for i1=1:m
%         display(i1,'i')
        K_i=K{:,:,t,i1};
        P_i_zz=Pzz{:,:,t,i1};
        x_hat(:,t,i1)=x_hat_predict(:,t-1,i1)+ K_i*(z{:,t,i1}-z_hat{:,t,i1});
       
       
        P_i=P_predict(:,:,t-1,i1)-K_i*(P_i_zz)*K_i';
        x_hatL(:,i1,1)=x_hat(:,t,i1);
        P_L(:,:,i1,1)=P_i;
%          
%          display(P_L(:,:,i1,1),'P_l')
    end
    
    for l=1:L
        for j=1:m
            
            neighbors_j=find(adj(:,j));
            sumx=zeros(n,1);
            sumP=0;
            for i=neighbors_j'
               sumx=sumx+adj(i,j)*x_hatL(:,i,l);
               sumP=sumP+adj(i,j)*P_L(:,:,i,l);
            end
            x_hatL(:,j,l+1)=sumx;
            P_L(:,:,j,l+1)=sumP;
           
        end
    end
    for i=1:m
        x_hat(:,t,i)=x_hatL(:,i,L+1);
        P(:,:,t,i)=P_L(:,:,i,L+1);  
    end
    
end