% testando o algoritmo implementado para
% o método dos gradientes conjugados

matriz = rand(3,3);
A_ = matriz'*matriz;
x0 = [1;1;1];
b_ = [-1;-1;3];

x_estrela = grad_conj(b_, A_, x0);
disp(A_);
disp(x_estrela);
disp(A_ * x_estrela);
disp('-------\n');

% Importando os dados
load 'C:\Users\B43642\Downloads\Data82.mat' A m;
N = sqrt(size(A,2));
H_ = A'*A;
% disp(size(H_));
x_zero = ones(size(H_, 2), 1);
% disp(size(x_zero));
b_ = A'*m(:);
x_min = grad_conj(b_, H_, x_zero);
disp(x_min);

imagesc(reshape(x_min, N, N));

function x_ = grad_conj(b, H, x_0)
    x_cur = x_0;
    epsilon = 0.001; % mudar isso aqui caso necessário
    k_max = 1000; % mudar caso necessário tbm
    r = b - (H*x_0);
    ro = r'*r;
    k = 1;
    while (sqrt(ro)>epsilon*norm(b,2) && k < k_max)
        if (k == 1)
            d_cur = r;
        else
            d_last = d_cur;
            beta = (d_last'*H*r)/(d_last'*H*d_last);
            d_cur = r + beta*d_last;
        end
        w = H*d_cur;
        alpha = (r'*d_cur)/(d_cur'*w);
        x_cur = x_cur + alpha*d_cur;
        r = b - H*x_cur;
        ro = r'*r;
        k = k+1;
    end
    x_ = x_cur;
end

