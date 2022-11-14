% Importando os dados
load 'C:\Users\B43642\Downloads\Data82.mat' A m;

N = sqrt(size(A,2));

% Apenas resolvendo o sistema
tic;
x1 = A\m(:);
t1 = toc;

% Usando mínimos quadrados
tic;
ATA =  A'*A;
b_ = A'*m(:);
x2 = ATA\b_;
t2 = toc;

% Usando método dos gradientes conjugados
% Com x_0 = (1,1,...,1)
tic;
H_ = A'*A;
x_zero = ones(size(H_, 2), 1);
b_ = A'*m(:);
x3 = grad_conj(b_, H_, x_zero);
t3 = toc;

% Usando método dos gradientes conjugados
% Com x_0 = (0,0,...,0)
tic;
H_ = A'*A;
x_zero = zeros(size(H_, 2), 1);
b_ = A'*m(:);
x4 = grad_conj(b_, H_, x_zero);
t4 = toc;

% Usando a função pcg
% Sem o parâmetro de regularização
tic;
fun1 = @(x) A'*(A*x);
b_ = A'*m(:);
x5 = pcg(fun1,b_);
t5 = toc;

% Com o parâmetro de regularização
tic;
alpha = 10;  % parâmetro de regularização
fun2 = @(x) A'*(A*x) + alpha*x;
b_ = A'*m(:);
x6 = pcg(fun2,b_);
t6 = toc;

disp('Tempo apenas resolvendo o sistema:');disp(t1);
disp('Tempo usando mínimos quadrados:'); disp(t2);
disp('Tempo método dos gradientes conjugados com x_0 = (1,...,1):'); disp(t3);
disp('Tempo método dos gradientes conjugados com x_0 = (0,...,0):'); disp(t4);
disp('Tempo função pcg sem parâmetro de regularização:'); disp(t5);
disp('Tempo função pcg com parâmetro de regularização:'); disp(t6);
%imagesc(reshape(x, N, N));

figure
subplot(2,3,1)
imagesc(reshape(x1, N, N))
colormap bone
axis square
axis off
title({'Resolvendo diretamente o sistema'; strcat('(',string(t1),' s)')})
subplot(2,3,2)
imagesc(reshape(x2, N, N))
colormap bone
axis square
axis off
title({'Usando mínimos quadrados'; strcat('(',string(t2),' s)')})
subplot(2,3,3)
imagesc(reshape(x3, N, N))
colormap bone
axis square
axis off
title({'Usando o método dos gradientes'; 'conjugados com x_0 = (1,...,1)'; strcat('(',string(t3),' s)')})
subplot(2,3,4)
imagesc(reshape(x4, N, N))
colormap bone
axis square
axis off
title({'Usando o método dos gradientes'; 'conjugados com x_0 = (0,...,0)'; strcat('(',string(t4),' s)')})
subplot(2,3,5)
imagesc(reshape(x5,N,N))
colormap bone
axis square
axis off
title({'Usando a função pcg'; 'sem parâmetro de regularização'; strcat('(',string(t5),' s)')})
subplot(2,3,6)
imagesc(reshape(x6,N,N))
colormap bone
axis square
axis off
title({'Usando a função pcg'; 'com parâmetro de regularização'; strcat('(',string(t6),' s)')})

% Funções
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
