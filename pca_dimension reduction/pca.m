load mnist_uint8.mat;
X = double(train_x(1:1000, :)');
y = train_y(1:1000, :);
[~, label] = max(y, [], 2);
n = size(X, 2);                  %size里1是行数，2是列数
k = 256;                         %设置降维后维度

% 求均值 中心化
m = mean(X, 2);                   %mean里1是列数，2是行数
X_cen = X - m* ones(1, n);        

% 求协方差阵
C = X_cen * X_cen' / n;

% 特征值分解 排序
[V, D] = eig(C);
[V_sort, index] = sort(diag(V), 'descend');

% 构成变换矩阵
W_pca = D(:,index(1:k));

%降维后数据
X_l = W_pca' * X_cen; 

%绘制图形
b = X_cen(:, 1)';
b = reshape(b, [28,28]);
b_l = X_l(:, 1)';
b_l = reshape(b_l, [16,16]);

figure(1);
subplot(1, 2, 1);
imshow(b,'InitialMagnification','fit');
subplot(1, 2, 2);
imshow(b_l,'InitialMagnification','fit');


figure(2);
X_plot = X_l';
p = tsne(X_plot,'Algorithm','barneshut','Distance', 'cosine');
gscatter(p(:,1), p(:,2), label);