clear, clc

k = 5;
dim = [2, 3, 5];
order = 4:9;
cond_number = [1.8069, 2.0763, 2.3053, 2.4049, 3.0920, 3.3831;
    1.7945, 1.9920, 2.2369, 2.4164, 2.7822, 3.3816;
    1.8160, 2.0519, 2.3532, 2.7239, 3.2440, 3.7248];
colors = ['r', 'b', 'k'];

for i = 1:3
    d = dim(i);
    plot(order, cond_number(i, :), '.-', 'Color', colors(i), 'DisplayName', ['d=' num2str(d)]); hold on;
    
end
plot(order, 1 + (order)./k, '--k', 'DisplayName', '$1 + \frac{p}{k}$');

legend('Location','northwest', 'Interpreter', 'latex', 'FontSize',14);

