
function draw()
    [X, Y, tt] = get_data('new.txt', 0);
    hold on
    f2 = plot(X,Y,'Color',[0.75,0.75,1]);
    f1 = plot(X,smooth(Y),'Color',[0,0,1]);
    title(tt)
    hold off
    grid on
end

% 0 - loss
% 1 - mae
% 2 - rmse
function [X, Y, title] = get_data(path, type)
    data = importdata(path);
    X = data(:,1)';
    if type == 0
        Y = data(:,2)';
        title = 'loss'
    end
    if type == 1
        Y = data(:,3)';
        title = 'mae'
    end
    if type == 2
        Y = data(:,4);
        title = 'rmse'
    end
end