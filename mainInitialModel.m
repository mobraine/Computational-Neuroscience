    clear all;
    

    % statement of work: I did parameter tuning by hand. TA has helped me
    % understand different curves and their contribution to the final
    % current. Then by calculation and observation I found some appropriate ranges of
    % parameter values. For example, decrease v0i will always decrease steady xi
    % at -40 mV and -20 mV. Depending on the value of v0i, e.g. -20mV >v0i >
    % -40mV, decrease bi will decrease steady xi at -40mV but increase
    % steady xi at -20mV. I try to calculate the solution by calculating
    % steady current for -40mV and -20mV, but the systems of equations are not enough to solve
    % for the solution. Through these kinds of observation and
    % calculation, I got to the current model fit.

    % load data
    load('CookAssignemnt1UnknownCurrent.mat') 

    % Model parameters
    ba = .3;  % steady state channels open per mV
    v0a = -40 % -40 -48;  % mV
    taua = 2.2;  % msec

    gBar = .002;  % mS
    Er = 12;   % mV
    
    bi = .2 % steady state channels close per mV
    v0i = -30 % -41;  % mV
    taui = 3;  % msec

    figure(1);
    clf;

    % Plot steady state curves
    
    % steady state activation curves
    subplot(6,2,1);
    hold on;
    grid on;
    v = -80 : 20;
    plot(v, twoParamSig(v, [ba v0a]), 'LineWidth', 1.5);
    % xi activation curve
    plot(v, 1 - twoParamSig(v, [bi v0i]), 'LineWidth', 1.5);
    xlabel('mV');
    legend('xa(t = inf)', 'xi(t = inf)', 'Location', 'best');

    % model parameters
    subplot(6,2,2);
    hold on;
    grid on;
    set(gca, 'Visible', 'off');
    text(0,0,['ba = ' num2str(ba)]);
    text(0,1,['v0a = ' num2str(v0a) ' mV']);
    text(0,2,['taua = ' num2str(taua) ' msec']);
    text(0,3,['gBar = ' num2str(gBar) ' mS']);
    text(0,4,['Er = ' num2str(Er) ' mV']);
    text(0,5,['gBar = ' num2str(gBar) ' mS']);
    
    text(0,6,['bi = ' num2str(bi)]);
    text(0,7,['v0i = ' num2str(v0i) ' mV']);
    text(0,8,['taui = ' num2str(taui) ' msec']);


    ylim([0 6]);
    xlim([0 4]);
             
    % run v-clamp simulation
    
    dt = t(2) - t(1);

    for vStepIndex = 1 : size(vStep,2)
        
        v = vStep(:,vStepIndex);
        xa = zeros(size(v));  g = zeros(size(v));  i = zeros(size(v));
        xi = zeros(size(v));
        
        for j = 1 : length(t)
            xaInf = twoParamSig(v(j), [ba v0a]);
            xiInf = 1 - twoParamSig(v(j), [bi v0i]);
            if j > 1
                % advance channel activation values for each time point
                xa(j) = xa(j-1) + (xaInf - xa(j-1)) * (1 - exp(-dt/taua));
                xi(j) = xi(j-1) + (xiInf - xi(j-1)) * (1 - exp(-dt/taui));
            else
                % first time point, we assume channel activation is at steady state
                xa(j) = xaInf;
                xi(j) = xiInf;
            end
            g(j) = gBar * xa(j) * xi(j);
            i(j) = g(j) * (v(j) - Er); 
        end
        subplot(6,1,2);
        hold on;
        grid on;
        plot(t,v, 'LineWidth', 1.5);
        axis tight;
        ylabel('V step (mV)');

        subplot(6,1,3);
        hold on;
        grid on;
        plot(t,xa, 'LineWidth', 1.5);
        axis tight;
        ylabel('xa');


        subplot(6,1,4);
        hold on;
        grid on;
        plot(t,xi, 'LineWidth', 1.5);
        axis tight;
        ylabel('xi');

        subplot(6,1,5);
        hold on;
        grid on;
        plot(t,g, 'LineWidth', 1.5);
        axis tight;
        ylabel('g (mS)');

        subplot(6,1,6);
        hold on;
        grid on;
        % plot true current
        pData = plot(t,iUnknownCurrent(:,vStepIndex), 'LineWidth', 2);
        % plot model current
        ax = gca;
        ax.ColorOrderIndex = vStepIndex; % set color to same as true current
        pModel = plot(t,i, '--', 'LineWidth', 2);
        axis tight;
        xlabel('msec');
        ylabel('i (mA)'); 
        legend([pData pModel], 'Data', 'Model', 'Location', 'best');
    end
    


    
function y = twoParamSig(x, params)
    
    b = params(1);
    x0 = params(2);
    
    y = 1 ./ (1 + exp(-b .* (x - x0)));
end      
    