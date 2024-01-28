function Repair()
global Clock             % simulation clock
global NextFailure      % time of next failure event
global NextRepair       % time of next repair event
global S                % system state
global Slast            % previous value of the system state
global Tlast            % time of previous state change
global Area             % area under S(t) curve
% Repair event
% Update state and schedule future events
    S = S + 1;
    if (S == 1)
        NextRepair = Clock + 2.5;
        NextFailure = Clock + ceil(6 * rand());
    end
    
% Update area under the S(t) curve
    Area = Area + Slast * (Clock - Tlast);
    Tlast = Clock;
    Slast = S;
end
