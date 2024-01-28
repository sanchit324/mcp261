function y=Timer()
Infinity = 1000000;
global Clock             % simulation clock
global NextFailure      % time of next failure event
global NextRepair       % time of next repair event

% Determine the next event and advance time
    if NextFailure < NextRepair
        y = 'Failure';
        Clock = NextFailure;
        NextFailure = Infinity;
    else
        y = 'Repair';
        Clock = NextRepair;
        NextRepair = Infinity;
    end
end