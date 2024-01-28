global Clock             % simulation clock
global NextFailure      % time of next failure event
global NextRepair       % time of next repair event
global S                % system state
global Slast            % previous value of the system state
global Tlast            % time of previous state change
global Area             % area under S(t) curve

% Program to generate a sample path for the TTF example

    Infinity = 1000000;
    s=rng('default');
    
% Define and initialize replication variables
    SumS = 0;
    SumY = 0;
    
    for Rep = 1:60
    
% Initialize the state and statistical variables
        S = 2;
        Slast = 2;
        Clock = 0;
        Tlast = 0;
        Area = 0;
    
% Schedule the initial failure event
        NextFailure = ceil(6 * rand());
        NextRepair = Infinity;
    
% Advance time and execute events until the system fails
        while ~(S == 0)
            NextEvent = Timer;
            switch NextEvent
                case 'Failure'
                    Failure;
                case 'Repair'
                    Repair;
            end
        end
    
% Accumulate replication statistics
        SumS = SumS + Area / Clock;
        SumY = SumY + Clock;
        Store_S(Rep, 1) = Area / Clock;
        Store_Y(Rep, 1) = Clock;
    end
% Display output
    msgbox (['Average failure at time ', num2str(SumY / 100) , ...
        ' with average # functional components ' , num2str(SumS / 100)]);
    
    xlswrite('matresults.xls', Store_S, 'Sheet1');
