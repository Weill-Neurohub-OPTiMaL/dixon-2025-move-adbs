%% Function Name: json_to_csv()
%
% Description: Converts json files to csv for python import.
%
% Inputs:
%     file : string
%         Time-Domain data, given in internal RC+S units.
%
% Outputs:
%     writes csv to file in the same location as the json
%
% Author: Tanner Chas Dixon, tanner.dixon@ucsf.edu.
% Date last updated: July 12, 2022
%---------------------------------------------------------

function [] = json_to_csv(log_segments)

session_number = input('Please enter the session ID: ');

data_dir = uigetdir;

% Load data and event timestamps
[~,...
~, ~, ~,...
~, ~, ~,...
~, ~, ~,...
~, ~, ~,...
AdaptiveData, ~, ~,...
~, ~, ~, ~,...
~, ~, ~, ~,...
~, AdaptiveStimSettings, ~,...
~] = ProcessRCS(data_dir, 2);

% log timestamp time-series
timestamp = AdaptiveData.newDerivedTime/1000;

% log session ID's
session_id = session_number*ones(size(timestamp));

% log block ID's
block_id = -ones(size(timestamp));

end_idx = length(timestamp);
block_start_ts = AdaptiveStimSettings.HostUnixTime( ...
    strcmp(AdaptiveStimSettings.adaptiveMode, 'Embedded'))/1000;
if length(block_start_ts) < 3
    error('Error: Not enough block transitions detected.')
end
block_edge_idx = [find(timestamp>block_start_ts(end-2), 1), ...
                  find(timestamp>block_start_ts(end-1), 1), ...
                  find(timestamp>block_start_ts(end), 1), ...
                  end_idx];
if any(diff(timestamp(block_edge_idx)) < 300)
    error(['Error: At least one of the blocks registered less than',... 
           ' five minutes duration.'])
end

block_stim_targets = [AdaptiveStimSettings.states(...
    strcmp(AdaptiveStimSettings.adaptiveMode, 'Embedded')...
    ).state0_AmpInMilliamps];
block_stim_targets = block_stim_targets(length(block_stim_targets) ...
                                        -[11,7,3]);
block_order = [];
for low_state_target = block_stim_targets
    if low_state_target < 1.9
        block_order = [block_order, 1];
    elseif low_state_target > 2
        block_order = [block_order, 2];
    else
        block_order = [block_order, 3];
    end
end

for block = 1:3
    s = block_edge_idx(block);
    e = block_edge_idx(block+1) - 1;
    block_id(s:e) = block_order(block);
end

% log segment ID's
segment_id = repmat({'none'}, size(timestamp));
if log_segments
    event_ts = readtable(fullfile(data_dir, 'event_ts.csv'));
    for segment = 1:height(event_ts)
        % block 1
        segment_mask = create_segment_mask(timestamp, ...
                                           event_ts.start_c1(segment), ...
                                           event_ts.end_c1(segment));
        if ~all(block_id(segment_mask) == 1)
            error(['Error: The ', event_ts.task{segment}, ' window for ', ...
                   'block 1 has been assigned the incorrect block ID.'])
        end
        segment_id(segment_mask) = event_ts.task(segment);
        % block 2
        segment_mask = create_segment_mask(timestamp, ...
                                           event_ts.start_c2(segment), ...
                                           event_ts.end_c2(segment));
        if ~all(block_id(segment_mask) == 2)
            error(['Error: The ', event_ts.task{segment}, ' window for ', ...
                   'block 2 has been assigned the incorrect block ID.'])
        end
        segment_id(segment_mask) = event_ts.task(segment);
        % block 3
        segment_mask = create_segment_mask(timestamp, ...
                                           event_ts.start_c3(segment), ...
                                           event_ts.end_c3(segment));
        if ~all(block_id(segment_mask) == 3)
            error(['Error: The ', event_ts.task{segment}, ' window for ', ...
                   'block 3 has been assigned the incorrect block ID.'])
        end
        segment_id(segment_mask) = event_ts.task(segment);
    end
end

% log LD feature inputs
pb_input = AdaptiveData.Ld0_featureInputs;
pb0 = pb_input(:,1);
pb1 = pb_input(:,2);
pb2 = pb_input(:,3);
pb3 = pb_input(:,4);

% log LD output time-series
output = AdaptiveData.Ld0_output;
output(output > (2^31)) = output(output > (2^31)) - 2^32;
output = output/(2^10);

% log state time-series
state = zeros(length(timestamp), 1);
for i = 1:length(timestamp)
    if strcmp(AdaptiveData.CurrentAdaptiveState{i}, 'No State')
        state(i) = -1;
    else
        state(i) = str2num(AdaptiveData.CurrentAdaptiveState{i}(7));
    end
end

% log stim time-series
stim = AdaptiveData.CurrentProgramAmplitudesInMilliamps(:,1);

% create table
valid_mask = (block_id > -1) & (stim < 3);
adaptive_table = table(timestamp, session_id, block_id, segment_id,...
                       pb0, pb1, pb2, pb3, output, state, stim);
adaptive_table = adaptive_table(valid_mask,:);

% write table to csv
writetable(adaptive_table, fullfile(data_dir, 'adaptive_table.csv'))

end


function segment_mask = create_segment_mask(timestamp, s, e)
segment_mask = false(size(timestamp));
segment_mask((timestamp>=s) & (timestamp<=e)) = true;
end
