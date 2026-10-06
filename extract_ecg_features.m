% =========================================================================
% AUTOMATED ECG FEATURE EXTRACTION PIPELINE (PARALLELIZED)
% =========================================================================
% Objective: Robust, production-grade MATLAB script to automate ECG feature 
% extraction using ECGdeli (v1.1) with parallel execution and WFDB support.
% =========================================================================

%% 1. Path Optimization & Environment Setup
clear; clc; close all;
fprintf('Initializing Environment...\n');

% Dynamically bind the entire ECGdeli-1.1 toolbox
ecgdeli_path = 'C:\MATLAB\ECGdeli-1.1';
if exist(ecgdeli_path, 'dir')
    addpath(genpath(ecgdeli_path));
    fprintf('SUCCESS: ECGdeli toolbox paths dynamically bound on client.\n');
else
    error('FATAL: ECGdeli path not found: %s', ecgdeli_path);
end

% Dynamically bind the WFDB toolbox
% NOTE: Restored to third_party because the wfdb-software-package-10.7.0 folder 
% contains the C-core, while the MATLAB wrapper (rdsamp.m) lives here:
wfdb_path = 'C:\Users\Acer\Desktop\Capstone\Capstone\third_party\wfdb\mcode';
if exist(wfdb_path, 'dir')
    addpath(wfdb_path);
    fprintf('SUCCESS: WFDB toolbox paths dynamically bound on client.\n');
else
    error('FATAL: WFDB path not found: %s', wfdb_path);
end

% Setup Parallel Pool and push paths to workers
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(4); 
end
parfevalOnAll(pool, @() addpath(genpath(ecgdeli_path)), 0);
parfevalOnAll(pool, @() addpath(wfdb_path), 0);
fprintf('SUCCESS: Toolbox paths pushed to parallel workers.\n');

%% 2. Directory Configuration
data_root_dir = 'C:\Users\Acer\Desktop\Capstone\Capstone\data';
mapping_file = fullfile(data_root_dir, 'raw', 'processed', 'ecg_mapping.csv');
features_file = fullfile(data_root_dir, 'raw', 'final_features_to_extract.txt');
output_file = fullfile(data_root_dir, 'extracted_ecg_features.csv');

%% 3. Input Parsing: Feature Filtering Matrix
fprintf('Parsing input targets and metadata...\n');
if exist(features_file, 'file')
    fileID = fopen(features_file, 'r');
    parsed_features = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    features_to_extract = string(parsed_features{1});
    % Removed makeValidName sanitization to perfectly preserve target feature strings
    fprintf('SUCCESS: Loaded %d target feature identifiers.\n', length(features_to_extract));
else
    error('FATAL: Features target file not found: %s', features_file);
end

%% 4. Index-Based File Traversal Setup
if exist(mapping_file, 'file')
    % Ingest 'ecg_mapping.csv'
    mapping_metadata = readtable(mapping_file, 'PreserveVariableNames', true);
    fprintf('SUCCESS: Loaded mapping metadata containing %d records.\n', height(mapping_metadata));
    
    % Validate necessary columns exist
    csv_cols = mapping_metadata.Properties.VariableNames;
    if ~ismember('subject_id', csv_cols) || ~ismember('waveform_path', csv_cols)
        error('FATAL: mapping CSV missing required columns "subject_id" or "waveform_path". Found: %s', strjoin(csv_cols, ', '));
    end
else
    error('FATAL: Mapping file not found: %s', mapping_file);
end

%% 4.5 Auto-Resume State Detection
if exist(output_file, 'file')
    fprintf('Resume state detected. Analyzing existing output file...\n');
    existing_data = readtable(output_file, 'PreserveVariableNames', true);
    if ismember('SubjectID', existing_data.Properties.VariableNames)
        completed_ids = existing_data.SubjectID;
        % Filter out completed IDs from mapping_metadata
        % Force consistent string typing to prevent CSV numeric-conversion mismatches (e.g. leading zeros)
        map_ids_str = strip(string(mapping_metadata.subject_id));
        comp_ids_str = strip(string(completed_ids));
        idx_to_keep = ~ismember(map_ids_str, comp_ids_str);
        mapping_metadata = mapping_metadata(idx_to_keep, :);
        fprintf('SUCCESS: Filtered out %d already-processed records.\n', sum(~idx_to_keep));
    end
end

num_records = height(mapping_metadata);
fprintf('Total remaining records to process: %d\n', num_records);

%% 5. Core Processing Pipeline & High-Availability Execution
fprintf('\n=======================================================\n');
fprintf('STARTING PARALLEL BATCH PROCESSING PIPELINE...\n');
fprintf('=======================================================\n');

batch_size = 1000; % Reduced for higher fault-tolerance during initial full-scale run
num_batches = ceil(num_records / batch_size);

% Pre-compute constant feature mapping outside of worker loops
f_extract_str = string(features_to_extract);

for batch_idx = 1:num_batches
    batch_start = (batch_idx - 1) * batch_size + 1;
    batch_end = min(batch_idx * batch_size, num_records);
    current_batch_size = batch_end - batch_start + 1;
    
    fprintf('\n--- Processing Batch %d of %d (Records %d to %d) ---\n', ...
        batch_idx, num_batches, batch_start, batch_end);
    
    % Slice the mapping metadata for the current batch
    batch_mapping = mapping_metadata(batch_start:batch_end, :);
    
    % Extract table columns to standard arrays to allow parfor slicing
    subject_ids = batch_mapping.subject_id;
    waveform_paths = batch_mapping.waveform_path;
    
    % Preallocate sliced arrays for parfor
    master_features = nan(current_batch_size, numel(features_to_extract));
    master_subject_ids = strings(current_batch_size, 1);
    successful_iterations = false(current_batch_size, 1);

    parfor i = 1:current_batch_size
    try
        % --- A. Traverse file references to resolve full system paths ---
        subject_id = subject_ids(i); 
        rel_file_path = waveform_paths(i);
        
        if iscell(rel_file_path), rel_file_path = rel_file_path{1}; end
        if iscell(subject_id), subject_id = subject_id{1}; end
        
        % The rel_file_path from MIMIC-IV usually omits the extension (.dat/.hea)
        % Note: The dataset sits inside 'raw' folder
        full_file_path = fullfile(data_root_dir, 'raw', char(rel_file_path));
        
        % --- B. Extract raw waveforms from WFDB format ---
        % Explicitly check for existence so missing files throw a clean MATLAB error, not a cryptic WFDB C-core crash
        if ~exist([full_file_path '.hea'], 'file') && ~exist([full_file_path '.dat'], 'file')
            error('WFDB record not found at: %s', full_file_path);
        end
        
        % Workaround for Windows: WFDBs C-core often misinterprets absolute paths with C:\ as URLs.
        % We cd into the directory on the worker, read by basename, and cd back.
        [record_dir, record_name, ~] = fileparts(full_file_path);
        
        old_pwd = pwd;
        try
            cd(record_dir);
            [ecg_signal, Fs, ~] = rdsamp(record_name);
            cd(old_pwd);
        catch wfdb_exception
            cd(old_pwd);
            rethrow(wfdb_exception);
        end
        
        % --- C. ECGdeli Integration Pipeline ---
        % NOTE: The actual ECGdeli API uses independent matrix functions rather than a struct.
        % 1. Noise-Mitigation / Filtering
        [ecg_filtered_baseline, ~] = ECG_Baseline_Removal(ecg_signal, Fs, 1, 0.5);
        [ecg_filtered_frq] = ECG_High_Low_Filter(ecg_filtered_baseline, Fs, 1, 40);
        ecg_filtered_frq = Notch_Filter(ecg_filtered_frq, Fs, 50, 1);
        [ecg_filtered_isoline, ~, ~, ~] = Isoline_Correction(ecg_filtered_frq);
        
        % 2. Full Morphological Cycle Delineations (Fiducial Point Table)
        [FPT_MultiChannel, FPT_Cell] = Annotate_ECG_Multi(ecg_filtered_isoline, Fs);
        
        % 3. Structural Feature Calculus
        [Amplitude_feature_12leads] = ExtractAmplitudeFeaturesFromFPT(FPT_Cell, ecg_filtered_isoline);
        [Timing_feature_12leads, Timing_feature_sync] = ExtractIntervalFeaturesFromFPT(FPT_Cell, FPT_MultiChannel);
        
        % --- Feature Aggregation ---
        row = nan(1, numel(features_to_extract));
        
        lead_names = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'};
        amp_names = {'P_Amp', 'Q_Amp', 'R_Amp', 'S_Amp', 'T_Amp'};
        
        % 1. Global Timing Features
        if ~isempty(Timing_feature_sync)
            qrs_vals = Timing_feature_sync(:, 2);
            qrs_vals = qrs_vals(~isnan(qrs_vals));
            if ~isempty(qrs_vals)
                idx = find(f_extract_str == "QRS_Dur_Global"); if ~isempty(idx), row(idx) = mean(qrs_vals); end
                idx = find(f_extract_str == "QRS_Dur_Global_iqr"); if ~isempty(idx), row(idx) = iqr(qrs_vals); end
                idx = find(f_extract_str == "QRS_Dur_Global_count"); if ~isempty(idx), row(idx) = length(qrs_vals); end
            end
            
            rr_vals = Timing_feature_sync(:, 8);
            rr_vals = rr_vals(~isnan(rr_vals));
            if ~isempty(rr_vals)
                idx = find(f_extract_str == "RR_Mean_Global"); if ~isempty(idx), row(idx) = mean(rr_vals); end
                idx = find(f_extract_str == "RR_Mean_Global_iqr"); if ~isempty(idx), row(idx) = iqr(rr_vals); end
                idx = find(f_extract_str == "RR_Mean_Global_count"); if ~isempty(idx), row(idx) = length(rr_vals); end
            end
        end
        
        % 2. Lead-wise Amplitude Features
        if ~isempty(Amplitude_feature_12leads)
            num_leads = min(length(lead_names), size(Amplitude_feature_12leads, 1));
            
            % Hoist FPT array concatenation out of the lead loop for efficiency
            FPT_mat = cat(3, FPT_Cell{:});
            
            for l_idx = 1:num_leads
                lname = lead_names{l_idx};
                for a_idx = 1:length(amp_names)
                    aname = amp_names{a_idx};
                    vals = squeeze(Amplitude_feature_12leads(l_idx, :, a_idx));
                    vals = vals(~isnan(vals));
                    
                    base_feat = string(sprintf('%s_%s', aname, lname));
                    
                    if ~isempty(vals)
                        idx = find(f_extract_str == base_feat); if ~isempty(idx), row(idx) = mean(vals); end
                        idx = find(f_extract_str == base_feat + "_iqr"); if ~isempty(idx), row(idx) = iqr(vals); end
                        idx = find(f_extract_str == base_feat + "_count"); if ~isempty(idx), row(idx) = length(vals); end
                    else
                        idx = find(f_extract_str == base_feat + "_count"); if ~isempty(idx), row(idx) = 0; end
                    end
                end
                
                % 3. ST_Elev Computation (J-point + 60ms minus QRS Onset baseline)
                st_feat = string(sprintf('ST_Elev_%s', lname));
                if size(FPT_mat, 3) >= l_idx
                    qrs_onsets = squeeze(FPT_mat(:, 4, l_idx));
                    j_points = squeeze(FPT_mat(:, 8, l_idx));
                    
                    st_vals = nan(size(j_points));
                    for b = 1:length(j_points)
                        q_on = qrs_onsets(b);
                        j_pt = j_points(b);
                        if ~isnan(q_on) && ~isnan(j_pt) && q_on > 0 && j_pt > 0
                            % J+60ms offset
                            j_plus = min(j_pt + round(0.06 * Fs), size(ecg_filtered_isoline, 1));
                            baseline = ecg_filtered_isoline(q_on, l_idx);
                            st_voltage = ecg_filtered_isoline(j_plus, l_idx);
                            st_vals(b) = st_voltage - baseline;
                        end
                    end
                    st_vals = st_vals(~isnan(st_vals));
                    
                    if ~isempty(st_vals)
                        idx = find(f_extract_str == st_feat); if ~isempty(idx), row(idx) = mean(st_vals); end
                        idx = find(f_extract_str == st_feat + "_iqr"); if ~isempty(idx), row(idx) = iqr(st_vals); end
                        idx = find(f_extract_str == st_feat + "_count"); if ~isempty(idx), row(idx) = length(st_vals); end
                    else
                        idx = find(f_extract_str == st_feat + "_count"); if ~isempty(idx), row(idx) = 0; end
                    end
                else
                    idx = find(f_extract_str == st_feat + "_count"); if ~isempty(idx), row(idx) = 0; end
                end
            end
        end
        
        % --- E. Aggregate ---
        master_features(i, :) = row;
        master_subject_ids(i) = string(subject_id);
        successful_iterations(i) = true;
        
        if mod(i, 100) == 0
            fprintf('[SUCCESS] %d/%d processed in current batch\n', i, current_batch_size);
        end
        
    catch exception
        % --- High-Availability Trap ---
        fprintf('[WARNING] Subject %s failed (idx %d): %s\n', ...
            string(subject_ids(i)), i, exception.message);
    end
    end % end parfor
    
    %% 6. Structured Vector Storage & Export for Batch
    fprintf('Isolating successful records and exporting batch...\n');

    % Reconstruct the master table from the parfor arrays
    master_table = array2table(master_features, 'VariableNames', features_to_extract);
    master_table.SubjectID = master_subject_ids;

    % Filter down to only successfully processed rows
    master_table = master_table(successful_iterations, :);

    if height(master_table) > 0
        % Reorder SubjectID to be the first column
        master_table = movevars(master_table, 'SubjectID', 'Before', 1);
        
        write_mode = 'append';
        if ~exist(output_file, 'file')
            write_mode = 'overwrite';
        end
        writetable(master_table, output_file, 'WriteMode', write_mode);
        fprintf('SUCCESS: Batch %d exported to:\n%s\n', batch_idx, output_file);
    else
        warning('No viable ECG features were extracted in batch %d.', batch_idx);
    end
    
    % Clear batch variables from memory to prevent RAM bloat
    clear master_features master_subject_ids successful_iterations master_table batch_mapping subject_ids waveform_paths;
end % end outer batch loop

fprintf('=======================================================\n');
fprintf('ALL BATCHES COMPLETE.\n');
fprintf('=======================================================\n\n');

%% 7. Resource Cleanup
fprintf('\nCleaning up parallel environment...\n');
pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
    fprintf('SUCCESS: Parallel pool shut down. Compute resources freed.\n');
end
