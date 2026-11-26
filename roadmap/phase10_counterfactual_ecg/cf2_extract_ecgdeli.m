function cf2_extract_ecgdeli(task_file, out_dir, n_workers, chunk_size)
% CF2_EXTRACT_ECGDELI  Phase 10, CF2: ECGdeli features + median beats for a task list (legacy method).
%
%   cf2_extract_ecgdeli(task_file, out_dir)            % 6 workers, 500 ECGs per chunk
%   cf2_extract_ecgdeli(task_file, out_dir, 8, 500)
%
% task_file : CSV with columns record_id, dataset, path (absolute WFDB path without extension), priority
%             (written by: python -m roadmap.phase10_counterfactual_ecg.cf2_01_extract --write-tasks)
% out_dir   : one shard per chunk is written here; finished chunks are skipped, so a stopped run resumes.
%
% Signal chain = src/matlab/extract_ecg_features.m (the legacy extractor behind the study's features):
%   ECG_Baseline_Removal(1, 0.5) -> ECG_High_Low_Filter(1, 40) -> Notch_Filter(50) -> Isoline_Correction
%   -> Annotate_ECG_Multi -> ExtractAmplitudeFeaturesFromFPT / ExtractIntervalFeaturesFromFPT,
%   ST_Elev = signal(J + 60 ms) - signal(QRS onset), same feature names and aggregation (mean, iqr, count).
% Two changes from the legacy script:
%   1. leads are reordered BY THE NAMES IN THE .hea HEADER to I, II, III, aVR, aVL, aVF, V1-V6
%      (MIMIC stores aVF before aVL; the legacy script read by position, which caused the Phase 5 label swap).
%      Feature names in these outputs are therefore already correct: do NOT apply fix_lead_labels to them.
%   2. a median beat per lead is added: beats cut from QRS onset - 200 ms to QRS onset + 600 ms (400 samples at
%      500 Hz) on the ECGdeli multichannel QRS onsets, beats correlating < 0.90 with the first median dropped,
%      median of the rest, baseline = value at QRS onset (the same reference as ST_Elev).
%
% Per chunk it writes  <chunk>_features.csv  (record_id, dataset, status, error, beat QC, ECGdeli features)
%                      <chunk>_beats.bin     (float32, n x 12 x 400, row-major [ecg, lead, sample], mV; NaN if none)

if nargin < 3 || isempty(n_workers), n_workers = 6; end
if nargin < 4 || isempty(chunk_size), chunk_size = 500; end

ecgdeli_path = 'C:\MATLAB\ECGdeli-1.1';
wfdb_path = 'C:\Users\Acer\Desktop\Capstone\Capstone\third_party\wfdb\mcode';
features_file = 'C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\final_features_to_extract.txt';
addpath(genpath(ecgdeli_path)); addpath(wfdb_path);

fid = fopen(features_file, 'r'); parsed = textscan(fid, '%s', 'Delimiter', '\n'); fclose(fid);
feat_names = string(parsed{1}); feat_names = feat_names(strlength(feat_names) > 0);
tasks = readtable(task_file, 'TextType', 'string', 'Delimiter', ',');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= n_workers
    delete(pool); pool = parpool(n_workers);
end
parfevalOnAll(pool, @() addpath(genpath(ecgdeli_path)), 0);
parfevalOnAll(pool, @() addpath(wfdb_path), 0);

n = height(tasks); n_chunks = ceil(n / chunk_size);
fprintf('%d ECGs, %d chunks of %d, %d workers -> %s\n', n, n_chunks, chunk_size, n_workers, out_dir);
t_all = tic; done = 0;
for c = 1:n_chunks
    rows = (c - 1) * chunk_size + 1 : min(c * chunk_size, n);
    name = sprintf('%05d_p%d', c - 1, min(tasks.priority(rows)));
    csv_out = fullfile(out_dir, [name '_features.csv']);
    if exist(csv_out, 'file'), continue; end
    t_chunk = tic;
    paths = tasks.path(rows); m = numel(rows);
    F = nan(m, numel(feat_names)); B = nan(m, 12, 400, 'single'); Q = nan(m, 6);
    status = strings(m, 1); err = strings(m, 1);
    parfor i = 1:m
        [F(i, :), B(i, :, :), Q(i, :), status(i), err(i)] = process_one(char(paths(i)), feat_names);
    end
    T = array2table(F, 'VariableNames', cellstr(feat_names));
    meta = table(tasks.record_id(rows), tasks.dataset(rows), status, err, Q(:, 1), Q(:, 2), Q(:, 3), Q(:, 4), ...
        Q(:, 5), Q(:, 6), 'VariableNames', {'record_id', 'dataset', 'status', 'error', 'n_beats_detected', ...
        'n_beats_used', 'template_corr', 'hr', 'j_offset_ms', 't_off_offset_ms'});
    fb = fopen(fullfile(out_dir, [name '_beats.bin']), 'w');
    fwrite(fb, permute(B, [3 2 1]), 'single');   % sample fastest -> row-major [ecg, lead, sample]
    fclose(fb);
    writetable([meta T], [csv_out '.tmp'], 'FileType', 'text');
    movefile([csv_out '.tmp'], csv_out);          % the CSV last: it marks the chunk complete
    done = done + m;
    rate = done / toc(t_all);
    fprintf('%s: %d ECGs in %.0f s (%d ok) | %d/%d | %.1f ECG/s | ETA %.0f min\n', name, m, toc(t_chunk), ...
        sum(status == "ok"), rows(end), n, rate, (n - rows(end)) / max(rate, eps) / 60);
end
fprintf('ALL CHUNKS COMPLETE in %.1f min\n', toc(t_all) / 60);
end


function [row, beat, q, status, err] = process_one(full_path, feat_names)
row = nan(1, numel(feat_names)); beat = nan(1, 12, 400, 'single'); q = nan(1, 6); err = "";
lead_order = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'};
try
    names = header_leads([full_path '.hea']);
    [record_dir, record_name, ~] = fileparts(full_path);
    old = pwd;
    try
        cd(record_dir); [sig, Fs, ~] = rdsamp(record_name); cd(old);
    catch e
        cd(old); rethrow(e);
    end
    [ok, idx] = ismember(upper(lead_order), upper(names));
    if ~all(ok), error('missing leads: %s', strjoin(lead_order(~ok), ',')); end
    sig = sig(:, idx);                                       % change 1: order by header name
    if Fs ~= 500, error('fs=%g', Fs); end
    sig(isnan(sig)) = 0;

    % --- legacy chain (unchanged) ---
    [x, ~] = ECG_Baseline_Removal(sig, Fs, 1, 0.5);
    x = ECG_High_Low_Filter(x, Fs, 1, 40);
    x = Notch_Filter(x, Fs, 50, 1);
    [x, ~, ~, ~] = Isoline_Correction(x);
    [FPT_M, FPT_Cell] = Annotate_ECG_Multi(x, Fs);
    Amp = ExtractAmplitudeFeaturesFromFPT(FPT_Cell, x);
    [~, Tsync] = ExtractIntervalFeaturesFromFPT(FPT_Cell, FPT_M);
    row = legacy_features(row, feat_names, Amp, Tsync, FPT_Cell, x, Fs, lead_order);

    % --- change 2: median beat on multichannel QRS onsets ---
    qon = FPT_M(:, 4); qon = qon(qon > 100 & qon + 300 <= size(x, 1));   % -200 ms .. +600 ms window
    q(1) = size(FPT_M, 1);
    if numel(qon) >= 3
        beats = zeros(numel(qon), 12, 400);
        for b = 1:numel(qon)
            seg = x(qon(b) - 100 : qon(b) + 299, :);              % 400 samples, QRS onset at sample 101
            beats(b, :, :) = (seg - seg(101, :))';                % baseline at QRS onset
        end
        tmpl = squeeze(median(beats, 1));
        r = arrayfun(@(b) corr_flat(squeeze(beats(b, :, :)), tmpl), 1:numel(qon));
        keep = r >= 0.90;
        q(3) = median(r);
        if sum(keep) >= 3
            beat(1, :, :) = single(median(beats(keep, :, :), 1));
            q(2) = sum(keep);
        end
    end
    rr = diff(FPT_M(:, 6)) / Fs;
    if ~isempty(rr), q(4) = 60 / median(rr); end
    q(5) = 1000 * median(FPT_M(:, 8) - FPT_M(:, 4), 'omitnan') / Fs;   % J point, ms after QRS onset
    if size(FPT_M, 2) >= 12
        q(6) = 1000 * median(FPT_M(:, 12) - FPT_M(:, 4), 'omitnan') / Fs; % T offset, ms after QRS onset
    end
    if isnan(q(2)), status = "no_median_beat"; else, status = "ok"; end
catch e
    status = "error"; err = string(e.message);
end
end


function names = header_leads(hea)
txt = fileread(hea); lines = regexp(txt, '\r?\n', 'split');
hdr = strsplit(strtrim(lines{1})); n_sig = str2double(hdr{2});
names = cell(1, n_sig);
for k = 1:n_sig
    parts = strsplit(strtrim(lines{k + 1}));
    names{k} = parts{end};
end
end


function r = corr_flat(a, b)
a = a(:) - mean(a(:)); b = b(:) - mean(b(:));
r = (a' * b) / (norm(a) * norm(b) + eps);
end


function row = legacy_features(row, f, Amp, Tsync, FPT_Cell, x, Fs, lead_names)
% Identical aggregation to src/matlab/extract_ecg_features.m (mean, iqr, count per feature).
    function row = put(row, name, vals)
        i1 = find(f == name); i2 = find(f == name + "_iqr"); i3 = find(f == name + "_count");
        if isempty(vals)
            if ~isempty(i3), row(i3) = 0; end
        else
            if ~isempty(i1), row(i1) = mean(vals); end
            if ~isempty(i2), row(i2) = iqr(vals); end
            if ~isempty(i3), row(i3) = numel(vals); end
        end
    end
if ~isempty(Tsync)
    v = Tsync(:, 2); row = put(row, "QRS_Dur_Global", v(~isnan(v)));
    v = Tsync(:, 8); row = put(row, "RR_Mean_Global", v(~isnan(v)));
end
amp_names = {'P_Amp', 'Q_Amp', 'R_Amp', 'S_Amp', 'T_Amp'};
if ~isempty(Amp)
    FPT_mat = cat(3, FPT_Cell{:});
    for l = 1:min(12, size(Amp, 1))
        for a = 1:numel(amp_names)
            v = squeeze(Amp(l, :, a)); v = v(~isnan(v));
            row = put(row, string(sprintf('%s_%s', amp_names{a}, lead_names{l})), v);
        end
        st = [];
        if size(FPT_mat, 3) >= l
            qo = squeeze(FPT_mat(:, 4, l)); jp = squeeze(FPT_mat(:, 8, l));
            for b = 1:numel(jp)
                if ~isnan(qo(b)) && ~isnan(jp(b)) && qo(b) > 0 && jp(b) > 0
                    jj = min(jp(b) + round(0.06 * Fs), size(x, 1));
                    st(end + 1) = x(jj, l) - x(qo(b), l); %#ok<AGROW>
                end
            end
        end
        row = put(row, string(sprintf('ST_Elev_%s', lead_names{l})), st);
    end
end
end
