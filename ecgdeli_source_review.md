# ECGdeli Feature Functions Source Review

As requested, I have pulled the exact source code for the two feature extraction functions directly from your local `ECGdeli-1.1` repository (`C:\MATLAB\ECGdeli-1.1\ECG_Processing`). 

It looks like these were indeed added to this local version of the toolbox by someone (they carry a 2022 copyright by Claudia Nagel from KIT), which explains why they might not be in the public GitHub branch you checked.

## 1. `ExtractAmplitudeFeaturesFromFPT.m`

This function extracts amplitudes directly at the sample indices of the P wave, Q peak, R peak, S peak, and T wave. Notice it returns a 3D matrix (`feature_12leads`) where the 3rd dimension corresponds exactly to `[P, Q, R, S, T]`.

```matlab
function [feature_12leads] = ExtractAmplitudeFeaturesFromFPT(FPT, signal)
%1st dimension: leads, 2nd dimension: beats, 3rd dimension: features (P
%wave, Q peak, R peak, S peak, T wave)

FPT_mat = cat(3,FPT{:});
%% extract features lead by lead
feature_12leads = zeros(12,size(FPT_mat,1), 5); % initialize feature vector
FPT_mat(FPT_mat>size(signal,1)) = size(signal,1); % correct FPT annotations exceeding the signal length
FPT_mat(FPT_mat<=0) = 1;

for leadID = 1:size(signal,2)
    feature_12leads(leadID,:,1) = signal(FPT_mat(:,2,leadID),leadID); % P wave amplitude
    feature_12leads(leadID,:,2) = signal(FPT_mat(:,5,leadID),leadID); % Q peak amplitude
    feature_12leads(leadID,:,3) = signal(FPT_mat(:,6,leadID),leadID); % R peak amplitude
    feature_12leads(leadID,:,4) = signal(FPT_mat(:,7,leadID),leadID); % S peak amplitude
    feature_12leads(leadID,:,5) = signal(FPT_mat(:,11,leadID),leadID); % T wave amplitude
end

end
```

## 2. `ExtractIntervalFeaturesFromFPT.m`

This function defines the timing features. Look specifically at the `features_sync` block at the bottom: 
- `features_sync(:,2)` is explicitly `qrsd` (QRS duration).
- `features_sync(:,8)` is explicitly `rr_multi` (RR interval).
This perfectly matches our pipeline's hardcoded indices, so we are safe there.

```matlab
function [features_leadwise, features_sync] = ExtractIntervalFeaturesFromFPT(FPT, FPT_MultiChannel)

FPT_mat = cat(3,FPT{:}); % reshape FPT table
%% extract features lead by lead
features_leadwise = zeros(size(FPT,1),size(FPT_mat,1), 7);
features_leadwise(:,:,1) = 2.*squeeze(FPT_mat(:,3,:)-FPT_mat(:,1,:))'; %p wave duration
features_leadwise(:,:,2) = 2.*squeeze(FPT_mat(:,8,:)-FPT_mat(:,4,:))'; % qrs duration
features_leadwise(:,:,3) = 2.*squeeze(FPT_mat(:,12,:)-FPT_mat(:,10,:))'; % t wave duration
features_leadwise(:,:,4) = 2.*squeeze(FPT_mat(:,4,:)-FPT_mat(:,1,:))'; % pq interval
features_leadwise(:,:,5) = 2.*squeeze(FPT_mat(:,6,:)-FPT_mat(:,1,:))'; % pr interval
features_leadwise(:,:,6) = 2.*squeeze(FPT_mat(:,12,:)-FPT_mat(:,4,:))'; % qt interval
rr = 2.*squeeze(diff(FPT_mat(:,6,:)))';
rr(:,end+1) = rr(:,end);
features_leadwise(:,:,7) = rr; % rr interval
%% find features synchronized over 12 leads
FPT_mat(FPT_mat<5) = NaN;

features_sync = zeros(size(FPT_mat,1),8);
% synchronized timing features, e.g. for P wave duration, are calculated as the 2nd latest detectable P wave ending
% across all leads minus the 2nd earliest detectable P wave beginning of
% all leads. In this way, possibly wrong annotated waveform on- and offsets
% in single leads won't impair the feature calculation

pwd = 2.*(maxk(squeeze(FPT_mat(:,3,:)),2,2)-mink(squeeze(FPT_mat(:,1,:)),2,2));
features_sync(:,1) = pwd(:,2); % p wave duration
qrsd =  2.*(maxk(squeeze(FPT_mat(:,8,:)),2,2)-mink(squeeze(FPT_mat(:,4,:)),2,2));
features_sync(:,2) = qrsd(:,2); % qrs duration
tdur = 2.*(maxk(squeeze(FPT_mat(:,12,:)),2,2)-mink(squeeze(FPT_mat(:,10,:)),2,2));
features_sync(:,3) = tdur(:,2); % t wave duration
pqi = 2.*(maxk(squeeze(FPT_mat(:,4,:)),2,2)-mink(squeeze(FPT_mat(:,1,:)),2,2));
features_sync(:,4) = pqi(:,2); % PQ interval
pri = 2.*(maxk(squeeze(FPT_mat(:,6,:)),2,2)-mink(squeeze(FPT_mat(:,1,:)),2,2));
features_sync(:,5) = pri(:,2); % PR interval
qti = 2.*(maxk(squeeze(FPT_mat(:,12,:)),2,2)-mink(squeeze(FPT_mat(:,4,:)),2,2));
features_sync(:,6) = qti(:,2); % QT interval
rr_multi = 2.*squeeze(diff(FPT_MultiChannel(:,6)));
rr_multi(end+1,1) = rr_multi(end,1); % RR interval
features_sync(:,7) = 1000.*(features_sync(:,6)./1000+0.154.*(1-0.001.*rr_multi)); % QTc (framingham corrections)
features_sync(:,8) = rr_multi;
end
```

## ST Elevation Calculation (J-Point)

Since we now know that `FPT_mat(:,4,:)` is the QRS onset and `FPT_mat(:,8,:)` is the QRS offset (J-point), I can build a custom ST Elevation calculator directly into our pipeline script. 
We can measure the isoelectric baseline at the QRS onset, measure the ST voltage at J-point + 60ms (which is `0.06 * Fs` samples), and subtract them.
