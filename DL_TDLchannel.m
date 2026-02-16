%Title  : Deep Learning-Based Detector over 3GPP TDL-A Channel
%  Author : Dr. Emad A. Hussien , Dr.Musab T. S. Al-Kaltakchi & Dr. Amal
%  Ibrahim Nasser
%  This program implements a BiLSTM-based deep learning detector for
%  OFDM transmission over a 3GPP TDL-A fading channel (TR 38.901).
%
%  It compares:
%     1) Zero-Forcing (ZF) Equalizer
%     2) BiLSTM Deep Learning Detector
%
%  Outputs:
%     - Channel impulse response
%     - Constellation diagrams
%     - BER vs SNR curve
%
%  MATLAB Version: R2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all;

Nfft        = 64;           % FFT size
cpLen       = 16;           % Cyclic prefix
M           = 4;            % 4-QAM
numSymbols  = 2000;         % Number of OFDM symbols
bitsPerSym  = log2(M);      
EbN0dB      = 0:2:30;       

%% ========================================================================
% STEP 2: 3GPP TDL-A Channel Setup
%==========================================================================
carrierFreq = 3.5e9;
sampleRate  = 15.36e6;
maxDoppler  = 100;

tdl = nrTDLChannel;
tdl.DelayProfile = 'TDL-A';
tdl.DelaySpread  = 300e-9;
tdl.MaximumDopplerShift = maxDoppler;
tdl.SampleRate = sampleRate;
tdl.NumTransmitAntennas = 1;
tdl.NumReceiveAntennas = 1;

%% ========================================================================
% STEP 3: Generate Binary Data
%==========================================================================
numBits = numSymbols * Nfft * bitsPerSym;
dataBits = randi([0 1], numBits, 1);

%% ========================================================================
% STEP 4: QAM Modulation
%==========================================================================
qamSymbols = qammod(dataBits, M, ...
    'InputType','bit', ...
    'UnitAveragePower',true);

qamSymbols = reshape(qamSymbols, Nfft, []);

%% ========================================================================
% STEP 5: OFDM Modulation
%==========================================================================
ifftSignal = ifft(qamSymbols, Nfft);
txSignal = [ifftSignal(end-cpLen+1:end,:); ifftSignal];
txSignal = txSignal(:);

%% ========================================================================
% STEP 6: Plot Transmitted Constellation
%==========================================================================
figure;
scatter(real(qamSymbols(:)), imag(qamSymbols(:)), '.');
grid on;
title('Figure 1: Transmitted QAM Constellation');
xlabel('In-Phase');
ylabel('Quadrature');

%% ========================================================================
% STEP 7: Initialize BER Arrays
%==========================================================================
BER_ZF = zeros(size(EbN0dB));
BER_DL = zeros(size(EbN0dB));

%% ========================================================================
% STEP 8: Loop Over SNR Values
%==========================================================================
for snrIdx = 1:length(EbN0dB)

    reset(tdl);

    % Pass signal through 3GPP TDL-A channel
    [rxChan, pathGains] = tdl(txSignal);

    % Plot channel impulse response once
    if snrIdx == 1
        h = squeeze(pathGains(1,:,:));
        figure;
        stem(abs(h));
        title('Figure 2: 3GPP TDL-A Channel Impulse Response');
        xlabel('Tap Index');
        ylabel('|h(n)|');
        grid on;
    end

    % Add AWGN
    rxSignal = awgn(rxChan, EbN0dB(snrIdx), 'measured');

    % OFDM Demodulation
    rxMatrix = reshape(rxSignal, Nfft+cpLen, []);
    rxMatrix = rxMatrix(cpLen+1:end,:);
    rxFFT = fft(rxMatrix, Nfft);

    %% ================= Perfect Channel Estimation =================
    h_time = squeeze(pathGains(1,:,:));
    H = fft(h_time, Nfft);
    H = repmat(H.',1,size(rxFFT,2));

    %% ================= Zero-Forcing Equalization ===================
    Xhat_ZF = rxFFT ./ H;

    rxBits_ZF = qamdemod(Xhat_ZF(:), M, ...
        'OutputType','bit', ...
        'UnitAveragePower',true);

    BER_ZF(snrIdx) = mean(rxBits_ZF ~= dataBits);

    %% ================= Prepare DL Dataset ==========================
    X_real = real(Xhat_ZF(:));
    X_imag = imag(Xhat_ZF(:));
    DL_input = [X_real X_imag];

    Y_real = real(qamSymbols(:));
    Y_imag = imag(qamSymbols(:));
    DL_target = [Y_real Y_imag];

    %% ================= Train BiLSTM Once ===========================
    if snrIdx == 1

        layers = [
            sequenceInputLayer(2)
            bilstmLayer(30,'OutputMode','sequence')
            dropoutLayer(0.5)
            fullyConnectedLayer(2)
            regressionLayer];

        options = trainingOptions('adam', ...
            'MaxEpochs',75, ...
            'MiniBatchSize',128, ...
            'InitialLearnRate',0.05, ...
            'LearnRateDropFactor',0.1, ...
            'Verbose',false);

        net = trainNetwork( ...
            num2cell(DL_input',1), ...
            num2cell(DL_target',1), ...
            layers, options);
    end

    %% ================= DL Detection ================================
    DL_pred = predict(net, num2cell(DL_input',1));
    DL_pred = cell2mat(DL_pred);

    DL_complex = DL_pred(1,:) + 1i*DL_pred(2,:);

    rxBits_DL = qamdemod(DL_complex.', M, ...
        'OutputType','bit', ...
        'UnitAveragePower',true);

    BER_DL(snrIdx) = mean(rxBits_DL ~= dataBits);

    fprintf('SNR = %d dB completed\n', EbN0dB(snrIdx));
end

%% ========================================================================
% STEP 9: Plot ZF Constellation
%==========================================================================
figure;
scatter(real(Xhat_ZF(:)), imag(Xhat_ZF(:)), '.');
grid on;
title('Figure 3: Constellation After ZF Equalization');
xlabel('In-Phase');
ylabel('Quadrature');

%% ========================================================================
% STEP 10: Plot DL Constellation
%==========================================================================
figure;
scatter(real(DL_complex), imag(DL_complex), '.');
grid on;
title('Figure 4: Constellation After BiLSTM Detection');
xlabel('In-Phase');
ylabel('Quadrature');

%% ========================================================================
% STEP 11: Plot BER Performance
%==========================================================================
figure;
semilogy(EbN0dB, BER_ZF, '-o','LineWidth',2); hold on;
semilogy(EbN0dB, BER_DL, '-s','LineWidth',2);
grid on;
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
legend('ZF Equalizer','BiLSTM Detector');
title('Figure 5: BER Performance over 3GPP TDL-A Channel');

%% ========================================================================
% STEP 12: Save Figures
%==========================================================================
savefig(1,'Fig1_TxConstellation.fig');
savefig(2,'Fig2_ChannelResponse.fig');
savefig(3,'Fig3_ZFConstellation.fig');
savefig(4,'Fig4_DLConstellation.fig');
savefig(5,'Fig5_BER.fig');

disp('Simulation Completed Successfully.');