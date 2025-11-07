% pca_demo_matlab.m
%
% Reproduces the Python PCA demo pipeline in MATLAB.
% - Loads wine_ucirepo_186.csv (expected in repo root)
% - Quick EDA (head, summary, histograms, correlation heatmap)
% - Binary target: good = quality >= 7
% - Stratified train/test split (80/20)
% - Scale features (z-score)
% - Baseline RandomForest (TreeBagger) on scaled features
% - PCA: choose n components to retain 95% variance, train on PCA features
% - Save plots and results_summary.txt into pca_outputs/
%
% Usage:
%   Open MATLAB, change directory to project root, then run:
%     run('pca_demo_matlab.m')
%
% Notes:
% - Requires Statistics and Machine Learning Toolbox for TreeBagger, cvpartition
% - If you lack the toolbox, replace TreeBagger with a suitable classifier

try
    % Create output folder
    outdir = fullfile(pwd, 'pca_outputs');
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    % Load data
    csvPath = fullfile(pwd, 'wine_ucirepo_186.csv');
    if ~exist(csvPath, 'file')
        error('CSV file not found. Please run main.py or add wine_ucirepo_186.csv to the project folder.');
    end
    T = readtable(csvPath);

    % Quick EDA
    fprintf('\n=== Quick EDA ===\n');
    fprintf('Rows: %d, Columns: %d\n', size(T,1), size(T,2));
    disp(head(T, 6));
    disp(summary(T));

    % Feature names and data matrix
    if ~ismember('quality', T.Properties.VariableNames)
        error('CSV must contain a column named quality');
    end
    featNames = setdiff(T.Properties.VariableNames, {'quality'});
    X = T{:, featNames};
    y_raw = T.quality;

    % Histograms (one figure)
    try
        fh = figure('Visible','off');
        nFeat = numel(featNames);
        nCols = min(4, nFeat);
        nRows = ceil(nFeat / nCols);
        for i = 1:nFeat
            subplot(nRows, nCols, i);
            histogram(X(:, i));
            title(strrep(featNames{i}, '_', '\_'));
        end
        sgtitle('Feature histograms');
        saveas(fh, fullfile(outdir, 'feature_histograms.png'));
        close(fh);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    % Correlation heatmap
    try
        R = corrcoef(X);
        fh = figure('Visible','off');
        heatmap(featNames, featNames, round(R,2), 'Colormap', parula);
        title('Feature correlation matrix');
        saveas(fh, fullfile(outdir, 'correlation_heatmap.png'));
        close(fh);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    % Binary target (good = quality >= 7)
    y = double(y_raw >= 7); % 1 for good, 0 otherwise

    % Stratified train/test split (80/20)
    try
        cv = cvpartition(y, 'HoldOut', 0.2);
    catch
        % Fallback if cvpartition signature differs
        n = numel(y);
        idx = randperm(n);
        testN = round(0.2 * n);
        testIdx = idx(1:testN);
        trainIdx = idx(testN+1:end);
        cv = struct('Training', @(i) trainIdx, 'Test', @(i) testIdx);
    end
    trainIdx = training(cv);
    testIdx = test(cv);

    X_train = X(trainIdx, :);
    X_test = X(testIdx, :);
    y_train = y(trainIdx);
    y_test = y(testIdx);

    % Scale features (z-score) using train stats
    mu = mean(X_train, 1);
    sigma = std(X_train, [], 1);
    sigma(sigma == 0) = 1; % avoid division by zero
    X_train_s = (X_train - mu) ./ sigma;
    X_test_s = (X_test - mu) ./ sigma;

    % Baseline RandomForest (TreeBagger)
    fprintf('\nTraining baseline RandomForest (TreeBagger)...\n');
    try
        % Convert classes to cellstr for TreeBagger classification
        classNames = {'0','1'};
        y_train_str = arrayfun(@(v) num2str(v), y_train, 'UniformOutput', false);
        model_baseline = TreeBagger(100, X_train_s, y_train_str, 'Method', 'classification', 'OOBPrediction', 'On');

        [pred_labels_cell, scores] = predict(model_baseline, X_test_s);
        y_pred = str2double(pred_labels_cell);
    catch ME
        warning(ME.identifier, '%s', ME.message);
        model_baseline = [];
        y_pred = zeros(size(y_test));
    end

    % Baseline metrics & confusion matrix
    acc_baseline = mean(y_pred == y_test);
    cm_baseline = confusionmat(y_test, y_pred);
    fprintf('Baseline accuracy: %.4f\n', acc_baseline);
    disp('Confusion matrix (rows=true, cols=predicted):');
    disp(cm_baseline);

    % Save baseline confusion matrix figure
    try
        fh = figure('Visible','off');
        confusionchart(y_test, y_pred);
        title('Confusion Matrix - Baseline');
        saveas(fh, fullfile(outdir, 'confusion_baseline.png'));
        close(fh);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    % PCA on scaled features (fit on combined train+test or train only)
    fprintf('\nFitting PCA and computing explained variance...\n');
    try
        % Fit PCA on training set
        [coeff, score_train, latent, tsquared, explained] = pca(X_train_s);
        cumExplained = cumsum(explained);
        nComp95 = find(cumExplained >= 95, 1, 'first');
        if isempty(nComp95)
            nComp95 = size(coeff,2);
        end

        % Save explained variance plot
        fh = figure('Visible','off');
        plot(1:numel(cumExplained), cumExplained, '-o'); hold on;
        yline(90, '--', 'Color', [0.5 0.5 0.5]);
        yline(95, '--r');
        xlabel('Number of components'); ylabel('Cumulative explained variance (%)');
        title('PCA cumulative explained variance');
        legend('Cumulative','90%','95%','Location','best');
        saveas(fh, fullfile(outdir, 'explained_variance.png'));
        close(fh);

        % Project train and test onto first n components
        coeff_n = coeff(:, 1:nComp95);
        X_train_p = X_train_s * coeff_n;
        X_test_p = X_test_s * coeff_n;
        fprintf('Selected n_components for 95%% variance: %d\n', nComp95);
    catch ME
        warning(ME.identifier, '%s', ME.message);
        coeff = [];
        X_train_p = X_train_s;
        X_test_p = X_test_s;
        nComp95 = size(X_train_p,2);
    end

    % Train RandomForest on PCA features
    fprintf('\nTraining RandomForest on PCA features (n=%d)...\n', nComp95);
    try
        y_train_str = arrayfun(@(v) num2str(v), y_train, 'UniformOutput', false);
        model_pca = TreeBagger(100, X_train_p, y_train_str, 'Method', 'classification', 'OOBPrediction', 'On');
        [pred_labels_cell_pca, scores_pca] = predict(model_pca, X_test_p);
        y_pred_pca = str2double(pred_labels_cell_pca);
    catch ME
        warning(ME.identifier, '%s', ME.message);
        model_pca = [];
        y_pred_pca = zeros(size(y_test));
    end

    acc_pca = mean(y_pred_pca == y_test);
    cm_pca = confusionmat(y_test, y_pred_pca);
    fprintf('PCA model accuracy: %.4f\n', acc_pca);
    disp('Confusion matrix (rows=true, cols=predicted):');
    disp(cm_pca);

    % Save PCA confusion matrix
    try
        fh = figure('Visible','off');
        confusionchart(y_test, y_pred_pca);
        title('Confusion Matrix - PCA');
        saveas(fh, fullfile(outdir, 'confusion_pca.png'));
        close(fh);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    % 2-component PCA scatter (visualization)
    try
        coeff2 = coeff(:, 1:min(2, size(coeff,2)));
        X_all_s = [X_train_s; X_test_s];
        y_all = [y_train; y_test];
        scores2 = X_all_s * coeff2;
        fh = figure('Visible','off');
        gscatter(scores2(:,1), scores2(:,2), y_all, 'rb', 'ox');
        xlabel('PC1'); ylabel('PC2'); title('PCA (2 components) scatter - colored by good (1)');
        legend({'Not good','Good'}, 'Location', 'best');
        saveas(fh, fullfile(outdir, 'pca_scatter.png'));
        close(fh);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    % Save textual summary
    try
        fid = fopen(fullfile(outdir, 'results_summary.txt'), 'w');
        fprintf(fid, 'Wine Quality PCA - Results summary\n');
        fprintf(fid, 'Run at: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, 'Baseline (TreeBagger on scaled features)\n');
        fprintf(fid, 'Accuracy: %.4f\n', acc_baseline);
        fprintf(fid, 'Confusion matrix (rows=true, cols=pred):\n');
        fprintf(fid, '%d %d\n', cm_baseline');
        fprintf(fid, '\nPCA (n_components=%d) + TreeBagger\n', nComp95);
        fprintf(fid, 'Accuracy: %.4f\n', acc_pca);
        fprintf(fid, 'Confusion matrix (rows=true, cols=pred):\n');
        fprintf(fid, '%d %d\n', cm_pca');
        fprintf(fid, '\nNote: Plots saved to folder: %s\n', outdir);
        fclose(fid);
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end

    fprintf('\nDone. Outputs saved to folder: %s\n', outdir);

    %% --- Display saved figures in visible MATLAB windows ---
    try
        imgList = { 'feature_histograms.png', 'correlation_heatmap.png', 'explained_variance.png', ...
                    'pca_scatter.png', 'confusion_baseline.png', 'confusion_pca.png' };
        for i = 1:numel(imgList)
            imgPath = fullfile(outdir, imgList{i});
            if exist(imgPath, 'file')
                try
                    I = imread(imgPath);
                    figure('Name', imgList{i}, 'NumberTitle', 'off');
                    imshow(I);
                    title(strrep(imgList{i}, '_', '\_'));
                catch ME_img
                    % fallback: show with imagesc for non-RGB/PNG
                    try
                        figure('Name', imgList{i}, 'NumberTitle', 'off');
                        imagesc(imread(imgPath)); axis image off;
                        title(strrep(imgList{i}, '_', '\_'));
                    catch
                        warning(ME_img.identifier, '%s', ME_img.message);
                    end
                end
            end
        end
    catch ME_show
        warning(ME_show.identifier, '%s', ME_show.message);
    end

    %% --- Print textual summary to the command window ---
    try
        summaryPath = fullfile(outdir, 'results_summary.txt');
        if exist(summaryPath, 'file')
            fprintf('\n=== Results summary (from %s) ===\n\n', summaryPath);
            txt = fileread(summaryPath);
            fprintf('%s\n', txt);
        else
            fprintf('\nNo results_summary.txt found at %s\n', summaryPath);
        end
    catch ME_print
        warning(ME_print.identifier, '%s', ME_print.message);
    end
catch ME_main
    fprintf('Error running pca_demo_matlab.m: %s\n', ME_main.message);
    for k = 1:numel(ME_main.stack)
        fprintf('  at %s (line %d)\n', ME_main.stack(k).name, ME_main.stack(k).line);
    end
end
