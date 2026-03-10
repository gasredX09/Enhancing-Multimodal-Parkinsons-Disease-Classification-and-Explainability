function Y = read_SVC_file_2(pth_svc, do_norm)
%READ_SVC_FILE Loads *.svc handwriting data into a matrix.
% pth_svc   - path to the *.svc file
% do_norm   - if 1: X shifted to start at 0; Y mean-centered (default 0)
% Y columns: [X, Y, time, on/off, azimuth, altitude, pressure]

    if nargin < 2 || isempty(do_norm)
        do_norm = 0;
    end

    if ~exist(pth_svc,'file')
        error('File %s does not exist.', pth_svc);
    end

    FID = fopen(pth_svc,'r');
    contents = textscan(FID, '%n%n%n%d8%n%n%n', 'HeaderLines', 1);
    fclose(FID);

    min_TS = min(contents{3});

    Y = [contents{2}, contents{1}, contents{3} - min_TS, ...
         double(contents{4}), contents{5}, contents{6}, contents{7}];

    % Optional unit conversion (only if needed)
    % Y(:,1:2) = Y(:,1:2) * 10 / 5080;

    if do_norm
        Y(:,1) = Y(:,1) - min(Y(:,1));
        Y(:,2) = Y(:,2) - mean(Y(:,2));
    end
end