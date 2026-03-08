
% function Y = read_SVC_file(pth_svc, norm)
pth_svc = 'C:\Users\megha\OneDrive\Attachments\Desktop\Homework\Projects in BME AI\PaHaW\PaHaW_matlab\00001__1_1.svc'
% Y = read_SVC_file(pth_svc, norm)

% This function loads *.svc data into matrix.
% pth_svc   - path to the *.svc file
% norm      - if set to 1 the x-coordinate will be normalized to 0 and
%             y-coordinate will be subtracted by its mean (default: 0)
% Y         - matrix with the SVC data
% 
% Y(:,1)    - X coordinate
% Y(:,2)    - Y coordinate
% Y(:,3)    - time stamp
% Y(:,4)    - on/off state
% Y(:,5)    - azimuth
% Y(:,6)    - altitude
% Y(:,7)    - pressure

%% Paths and variables
if((nargin < 2) || isempty(norm))
    norm = 0;
end

%% Read the data from *.svc file
if(~exist(pth_svc,'file'))
    error(['File ' pth_svc ' does not exist.']);
end

FID = fopen(pth_svc,'r');
contents = textscan(FID,'%n%n%n%d8%n%n%n','HeaderLines',1);
fclose(FID);

%% Get the initial time
min_TS = min(contents{3});

%% Store the data into the matrix
Y = [contents{2}, contents{1}, contents{3}-min_TS, cast(contents{4}, ...
    'double'), contents{5}, contents{6}, contents{7}];

%% Transform coordination units to millimeters 
%Y(:,1:2) = Y(:,1:2))*10/(5080); 

%% Normalize X-Y coordinates if necessary
if(norm)
    Y(:,1) = Y(:,1)-min(Y(:,1));
    Y(:,2) = Y(:,2)-mean(Y(:,2));
end