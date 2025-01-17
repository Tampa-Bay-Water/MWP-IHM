function hmat = ReadHead(fn,hsdays,layers,tsteps,cells)
% ReadHead: read headfile by layers and/or by CellIDs
% fn = path to head file as string
% hsdays =  number of hotstart in days (tsteps)
% layers = the vector which elements are layers to read
% tsteps = the vector which elements are tsteps to read
% cells = the vector which elements are cellid's to read
%         (CellIDs need NOT to be sorted)
% hmat = return 3-D matrix of heads read from the read file with these
%        dimensions: length(cells) X length(tsteps) X length(layers).
%        Return heads are ordered by cellid, time step and layer along
%        each respective dimension.

if nargin<2 || isempty(hsdays), hsdays = 0; end
if nargin<3 || isempty(layers), layers = 1:3; end
if nargin<4 || isempty(tsteps), tsteps = 1:(40*365.25); end

debug = false;
layers = layers(:);
tsteps = tsteps(:);

global nlays nrows ncols nbytes hbytes nwords;
global cellord;

% if exist('nlays','var')~=1 || isempty(nlays)
if isempty(nlays)
  nwords = 4;
  nrows = 207;
  ncols = 183;
  nlays = 3;
  hbytes = 44;
  nbytes = nrows*ncols*nwords+hbytes; % one layer
end

fid = fopen(fn);
if fid==-1
  disp(['ReadHead: Can''t open ' fn ' for binary read!']);
  return
end

% sort layers and tsteps
layers =sort(layers);
tsteps = sort(tsteps);

% determine cell positions as element numbers
if nargin >= 5 && ~isempty(cells)
  if isempty(cellord)
    for i = 1:nrows
      for j = 1: ncols
        cellord(j,i) = i*1000+j;
      end
    end
    cellord = reshape(cellord,nrows*ncols,1);
  end
  elems = arrayfun(@(y) find(cellord(:,1)==y),cells);
end

% loop to read
hmat = cell(length(tsteps),length(layers));
for i=1:length(tsteps)
  t = tsteps(i)-1;
  for j=1:length(layers)
    l = layers(j)-1;
    offset = ((hsdays+t)*nlays+l)*nbytes;
    if ~debug
      offset = offset + hbytes;
    end
    status = fseek(fid, offset, 'bof');
    if status==-1
      if debug, disp(['ReadHead: ' ferror(fid)]); end
      break;
    end
    if debug
      [TimeStep,Period,PeriodTime,TotalTime,Text,Columns,Rows,Layer]...
        = ReadHeadParams(fid);
      fprintf('TimeStep=%d,Period=%d,PeriodTime=%f,TotalTime=%f,Layer=%d\n',...
        TimeStep,Period,PeriodTime,TotalTime,Layer);
    end
    if nargin >= 5 && ~isempty(cells)
      % read head by list of cellids
      temp = fread(fid,nrows*ncols,'float32');
      hmat{i,j} = temp(elems);
    else
      % read head for all cells in a layer 
      hmat{i,j} = fread(fid,nrows*ncols,'float32');
    end
  end
  if status==-1
      if debug, fprintf('ReadHead: Found the end of file while reading at timestep %d\n',t+1); end
      tsteps = tsteps(tsteps<=t);
      break;
  end
end
fclose(fid);

if nargin >= 5 && ~isempty(cells)
	hmat = reshape(cell2mat(hmat),length(cells),length(tsteps),length(layers));
    % support 1 cell in URM - reduce to 2-D matrix
    if length(cells)==1
        hmat = reshape(hmat,length(tsteps),length(layers));
    end
else
	hmat = reshape(cell2mat(hmat),nrows*ncols,length(tsteps),length(layers));
    % support 1 tstep in URM - reduce to 2-D matrix
    if length(tsteps)==1 || length(layers)==1
      hmat = reshape(hmat,size(hmat,1),numel(hmat)/size(hmat,1));
      
%       if length(layers)==1
%         hmat = hmat(:,1);
%       else
%         hmat = reshape(hmat,size(hmat,1),length(layers));
%       end
%     else
%       if length(layers)==1
%         hmat = reshape(hmat,size(hmat,1),length(tsteps));
%       end
    end
end

    function [TimeStep,Period,PeriodTime,TotalTime,Text,Columns,Rows,Layers] = ReadHeadParams(fid)
    % get headfile parameters
    % header for each time step is 44 bytes of the following information
    % TimeStep As Integer (4)
    % Period As Integer (4)
    % PeriodTime As Single (4)
    % TotalTime As Single (4)
    % Text As String 16 bytes
    % ByRef Columns As Integer (4)
    % ByRef Rows As Integer (4)
    % ByRef Layers As Integer (4)
    TimeStep = fread(fid,1,'int32');
    Period = fread(fid,1,'int32');
    PeriodTime = fread(fid,1,'float32');
    TotalTime = fread(fid,1,'float32');
    Text = fread(fid,16,'uint8=>char')';
    Columns = fread(fid,1,'int32');
    Rows = fread(fid,1,'int32');
    Layers = fread(fid,1,'int32');
    end
end
