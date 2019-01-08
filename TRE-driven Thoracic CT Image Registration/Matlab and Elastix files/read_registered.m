function registered_indices = read_registered(registered_path)
% registered_indices = read_registered(registered_path)
%Takes the path of the registered text file (result of transformix) and
%reads the part related to output indices only.
%Parameters:
%   registered_path: string/char array
%       the path where the text file is located.

%Returns:
%   registered_indices: 3* number of features (lines).
%       the numbers only that correspond to output indices


%reading moving features
fid             = fopen(registered_path);
content_cell    = textscan(fid, '%s', 'delimiter', '\n');
xm              = zeros(length(content_cell{1}),1);
ym              = zeros(length(content_cell{1}),1);
zm              = zeros(length(content_cell{1}),1);

%for all lines
for i=1:length(content_cell{1})
content= content_cell{1}{i};

%finding the place where the registered indices are located
start_index=  strfind(content, 'OutputIndexFixed = [ ');
end_index = strfind(content, ' ]	; OutputPoint');

%extracting numbers part only
numbers = content(start_index + 21 : end_index-1);
numbers_cell = split(numbers, ' ');

%changing datatype from string/char array to double
xm(i) = str2double(numbers_cell{1});
ym(i) = str2double(numbers_cell{2});
zm(i) = str2double(numbers_cell{3});
    
end

registered_indices = [xm, ym, zm];
end

