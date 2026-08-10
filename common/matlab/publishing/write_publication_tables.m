function wrote = write_publication_tables( ...
        values,output_files,requested_axes,canonical_axes,expected_rows, ...
        warning_id,warning_message,prepublish_validator)
%WRITE_PUBLICATION_TABLES Atomically publish tables only for an exact grid.
%   A valid subset is returned by its caller but must not replace canonical
%   publication CSVs.  Axis order is immaterial; completeness and uniqueness
%   are required for every requested axis.

if nargin < 8 || isempty(prepublish_validator)
    prepublish_validator = @() [];
end
validate_inputs(values,output_files,requested_axes,canonical_axes, ...
    expected_rows,warning_id,warning_message,prepublish_validator);

full_grid = true;
for axis_index = 1:numel(requested_axes)
    full_grid = full_grid && is_complete_unique_axis( ...
        requested_axes{axis_index},canonical_axes{axis_index});
end
if ~full_grid
    warning(char(warning_id),'%s',char(warning_message));
    wrote = false;
    return
end

for value_index = 1:numel(values)
    if height(values{value_index}) ~= expected_rows(value_index)
        error('write_publication_tables:IncompleteGrid', ...
            ['A full-grid publication table has %d rows; the output ' ...
             'contract requires %d.'], ...
            height(values{value_index}),expected_rows(value_index));
    end
end

temporary_files = strings(size(output_files));
backup_files = strings(size(output_files));
had_original = false(size(output_files));
for file_index = 1:numel(output_files)
    folder = fileparts(char(output_files{file_index}));
    if ~exist(folder,'dir'), mkdir(folder); end
    if is_symbolic_link(folder)
        error('write_publication_tables:NonRegularTarget', ...
            'A publication output directory is a symbolic link: %s',folder);
    end
    reject_nonregular_target(output_files{file_index});
    temporary_files(file_index) = string(tempname(folder)) + ".csv";
    backup_files(file_index) = string(tempname(folder)) + ".bak";
end
cleanup = onCleanup(@() delete_files_if_present( ...
    [temporary_files(:); backup_files(:)]));

% Serialize every table before replacing any canonical output.  Each final
% replacement is then an atomic move within its destination directory.
for file_index = 1:numel(output_files)
    writetable(values{file_index},char(temporary_files(file_index)));
end
expected_bytes = cell(size(output_files));
for file_index = 1:numel(output_files)
    expected_bytes{file_index} = read_file_bytes(temporary_files(file_index));
end

% Keep byte-for-byte rollback copies before replacing the first target.  A
% multi-file publication cannot be one filesystem rename, but on any reported
% replacement failure this restores the complete previous generation instead
% of leaving a mixture of new and old tables.
original_bytes = cell(size(output_files));
for file_index = 1:numel(output_files)
    target = char(output_files{file_index});
    had_original(file_index) = isfile(target);
    if had_original(file_index)
        original_bytes{file_index} = read_file_bytes(target);
        [ok,message] = copyfile( ...
            target,char(backup_files(file_index)),'f');
        if ~ok
            error('write_publication_tables:BackupFailed','%s',message);
        end
        if ~isequal(read_file_bytes(backup_files(file_index)), ...
                original_bytes{file_index})
            error('write_publication_tables:BackupFailed', ...
                'A rollback copy does not match its original table.');
        end
    end
end

% Revalidate caller-owned inputs after every output and rollback copy exists.
% This is deliberately the last operation before the first replacement.
prepublish_validator();

try
    for file_index = 1:numel(output_files)
        [ok,message] = movefile(char(temporary_files(file_index)), ...
            char(output_files{file_index}),'f');
        if ~ok
            error('write_publication_tables:MoveFailed','%s',message);
        end
    end
    for file_index = 1:numel(output_files)
        if ~isfile(output_files{file_index}) ...
                || ~isequal(read_file_bytes(output_files{file_index}), ...
                    expected_bytes{file_index})
            error('write_publication_tables:PostPublishMismatch', ...
                'A published table failed post-publication verification.');
        end
    end
catch publication_error
    rollback_messages = rollback_outputs( ...
        output_files,backup_files,had_original,original_bytes);
    if ~isempty(rollback_messages)
        error('write_publication_tables:RollbackFailed', ...
            'Publication failed (%s); rollback also failed: %s', ...
            publication_error.message,strjoin(rollback_messages,' | '));
    end
    rethrow(publication_error)
end
delete_files_if_present(backup_files);
clear cleanup
wrote = true;
end


function validate_inputs(values,output_files,requested_axes,canonical_axes, ...
        expected_rows,warning_id,warning_message,prepublish_validator)
valid_values = iscell(values) && ~isempty(values) ...
    && all(cellfun(@istable,values));
valid_files = iscell(output_files) && numel(output_files) == numel(values) ...
    && all(cellfun(@is_text_scalar,output_files));
valid_rows = isnumeric(expected_rows) && isreal(expected_rows) ...
    && isvector(expected_rows) && numel(expected_rows) == numel(values) ...
    && all(isfinite(expected_rows)) && all(expected_rows >= 0) ...
    && all(expected_rows == fix(expected_rows));
valid_axes = iscell(requested_axes) && iscell(canonical_axes) ...
    && ~isempty(requested_axes) ...
    && numel(requested_axes) == numel(canonical_axes);
if ~valid_values || ~valid_files || ~valid_rows || ~valid_axes ...
        || ~is_text_scalar(warning_id) || ~is_text_scalar(warning_message) ...
        || ~isa(prepublish_validator,'function_handle')
    error('write_publication_tables:InvalidContract', ...
        'Publication output arguments do not satisfy the table contract.');
end
end


function complete = is_complete_unique_axis(requested,canonical)
if ischar(requested), requested = string(requested); end
if ischar(canonical), canonical = string(canonical); end
same_type = (isstring(requested) && isstring(canonical)) ...
    || (isnumeric(requested) && isnumeric(canonical));
if ~same_type || ~isvector(requested) || ~isvector(canonical) ...
        || isempty(canonical)
    error('write_publication_tables:InvalidContract', ...
        'Requested and canonical axes must be nonempty compatible vectors.');
end
requested = requested(:);
canonical = canonical(:);
if numel(unique(canonical,'stable')) ~= numel(canonical)
    error('write_publication_tables:InvalidContract', ...
        'A canonical publication axis contains duplicate values.');
end
complete = numel(requested) == numel(canonical) ...
    && numel(unique(requested,'stable')) == numel(requested) ...
    && all(ismember(canonical,requested));
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end


function delete_files_if_present(paths)
for path = paths(:).'
    if isfile(path), delete(path); end
end
end


function messages = rollback_outputs( ...
        output_files,backup_files,had_original,original_bytes)
messages = strings(0,1);
for file_index = 1:numel(output_files)
    target = char(output_files{file_index});
    if had_original(file_index)
        if ~isfile(backup_files(file_index))
            messages(end+1,1) = "Missing backup for " + string(target); %#ok<AGROW>
            continue
        end
        [ok,message] = movefile( ...
            char(backup_files(file_index)),target,'f');
        if ~ok
            messages(end+1,1) = string(message); %#ok<AGROW>
        elseif ~isequal(read_file_bytes(target),original_bytes{file_index})
            messages(end+1,1) = ...
                "Restored table differs from its backup: " + string(target); %#ok<AGROW>
        end
    elseif isfile(target)
        try
            delete(target);
        catch exception
            messages(end+1,1) = string(exception.message); %#ok<AGROW>
        end
    end
end
end


function bytes = read_file_bytes(path)
file = fopen(char(string(path)),'rb');
if file < 0
    error('write_publication_tables:ReadFailed', ...
        'Cannot read a publication table: %s',char(string(path)));
end
cleanup = onCleanup(@() fclose(file));
bytes = fread(file,Inf,'*uint8');
clear cleanup
end


function reject_nonregular_target(path)
path = char(string(path));
entry = javaObject('java.io.File',path);
if is_symbolic_link(path) || (entry.exists() && ~entry.isFile())
    error('write_publication_tables:NonRegularTarget', ...
        'Publication targets must be absent or regular files: %s',path);
end
end


function symbolic = is_symbolic_link(path)
entry = javaObject('java.io.File',char(string(path)));
symbolic = javaMethod('isSymbolicLink','java.nio.file.Files',entry.toPath());
end
