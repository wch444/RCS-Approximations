function publish_mixed_order_pair(outdir, output_stem, results, out, ...
        ord_names, sites, trials, provenance, publication_hook)
%PUBLISH_MIXED_ORDER_PAIR Transactionally publish one MAT/CSV generation.
% The optional hook is used only by fault-injection tests.  Production calls
% omit it and therefore follow the exact same numerical and file schemas.

if nargin < 9 || isempty(publication_hook)
    publication_hook = @(~,~) [];
end
if ~isa(publication_hook,'function_handle')
    error('publish_mixed_order_pair:InvalidHook', ...
        'publication_hook must be a function handle.');
end

require_regular_directory(outdir,'mixed-order output directory');
final_csv = fullfile(outdir, output_stem + ".csv");
final_mat = fullfile(outdir, output_stem + ".mat");
temporary_csv = [tempname(outdir), '.csv'];
temporary_mat = [tempname(outdir), '.mat'];
final_files = {final_mat,final_csv};
temporary_files = {temporary_mat,temporary_csv};
backup_files = {[tempname(outdir),'.mat.bak'], ...
    [tempname(outdir),'.csv.bak']};
restore_files = {[tempname(outdir),'.mat.restore'], ...
    [tempname(outdir),'.csv.restore']};
cleanup = onCleanup(@() cleanup_temporary_files( ...
    temporary_files{:},backup_files{:},restore_files{:}));

writetable(results, temporary_csv);
save(temporary_mat, 'out', 'results', 'ord_names', 'sites', 'trials', ...
    'provenance', '-v7');

expected_bytes = {read_file_bytes(temporary_mat), ...
    read_file_bytes(temporary_csv)};
lock_file = fullfile(outdir,'.mixed-order-producer.lock');
lock_cleanup = acquire_publication_lock(lock_file); %#ok<NASGU>
had_original = false(1,numel(final_files));
original_bytes = cell(1,numel(final_files));
for file_index = 1:numel(final_files)
    require_regular_target(final_files{file_index});
    had_original(file_index) = isfile(final_files{file_index});
    if had_original(file_index)
        original_bytes{file_index} = read_file_bytes(final_files{file_index});
        [ok,message] = copyfile( ...
            final_files{file_index},backup_files{file_index},'f');
        if ~ok || ~isequal(read_file_bytes(backup_files{file_index}), ...
                original_bytes{file_index})
            error('publish_mixed_order_pair:BackupFailed', ...
                'Could not snapshot %s before publication: %s', ...
                final_files{file_index},message);
        end
    end
end

try
    % The CSV is the externally consumed pair marker and moves last.
    for file_index = 1:numel(final_files)
        promote_file(temporary_files{file_index},final_files{file_index});
        publication_hook("after_replace",file_index);
    end
    for file_index = 1:numel(final_files)
        if is_symbolic_link(final_files{file_index}) ...
                || ~isfile(final_files{file_index}) ...
                || ~isequal(read_file_bytes(final_files{file_index}), ...
                    expected_bytes{file_index})
            error('publish_mixed_order_pair:PostCommitVerificationFailed', ...
                'Committed mixed-order artifact failed verification: %s', ...
                final_files{file_index});
        end
    end
    publication_hook("postcheck",0);
catch publication_error
    rollback_errors = rollback_mixed_order_outputs( ...
        final_files,had_original,backup_files,restore_files,original_bytes);
    if ~isempty(rollback_errors)
        rollback_error = MException('publish_mixed_order_pair:RollbackFailed', ...
            'Mixed-order rollback was incomplete: %s', ...
            strjoin(rollback_errors,' | '));
        publication_error = addCause(publication_error,rollback_error);
    end
    rethrow(publication_error);
end
clear lock_cleanup
clear cleanup
end


function require_regular_directory(path,label)
if is_symbolic_link(path)
    error('publish_mixed_order_pair:UnsafeOutputPath', ...
        '%s must not be a symbolic link: %s',label,path);
end
if ~isfolder(path)
    if exist(path,'file') ~= 0
        error('publish_mixed_order_pair:UnsafeOutputPath', ...
            '%s is not a directory: %s',label,path);
    end
    [ok,message] = mkdir(path);
    if ~ok
        error('publish_mixed_order_pair:OutputDirectoryFailed', ...
            'Could not create %s: %s',label,message);
    end
end
if is_symbolic_link(path) || ~isfolder(path)
    error('publish_mixed_order_pair:UnsafeOutputPath', ...
        '%s must be a regular directory: %s',label,path);
end
end


function require_regular_target(path)
entry = javaObject('java.io.File',char(string(path)));
if is_symbolic_link(path) || (entry.exists() && ~entry.isFile())
    error('publish_mixed_order_pair:UnsafeOutputTarget', ...
        'Mixed-order publication target is not a regular file: %s',path);
end
end


function cleanup = acquire_publication_lock(lock_file)
if is_symbolic_link(lock_file) || exist(lock_file,'file') ~= 0
    error('publish_mixed_order_pair:PublicationLocked', ...
        'Another mixed-order producer holds the publication lock: %s',lock_file);
end
lock_path = javaObject('java.io.File',char(string(lock_file))).toPath();
options = javaObject('java.util.HashSet');
options.add(javaMethod('valueOf','java.nio.file.StandardOpenOption','CREATE_NEW'));
options.add(javaMethod('valueOf','java.nio.file.StandardOpenOption','WRITE'));
attributes = javaArray('java.nio.file.attribute.FileAttribute',0);
try
    channel = javaMethod('open','java.nio.channels.FileChannel', ...
        lock_path,options,attributes);
catch lock_error
    error('publish_mixed_order_pair:PublicationLocked', ...
        'Could not acquire mixed-order publication lock %s: %s', ...
        lock_file,lock_error.message);
end
identifier = javaMethod('randomUUID','java.util.UUID');
token = char(identifier.toString());
try
    buffer = javaMethod('wrap','java.nio.ByteBuffer',int8(token));
    while buffer.hasRemaining()
        channel.write(buffer);
    end
    channel.force(true);
    if ~strcmp(read_lock_token(lock_file),token)
        error('publish_mixed_order_pair:PublicationLockFailed', ...
            'Mixed-order publication lock failed token verification.');
    end
catch lock_error
    channel.close();
    if strcmp(read_lock_token(lock_file),token)
        delete(lock_file);
    end
    error('publish_mixed_order_pair:PublicationLockFailed', ...
        'Could not initialize mixed-order publication lock %s: %s', ...
        lock_file,lock_error.message);
end
cleanup = onCleanup(@() release_publication_lock( ...
    channel,lock_file,token));
end


function token = read_lock_token(lock_file)
token = '';
if ~isfile(lock_file) || is_symbolic_link(lock_file)
    return
end
file_id = fopen(lock_file,'rb');
if file_id < 0
    return
end
cleanup = onCleanup(@() fclose(file_id));
token = char(fread(file_id,Inf,'*uint8').');
clear cleanup
end


function release_publication_lock(channel,lock_file,token)
channel.close();
if strcmp(read_lock_token(lock_file),token)
    delete(lock_file);
end
end


function promote_file(source,target)
[ok,message] = movefile(source,target,'f');
if ~ok
    error('publish_mixed_order_pair:AtomicWriteFailed', ...
        'Could not promote mixed-order artifact %s: %s',target,message);
end
end


function rollback_errors = rollback_mixed_order_outputs( ...
        final_files,had_original,backup_files,restore_files,original_bytes)
rollback_errors = {};
for file_index = 1:numel(final_files)
    target = final_files{file_index};
    try
        if had_original(file_index)
            [ok,message] = copyfile( ...
                backup_files{file_index},restore_files{file_index},'f');
            if ~ok
                error('publish_mixed_order_pair:RollbackCopyFailed', ...
                    'Could not prepare rollback copy for %s: %s',target,message);
            end
            promote_file(restore_files{file_index},target);
            if is_symbolic_link(target) || ~isfile(target) ...
                    || ~isequal(read_file_bytes(target), ...
                        original_bytes{file_index})
                error('publish_mixed_order_pair:RollbackVerificationFailed', ...
                    'Restored mixed-order artifact failed verification: %s',target);
            end
        elseif is_symbolic_link(target) || isfile(target)
            delete(target);
        elseif exist(target,'file') ~= 0
            error('publish_mixed_order_pair:RollbackUnsafeTarget', ...
                'Cannot remove non-file mixed-order target: %s',target);
        end
        if had_original(file_index) ~= isfile(target) ...
                || is_symbolic_link(target)
            error('publish_mixed_order_pair:RollbackVerificationFailed', ...
                'Mixed-order target has the wrong state after rollback: %s',target);
        end
    catch rollback_error
        rollback_errors{end+1} = sprintf('%s: %s', ...
            target,rollback_error.message); %#ok<AGROW>
    end
end
end


function cleanup_temporary_files(varargin)
for file_index = 1:nargin
    if isfile(varargin{file_index})
        delete(varargin{file_index});
    end
end
end


function value = is_symbolic_link(path)
entry = javaObject('java.io.File',char(string(path)));
value = javaMethod('isSymbolicLink','java.nio.file.Files',entry.toPath());
end


function bytes = read_file_bytes(path)
file_id = fopen(path, 'rb');
if file_id < 0
    error('publish_mixed_order_pair:ReadFailed', ...
        'Cannot read mixed-order artifact: %s', path);
end
cleanup = onCleanup(@() fclose(file_id));
bytes = fread(file_id,Inf,'*uint8');
clear cleanup
end
