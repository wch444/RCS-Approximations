function hex_digest = sha256_file(path)
%SHA256_FILE Compute a live SHA-256 digest without loading the file at once.

path = char(string(path));
file_id = fopen(path,'rb');
if file_id < 0
    error('ExpediaHash:OpenFailed','Cannot open file for hashing: %s',path);
end
cleanup = onCleanup(@() fclose(file_id));
digest = java.security.MessageDigest.getInstance('SHA-256');
while true
    bytes = fread(file_id,1024*1024,'*uint8');
    if isempty(bytes), break; end
    digest.update(typecast(bytes(:),'int8'));
end
raw_digest = digest.digest();
unsigned_digest = typecast(raw_digest(:),'uint8');
hex_digest = lower(reshape(dec2hex(unsigned_digest,2).',1,[]));
clear cleanup
end
