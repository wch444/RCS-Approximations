function summary = validate_expedia_matrix(data,varargin)
%VALIDATE_EXPEDIA_MATRIX Enforce the frozen 20-column session schema.
%   The outside row must be last in every contiguous session. Product rows
%   must be in strictly increasing displayed-position order. Exactly one row
%   (a product or the outside row) must be chosen in every session.

config = expedia_experiment_config();
parser = inputParser;
parser.FunctionName = 'validate_expedia_matrix';
addParameter(parser,'ExpectedSessions',[],@valid_optional_count);
addParameter(parser,'ExpectedSite',[],@valid_optional_count);
addParameter(parser,'Label','data',@is_text_scalar);
parse(parser,varargin{:});
options = parser.Results;
label = char(string(options.Label));

if ~isnumeric(data) || ~isreal(data) || ~ismatrix(data) || isempty(data)
    error('ExpediaData:InvalidMatrix','%s must be a nonempty real numeric matrix.',label);
end
if size(data,2) ~= config.expectedColumns
    error('ExpediaData:WrongColumnCount', ...
        '%s has %d columns; the no-position-encoding schema requires exactly %d.', ...
        label,size(data,2),config.expectedColumns);
end
if any(~isfinite(data(:)))
    error('ExpediaData:NonfiniteValue','%s contains NaN or Inf.',label);
end

query_ids = data(:,config.queryCol);
if any(query_ids <= 0) || any(query_ids ~= fix(query_ids))
    error('ExpediaData:InvalidQueryId','%s has nonpositive or noninteger query ids.',label);
end
starts = [1; find(diff(query_ids) ~= 0)+1];
ends = [starts(2:end)-1; size(data,1)];
session_ids = query_ids(starts);
if numel(unique(session_ids)) ~= numel(session_ids)
    error('ExpediaData:NoncontiguousQuery', ...
        '%s contains a query id in more than one row block.',label);
end
num_sessions = numel(starts);
if ~isempty(options.ExpectedSessions) && num_sessions ~= options.ExpectedSessions
    error('ExpediaData:WrongSessionCount', ...
        '%s has %d sessions; expected %d.', ...
        label,num_sessions,options.ExpectedSessions);
end

product_mask = data(:,config.itemCol) ~= 0;
site_values = unique(data(product_mask,config.siteCol));
if numel(site_values) ~= 1 || site_values ~= fix(site_values)
    error('ExpediaData:InvalidSite', ...
        '%s product rows must contain exactly one integer site id.',label);
end
if ~isempty(options.ExpectedSite) && site_values ~= options.ExpectedSite
    error('ExpediaData:WrongSite','%s contains site %d; expected site %d.', ...
        label,site_values,options.ExpectedSite);
end

choices = data(:,config.choiceCol);
clicks = data(:,config.clickCol);
if any(~ismember(choices,[0 1]))
    error('ExpediaData:InvalidChoice','%s choice indicators must be binary.',label);
end
if any(~ismember(clicks,[0 1]))
    error('ExpediaData:InvalidClick','%s click indicators must be binary.',label);
end

num_products = 0;
num_booked = 0;
queries_click_gt4 = 0;
queries_click_zero = 0;
max_click_count = 0;
max_products = 0;
for t = 1:num_sessions
    rows = data(starts(t):ends(t),:);
    item = rows(:,config.itemCol);
    position = rows(:,config.positionCol);
    outside_by_item = item == 0;
    outside_by_position = position == config.outsidePosition;
    if sum(outside_by_item) ~= 1 || sum(outside_by_position) ~= 1 ...
            || ~isequal(outside_by_item,outside_by_position) ...
            || ~outside_by_item(end)
        error('ExpediaData:InvalidOutsideRow', ...
            '%s query %.0f must end in exactly one item-0/position-%d outside row.', ...
            label,session_ids(t),config.outsidePosition);
    end
    product_rows = rows(1:end-1,:);
    if isempty(product_rows)
        error('ExpediaData:EmptyOfferSet','%s query %.0f has no products.',label,session_ids(t));
    end
    product_ids = product_rows(:,config.itemCol);
    product_positions = product_rows(:,config.positionCol);
    if any(product_ids <= 0) || any(product_ids ~= fix(product_ids)) ...
            || numel(unique(product_ids)) ~= numel(product_ids)
        error('ExpediaData:InvalidProductId', ...
            '%s query %.0f has invalid or duplicate product ids.',label,session_ids(t));
    end
    if any(product_positions < 0) || any(product_positions ~= fix(product_positions)) ...
            || any(diff(product_positions) <= 0)
        error('ExpediaData:InvalidPosition', ...
            '%s query %.0f product positions must be strictly increasing nonnegative integers.', ...
            label,session_ids(t));
    end
    if sum(rows(:,config.choiceCol)) ~= 1
        error('ExpediaData:InvalidOutcome', ...
            '%s query %.0f must have exactly one chosen product-or-outside row.', ...
            label,session_ids(t));
    end
    if any(abs(product_rows(:,config.featureCols(1))-1) > 1e-12) ...
            || any(abs(rows(end,config.featureCols)) > 1e-12)
        error('ExpediaData:InvalidIntercept', ...
            '%s query %.0f must have product intercept 1 and zero outside features.', ...
            label,session_ids(t));
    end
    if rows(end,config.clickCol) ~= 0
        error('ExpediaData:OutsideClick','%s query %.0f has an outside-row click.', ...
            label,session_ids(t));
    end
    if rows(end,config.siteCol) ~= 0 || rows(end,7) ~= 0
        error('ExpediaData:OutsideNotZero', ...
            '%s query %.0f outside site and origin-price fields must be zero.', ...
            label,session_ids(t));
    end

    click_count = sum(product_rows(:,config.clickCol));
    queries_click_gt4 = queries_click_gt4 + (click_count > 4);
    queries_click_zero = queries_click_zero + (click_count == 0);
    max_click_count = max(max_click_count,click_count);
    num_products = num_products+size(product_rows,1);
    max_products = max(max_products,size(product_rows,1));
    num_booked = num_booked+any(product_rows(:,config.choiceCol) == 1);
end

summary = struct();
summary.label = label;
summary.site = site_values;
summary.rows = size(data,1);
summary.columns = size(data,2);
summary.sessions = num_sessions;
summary.products = num_products;
summary.bookedSessions = num_booked;
summary.outsideSessions = num_sessions-num_booked;
summary.queriesClickGreaterThan4 = queries_click_gt4;
summary.queriesClickZero = queries_click_zero;
summary.maxClickCount = max_click_count;
summary.maxProducts = max_products;
summary.sessionIds = session_ids;
end


function valid = valid_optional_count(value)
valid = isempty(value) || (isnumeric(value) && isreal(value) && isscalar(value) ...
    && isfinite(value) && value >= 0 && value == fix(value));
end


function valid = is_text_scalar(value)
valid = (ischar(value) && isrow(value)) ...
    || (isstring(value) && isscalar(value) && ~ismissing(value));
end
