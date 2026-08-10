function config = dataset_config(dataset_name)
%DATASET_CONFIG Canonical matrix and market layout for empirical workflows.

name = validatestring(char(dataset_name), {'expedia', 'trivago'});
config.name = string(name);
config.feature_start = 8;
config.matrix_columns = [];
config.data_schema_version = 1;

switch name
    case 'expedia'
        config.markets = ["5", "14", "15", "24", "32"];
        config.feature_end = 20;
        config.data_pattern = '10k_sample_%s_%d.mat';
    case 'trivago'
        config.markets = ["BR", "US", "JP", "UK", "DE"];
        config.feature_end = 18;
        config.matrix_columns = 23;
        config.data_schema_version = 2;
        config.data_pattern = '10k_sample_%s_%d_p2000.mat';
end
end
