function config = expedia_experiment_config()
%EXPEDIA_EXPERIMENT_CONFIG Frozen protocol for the four-model Expedia grid.

config = struct();
config.schemaVersion = 'expedia_four_model_fit_v1';
config.resultSchemaVersion = 'expedia_four_model_result_v1';
config.protocol = 'expedia_no_click_filter_no_position_v1';
config.rawInputSha256 = '2725f98ba20252dc4153baf92675adfc1abbe98160ab45825929a721dfc238e7';
config.sites = [5 14 15 24 32];
config.trials = 1:10;
config.modelIds = ["sqmnl" "nsmnl" "iml" "imrcs"];
config.dataPattern = '10k_sample_%d_%d.mat';
config.manifestName = 'expedia_preprocess_manifest.json';
config.expectedColumns = 20;
config.featureCols = 8:20;
config.matrixColumns = ["srch_id" "prop_id" "site_id" "position" ...
    "booking_bool" "click_bool" "origin_price" "b_0" "price_usd" ...
    "prop_starrating" "prop_location_score1" "prop_review_score" ...
    "srch_length_of_stay" "prop_log_historical_price" "promotion_flag" ...
    "srch_booking_window" "srch_adults_count" "srch_children_count" ...
    "prop_brand_bool" "srch_saturday_night_bool"];
config.modelFeatureColumns = config.matrixColumns(config.featureCols);
config.stageSize = 10;
config.expectedTrainSessions = 6000;
config.expectedTestSessions = 4000;
config.queryCol = 1;
config.itemCol = 2;
config.siteCol = 3;
config.positionCol = 4;
config.choiceCol = 5;
config.clickCol = 6;
config.outsidePosition = 100;
config.probabilityTolerance = 1e-8;
config.monotonicityTolerance = 1e-7;
config.emRelativeTolerance = 1e-6;
config.imrcsOuterMaxIter = 100;
config.directMaxIter = 500;
config.innerMaxIter = 500;
config.optimizerOptimalityTolerance = 1e-6;
config.seedBase = 2026081600;
end
