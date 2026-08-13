//! Model version helpers.
//!
//! Corresponds to `cpp/neuralnet/modelversion.h` and `cpp/neuralnet/modelversion.cpp`.

use crate::inputs::*;

pub const LATEST_MODEL_VERSION_IMPLEMENTED: i32 = 17;
pub const LATEST_INPUTS_VERSION_IMPLEMENTED: i32 = 7;
pub const DEFAULT_MODEL_VERSION: i32 = 17;

pub const OLDEST_MODEL_VERSION_IMPLEMENTED: i32 = 3;
pub const OLDEST_INPUTS_VERSION_IMPLEMENTED: i32 = 3;

fn unsupported_model_version(model_version: i32) -> String {
    format!(
        "NNModelVersion: Model version not currently implemented or supported: {}",
        model_version
    )
}

/// Returns the `NNInputs` feature version consumed by a given model version.
pub fn get_inputs_version(model_version: i32) -> Result<i32, String> {
    match model_version {
        8..=17 => Ok(7),
        7 => Ok(6),
        6 => Ok(5),
        5 => Ok(4),
        3 | 4 => Ok(3),
        _ => Err(unsupported_model_version(model_version)),
    }
}

/// Number of spatial input feature planes for a model version.
pub fn get_num_spatial_features(model_version: i32) -> Result<i32, String> {
    match model_version {
        8..=17 => Ok(NUM_FEATURES_SPATIAL_V7),
        7 => Ok(NUM_FEATURES_SPATIAL_V6),
        6 => Ok(NUM_FEATURES_SPATIAL_V5),
        5 => Ok(NUM_FEATURES_SPATIAL_V4),
        3 | 4 => Ok(NUM_FEATURES_SPATIAL_V3),
        _ => Err(unsupported_model_version(model_version)),
    }
}

/// Number of global input features for a model version.
pub fn get_num_global_features(model_version: i32) -> Result<i32, String> {
    match model_version {
        8..=17 => Ok(NUM_FEATURES_GLOBAL_V7),
        7 => Ok(NUM_FEATURES_GLOBAL_V6),
        6 => Ok(NUM_FEATURES_GLOBAL_V5),
        5 => Ok(NUM_FEATURES_GLOBAL_V4),
        3 | 4 => Ok(NUM_FEATURES_GLOBAL_V3),
        _ => Err(unsupported_model_version(model_version)),
    }
}

/// Number of SGF metadata encoder input channels for a meta-encoder version.
pub fn get_num_input_meta_channels(meta_encoder_version: i32) -> Result<i32, String> {
    match meta_encoder_version {
        0 => Ok(0),
        1 => Ok(192),
        _ => Err(format!(
            "NNModelVersion: metaEncoderVersion not currently implemented or supported: {}",
            meta_encoder_version
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_bounds() {
        assert_eq!(LATEST_MODEL_VERSION_IMPLEMENTED, 17);
        assert_eq!(OLDEST_MODEL_VERSION_IMPLEMENTED, 3);
        assert_eq!(LATEST_INPUTS_VERSION_IMPLEMENTED, 7);
        assert_eq!(OLDEST_INPUTS_VERSION_IMPLEMENTED, 3);
        assert_eq!(DEFAULT_MODEL_VERSION, 17);
    }

    #[test]
    fn test_get_inputs_version() {
        assert_eq!(get_inputs_version(3).unwrap(), 3);
        assert_eq!(get_inputs_version(4).unwrap(), 3);
        assert_eq!(get_inputs_version(5).unwrap(), 4);
        assert_eq!(get_inputs_version(6).unwrap(), 5);
        assert_eq!(get_inputs_version(7).unwrap(), 6);
        assert_eq!(get_inputs_version(8).unwrap(), 7);
        assert_eq!(get_inputs_version(17).unwrap(), 7);
        assert!(get_inputs_version(2).is_err());
        assert!(get_inputs_version(18).is_err());
    }

    #[test]
    fn test_get_num_spatial_features() {
        assert_eq!(
            get_num_spatial_features(3).unwrap(),
            NUM_FEATURES_SPATIAL_V3
        );
        assert_eq!(
            get_num_spatial_features(8).unwrap(),
            NUM_FEATURES_SPATIAL_V7
        );
        assert!(get_num_spatial_features(1).is_err());
    }

    #[test]
    fn test_get_num_global_features() {
        assert_eq!(get_num_global_features(4).unwrap(), NUM_FEATURES_GLOBAL_V3);
        assert_eq!(get_num_global_features(9).unwrap(), NUM_FEATURES_GLOBAL_V7);
        assert!(get_num_global_features(100).is_err());
    }

    #[test]
    fn test_get_num_input_meta_channels() {
        assert_eq!(get_num_input_meta_channels(0).unwrap(), 0);
        assert_eq!(get_num_input_meta_channels(1).unwrap(), 192);
        assert!(get_num_input_meta_channels(2).is_err());
    }
}
