from schema_validator import ModelSchema


class DrApiAttrs:
    DR_TEST_CHECK_MAP = {
        ModelSchema.NULL_VALUE_IMPUTATION_KEY: "nullValueImputation",
        ModelSchema.SIDE_EFFECTS_KEY: "sideEffects",
        ModelSchema.PREDICTION_VERIFICATION_KEY: "predictionVerificationCheck",
        ModelSchema.PERFORMANCE_KEY: "performanceCheck",
        ModelSchema.STABILITY_KEY: "stabilityCheck",
    }

    @classmethod
    def to_dr_test_check(cls, check_name):
        return cls.DR_TEST_CHECK_MAP[check_name]
