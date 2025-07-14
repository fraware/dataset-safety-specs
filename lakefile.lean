import Lake
open Lake DSL

package dataset_safety_specs {
  srcDir := "src"
}

@[default_target]
lean_lib DatasetSafetySpecs {
  roots := #[`DatasetSafetySpecs.DatasetSafetySpecs, `DatasetSafetySpecs.Lineage, `DatasetSafetySpecs.Policy, `DatasetSafetySpecs.Optimizer, `DatasetSafetySpecs.Shape, `DatasetSafetySpecs.Guard]
}

lean_exe extract_guard {
  root := `DatasetSafetySpecs.ExtractGuard
}

lean_exe shapesafe_verify {
  root := `DatasetSafetySpecs.ShapeSafeVerify
}

lean_exe test_suite {
  root := `DatasetSafetySpecs.TestSuite
}

lean_exe benchmark_suite {
  root := `DatasetSafetySpecs.BenchmarkSuite
}

lean_exe lineage_regression_test {
  root := `DatasetSafetySpecs.LineageRegressionTest
}
