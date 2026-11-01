import unittest

from compat.gtx1080ti import assess_device, estimate_weight_memory_gib


class WeightEstimateTests(unittest.TestCase):
    def test_fp16_weight_floor_for_3344m_parameters(self):
        self.assertAlmostEqual(estimate_weight_memory_gib(), 6.23, places=2)

    def test_rejects_unknown_dtype(self):
        with self.assertRaises(ValueError):
            estimate_weight_memory_gib(dtype="int8")

    def test_rejects_negative_parameter_count(self):
        with self.assertRaises(ValueError):
            estimate_weight_memory_gib(parameters=-1)


class DeviceAssessmentTests(unittest.TestCase):
    def test_pascal_with_triton_model_path_is_blocked(self):
        report = assess_device(
            name="NVIDIA GeForce GTX 1080 Ti",
            capability=(6, 1),
            total_vram_gib=11.0,
            free_vram_gib=9.0,
            architectures=["sm_50", "sm_60", "sm_61", "sm_70"],
            bf16_supported=False,
            triton_installed=True,
        )
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("TRITON_ARCH_UNSUPPORTED", report["blockers"])
        self.assertFalse(report["bf16_supported"])
        self.assertFalse(report["model_inference_verified"])

    def test_missing_pytorch_arch_is_blocked(self):
        report = assess_device(
            name="Test GPU",
            capability=(6, 1),
            total_vram_gib=11.0,
            free_vram_gib=9.0,
            architectures=["sm_50", "sm_70"],
            bf16_supported=False,
            triton_installed=False,
            model_uses_triton_kernels=False,
        )
        self.assertIn("PYTORCH_ARCH_MISSING", report["blockers"])
        self.assertEqual(report["status"], "BLOCKED")

    def test_same_major_lower_minor_arch_target_is_compatible(self):
        report = assess_device(
            name="Pascal GPU",
            capability=(6, 1),
            total_vram_gib=11.0,
            free_vram_gib=9.0,
            architectures=["sm_60", "sm_70"],
            bf16_supported=False,
            triton_installed=False,
            model_uses_triton_kernels=False,
        )
        self.assertNotIn("PYTORCH_ARCH_MISSING", report["blockers"])

    def test_reports_model_smoke_test_required_when_no_static_blocker(self):
        report = assess_device(
            name="Newer GPU",
            capability=(8, 6),
            total_vram_gib=24.0,
            free_vram_gib=20.0,
            architectures=["sm_86"],
            bf16_supported=True,
            triton_installed=True,
        )
        self.assertEqual(report["status"], "NEEDS_MODEL_SMOKE_TEST")
        self.assertFalse(report["model_inference_verified"])

    def test_insufficient_free_vram_is_blocked(self):
        report = assess_device(
            name="Small GPU",
            capability=(8, 0),
            total_vram_gib=8.0,
            free_vram_gib=4.0,
            architectures=["sm_80"],
            bf16_supported=True,
            triton_installed=True,
        )
        self.assertIn("INSUFFICIENT_FREE_VRAM_FOR_WEIGHTS", report["blockers"])


if __name__ == "__main__":
    unittest.main()
