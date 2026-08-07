import numpy as np
import SimpleITK as sitk


class ClassicalBRegistration:
    """Classical registration using SimpleITK B-Spline"""

    def __init__(
        self,
        mesh_size=(2, 2),
        scale_factors=(1, 2, 4),
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(4, 2, 0),
        learning_rate=5.0,
        numberOfIterations=100,
    ):
        self.mesh_size = list(mesh_size)
        self.scale_factors = list(scale_factors)
        self.shrink_factors = list(shrink_factors)
        self.smoothing_sigmas = list(smoothing_sigmas)
        self.learning_rate = learning_rate
        self.numberOfIterations = numberOfIterations

        self.registration_method = sitk.ImageRegistrationMethod()
        self.bspline_transform = None

    def _command_iteration(self):
        """Callback invoked each optimizer iteration."""
        method = self.registration_method

        if method.GetOptimizerIteration() == 0:
            print(self.bspline_transform)

        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")

    def _command_multi_iteration(self):
        """Callback invoked before starting a multi-resolution level."""
        method = self.registration_method
        if method.GetCurrentLevel() > 0:
            print(
                "Optimizer stop condition: "
                f"{method.GetOptimizerStopConditionDescription()}"
            )
            print(f" Iteration: {method.GetOptimizerIteration()}")
            print(f" Metric value: {method.GetMetricValue()}")

        print("--- Resolution Changing ---")

    @staticmethod
    def extract_2d(img_3d):
        """Extract 2D slice and normalize directions for 2D registration"""
        if img_3d.GetDimension() == 3:
            size = list(img_3d.GetSize())
            size[2] = 0
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex([0, 0, 0])
            extractor.SetDirectionCollapseToStrategy(
                sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOGUESS
            )
            img_2d = extractor.Execute(img_3d)
        else:
            img_2d = img_3d

        img_2d.SetDirection((1.0, 0.0, 0.0, 1.0))
        img_2d.SetOrigin((0.0, 0.0))

        return img_2d

    @staticmethod
    def convert_2d_dvf_to_3d_like(dvf_2d, reference_3d):
        """
        Embed 2D displacement field (2 components) into a 3D vector volume
        (3 components: dx, dy, 0) preserving original 3D spatial metadata.
        """
        # Shape: (H, W, 2)
        arr_2d = sitk.GetArrayFromImage(dvf_2d)

        # Pad z-axis displacement with zeros -> shape becomes (1, H, W, 3)
        z_zeros = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 1), dtype=arr_2d.dtype)
        arr_3d_vec = np.concatenate([arr_2d, z_zeros], axis=-1)[None, ...]

        # Convert back to SimpleITK vector image
        dvf_3d = sitk.GetImageFromArray(arr_3d_vec, isVector=True)
        dvf_3d.SetSpacing(reference_3d.GetSpacing())
        dvf_3d.SetOrigin(reference_3d.GetOrigin())
        dvf_3d.SetDirection(reference_3d.GetDirection())

        return dvf_3d

    def register(self, fixed_2d, moving_2d):
        """Executes the BSpline registration pipeline in 2D."""
        self.bspline_transform = sitk.BSplineTransformInitializer(
            fixed_2d, self.mesh_size
        )

        print(
            f"Initial Number of Parameters: {self.bspline_transform.GetNumberOfParameters()}"
        )

        R = self.registration_method
        R.SetMetricAsJointHistogramMutualInformation()
        R.SetOptimizerAsGradientDescentLineSearch(
            self.learning_rate,
            self.numberOfIterations,
            convergenceMinimumValue=1e-4,
            convergenceWindowSize=5,
        )
        R.SetInterpolator(sitk.sitkLinear)
        R.SetInitialTransformAsBSpline(
            self.bspline_transform, inPlace=True, scaleFactors=self.scale_factors
        )

        R.SetShrinkFactorsPerLevel(self.shrink_factors)
        R.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)

        R.AddCommand(sitk.sitkIterationEvent, self._command_iteration)
        R.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, self._command_multi_iteration
        )

        out_tx = R.Execute(fixed_2d, moving_2d)

        return out_tx

    def register_arrays(self, fixed_np, moving_np):
        """
        Runs the same B-spline pipeline as `run()`, but on already-preprocessed
        numpy arrays (as produced by `preprocess_dataset`) with spacing=1,
        origin=0, identity direction - so physical space and pixel-index space
        coincide, and the resulting DVF is pixel-space displacement, directly
        comparable to the deep-learning models' predicted DVFs (both are then
        used the same way by EvaluationMetric/map_coordinates).

        SimpleITK's Execute(fixed, moving) returns a transform T used to pull
        moving into fixed's frame (moving(x + D(x)) ~ fixed(x)), the opposite
        direction from this codebase's pred_dvf convention
        (fixed(x + pred_dvf(x)) ~ moving(x), matching gt_dvf/the DL models) -
        negated below to match. Verified empirically: warping fixed with the
        negated field reduces MSE against moving below the no-registration
        baseline; the un-negated field made it worse than no registration.

        Returns
        -------
        pred_dvf : np.ndarray (H, W, 2)
        """
        fixed_2d = sitk.GetImageFromArray(fixed_np.astype(np.float32))
        moving_2d = sitk.GetImageFromArray(moving_np.astype(np.float32))
        for img in (fixed_2d, moving_2d):
            img.SetSpacing((1.0, 1.0))
            img.SetOrigin((0.0, 0.0))
            img.SetDirection((1.0, 0.0, 0.0, 1.0))

        out_tx = self.register(fixed_2d, moving_2d)

        dvf_filter = sitk.TransformToDisplacementFieldFilter()
        dvf_filter.SetReferenceImage(fixed_2d)
        dvf_2d = dvf_filter.Execute(out_tx)

        return -sitk.GetArrayFromImage(dvf_2d)

    def run(self, fixed_path, moving_path, output_tx_path=None, output_dvf_path=None):
        """High-level pipeline: loads images, registers 2D slices, and outputs the DVF"""

        fixed_3d = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
        moving_3d = sitk.ReadImage(moving_path, sitk.sitkFloat32)

        fixed_2d = self.extract_2d(fixed_3d)
        moving_2d = self.extract_2d(moving_3d)

        out_tx = self.register(fixed_2d, moving_2d)

        # 1. Compute 2D Displacement Field from B-spline transform
        dvf_filter = sitk.TransformToDisplacementFieldFilter()
        dvf_filter.SetReferenceImage(fixed_2d)
        dvf_2d = dvf_filter.Execute(out_tx)

        # 2. Convert 2D DVF to 3D Vector Image (dx, dy, 0.0)
        dvf_3d = self.convert_2d_dvf_to_3d_like(dvf_2d, fixed_3d)

        # 3. Save outputs
        if output_tx_path:
            sitk.WriteTransform(out_tx, output_tx_path)

        if output_dvf_path:
            sitk.WriteImage(dvf_3d, output_dvf_path)

        return out_tx, dvf_3d
