from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
from pydantic import BaseModel, Field

from core.state import Task, TaskResult, ConversationState
from tools.shared import toolify_agent, _cs

# Where to write output registered images & transforms
OUTDIR = "registration_outputs"


# ============================================================
# Args schema (for toolify_agent + LLM)
# ============================================================

class ImageRegistrationArgs(BaseModel):
    fixed_image_path: str = Field(
        ...,
        description=(
            "Path to the fixed/reference image on disk (e.g., NIfTI, MHA, or other SimpleITK-supported format). "
            "This image defines the target space."
        ),
    )
    moving_image_path: str = Field(
        ...,
        description=(
            "Path to the moving image to be registered to the fixed image. "
            "Must have a similar modality/contrast for best results."
        ),
    )
    transform_type: str = Field(
        "rigid",
        description="Type of transform: 'rigid' or 'affine'. Default is 'rigid'.",
    )
    output_basename: Optional[str] = Field(
        None,
        description=(
            "Optional base name for output files (without extension). "
            "If not provided, one will be constructed from fixed/moving filenames."
        ),
    )


# ============================================================
# Core registration logic (Sync – run in a thread)
# ============================================================

def _run_registration_sync(
    fixed_image_path: str,
    moving_image_path: str,
    transform_type: str,
    output_basename: Optional[str],
    out_root: Path,
) -> dict:
    """
    Synchronous registration using SimpleITK (ITK-based).
    This is called inside a thread via asyncio.to_thread().
    """

    fixed_path = Path(fixed_image_path).expanduser().resolve()
    moving_path = Path(moving_image_path).expanduser().resolve()

    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed image does not exist: {fixed_path}")
    if not moving_path.exists():
        raise FileNotFoundError(f"Moving image does not exist: {moving_path}")

    # Load images
    fixed = sitk.ReadImage(str(fixed_path))
    moving = sitk.ReadImage(str(moving_path))

    # Choose transform type
    transform_type = (transform_type or "rigid").lower()
    if transform_type not in {"rigid", "affine"}:
        raise ValueError(f"Unsupported transform_type '{transform_type}'. Use 'rigid' or 'affine'.")

    # Initial transform (centered)
    if transform_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform() if fixed.GetDimension() == 3 else sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:  # affine
        if fixed.GetDimension() == 3:
            initial_transform = sitk.AffineTransform(3)
        else:
            initial_transform = sitk.AffineTransform(2)
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            initial_transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

    # Set up registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute
    final_transform = registration_method.Execute(fixed, moving)

    # Resample moving into fixed space
    registered_moving = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    # Decide output filenames
    out_root.mkdir(parents=True, exist_ok=True)

    if output_basename:
        base = output_basename
    else:
        base = f"reg_{fixed_path.stem}_to_{moving_path.stem}"

    registered_path = out_root / f"{base}_registered.nii.gz"
    transform_path = out_root / f"{base}_transform.tfm"

    sitk.WriteImage(registered_moving, str(registered_path))
    sitk.WriteTransform(final_transform, str(transform_path))

    return {
        "registered_image": str(registered_path),
        "transform": str(transform_path),
        "fixed_image": str(fixed_path),
        "moving_image": str(moving_path),
        "transform_type": transform_type,
        "metric_value": float(registration_method.GetMetricValue()),
        "iterations": int(registration_method.GetOptimizerIteration()),
    }


# ============================================================
# Agent implementation
# ============================================================

class ImageRegistrationAgent:
    name = "image_registration"
    model = None  # No LLM; this is purely a tool/compute agent.

    async def run(self, task: Task, state: ConversationState) -> TaskResult:
        kw = task.kwargs or {}

        fixed_image_path: str = kw.get("fixed_image_path")
        moving_image_path: str = kw.get("moving_image_path")
        transform_type: str = kw.get("transform_type", "rigid")
        output_basename: Optional[str] = kw.get("output_basename")

        if not fixed_image_path or not moving_image_path:
            return TaskResult(
                output=(
                    "Both fixed_image_path and moving_image_path are required for registration. "
                    "The fixed image is the reference; the moving image will be aligned to it."
                ),
                artifacts={},
            )

        out_root = Path(OUTDIR).expanduser().resolve()

        try:
            # Offload heavy registration to a worker thread
            res = await asyncio.to_thread(
                _run_registration_sync,
                fixed_image_path,
                moving_image_path,
                transform_type,
                output_basename,
                out_root,
            )
        except Exception as e:
            return TaskResult(
                output=f"ImageRegistration error: {e}",
                artifacts={},
            )

        summary = (
            "Registration completed.\n"
            f"- Fixed image: {res['fixed_image']}\n"
            f"- Moving image: {res['moving_image']}\n"
            f"- Transform type: {res['transform_type']}\n"
            f"- Metric value (final): {res['metric_value']:.4f}\n"
            f"- Optimizer iterations: {res['iterations']}\n"
            f"- Registered image: {res['registered_image']}\n"
            f"- Transform file: {res['transform']}"
        )

        return TaskResult(
            output={
                "text": summary,
                "tool": self.name,
                "fixed_image": res["fixed_image"],
                "moving_image": res["moving_image"],
                "transform_type": res["transform_type"],
                "metric_value": res["metric_value"],
                "iterations": res["iterations"],
                "registered_image": res["registered_image"],
                "transform": res["transform"],
                "output_dir": str(out_root),
            },
            artifacts={
                "files": [res["registered_image"], res["transform"]],
                "output_dir": str(out_root),
            },
        )


# ============================================================
# Config + tool wrapper (mirrors idc_download style)
# ============================================================

_REG: Optional[ImageRegistrationAgent] = None


def configure_image_registration_tool():
    """Call this once at startup to configure the image registration tool."""
    global _REG
    _REG = ImageRegistrationAgent()


@toolify_agent(
    name="image_registration",
    description=(
        "Perform image registration using ITK (via SimpleITK). "
        "The user must provide fixed_image_path (reference) and moving_image_path. "
        "The moving image will be aligned to the fixed image. "
        "Supports 'rigid' and 'affine' transforms."
    ),
    args_schema=ImageRegistrationArgs,
    timeout_s=900,  # 15 minutes; registration can be heavy
)
async def image_registration_runner(
    fixed_image_path: str,
    moving_image_path: str,
    transform_type: str = "rigid",
    output_basename: Optional[str] = None,
):
    if _REG is None:
        raise RuntimeError(
            "Image registration tool not configured. "
            "Call configure_image_registration_tool() first."
        )

    kwargs = {
        "fixed_image_path": fixed_image_path,
        "moving_image_path": moving_image_path,
        "transform_type": transform_type,
        "output_basename": output_basename,
    }
    task = Task(
        user_msg="Register moving image to fixed image",
        files=[],
        kwargs=kwargs,
    )
    return await _REG.run(task, _cs())
