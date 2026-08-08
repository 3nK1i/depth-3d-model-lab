from __future__ import annotations

from core.contracts import ImageSize
from core.reconstruction import (
    CameraGeometryPurpose,
    camera_geometry_profile_from_json,
    camera_geometry_profile_to_json,
    estimate_camera_geometry_from_horizontal_fov,
)


def test_estimated_camera_profile_round_trip_is_stable_across_common_resolutions_and_fovs() -> None:
    sizes = (ImageSize(320, 240), ImageSize(640, 480), ImageSize(1280, 720), ImageSize(1920, 1080))
    fovs = (45.0, 60.0, 70.0, 80.0, 100.0)
    for iteration in range(20):
        for size in sizes:
            for fov in fovs:
                profile = estimate_camera_geometry_from_horizontal_fov(
                    profile_id=f"stress-{iteration}-{size.width}x{size.height}-{int(fov)}",
                    camera_id="stress-camera",
                    image_size=size,
                    horizontal_fov_degrees=fov,
                    focal_length_relative_uncertainty=0.25,
                )
                restored = camera_geometry_profile_from_json(camera_geometry_profile_to_json(profile))
                assert restored.as_dict() == profile.as_dict()
                assert restored.assess(CameraGeometryPurpose.VISUAL_TRACKING).allowed is True
                assert restored.camera.intrinsics.matrix[0, 0] > 0.0
