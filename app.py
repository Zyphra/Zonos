from datetime import timedelta

from union import Resources, ImageSpec
from union.app import App, ScalingMetric

zonos_image = ImageSpec(
    base_image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    requirements="uv.lock",
    apt_packages=["espeak-ng"],
    builder="unionai",
)


zonos_app = App(
    name="zonos-gradio",
    container_image=zonos_image,
    limits=Resources(cpu="2", mem="8Gi", gpu="1"),
    port=8080,
    include=["./gradio_interface.py"],
    args=["python", "gradio_interface.py"],
    min_replicas=0,
    max_replicas=1,
    scaledown_after=timedelta(minutes=2),
)