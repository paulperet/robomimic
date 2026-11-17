"""Check the isolated Kaggle CUDA environment and optional headless renderer."""

import argparse
import os


def check_cuda():
    import torch
    import torchvision
    from diffusers.training_utils import EMAModel
    from transformers import CLIPTextModel, CLIPTokenizer

    import robomimic.algo  # register algorithms and check their imports
    from robomimic.utils.torch_utils import get_torch_device

    print(f"PyTorch {torch.__version__}; torchvision {torchvision.__version__}")
    print(f"CUDA wheel runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Enable a GPU accelerator in Kaggle settings.")

    device = get_torch_device(try_to_use_cuda=True)
    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA, selected {device}.")
    properties = torch.cuda.get_device_properties(device)
    print(f"Training device: {device} ({properties.name}, {properties.total_memory / 2**30:.1f} GiB)")
    print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
    print(f"Wheel architectures: {torch.cuda.get_arch_list()}")

    # Exercise the same vision backbone used by image BC, including cuDNN backward.
    # CUDA availability alone does not prove the wheel can execute on this GPU.
    network = torchvision.models.resnet18(weights=None).to(device).train()
    image = torch.rand(2, 3, 84, 84, device=device)
    loss = network(image).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss).item() or not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in network.parameters()
    ):
        raise RuntimeError("The CUDA vision forward/backward check produced invalid values.")
    print("CUDA ResNet18 forward/backward and algorithm imports passed.")


def check_renderer():
    import sys
    from types import SimpleNamespace

    import numpy as np
    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
    from robomimic.envs.env_robosuite import EnvRobosuite

    if sys.platform.startswith("linux"):
        import egl_probe
        print(f"EGL probe devices: {egl_probe.get_available_devices()}")

    # Exercise robosuite's actual rendering path, including the jump from policy
    # observations to 512px video. A small mujoco.Renderer test misses this bug.
    sim = MjSim.from_xml_string("""
        <mujoco>
          <visual><global offwidth="84" offheight="84"/></visual>
          <worldbody>
            <light pos="0 0 3"/>
            <camera name="agentview" pos="0 -3 2" xyaxes="1 0 0 0 .55 .83"/>
            <geom type="plane" size="2 2 .1" rgba=".3 .3 .3 1"/>
            <geom type="sphere" pos="0 0 .3" size=".3" rgba="1 .1 .1 1"/>
          </worldbody>
        </mujoco>
    """)
    try:
        context = MjRenderContextOffscreen(sim, device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")))
        original_con = context.con
        # Use the real robomimic wrapper with a minimal scene, without loading
        # a dataset, policy or full robot environment.
        env = EnvRobosuite.__new__(EnvRobosuite)
        env.env = SimpleNamespace(sim=sim)
        env._init_kwargs = {"camera_names": ["agentview"]}
        initial = None
        for size in (84, 512, 84, 512):
            pixels = env.render(mode="rgb_array", height=size, width=size)
            if pixels.shape != (size, size, 3) or not np.isfinite(pixels).all() or pixels.std() == 0:
                raise RuntimeError("MuJoCo returned an invalid or uniform image.")
            if context.con is not original_con:
                raise RuntimeError("Video resizing unexpectedly replaced MjrContext.")
            if size == 84:
                if initial is None:
                    initial = pixels.copy()
                else:
                    # MSAA rounding can move edge pixels by one 8-bit level
                    # when the backing framebuffer changes dimensions.
                    np.testing.assert_allclose(initial.astype(np.int16), pixels.astype(np.int16), rtol=0, atol=1)
        print("Robosuite rendering passed: 84 -> 512 -> 84 -> 512 pixels; context retained.")
    finally:
        sim.free()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-render", action="store_true", help="Check training without simulator rendering.")
    args = parser.parse_args()

    # Override the notebook kernel's inherited inline backend before importing
    # robomimic.algo, which imports matplotlib.pyplot during registration.
    os.environ["MPLBACKEND"] = "Agg"

    # These must be set before importing MuJoCo or PyOpenGL in this process.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    check_cuda()
    if not args.skip_render:
        try:
            check_renderer()
        except Exception as error:
            raise RuntimeError(
                "Headless rendering failed. To train from the stored HDF5 images without "
                "simulation, set experiment.rollout.enabled=False and "
                "experiment.render_video=False in the training config; held-out validation "
                "still runs, but rollout success metrics and videos will be unavailable."
            ) from error


if __name__ == "__main__":
    main()
