"""Build an RSL-RL checkpoint (.pt) whose actor comes from an exported ONNX policy.

The ONNX file exported by ``play.py`` only holds the actor MLP and its observation normalizer. It has no critic, no
action noise and no optimizer state, so it cannot be passed to ``train.py --resume`` directly. This script copies the
actor weights (and the normalizer statistics, when present) from the ONNX file into a "template" checkpoint produced
by ``train.py`` for the target task. Everything else (critic, critic normalizer) is kept from the template, i.e. it is
randomly initialized.

It does not need Isaac Sim, only ``torch``, ``onnx`` and ``numpy``:

    python scripts/rsl_rl/onnx_to_checkpoint.py \
        --onnx <POLICY>.onnx \
        --template logs/rsl_rl/<EXPERIMENT>/<TEMPLATE_RUN>/model_0.pt \
        --output logs/rsl_rl/<EXPERIMENT>/from_onnx/model_0.pt

See docs/finetuning_onnx.md for the full procedure.
"""

import argparse
import copy
import os

import numpy as np
import onnx
import torch
from onnx import numpy_helper

# default of rsl_rl.networks.EmpiricalNormalization: y = (x - mean) / (std + eps)
NORMALIZER_EPS = 1e-2


def _constants(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    """Collect every constant tensor of the graph (initializers and Constant nodes)."""
    consts = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    consts[node.output[0]] = numpy_helper.to_array(attr.t)
    # fold "Add(const, const)", which is how "std + eps" appears when the exporter does not fold it
    for node in graph.node:
        if node.op_type == "Add" and all(name in consts for name in node.input):
            consts[node.output[0]] = consts[node.input[0]] + consts[node.input[1]]
    return consts


def read_onnx_policy(path: str) -> dict:
    """Extract the linear layers and the input normalization of a feed-forward ONNX policy.

    Returns:
        A dictionary with ``layers`` (list of ``(weight [out, in], bias [out])``), ``mean`` and ``divisor`` (the
        normalizer constants, or None when the ONNX file has no normalizer) and ``activations`` (set of op types).
    """
    model = onnx.load(path)
    graph = model.graph
    consts = _constants(graph)
    const_names = set(consts)
    graph_inputs = [i.name for i in graph.input if i.name not in const_names]
    if len(graph_inputs) != 1:
        raise ValueError(
            f"Expected a feed-forward policy with a single input, got inputs {graph_inputs}."
            " Recurrent (LSTM/GRU) policies are not supported."
        )

    layers: list[list[np.ndarray | None]] = []
    mean = divisor = None
    activations = set()
    pending_matmul: dict[str, int] = {}  # MatMul output name -> index in `layers`

    for node in graph.node:
        op = node.op_type
        if op == "Gemm":
            attrs = {a.name: a for a in node.attribute}
            weight = consts[node.input[1]].astype(np.float32)
            if "alpha" in attrs and attrs["alpha"].f != 1.0:
                weight = weight * attrs["alpha"].f
            if not ("transB" in attrs and attrs["transB"].i == 1):
                weight = weight.T
            bias = consts[node.input[2]].astype(np.float32) if len(node.input) > 2 else np.zeros(weight.shape[0])
            if "beta" in attrs and attrs["beta"].f != 1.0:
                bias = bias * attrs["beta"].f
            layers.append([weight, bias.reshape(-1)])
        elif op == "MatMul":
            if node.input[1] not in consts:
                raise ValueError(f"MatMul node '{node.name}' has a non-constant weight, cannot convert.")
            layers.append([consts[node.input[1]].astype(np.float32).T, None])
            pending_matmul[node.output[0]] = len(layers) - 1
        elif op == "Add" and any(name in pending_matmul for name in node.input):
            data, const = (node.input[0], node.input[1]) if node.input[0] in pending_matmul else node.input[::-1]
            layers[pending_matmul.pop(data)][1] = consts[const].astype(np.float32).reshape(-1)
        elif op == "Sub" and not layers and node.input[1] in consts:
            mean = consts[node.input[1]].astype(np.float32).reshape(1, -1)
        elif op == "Div" and not layers and node.input[1] in consts:
            divisor = consts[node.input[1]].astype(np.float32).reshape(1, -1)
        elif op in ("LSTM", "GRU", "RNN"):
            raise ValueError("Recurrent (LSTM/GRU) policies are not supported.")
        elif op not in ("Constant", "Add", "Identity", "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Cast", "Shape"):
            activations.add(op)

    if not layers:
        raise ValueError("No linear layer (Gemm or MatMul) found in the ONNX graph.")
    for index, (weight, bias) in enumerate(layers):
        if bias is None:
            layers[index][1] = np.zeros(weight.shape[0], dtype=np.float32)
    if (mean is None) != (divisor is None):
        raise ValueError("Found only half of an input normalizer (Sub without Div, or the opposite). Cannot convert.")

    return {"layers": [tuple(layer) for layer in layers], "mean": mean, "divisor": divisor, "activations": activations}


def actor_layer_keys(state_dict: dict) -> list[str]:
    """Return the 'actor.<i>' prefixes of the linear layers of the actor, in forward order."""
    indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("actor.") and key.endswith(".weight") and state_dict[key].dim() == 2
    )
    return [f"actor.{index}" for index in indices]


def main():
    parser = argparse.ArgumentParser(description="Copy the actor of an ONNX policy into an RSL-RL checkpoint.")
    parser.add_argument("--onnx", required=True, help="ONNX policy to take the actor from.")
    parser.add_argument("--template", required=True, help="RSL-RL checkpoint (model_<N>.pt) of the target task.")
    parser.add_argument("--output", required=True, help="Checkpoint to write, e.g. logs/rsl_rl/<EXP>/<RUN>/model_0.pt")
    parser.add_argument(
        "--action_std",
        type=float,
        default=None,
        help="Initial action noise (std) to store in the checkpoint. Default: keep the template value.",
    )
    parser.add_argument(
        "--normalizer_count",
        type=int,
        default=1_000_000_000,
        help=(
            "Sample count stored in the actor observation normalizer. The larger it is, the slower the statistics"
            " taken from the ONNX file drift during fine-tuning."
        ),
    )
    args = parser.parse_args()

    policy = read_onnx_policy(args.onnx)
    checkpoint = torch.load(args.template, map_location="cpu", weights_only=False)
    checkpoint = copy.deepcopy(checkpoint)
    state = checkpoint["model_state_dict"]

    # -- actor MLP
    keys = actor_layer_keys(state)
    onnx_shapes = [tuple(weight.shape) for weight, _ in policy["layers"]]
    template_shapes = [tuple(state[f"{key}.weight"].shape) for key in keys]
    if onnx_shapes != template_shapes:
        raise SystemExit(
            "[ERROR] The ONNX actor does not match the actor of the template checkpoint.\n"
            f"  ONNX layers     (out, in): {onnx_shapes}\n"
            f"  template layers (out, in): {template_shapes}\n"
            "The observation size, the action size and the hidden layers must all be the same."
        )
    for key, (weight, bias) in zip(keys, policy["layers"]):
        state[f"{key}.weight"] = torch.from_numpy(np.ascontiguousarray(weight)).to(state[f"{key}.weight"].dtype)
        state[f"{key}.bias"] = torch.from_numpy(np.ascontiguousarray(bias)).to(state[f"{key}.bias"].dtype)
    print(f"[INFO] Copied {len(keys)} actor layers: {onnx_shapes}")
    print(f"[INFO] Activations found in the ONNX graph: {sorted(policy['activations'])}")
    print("       Make sure they match 'policy.activation' of the task (the checkpoint does not store it).")

    # -- actor observation normalizer
    has_template_normalizer = "actor_obs_normalizer._mean" in state
    if policy["mean"] is not None and not has_template_normalizer:
        raise SystemExit(
            "[ERROR] The ONNX file has an observation normalizer but the template checkpoint does not.\n"
            "Train the template with 'actor_obs_normalization=True'."
        )
    if has_template_normalizer:
        if policy["mean"] is not None:
            mean = torch.from_numpy(policy["mean"])
            std = torch.from_numpy(policy["divisor"]) - NORMALIZER_EPS
            print("[INFO] Copied the observation normalizer from the ONNX file.")
        else:
            # identity: (x - 0) / (std + eps) == x
            mean = torch.zeros_like(state["actor_obs_normalizer._mean"])
            std = torch.full_like(state["actor_obs_normalizer._std"], 1.0 - NORMALIZER_EPS)
            print("[WARN] The ONNX file has no observation normalizer: the checkpoint normalizer was set to identity.")
            print("       It will start adapting to the data during training, which changes what the actor sees.")
        if mean.shape != state["actor_obs_normalizer._mean"].shape:
            raise SystemExit(
                f"[ERROR] Normalizer shape mismatch: ONNX {tuple(mean.shape)} vs template"
                f" {tuple(state['actor_obs_normalizer._mean'].shape)}."
            )
        state["actor_obs_normalizer._mean"] = mean
        state["actor_obs_normalizer._std"] = std
        state["actor_obs_normalizer._var"] = std**2
        state["actor_obs_normalizer.count"] = torch.tensor(args.normalizer_count, dtype=torch.long)

    # -- action noise
    if args.action_std is not None:
        if "std" in state:
            state["std"] = torch.full_like(state["std"], args.action_std)
        elif "log_std" in state:
            state["log_std"] = torch.full_like(state["log_std"], float(np.log(args.action_std)))
        else:
            raise SystemExit("[ERROR] --action_std was given but the checkpoint has no 'std' or 'log_std' parameter.")
        print(f"[INFO] Action noise std set to {args.action_std}.")

    # -- training state: start a fresh run (iteration 0, no Adam moments from the template)
    checkpoint["iter"] = 0
    checkpoint["infos"] = None
    if "optimizer_state_dict" in checkpoint:
        checkpoint["optimizer_state_dict"]["state"] = {}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"[INFO] Checkpoint written to: {args.output}")
    print("[INFO] The critic is NOT in the ONNX file: it keeps the (random) weights of the template.")


if __name__ == "__main__":
    main()
