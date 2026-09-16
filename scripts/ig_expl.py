from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoModelForImageTextToText, AutoProcessor


def build_prompt(amr: str) -> str:
    return "\n".join([
        "Your job is to determine whether an Abstract Meaning Representation "
        "(AMR) correctly corresponds to the provided image.",
        "The AMR describes a semantic interpretation involving entities, events, "
        "predicate-argument relations, attachment, coreference, scope, or coordination.",
        "Compare the semantic structure expressed by the AMR with the visual scene in the image.",
        "Answer Yes only if the image supports the semantic relations expressed by the AMR.",
        "Answer No if any semantically important relation in the AMR is inconsistent with the image.",
        "",
        "AMR:",
        amr,
        "",
        'Return only "Yes" or "No". Do not provide explanations, Markdown, comments, or additional text.',
    ])


def load_model(card):
    processor = AutoProcessor.from_pretrained(
        card,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        card,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, processor


def prepare_inputs(processor, image, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )


def get_yes_no_ids(tokenizer):
    def find(candidates):
        for text in candidates:
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if len(ids) == 1:
                return ids[0]

        raise ValueError(
            f"Single token not found: {candidates}"
        )

    return (
        find(["Yes", " Yes"]),
        find(["No", " No"]),
    )


def score(model, inputs, yes_id, no_id):
    logits = model(
        **inputs,
        use_cache=False,
    ).logits[:, -1]

    return (
        logits[:, yes_id]
        - logits[:, no_id]
    )


def find_amr_span(tokenizer, input_ids, amr):
    full_ids = input_ids[0].tolist()

    amr_ids = tokenizer.encode(
        amr,
        add_special_tokens=False,
    )

    for start in range(
        len(full_ids) - len(amr_ids) + 1
    ):
        if full_ids[
            start:start + len(amr_ids)
        ] == amr_ids:
            return (
                start,
                start + len(amr_ids),
                amr_ids,
            )

    raise ValueError(
        "AMR not found in input."
    )


# ============================================================
# Image IG
# ============================================================

def compute_image_ig(
    model,
    inputs,
    yes_id,
    no_id,
    steps,
):
    # Qwen uses [patches, features].
    # Add fake batch dimension for Captum.
    pixels = (
        inputs["pixel_values"]
        .detach()
        .unsqueeze(0)
    )

    fixed = dict(inputs)

    def forward(x):
        current = dict(fixed)
        current["pixel_values"] = x.squeeze(0)

        return score(
            model,
            current,
            yes_id,
            no_id,
        )

    ig = IntegratedGradients(forward)

    attr = ig.attribute(
        pixels,
        baselines=torch.zeros_like(pixels),
        n_steps=steps,
        internal_batch_size=1,
    )

    return attr.squeeze(0)


def make_heatmap(attr, inputs):
    # Signed:
    # + -> Yes
    # - -> No
    patch_scores = (
        attr.float()
        .sum(dim=-1)
    )

    t, h, w = [
        int(x)
        for x in inputs[
            "image_grid_thw"
        ][0]
    ]

    heatmap = (
        patch_scores
        .reshape(t, h, w)
        .mean(dim=0)
        .detach()
        .cpu()
        .numpy()
    )

    # Visualization only.
    return gaussian_filter(
        heatmap,
        sigma=1.5,
    )


def save_heatmap(image, heatmap, path):
    magnitude = np.abs(heatmap)

    limit = max(
        float(magnitude.max()),
        1e-8,
    )

    threshold = np.quantile(
        magnitude,
        0.70,
    )

    alpha = np.clip(
        (magnitude - threshold)
        / (limit - threshold + 1e-8),
        0,
        1,
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.imshow(image)

    overlay = ax.imshow(
        heatmap,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        alpha=alpha * 0.75,
        extent=(
            0,
            image.width,
            image.height,
            0,
        ),
    )

    ax.axis("off")

    fig.colorbar(
        overlay,
        ax=ax,
        label="No ← IG → Yes",
    )

    fig.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# AMR IG
# ============================================================

def compute_amr_ig(
    model,
    processor,
    inputs,
    amr,
    yes_id,
    no_id,
    steps,
):
    tokenizer = processor.tokenizer
    input_ids = inputs["input_ids"]

    start, end, _ = find_amr_span(
        tokenizer,
        input_ids,
        amr,
    )

    baseline = input_ids.clone()

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    # Keep image tokens and instructions.
    # Remove only AMR content.
    baseline[:, start:end] = pad_id

    fixed = dict(inputs)

    def forward(ids):
        current = dict(fixed)
        current["input_ids"] = ids

        return score(
            model,
            current,
            yes_id,
            no_id,
        )

    lig = LayerIntegratedGradients(
        forward,
        model.get_input_embeddings(),
    )

    return lig.attribute(
        input_ids,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=1,
    )


def get_amr_units(amr):
    # Concepts + relations only.
    # Variable names and parentheses are ignored.
    return re.findall(
        r":[A-Za-z0-9_-]+"
        r"|(?<=/\s)[A-Za-z][A-Za-z0-9_-]*",
        amr,
    )


def save_amr_ig(
    tokenizer,
    input_ids,
    attribution,
    amr,
    path,
):
    # Signed scalar attribution per tokenizer token.
    scores = (
        attribution.float()
        .sum(dim=-1)[0]
        .detach()
        .cpu()
    )

    start, _, amr_ids = find_amr_span(
        tokenizer,
        input_ids,
        amr,
    )

    scores = scores[
        start:start + len(amr_ids)
    ]

    results = []
    pos = 0

    for unit in get_amr_units(amr):
        found = False

        for text in [
            unit,
            " " + unit,
        ]:
            unit_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            for i in range(
                pos,
                len(amr_ids)
                - len(unit_ids)
                + 1,
            ):
                if (
                    amr_ids[
                        i:i + len(unit_ids)
                    ]
                    == unit_ids
                ):
                    value = scores[
                        i:i + len(unit_ids)
                    ].sum().item()

                    results.append(
                        (unit, value)
                    )

                    pos = i + len(unit_ids)
                    found = True
                    break

            if found:
                break

    max_abs = max(
        (
            abs(v)
            for _, v in results
        ),
        default=1.0,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "# + = Yes, - = No\n\n"
        )

        for unit, value in results:
            f.write(
                f"{value / max_abs:+.4f}\t"
                f"{unit}\n"
            )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-27B",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--amr",
        required=True,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ig"),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, processor = load_model(
        args.model
    )

    device = next(
        model.parameters()
    ).device

    image = Image.open(
        args.image
    ).convert("RGB")

    inputs = prepare_inputs(
        processor,
        image,
        build_prompt(args.amr),
    )

    inputs = {
        k: (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in inputs.items()
    }

    yes_id, no_id = get_yes_no_ids(
        processor.tokenizer
    )

    with torch.no_grad():
        value = score(
            model,
            inputs,
            yes_id,
            no_id,
        ).item()

    print(
        f"Prediction: "
        f"{'Yes' if value > 0 else 'No'} "
        f"({value:.3f})"
    )

    # Image
    image_attr = compute_image_ig(
        model,
        inputs,
        yes_id,
        no_id,
        args.steps,
    )

    heatmap = make_heatmap(
        image_attr,
        inputs,
    )

    save_heatmap(
        image,
        heatmap,
        args.output / "image_ig.png",
    )

    # AMR
    amr_attr = compute_amr_ig(
        model,
        processor,
        inputs,
        args.amr,
        yes_id,
        no_id,
        args.steps,
    )

    save_amr_ig(
        processor.tokenizer,
        inputs["input_ids"],
        amr_attr,
        args.amr,
        args.output / "amr_ig.txt",
    )

    print(
        f"Saved to {args.output}"
    )


if __name__ == "__main__":
    main()