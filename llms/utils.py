import argparse
from typing import Any
import math

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from PIL import Image as PILImage

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    num_outputs: int = 1,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
                num_outputs=num_outputs,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) or isinstance(p, PILImage.Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
            # n=1  # Gemini only supports 1 output for now
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response

def call_llm_with_self_certainty(
    lm_config: lm_config.LMConfig,
    prompt: str,
    num_outputs: int = 5,
    use_cross_entropy: bool = True,
) -> list[tuple[str, float]]:
    assert lm_config.provider == "openai"
    assert lm_config.mode == "completion", "Self-certainty computation is currently implemented only for completion mode"

    import openai

    openai.api_key = lm_config.gen_config["api_key"]
    results = []

    for _ in range(num_outputs):
        response = openai.Completion.create(
            engine=lm_config.model,
            prompt=prompt,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
            logprobs=lm_config.gen_config.get("logprobs", 20),
            echo=False,
        )

        text = response["choices"][0]["text"]
        tokens = response["choices"][0]["logprobs"]["tokens"]
        token_logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        V = len(token_logprobs)
        n = 1

        # cross-entropy self-certainty
        if use_cross_entropy:
            ce_score = - (1 / (n * V)) * sum([lp for lp in token_logprobs if lp is not None])
            results.append((text.strip(), ce_score))
        else:
            sc_score = - (1 / (n * V)) * sum([
                math.log(V * math.exp(lp)) for lp in token_logprobs if lp is not None
            ])
            results.append((text.strip(), sc_score))

    return results
