"""
HumanCellAI - Ollama Local Model Test
"""

from __future__ import annotations

import sys

import ollama


MODEL_NAME = "qwen2:1.5b"


def main() -> None:
    print("=" * 72)
    print("HumanCellAI - Ollama Local Model Test")
    print("=" * 72)
    print("Model:", MODEL_NAME)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise scientific assistant. "
                        "Answer clearly and do not invent unsupported claims."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Explain single-cell RNA sequencing "
                        "in two clear sentences."
                    ),
                },
            ],
            options={
                "temperature": 0.1,
            },
        )

        answer = response["message"]["content"].strip()

        print("\n" + "=" * 72)
        print("Local model connection successful")
        print("=" * 72)
        print("\nResponse:\n")
        print(answer)

        if not answer:
            print("\nWARNING: Ollama returned an empty response.")

    except ConnectionError:
        print(
            "\nERROR: Ollama is not running.\n"
            "Open Ollama or run the model once using:\n"
            '"%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe" run qwen2:1.5b'
        )
        sys.exit(1)

    except ollama.ResponseError as error:
        print("\nOLLAMA ERROR:")
        print(error)

        if getattr(error, "status_code", None) == 404:
            print(
                f"\nThe model is missing. Run:\n"
                f'"%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe" '
                f"pull {MODEL_NAME}"
            )

        sys.exit(1)

    except Exception as error:
        print("\nUNEXPECTED ERROR:")
        print(type(error).__name__, "-", error)
        sys.exit(1)


if __name__ == "__main__":
    main()