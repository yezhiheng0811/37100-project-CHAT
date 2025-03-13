# Ollama Server

A local Ollama server is needed for the embedding database and LLM inference. Download and install Ollama and the CLI [here](https://ollama.com/).

## macOS

Run `launchctl setenv` to set `OLLAMA_ORIGINS` in order to allow requests originating from the Chrome extension:

```bash
launchctl setenv OLLAMA_ORIGINS "chrome-extension://*"
```

## Load Fine-tuned Model

To load our fine-tuned model, navigate to the project directory and run the following command:

```bash
cd /path/to/your/project
ollama create 37100-finetuned -f finetuned_model/Modelfile
```

## Run Ollama Server

If the server is not running, start it with:

```bash
OLLAMA_ORIGINS=chrome-extension://* ollama serve
```

# Chrome Extension

## Installation

Our Chrome extension can be loaded from the `chrome_extension/dist` directory. To learn how to load an unpacked extension, please following the [this tutorial](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked). 

## Acknowledgement

This extension is inspired from [this article](https://medium.com/@andrewnguonly/local-llm-in-the-browser-powered-by-ollama-236817f335da) and built upon [the Lumos extension.](https://github.com/andrewnguonly/Lumos)


# Example Usage
URL: [The Guardian](https://www.theguardian.com/us-news/2025/mar/08/america-vetoes-g7-proposal-to-combat-russias-shadow-fleet-of-oil-tankers)

Question: Why did the US veto Canada’s proposal to combat Russia’s shadow fleet of oil tankers?