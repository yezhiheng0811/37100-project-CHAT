export interface ToolConfig {
  [key: string]: {
    enabled: boolean;
    prefix: string;
  };
}


export interface ContentConfig {
  [key: string]: {
    chunkSize: number;
    chunkOverlap: number;
    selectors: string[];
    selectorsAll: string[];
  };
}

const DEFAULT_CHUCK_SIZE = 1000;
const DEFAULT_CHUNK_OVERLAP = 0;

export const defaultContentConfig: ContentConfig = {
  default: {
    chunkSize: DEFAULT_CHUCK_SIZE,
    chunkOverlap: DEFAULT_CHUNK_OVERLAP,
    selectors: ["body"],
    selectorsAll: [],
  },
};

export const DEFAULT_HOST = "http://localhost:11434";
export const DEFAULT_MODEL = "37100-finetuned:latest";
export const DEFAULT_KEEP_ALIVE = "60m";
export const DEFAULT_CONTENT_CONFIG = JSON.stringify(
  defaultContentConfig,
  null,
  2,
);
export const DEFAULT_VECTOR_STORE_TTL_MINS = 60;
export const DEFAULT_TOOL_CONFIG: ToolConfig = {
  Calculator: {
    enabled: true,
    prefix: "calculate:",
  },
};

export interface ChatOptions {
  ollamaModel: string;
  ollamaHost: string;
  contentConfig: ContentConfig;
  vectorStoreTTLMins: number;
  toolConfig: ToolConfig;
}

export const getChatOptions = (): ChatOptions => {
  return {
    ollamaModel: DEFAULT_MODEL,
    ollamaHost: DEFAULT_HOST,
    contentConfig: JSON.parse(
      DEFAULT_CONTENT_CONFIG,
    ) as ContentConfig,
    vectorStoreTTLMins: DEFAULT_VECTOR_STORE_TTL_MINS,
    toolConfig: DEFAULT_TOOL_CONFIG,
  };
};

/**
 * Ollama API connectivity check.
 *
 * @param {string} host Ollama host.
 * @return {[boolean, string[], string]} Tuple of connected status, available models, and an optional error message.
 */
export const apiConnected = async (
  host: string,
): Promise<[boolean, string[], string]> => {
  let resp;
  const errMsg = "Unable to connect to Ollama API. Check Ollama server.";

  try {
    resp = await fetch(`${host}/api/tags`);
  } catch (e) {
    return [false, [], errMsg];
  }

  if (resp.ok) {
    const data = await resp.json();
    const modelOptions = data.models.map(
      (model: { name: string }) => model.name,
    );
    // successfully connected
    return [true, modelOptions, ""];
  }

  return [false, [], errMsg];
};