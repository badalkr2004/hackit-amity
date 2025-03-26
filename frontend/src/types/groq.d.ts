declare module 'groq-sdk' {
  interface GroqOptions {
    messages: Array<{
      role: string;
      content: string;
    }>;
    model: string;
    temperature: number;
    max_completion_tokens: number;
    top_p: number;
    stream: boolean;
    stop: null | string[];
  }

  interface GroqResponse {
    choices: Array<{
      delta: {
        content?: string;
      };
    }>;
  }

  interface GroqConfig {
    apiKey: string;
    model?: string;
    dangerouslyAllowBrowser?: boolean;
  }

  class Groq {
    constructor(config: GroqConfig);
    chat: {
      completions: {
        create(options: GroqOptions): AsyncIterable<GroqResponse>;
      };
    };
  }

  export default Groq;
} 